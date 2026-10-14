"""CPU-only scientific invariants; no model download or GPU is required.

Extract dependency-free helpers so these checks also run on the controller Mac.
Runtime collective/optimizer validation still requires a real torchrun smoke test.
"""
import ast
import copy
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'tools/training/train_whisper_ddp.py'
module = ast.parse(SOURCE.read_text())
helpers = {'shard_indices', 'validate_resume_contract', 'merge_eval_payloads'}
namespace = {}
exec(compile(ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in helpers], type_ignores=[]), str(SOURCE), 'exec'), namespace)


class ScientificContracts(unittest.TestCase):
    def contract(self):
        return dict(batch_size=4, grad_accum=4, train_sha256='train',
                    eval_sha256='dev', full_eval_sha256='fulldev', model_dir='/model',
                    seed=42, lr=1e-5, max_steps=50000, warmup_steps=None,
                    weight_decay=.01, full_eval_every=3400)

    def test_single_to_four_gpu_preserves_effective_batch(self):
        old = self.contract()
        new = dict(old, batch_size=2, grad_accum=2, world_size=4, global_batch_size=16)
        namespace['validate_resume_contract'](old, new)
        new['grad_accum'] = 4
        with self.assertRaisesRegex(RuntimeError, 'Global batch'):
            namespace['validate_resume_contract'](old, new)

    def test_resume_rejects_data_and_scheduler_drift(self):
        old = self.contract()
        for key in ['train_sha256', 'eval_sha256', 'full_eval_sha256', 'model_dir', 'seed',
                    'lr', 'max_steps', 'warmup_steps', 'full_eval_every']:
            new = copy.deepcopy(old)
            new[key] = 'changed'
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                namespace['validate_resume_contract'](old, new)

    def test_nondivisible_and_empty_dev_shards_have_exact_coverage(self):
        for count in [1, 3, 17, 512, 4119]:
            shards = [namespace['shard_indices'](count, rank, 4) for rank in range(4)]
            flattened = sum(shards, [])
            self.assertEqual(sorted(flattened), list(range(count)))
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_merge_orders_predictions_and_weights_loss_by_tokens(self):
        # Metric scorer sees global prediction/reference pairs, never shard averages.
        seen = []
        def scorer(hyp, ref):
            seen.append((hyp, ref))
            return {'cer': .25}
        namespace['score_pairs'] = scorer
        payloads = [dict(records=[dict(audio='b', prediction='B', reference='b')],
                         tokens=3, loss_sum=6),
                    dict(records=[dict(audio='a', prediction='A', reference='a')],
                         tokens=1, loss_sum=4)]
        records, metrics = namespace['merge_eval_payloads'](payloads, ['a', 'b'])
        self.assertEqual([r['audio'] for r in records], ['a', 'b'])
        self.assertEqual(seen, [(['A', 'B'], ['a', 'b'])])
        self.assertEqual(metrics['teacher_forced_loss'], 2.5)
        self.assertEqual(metrics['utterances'], 2)

    def test_merge_rejects_duplicate_missing_or_extra_audio(self):
        namespace['score_pairs'] = lambda h, r: {'cer': 0}
        for audio, expected in [(['a', 'a'], ['a', 'b']), (['a'], ['a', 'b']),
                                (['a', 'c'], ['a', 'b']), (['a'], ['a', 'a'])]:
            payload = dict(records=[dict(audio=a, prediction=a, reference=a) for a in audio],
                           tokens=1, loss_sum=1)
            with self.subTest(audio=audio), self.assertRaisesRegex(RuntimeError, 'coverage'):
                namespace['merge_eval_payloads']([payload], expected)


if __name__ == '__main__':
    unittest.main()
