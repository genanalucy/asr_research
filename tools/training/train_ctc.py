#!/usr/bin/env python3
"""Offline XLS-R CTC baseline. No test labels or manifests are accepted."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from common import (
    append_jsonl, load_audio_16k, load_manifest, normalize_text,
    score_pairs, seed_everything, write_json,
)


def make_vocab(rows):
    chars = {c for row in rows for c in normalize_text(row['text']) if c != ' '}
    # Distinct literal vertical bars would be ambiguous with the CTC word delimiter.
    if '|' in chars:
        raise ValueError('Literal | in transcription: choose an explicit escaped orthography first')
    vocab = {'[PAD]': 0, '[UNK]': 1, '|': 2}
    vocab.update({c: i + 3 for i, c in enumerate(sorted(chars))})
    return vocab


def encode(text, vocab):
    return [vocab.get('|' if c == ' ' else c, 1) for c in normalize_text(text)]


def decode(ids, vocab):
    inverse = {v: k for k, v in vocab.items()}
    previous, output = None, []
    for token in ids:
        token = int(token)
        if token != previous and token != 0:
            output.append(' ' if token == 2 else inverse.get(token, '[UNK]'))
        previous = token
    return normalize_text(''.join(output))


def unknown_stats(rows, vocab):
    labels = [encode(r['text'], vocab) for r in rows]
    return {'characters': sum(map(len, labels)),
            'unknown_characters': sum(x.count(1) for x in labels),
            'utterances_with_unknown': sum(1 in x for x in labels)}


def manifest_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform = load_audio_16k(row)
        if isinstance(waveform, tuple):
            waveform = waveform[0]
        if isinstance(waveform, dict):
            waveform = waveform['array']
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size < 400 or not np.isfinite(waveform).all():
            raise ValueError(f'Invalid waveform: {row["audio"]}')
        return row, waveform


class Collate:
    def __init__(self, extractor, vocab):
        self.extractor, self.vocab = extractor, vocab

    def __call__(self, items):
        rows, waves = zip(*items)
        features = self.extractor(list(waves), sampling_rate=16000, padding=True,
                                  return_attention_mask=True, return_tensors='pt')
        encoded = [encode(row['text'], self.vocab) for row in rows]
        if any(not label for label in encoded):
            raise ValueError('Empty normalized transcription')
        labels = torch.full((len(rows), max(map(len, encoded))), -100, dtype=torch.long)
        for i, label in enumerate(encoded):
            labels[i, :len(label)] = torch.tensor(label)
        return dict(features), labels, list(rows)


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16,
                          enabled=device.type == 'cuda' and torch.cuda.is_bf16_supported())


def infer(model, loader, device, vocab, destination=None):
    was_training = model.training
    model.eval()
    predictions, references, records, losses = [], [], [], []
    with torch.no_grad():
        for features, labels, rows in loader:
            features = {k: v.to(device) for k, v in features.items()}
            with amp(device):
                result = model(**features, labels=labels.to(device))
            if not torch.isfinite(result.loss):
                raise RuntimeError('Non-finite evaluation loss')
            losses.extend([float(result.loss)] * len(rows))
            ids = result.logits.argmax(-1).cpu()
            lengths = model._get_feat_extract_output_lengths(features['attention_mask'].sum(-1)).cpu()
            for sequence, length, row in zip(ids, lengths, rows):
                prediction = decode(sequence[:int(length)], vocab)
                reference = normalize_text(row['text'])
                predictions.append(prediction)
                references.append(reference)
                records.append({'audio': row['audio'], 'sha256': row.get('sha256'),
                                'speaker_candidate': row.get('speaker_candidate'),
                                'prediction': prediction, 'reference': reference})
    metrics = dict(score_pairs(predictions, references))
    metrics.update({'loss': float(np.mean(losses)), 'utterances': len(records)})
    if destination:
        Path(destination).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records),
                                     encoding='utf-8')
    model.train(was_training)
    return metrics


def save_checkpoint(model, extractor, vocab, output):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    extractor.save_pretrained(output)
    write_json(output / 'vocab.json', vocab)


def checkpoint_witness(model, extractor, vocab, output, sample_batch, device, optimizer, scheduler):
    """Reload the final checkpoint and compare inference and its CTC decoding."""
    model.eval()
    features = {k: v[:1].to(device) for k, v in sample_batch[0].items()}
    with torch.no_grad(), amp(device):
        expected = model(**features).logits.detach().float().cpu()
    restored = Wav2Vec2ForCTC.from_pretrained(output, local_files_only=True).to(device)
    restored.eval()
    with torch.no_grad(), amp(device):
        actual = restored(**features).logits.detach().float().cpu()
    max_difference = float((expected - actual).abs().max())
    logits_equal = bool(torch.allclose(expected, actual, atol=0.002, rtol=0.002))
    decoding_equal = decode(expected.argmax(-1)[0], vocab) == decode(actual.argmax(-1)[0], vocab)
    restored.freeze_feature_encoder()
    restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=1e-4)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0)
    state = torch.load(output / 'training_state.pt', map_location='cpu', weights_only=True)
    restored_optimizer.load_state_dict(state['optimizer'])
    restored_scheduler.load_state_dict(state['scheduler'])
    expected_states, actual_states = optimizer.state_dict(), restored_optimizer.state_dict()
    optimizer_equal = expected_states['param_groups'] == actual_states['param_groups']
    tensor_count = exp_avg_count = 0
    for identifier, expected_state in expected_states['state'].items():
        actual_state = actual_states['state'].get(identifier, {})
        optimizer_equal &= expected_state.keys() == actual_state.keys()
        for name, expected_value in expected_state.items():
            actual_value = actual_state.get(name)
            if isinstance(expected_value, torch.Tensor):
                tensor_count += 1
                exp_avg_count += name == 'exp_avg'
                optimizer_equal &= isinstance(actual_value, torch.Tensor) and torch.equal(
                    expected_value.detach().cpu(), actual_value.detach().cpu())
            else:
                optimizer_equal &= expected_value == actual_value
    scheduler_equal = scheduler.state_dict() == restored_scheduler.state_dict()
    epoch_equal = scheduler.last_epoch == restored_scheduler.last_epoch == state['step']
    optimizer_equal = bool(optimizer_equal and exp_avg_count > 0)
    del restored_optimizer, restored_scheduler, restored, state
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'passed': logits_equal and decoding_equal and optimizer_equal and scheduler_equal and epoch_equal,
            'max_logit_difference': max_difference, 'decoded_text_equal': decoding_equal,
            'logits_allclose': logits_equal, 'optimizer_restored_equal': optimizer_equal,
            'optimizer_tensors_compared': tensor_count, 'exp_avg_tensors_compared': exp_avg_count,
            'scheduler_restored_equal': scheduler_equal, 'scheduler_epoch_equal': epoch_equal,
            'scheduler_last_epoch': scheduler.last_epoch}


def execute(args):
    output = Path(args.output)
    if (output / 'run_manifest.json').exists():
        raise ValueError('Output already contains a run; use a new directory to preserve evidence')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    started = time.time()
    train_rows, dev_rows = load_manifest(args.train_manifest), load_manifest(args.dev_manifest)
    vocab_rows = load_manifest(args.vocab_manifest)
    if not train_rows or not dev_rows or not vocab_rows:
        raise ValueError('All manifests must be non-empty')
    vocab = make_vocab(vocab_rows)
    if args.mode == 'sanity':
        # Deterministic, diverse short examples; never draw M0 examples from dev/test.
        train_rows = sorted((r for r in train_rows if 1 <= float(r['duration']) <= 10),
                            key=lambda r: (float(r['duration']), str(r['audio'])))
        seen, selected = set(), []
        for row in train_rows:
            key = row.get('sha256') or row['audio']
            if key not in seen:
                seen.add(key)
                selected.append(row)
            if len(selected) == args.sanity_examples:
                break
        if len(selected) < 16:
            raise ValueError('M0 requires at least 16 unique short examples')
        train_rows, dev_rows = selected, selected
    elif args.eval_limit > 0:
        # A fixed subset for training selection; sorted hash makes it reproducible.
        dev_rows = sorted(dev_rows, key=lambda r: hashlib.sha256(str(r['audio']).encode()).hexdigest())[:args.eval_limit]
    hashes = {str(r['sha256']) for r in train_rows if r.get('sha256')}
    if args.mode == 'train' and hashes & {str(r['sha256']) for r in dev_rows if r.get('sha256')}:
        raise ValueError('Train/dev audio SHA256 overlap')
    vocab_hashes = {str(r.get('sha256') or r['audio']) for r in vocab_rows}
    if any(str(r.get('sha256') or r['audio']) not in vocab_hashes for r in train_rows):
        raise ValueError('Training examples are not all members of the declared vocabulary training pool')
    manifest = {'status': 'running', 'model_family': 'XLS-R-300M-CTC',
                'started_unix': started, 'arguments': vars(args),
                'manifest_sha256': {k: manifest_sha(getattr(args, k)) for k in
                                    ('train_manifest', 'dev_manifest', 'vocab_manifest')},
                'train_utterances': len(train_rows), 'eval_utterances': len(dev_rows),
                'eval_scope': 'training-M0' if args.mode == 'sanity' else 'fixed-development-subset',
                'vocab_size': len(vocab), 'vocab_source': args.vocab_manifest,
                'unknown_counts': {'train': unknown_stats(train_rows, vocab),
                                   'eval': unknown_stats(dev_rows, vocab)},
                'torch': torch.__version__, 'visible_gpus': os.getenv('CUDA_VISIBLE_DEVICES')}
    write_json(output / 'run_manifest.json', manifest)
    write_json(output / 'vocab.json', vocab)
    write_json(output / 'selected_examples.json', {'train': [r['audio'] for r in train_rows],
                                                  'eval': [r['audio'] for r in dev_rows]})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
    if args.resume_checkpoint:
        checkpoint_vocab = json.loads((Path(args.resume_checkpoint) / 'vocab.json').read_text())
        checkpoint_config = json.loads((Path(args.resume_checkpoint) / 'config.json').read_text())
        if checkpoint_vocab != vocab:
            raise ValueError('Resume checkpoint vocabulary differs from frozen training vocabulary')
        if args.mode == 'train' and checkpoint_config.get('asr_run_mode') != 'train':
            raise ValueError('Formal training cannot inherit M0/sanity or unprovenanced weights')
    model = Wav2Vec2ForCTC.from_pretrained(
        args.resume_checkpoint or args.model_dir, vocab_size=len(vocab), pad_token_id=0,
        ctc_loss_reduction='mean', ctc_zero_infinity=True,
        ignore_mismatched_sizes=True, local_files_only=True,
    )
    model.config.asr_run_mode = args.mode
    model.config.asr_vocabulary_manifest_sha256 = manifest_sha(args.vocab_manifest)
    model.freeze_feature_encoder()
    if args.mode == 'sanity':
        # Removing augmentation/dropout is necessary for an interpretable overfit test.
        # Keep mask probabilities: HF uses them to construct masked_spec_embed.
        # Zeroing them AFTER creation produces a checkpoint with a parameter that
        # disappears upon restoration and breaks AdamW parameter-group matching.
        model.config.apply_spec_augment = False
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    collator = Collate(extractor, vocab)
    train_loader = DataLoader(Rows(train_rows), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=args.workers,
                              pin_memory=device.type == 'cuda')
    eval_loader = DataLoader(Rows(dev_rows), batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=collator, num_workers=args.workers)
    first_batch = next(iter(eval_loader))
    initial = infer(model, eval_loader, device, vocab, output / 'predictions_initial.jsonl')
    append_jsonl(output / 'metrics.jsonl', {'step': 0, 'eval': initial})
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.max_steps * args.warmup_ratio))
    def schedule(step):
        return step / warmup if step < warmup else max(0.0, (args.max_steps - step) / max(1, args.max_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    starting_step = 0
    if args.resume_checkpoint:
        state_path = Path(args.resume_checkpoint) / 'training_state.pt'
        if not state_path.exists():
            raise ValueError('Resume needs training_state.pt, use a final checkpoint')
        training_state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(training_state['optimizer'])
        scheduler.load_state_dict(training_state['scheduler'])
        starting_step = int(training_state['step'])
        if starting_step >= args.max_steps:
            raise ValueError('Resume max_steps must exceed checkpoint step')
        manifest['resume_note'] = 'Optimizer/scheduler restored; shuffled data stream restarts from the declared seed.'
    iterator = iter(train_loader)
    losses, best_cer = [], math.inf
    eval_every = args.eval_every if args.mode == 'train' else max(1, args.max_steps // 4)
    model.train()
    for step in range(starting_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                features, labels, rows = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                features, labels, rows = next(iterator)
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            output_lengths = model._get_feat_extract_output_lengths(features['attention_mask'].sum(-1))
            label_lengths = (labels != -100).sum(-1)
            repeats = ((labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] != -100)).sum(-1)
            if torch.any(output_lengths < label_lengths + repeats):
                raise ValueError('CTC alignment impossible: output frames shorter than labels/repeats')
            with amp(device):
                loss = model(**features, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at optimizer step {step}')
            step_loss += float(loss.detach()) / args.grad_accum
            (loss / args.grad_accum).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        losses.append(step_loss)
        if step == 1 or step % (10 if args.mode == 'sanity' else 100) == 0:
            record = {'step': step, 'train_loss': step_loss, 'lr': scheduler.get_last_lr()[0],
                      'gradient_norm': float(gradient_norm), 'elapsed_seconds': time.time() - started}
            append_jsonl(output / 'metrics.jsonl', record)
            print(json.dumps(record), flush=True)
        if step % eval_every == 0 or step == args.max_steps:
            metrics = infer(model, eval_loader, device, vocab, output / f'predictions_step{step}.jsonl')
            append_jsonl(output / 'metrics.jsonl', {'step': step, 'eval': metrics})
            print(json.dumps({'step': step, 'eval': metrics}), flush=True)
            if metrics['cer'] < best_cer:
                best_cer = metrics['cer']
                save_checkpoint(model, extractor, vocab, output / 'best')
                write_json(output / 'best_metrics.json', {'step': step, **metrics})
    final_metrics = infer(model, eval_loader, device, vocab, output / 'predictions_final.jsonl')
    save_checkpoint(model, extractor, vocab, output / 'final')
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'step': args.max_steps, 'seed': args.seed}, output / 'final' / 'training_state.pt')
    witness = checkpoint_witness(model, extractor, vocab, output / 'final', first_batch, device,
                                 optimizer, scheduler)
    write_json(output / 'checkpoint_witness.json', witness)
    if not witness['passed']:
        raise RuntimeError('Checkpoint roundtrip failed')
    write_json(output / 'final_metrics.json', final_metrics)
    gate = None
    if args.mode == 'sanity':
        window = max(1, len(losses) // 5)
        first_loss, last_loss = float(np.mean(losses[:window])), float(np.mean(losses[-window:]))
        finite_loss = all(math.isfinite(loss) for loss in losses)
        gate = {'passed': finite_loss and last_loss < first_loss * 0.8
                         and final_metrics['cer'] < initial['cer'] * 0.9
                         and final_metrics['cer'] <= 0.15 and witness['passed'],
                'loss_decreased_20pct': last_loss < first_loss * 0.8,
                'cer_improved_10pct_relative': final_metrics['cer'] < initial['cer'] * 0.9,
                'absolute_cer_at_most_0_15': final_metrics['cer'] <= 0.15,
                'initial_metrics': initial, 'final_metrics': final_metrics,
                'first_window_loss': first_loss, 'last_window_loss': last_loss,
                'checkpoint_roundtrip': witness, 'examples': len(train_rows),
                'note': 'Minimum engineering gate; does not establish held-out ASR performance.'}
        gate['pass'] = gate['passed']
        gate['metrics'] = final_metrics
        gate['finite_loss'] = finite_loss
        write_json(output / 'm0_gate.json', gate)
    manifest.update({'status': 'completed' if gate is None or gate['passed'] else 'gate_failed',
                     'elapsed_seconds': time.time() - started, 'final_metrics': final_metrics,
                     'best_dev_cer': best_cer, 'checkpoint_witness': witness,
                     'max_gpu_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None})
    write_json(output / 'run_manifest.json', manifest)
    return 0 if gate is None or gate['passed'] else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-manifest', 'dev-manifest', 'vocab-manifest', 'model-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--mode', choices=('sanity', 'train'), default='sanity')
    parser.add_argument('--max-steps', '--sanity-steps', dest='max_steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-limit', type=int, default=128)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--sanity-examples', type=int, choices=range(16, 33), default=24)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--resume-checkpoint', default=None)
    args = parser.parse_args()
    if min(args.max_steps, args.batch_size, args.eval_batch_size, args.grad_accum, args.eval_every) <= 0:
        parser.error('Steps and batch sizes must be positive')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    try:
        raise SystemExit(execute(arguments))
    except Exception as error:
        run_path = Path(arguments.output) / 'run_manifest.json'
        # Never overwrite evidence when refusing reuse of an existing directory.
        if str(error).startswith('Output already contains'):
            raise
        state = json.loads(run_path.read_text()) if run_path.exists() else {'arguments': vars(arguments)}
        state.update({'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
        run_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(run_path, state)
        raise
