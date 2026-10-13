"""Recompute dev metrics from private predictions; export only anonymous aggregates."""
import argparse
import collections
import json
import math
from pathlib import Path
from common import load_manifest, normalize_text, score_pairs, write_json


def close_metrics(actual, saved):
    for key in ['cer', 'wer']:
        if not math.isclose(actual[key], saved[key], abs_tol=1e-10):
            raise ValueError('Recomputed metric mismatch: '+key)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    a=p.parse_args()
    dev=load_manifest(a.root/'data/splits/baseline_v1/dev_monitor.jsonl')
    expected={r['audio']:normalize_text(r['text']) for r in dev}
    if len(expected)!=128:raise ValueError('Unexpected frozen monitoring sample count')
    output={'scope':'5h engineering pilots; same fixed dev-monitor subset; no test scores',
            'metrics_units':'fractions', 'models':{}, 'raw_predictions_location':'private server run directories',
            'limitations':['one training seed', '128 utterances from 2 development speakers',
                           '1500 optimizer updates; convergence not assumed',
                           'checkpoint selection uses this development subset',
                           'CER ignores whitespace but preserves case/punctuation/digits',
                           'WER scores whitespace tokens, not linguistically verified words']}
    for family in ['whisper','ctc']:
        root=a.root/'runs'/(family+'_5h_s42_20261013_v2')
        pipeline=json.loads((root/'pipeline_status.json').read_text())
        if pipeline['state']!='completed' or pipeline.get('baseline_exit_code')!=0:
            raise ValueError('Pipeline not completed: '+family)
        if pipeline.get('test_evaluated') is not False:raise ValueError('Unexpected test use')
        gate=json.loads((root/'m0/m0_gate.json').read_text())
        if not gate.get('passed'):raise ValueError('M0 did not pass')
        baseline=root/'baseline'
        evaluation=root/'unified_development' if family=='ctc' else baseline
        predictions=load_manifest(evaluation/'final_predictions.jsonl')
        if len(predictions)!=128 or {r['audio'] for r in predictions}!=set(expected):
            raise ValueError('Final predictions do not cover the same development set')
        if any(normalize_text(r['reference'])!=expected[r['audio']] for r in predictions):
            raise ValueError('Reference differs from frozen dev ground truth')
        recomputed=score_pairs([r['prediction'] for r in predictions],[r['reference'] for r in predictions])
        final=json.loads((evaluation/'final_metrics.json').read_text())
        close_metrics(recomputed,final)
        best=json.loads((baseline/'best_metrics.json').read_text())
        best_report={'step':best['step'],'cer':best['cer'],'wer':best['wer'],
                     'scope':'Original checkpoint monitoring; CTC used a single-speaker subset, not comparable across models'}
        if family=='whisper':
            best_rows=load_manifest(baseline/f"predictions_step_{best['step']}.jsonl")
            if len(best_rows)!=128 or {r['audio'] for r in best_rows}!=set(expected):raise ValueError('Best sample mismatch')
            if any(normalize_text(r['reference'])!=expected[r['audio']] for r in best_rows):raise ValueError('Best references mismatch')
            best_recomputed=score_pairs([r['prediction'] for r in best_rows],[r['reference'] for r in best_rows])
            close_metrics(best_recomputed,best)
            best_report.update(scope='Balanced dev-monitor; checkpoint-selected, not test')
        by=collections.defaultdict(list)
        for r in predictions:by[r['speaker_candidate']].append(r)
        speaker_metrics={f'dev_speaker_{i+1}':score_pairs([r['prediction'] for r in rows],[r['reference'] for r in rows])
                         for i,(_,rows) in enumerate(sorted(by.items()))}
        if sorted(v['utterances'] for v in speaker_metrics.values())!=[64,64]:raise ValueError('Dev speaker balance mismatch')
        logs=load_manifest(baseline/'metrics.jsonl')
        # Reject NaN/Inf recursively; irregular epoch logs are not converted into evidence.
        def inspect(value):
            if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite logged number')
            if isinstance(value,dict):
                for v in value.values():inspect(v)
            if isinstance(value,list):
                for v in value:inspect(v)
        for row in logs:inspect(row)
        train_logs=[r for r in logs if r.get('phase')=='train' or 'train_loss' in r]
        witness=json.loads((baseline/'checkpoint_witness.json').read_text()) if family=='ctc' else gate['checkpoint_restore']
        if not witness['passed']:raise ValueError('Checkpoint witness failed')
        output['models'][family]={
            'run_name':root.name,'pipeline_state':pipeline['state'],'m0_passed':True,
            'checkpoint_restored_verified':True,'final_step':1500,'evaluation_records':len(predictions),
            'final_metrics_recomputed':recomputed,'best_development_checkpoint':best_report,
            'development_speaker_metrics':speaker_metrics,
            'last_logged_train_loss':train_logs[-1].get('loss',train_logs[-1].get('train_loss')),
            'baseline_seconds':pipeline['baseline_seconds'],'m0_seconds':pipeline['m0_seconds'],
            'pipeline_started_utc':pipeline['started_utc'],'pipeline_completed_utc':pipeline['completed_utc'],
            'baseline_gpu_hours':pipeline['baseline_seconds']/3600,
            'original_pretrained_initialization':True,'test_evaluated':False,
            'verification':'128 final-step predictions and exact references matched fixed balanced dev manifest; final metrics recomputed; finite logs; normal exit',
            'evaluation_correction':'CTC final step independently re-evaluated on balanced dev; original records retained' if family=='ctc' else None}
    write_json(a.root/'audit/private/baseline_results_2026-10-13.json',output)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
