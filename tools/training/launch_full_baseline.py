#!/usr/bin/env python3
"""Launch one offline full-train Whisper baseline with real dependency and M0 gates."""
import argparse
import collections
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import load_manifest, normalize_text, test_metrics, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_record(path, model_dir):
    if not path.exists():
        return None
    audit = json.loads(path.read_text())
    records = audit.get('models', [audit])
    return next((record for record in records if
                 record.get('directory') == model_dir.name or
                 record.get('target_directory') == str(model_dir)), None)


def wait_dependencies(root, run, state, model_dir, audit_path, timeout):
    deadline = time.monotonic() + timeout
    cache_audit_path = root / 'audit/private/full_audio_cache.json'
    while True:
        record = model_record(audit_path, model_dir)
        cache = json.loads(cache_audit_path.read_text()) if cache_audit_path.exists() else {}
        if record and record.get('status') == 'offline_cpu_load_verified' and cache.get('status') == 'completed':
            if cache.get('failed') != 0 or not cache.get('expected_input_match') or cache.get('test_manifest_read'):
                raise RuntimeError('Full audio cache integrity gate failed')
            for split in ['train', 'dev']:
                manifest = root / 'data/splits/draft' / (split + '.jsonl')
                if digest(manifest) != cache['manifests'][split]['manifest_sha256']:
                    raise RuntimeError('Frozen manifest changed since audio validation')
            for name, expected in record['expected_files'].items():
                file = model_dir / name
                if not file.is_file() or file.stat().st_size != expected['size']:
                    raise RuntimeError('Model file missing or size differs: ' + name)
                wanted = expected.get('local_sha256') or expected.get('lfs_sha256')
                if not wanted or digest(file) != wanted:
                    raise RuntimeError('Model file hash differs: ' + name)
            state.update(model_revision=record['revision'], cache_validated=cache['processed'],
                         dependencies_verified_utc=utc_now())
            return
        if cache.get('status') in ['completed_with_errors', 'stopped_input_mismatch']:
            raise RuntimeError('Audio preparation failed; no training launched')
        if record and record.get('status') in ['failed', 'download_failed', 'verification_failed']:
            raise RuntimeError('Model preparation failed; no training launched')
        if time.monotonic() > deadline:
            raise TimeoutError('Model/cache dependency wait timed out')
        state.update(state='waiting_dependencies', model_status=record.get('status') if record else 'missing',
                     cache_status=cache.get('status', 'missing'), updated_utc=utc_now())
        write_json(run / 'pipeline_status.json', state)
        time.sleep(20)


def wait_gpu(gpu, run, state, timeout):
    deadline = time.monotonic() + timeout
    while True:
        output = subprocess.check_output([
            'nvidia-smi', '--id=' + str(gpu), '--query-gpu=memory.used',
            '--format=csv,noheader,nounits',
        ], text=True).strip()
        memory = int(output)
        if memory < 500:
            return
        if time.monotonic() > deadline:
            raise TimeoutError('Selected GPU stayed busy; no other process changed')
        state.update(state='waiting_gpu', gpu_memory_mib=memory, updated_utc=utc_now())
        write_json(run / 'pipeline_status.json', state)
        time.sleep(20)


def write_manifest(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return digest(path)


def prepare_inputs(root, run, model_dir):
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.tokenizer.set_prefix_tokens(language=None, task='transcribe', predict_timestamps=False)
    max_positions = json.loads((model_dir / 'config.json').read_text())['max_target_positions']
    summaries, rows_by_split = {}, {}
    manifests = run / 'inputs'
    manifests.mkdir()
    for split in ['train', 'dev']:
        rows = load_manifest(root / 'data/splits/draft' / (split + '.jsonl'))
        keep, exclusions = [], collections.Counter()
        for row in rows:
            text = normalize_text(row.get('text', ''))
            duration = float(row.get('duration', 0))
            if not math.isfinite(duration) or not 0 < duration <= 30:
                exclusions['invalid_or_over_30s'] += 1
            elif not text:
                exclusions['empty_text'] += 1
            elif 2 + len(processor.tokenizer.encode(text, add_special_tokens=False)) + 1 > max_positions:
                exclusions['target_exceeds_model_context'] += 1
            else:
                keep.append(row)
        if not keep:
            raise RuntimeError('No model-eligible examples: ' + split)
        rows_by_split[split] = keep
        summaries[split] = {
            'source_rows': len(rows), 'rows': len(keep), 'excluded': dict(exclusions),
            'hours': sum(row['duration'] for row in keep) / 3600,
            'speakers': len({row['speaker_candidate'] for row in keep}),
            'manifest_sha256': write_manifest(manifests / (split + '.jsonl'), keep),
        }
    for key in ['sha256', 'speaker_candidate']:
        if {row[key] for row in rows_by_split['train']} & {row[key] for row in rows_by_split['dev']}:
            raise RuntimeError('Training/development overlap: ' + key)
    by_speaker = collections.defaultdict(list)
    for row in sorted(rows_by_split['dev'], key=lambda row: row['sha256']):
        by_speaker[row['speaker_candidate']].append(row)
    monitor = []
    for index in range(256):
        for speaker in sorted(by_speaker):
            if index < len(by_speaker[speaker]):
                monitor.append(by_speaker[speaker][index])
    summaries['monitor'] = {'rows': len(monitor), 'manifest_sha256': write_manifest(manifests / 'dev_monitor.jsonl', monitor)}
    candidates = collections.defaultdict(list)
    for row in sorted(rows_by_split['train'], key=lambda row: row['sha256']):
        if 2 <= row['duration'] <= 8 and len(normalize_text(row['text'])) <= 120:
            candidates[row['speaker_candidate']].append(row)
    sanity = []
    index = 0
    while len(sanity) < 16:
        added = 0
        for speaker in sorted(candidates):
            if index < len(candidates[speaker]) and len(sanity) < 16:
                sanity.append(candidates[speaker][index])
                added += 1
        if not added:
            raise RuntimeError('Insufficient distinct short training examples for M0')
        index += 1
    summaries['sanity'] = {'rows': 16, 'manifest_sha256': write_manifest(manifests / 'sanity.jsonl', sanity)}
    summaries.update(test_read=False, scoring_policy='NFC and whitespace collapse; preserve case/punctuation/digits')
    write_json(manifests / 'summary.json', summaries)
    return summaries


def run_phase(command, phase, run, root, env, state, timeout):
    state.update(state=phase + '_running', updated_utc=utc_now())
    write_json(run / 'pipeline_status.json', state)
    started = time.monotonic()
    with (run / (phase + '.log')).open('w') as logfile:
        process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, env=env, cwd=root)
        state['child_pid'] = process.pid
        write_json(run / 'pipeline_status.json', state)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise TimeoutError('Only this launcher-owned child exceeded ' + phase + ' timeout')
    state.update(child_pid=None, **{phase + '_exit_code': code, phase + '_seconds': time.monotonic() - started})
    write_json(run / 'pipeline_status.json', state)
    if code != 0:
        raise RuntimeError(phase + ' failed; evidence saved, no automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--model', choices=['whisper-small', 'whisper-medium'], required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    if args.lr <= 0 or args.epochs < 2 or Path(args.run_name).name != args.run_name:
        parser.error('Valid run name, positive LR and epochs >= 2 required')
    os.umask(0o077)
    root = Path(__file__).resolve().parents[2]
    run = root / 'runs' / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    state = {'state': 'preparing', 'model': args.model, 'gpu': args.gpu, 'lr': args.lr,
             'launcher_pid': os.getpid(), 'started_utc': utc_now(), 'test_read': False}
    write_json(run / 'pipeline_status.json', state)
    try:
        model_dir = root / 'data/model_cache' / args.model
        audit = root / ('audit/private/whisper_medium_download.json' if args.model == 'whisper-medium'
                        else 'audit/private/model_download/download_manifest.json')
        wait_dependencies(root, run, state, model_dir, audit, 8 * 3600)
        lock_path = root / 'audit/private' / ('full_gpu_' + str(args.gpu) + '.lock')
        with lock_path.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                       ASR_AUDIO_CACHE=str(root / 'data/audio_cache/16k_float_mono_ffmpeg42_v1'))
            os.environ.update({key: env[key] for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']})
            test_metrics()
            summaries = prepare_inputs(root, run, model_dir)
            steps_per_epoch = math.ceil(summaries['train']['rows'] / 16)
            state.update(input_summary=summaries, steps_per_epoch=steps_per_epoch, max_steps=steps_per_epoch * args.epochs)
            code_dir = run / 'code'
            code_dir.mkdir()
            code_hashes = {}
            for filename in ['train_whisper.py', 'common.py']:
                shutil.copy2(root / 'tools/training' / filename, code_dir / filename)
                code_hashes[filename] = digest(code_dir / filename)
            state['code_sha256'] = code_hashes
            script = str(code_dir / 'train_whisper.py')
            shared = [sys.executable, '-u', script, '--model-dir', str(model_dir), '--seed', '42']
            m0 = shared + ['--mode', 'sanity', '--train-manifest', str(run / 'inputs/sanity.jsonl'),
                           '--output', str(run / 'm0'), '--max-steps', '400', '--batch-size', '2',
                           '--grad-accum', '1', '--lr', '0.00003', '--eval-every', '100']
            microbatch = 4 if args.model == 'whisper-small' else 2
            baseline = shared + [
                '--mode', 'train', '--train-manifest', str(run / 'inputs/train.jsonl'),
                '--dev-manifest', str(run / 'inputs/dev_monitor.jsonl'),
                '--full-dev-manifest', str(run / 'inputs/dev.jsonl'),
                '--output', str(run / 'baseline'), '--max-steps', str(steps_per_epoch * args.epochs),
                '--batch-size', str(microbatch), '--grad-accum', str(16 // microbatch),
                '--lr', str(args.lr), '--eval-limit', str(summaries['monitor']['rows']),
                '--eval-every', '1000', '--full-eval-every', str(steps_per_epoch),
                '--eval-batch-size', str(microbatch), '--workers', '2', '--skip-initial-eval',
                '--early-stopping-patience', '3', '--min-steps', str(steps_per_epoch * 3),
            ]
            write_json(run / 'launch_commands.json', {'m0': m0, 'baseline': baseline,
                'environment': {key: env[key] for key in ['CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'ASR_AUDIO_CACHE']},
                'baseline_initialization': 'fresh original pretrained checkpoint; no M0 weights',
                'checkpoint_selection': 'complete filtered development, not monitoring subset',
                'timeout_hours': {'dependencies': 8, 'm0': 4, 'baseline': 72}, 'test_read': False})
            wait_gpu(args.gpu, run, state, 8 * 3600)
            run_phase(m0, 'm0', run, root, env, state, 4 * 3600)
            gate = json.loads((run / 'm0/m0_gate.json').read_text())
            if gate.get('passed') is not True:
                raise RuntimeError('M0 gate did not pass; baseline forbidden')
            state['m0_passed'] = True
            write_json(run / 'pipeline_status.json', state)
            wait_gpu(args.gpu, run, state, 8 * 3600)
            run_phase(baseline, 'baseline', run, root, env, state, 72 * 3600)
            completion = json.loads((run / 'baseline/status.json').read_text())
            if completion.get('state') != 'completed' or not (run / 'baseline/best_full_dev_metrics.json').exists():
                raise RuntimeError('Baseline completion lacks real full-development results')
            state.update(state='completed', completed_utc=utc_now(), baseline_step=completion['step'])
            write_json(run / 'pipeline_status.json', state)
    except Exception as error:
        state.update(state='failed', error_type=type(error).__name__, error=str(error), updated_utc=utc_now())
        write_json(run / 'pipeline_status.json', state)
        raise


if __name__ == '__main__':
    main()
