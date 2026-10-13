#!/usr/bin/env python3
"""Offline, fully supervised Whisper baseline for languages without a language ID.

All CER/WER values are fractions. No test manifest is consumed. Sanity mode is a
deliberate training-set overfit check, never a held-out accuracy claim.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import (
    append_jsonl, load_audio_16k, load_manifest, normalize_text, score_pairs,
    seed_everything, write_json,
)


class SpeechDataset(Dataset):
    def __init__(self, rows, processor, label_prefix, max_tokens):
        self.rows = rows
        self.processor = processor
        self.label_prefix = label_prefix
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio = np.asarray(load_audio_16k(row), dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid decoded waveform: {row['audio']}")
        if len(audio) > 30 * 16000:
            raise ValueError(f"Actual duration exceeds 30s, refusing truncation: {row['audio']}")
        text = normalize_text(row['text'])
        ids = self.label_prefix + self.processor.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.processor.tokenizer.eos_token_id)
        if len(ids) > self.max_tokens:
            raise ValueError(f"Target exceeds {self.max_tokens} tokens: {row['audio']}")
        features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors='pt',
            padding='max_length', truncation=False,
        ).input_features[0]
        if features.shape != (self.processor.feature_extractor.feature_size, 3000):
            raise ValueError(f"Unexpected features {features.shape}: {row['audio']}")
        return {'input_features': features, 'labels': ids, 'row': row, 'text': text}


def collate(items):
    length = max(len(item['labels']) for item in items)
    labels = torch.full((len(items), length), -100, dtype=torch.long)
    for index, item in enumerate(items):
        labels[index, :len(item['labels'])] = torch.tensor(item['labels'])
    return {
        'input_features': torch.stack([item['input_features'] for item in items]),
        'labels': labels, 'rows': [item['row'] for item in items],
        'refs': [item['text'] for item in items],
    }


def filtered_rows(path, processor, label_prefix, max_tokens, log_path):
    accepted, rejected = [], 0
    for row in load_manifest(path):
        reason = None
        duration = float(row.get('duration', 0))
        text = normalize_text(row.get('text', ''))
        if not math.isfinite(duration) or duration <= 0:
            reason = 'invalid_duration'
        elif duration > 30:
            reason = 'over_30s_no_truncation'
        elif not text:
            reason = 'empty_normalized_text'
        elif len(label_prefix) + len(processor.tokenizer.encode(text, add_special_tokens=False)) + 1 > max_tokens:
            reason = 'target_too_long_no_truncation'
        if reason:
            rejected += 1
            append_jsonl(log_path, {'audio': row.get('audio'), 'reason': reason, 'manifest': str(path)})
        else:
            accepted.append(row)
    if not accepted:
        raise RuntimeError(f"No eligible rows in {path}")
    return accepted, rejected


def manifest_digest(rows):
    payload = '\n'.join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_generation_config(model):
    config = copy.deepcopy(model.generation_config)
    # Transformers 4.46.3 otherwise detects a language even when language=None.
    # This affects generation routing only, not model weights/config or tokenizer.
    config.is_multilingual = False
    config.language = None
    config.task = None
    config.forced_decoder_ids = None
    # In Transformers 4.46.3 the presence of lang_to_id alone triggers detection.
    if hasattr(config, 'lang_to_id'):
        delattr(config, 'lang_to_id')
    config.return_timestamps = False
    config.do_sample = False
    config.num_beams = 1
    config.use_cache = True
    return config


def forbid_language_detection(*args, **kwargs):
    raise RuntimeError('Language detection called: explicit no-language-token contract violated')


@torch.no_grad()
def evaluate(model, loader, processor, prefix, generation_config, device, output_path):
    model.eval()
    records, losses = [], []
    for batch in loader:
        features = batch['input_features'].to(device)
        labels = batch['labels'].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_features=features, labels=labels).loss
            decoder_input_ids = torch.tensor(prefix, device=device).unsqueeze(0).expand(len(features), -1)
            tokens = model.generate(
                input_features=features, decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
                max_new_tokens=model.config.max_target_positions - len(prefix),
            )
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite evaluation loss')
        losses.append((float(loss), int((labels != -100).sum())))
        hypotheses = processor.tokenizer.batch_decode(tokens.cpu(), skip_special_tokens=True)
        for row, ref, hyp in zip(batch['rows'], batch['refs'], hypotheses):
            hyp = normalize_text(hyp)
            record = {
                'audio': row['audio'], 'sha256': row.get('sha256'),
                'speaker_candidate': row.get('speaker_candidate'),
                'duration': row.get('duration'), 'reference': ref, 'prediction': hyp,
            }
            record.update(score_pairs([hyp], [ref]))
            records.append(record)
    metrics = score_pairs([r['prediction'] for r in records], [r['reference'] for r in records])
    metrics.update({
        'teacher_forced_loss': sum(loss * n for loss, n in losses) / sum(n for _, n in losses),
        'utterances': len(records),
    })
    output_path = Path(output_path)
    output_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    write_json(output_path.with_suffix('.metrics.json'), metrics)
    return metrics


def save_checkpoint(path, model, optimizer, scheduler, step, best_cer, contract, training_state=None):
    state = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'step': step, 'best_cer': best_cer,
        'contract': contract, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'training_state': training_state or {},
    }
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore_checkpoint(path, model, optimizer, scheduler, contract, return_tracking=False):
    # Only our own locally created checkpoints are accepted by this CLI.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['contract'] != contract:
        raise RuntimeError('Checkpoint/data/training contract mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    result = (int(state['step']), float(state['best_cer']))
    return (*result, state.get('training_state', {})) if return_tracking else result


@torch.no_grad()
def restore_witness(model, batch, device):
    model.eval()
    # A small actual forward witness, evaluated in float32 to avoid BF16 roundoff.
    return model(
        input_features=batch['input_features'][:1].to(device),
        labels=batch['labels'][:1].to(device),
    ).logits.detach().cpu()


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This controlled recipe requires CUDA with BF16 support')
    if args.max_steps < 2 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError('max-steps >= 2, batch-size >= 1, grad-accum >= 1 required')
    seed_everything(args.seed)
    device = torch.device('cuda')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_config.json').exists() and not args.resume:
        raise RuntimeError('Output already contains a run; choose a fresh directory or --resume')
    write_json(output / 'status.json', {'state': 'preparing', 'mode': args.mode})
    processor = WhisperProcessor.from_pretrained(args.model_dir, local_files_only=True)
    processor.tokenizer.language = None
    processor.tokenizer.set_prefix_tokens(language=None, task='transcribe', predict_timestamps=False)
    prefix = list(processor.tokenizer.prefix_tokens)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True)
    if len(prefix) != 3 or prefix[0] != model.config.decoder_start_token_id:
        raise RuntimeError(f'Unexpected language-free prefix: {prefix}')
    expected = [
        model.config.decoder_start_token_id,
        processor.tokenizer.convert_tokens_to_ids('<|transcribe|>'),
        processor.tokenizer.convert_tokens_to_ids('<|notimestamps|>'),
    ]
    if prefix != expected:
        raise RuntimeError(f'Prefix includes unexpected token: {prefix} != {expected}')
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    generation_config = make_generation_config(model)
    # Fail loudly if a future Transformers implementation reintroduces detection.
    model.detect_language = forbid_language_detection
    max_tokens = int(model.config.max_target_positions)
    train_rows, train_rejected = filtered_rows(
        args.train_manifest, processor, prefix[1:], max_tokens, output / 'excluded.jsonl',
    )
    full_rows = None
    if args.mode == 'sanity':
        if args.skip_initial_eval or args.full_dev_manifest or args.early_stopping_patience:
            raise ValueError('Sanity mode requires its original training-overfit gate')
        selected, seen = [], set()
        for row in train_rows:
            key = row.get('sha256') or row['audio']
            if float(row['duration']) <= 10 and key not in seen:
                selected.append(row)
                seen.add(key)
                if len(selected) == 16:
                    break
        if len(selected) < 16:
            raise RuntimeError('Sanity mode requires 16 distinct audio samples <= 10s')
        train_rows, eval_rows = selected, selected
        dev_rejected = 0
    else:
        if not args.dev_manifest:
            raise ValueError('--dev-manifest required in train mode')
        eval_rows, dev_rejected = filtered_rows(
            args.dev_manifest, processor, prefix[1:], max_tokens, output / 'excluded.jsonl',
        )
        # Deterministic subset spanning the manifest instead of just the first speaker.
        indices = np.linspace(0, len(eval_rows) - 1, min(args.eval_limit, len(eval_rows)), dtype=int)
        eval_rows = [eval_rows[index] for index in indices]
        train_hashes = {row.get('sha256') or row['audio'] for row in train_rows}
        if any((row.get('sha256') or row['audio']) in train_hashes for row in eval_rows):
            raise RuntimeError('Train/evaluation audio overlap')
        train_speakers = {row.get('speaker_candidate') for row in train_rows}
        if None in train_speakers or any(row.get('speaker_candidate') in train_speakers for row in eval_rows):
            raise RuntimeError('Missing speaker identity or train/development speaker overlap')
        if args.full_dev_manifest:
            full_rows, full_rejected = filtered_rows(
                args.full_dev_manifest, processor, prefix[1:], max_tokens, output / 'excluded.jsonl',
            )
            if any((row.get('sha256') or row['audio']) in train_hashes for row in full_rows):
                raise RuntimeError('Train/full-development audio overlap')
            if any(row.get('speaker_candidate') is None or row.get('speaker_candidate') in train_speakers for row in full_rows):
                raise RuntimeError('Missing speaker identity or train/full-development speaker overlap')
            full_keys = {row.get('sha256') or row['audio'] for row in full_rows}
            if not {(row.get('sha256') or row['audio']) for row in eval_rows} <= full_keys:
                raise RuntimeError('Monitoring examples must be contained in full development')
            if args.full_eval_every is None:
                args.full_eval_every = math.ceil(len(train_rows) / (args.batch_size * args.grad_accum))
    for name, rows in [('train_used', train_rows), ('eval_used', eval_rows)]:
        (output / f'{name}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8',
        )
    if full_rows is not None:
        (output / 'full_eval_used.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in full_rows), encoding='utf-8',
        )
    contract = {
        'model_dir': str(Path(args.model_dir).resolve()), 'mode': args.mode,
        'train_sha256': manifest_digest(train_rows), 'eval_sha256': manifest_digest(eval_rows),
        'seed': args.seed, 'lr': args.lr, 'max_steps': args.max_steps,
        'batch_size': args.batch_size, 'grad_accum': args.grad_accum,
        'warmup_steps': args.warmup_steps, 'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm,
        'full_eval_sha256': manifest_digest(full_rows) if full_rows is not None else None,
        'full_eval_every': args.full_eval_every,
        'early_stopping_patience': args.early_stopping_patience,
        'min_steps': args.min_steps,
    }
    config = dict(vars(args))
    config.update({
        'contract': contract, 'language_free_prefix_ids': prefix,
        'language_free_prefix_tokens': processor.tokenizer.convert_ids_to_tokens(prefix),
        'generation_is_multilingual_override': False,
        'model_is_still_multilingual': True, 'detect_language_forbidden': True,
        'train_utterances': len(train_rows), 'eval_utterances': len(eval_rows),
        'train_hours': sum(float(row['duration']) for row in train_rows) / 3600,
        'excluded_train': train_rejected, 'excluded_dev': dev_rejected,
        'full_eval_utterances': len(full_rows) if full_rows is not None else None,
        'excluded_full_dev': full_rejected if full_rows is not None else None,
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'evaluation_scope': 'training_overfit' if args.mode == 'sanity' else 'fixed_development_subset',
        'checkpoint_selection_scope': 'complete_filtered_development' if full_rows is not None else 'fixed_development_subset',
        'metric_units': 'fraction',
    })
    write_json(output / 'run_config.json', config)
    train_data = SpeechDataset(train_rows, processor, prefix[1:], max_tokens)
    eval_data = SpeechDataset(eval_rows, processor, prefix[1:], max_tokens)
    loader_options = {'num_workers': args.workers}
    if args.workers:
        loader_options.update(multiprocessing_context='spawn', persistent_workers=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate, **loader_options)
    eval_batch_size = args.eval_batch_size or args.batch_size
    eval_loader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=False, collate_fn=collate, **loader_options)
    full_loader = None
    if full_rows is not None:
        full_loader = DataLoader(
            SpeechDataset(full_rows, processor, prefix[1:], max_tokens),
            batch_size=eval_batch_size, shuffle=False, collate_fn=collate, **loader_options,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, int(args.max_steps * .05))

    def multiplier(step):
        if step < warmup:
            return max(step, 1) / warmup
        return max(0., (args.max_steps - step) / max(1, args.max_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    step, best_cer = 0, float('inf')
    resume_tracking = {}
    if args.resume:
        step, best_cer, resume_tracking = restore_checkpoint(args.resume, model, optimizer, scheduler, contract, return_tracking=True)
    initial = None
    if not args.skip_initial_eval:
        initial = evaluate(model, eval_loader, processor, prefix, generation_config, device, output / 'initial_predictions.jsonl')
        append_jsonl(output / 'metrics.jsonl', {'step': step, 'phase': 'initial', **initial})
    iterator = iter(train_loader)
    start_time = time.monotonic()
    checkpoint_witness = None
    interval = min(50, args.eval_every) if args.mode == 'sanity' else args.eval_every
    rolling_loss, rolling_batches = 0., 0
    optimizer_seconds = float(resume_tracking.get('optimizer_seconds', 0.))
    stale_evaluations = int(resume_tracking.get('stale_evaluations', 0))
    stopped_early = False
    if args.resume and not resume_tracking and (output / 'early_stopping.json').exists():
        stop_state = json.loads((output / 'early_stopping.json').read_text())
        if stop_state['step'] == step:
            stale_evaluations = stop_state['stale_evaluations']
    for step in range(step + 1, args.max_steps + 1):
        update_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(
                    input_features=batch['input_features'].to(device), labels=batch['labels'].to(device),
                ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite train loss at step {step}')
            (loss / args.grad_accum).backward()
            rolling_loss += float(loss.detach())
            rolling_batches += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f'Non-finite gradient norm at step {step}')
        optimizer.step()
        scheduler.step()
        optimizer_seconds += time.monotonic() - update_started
        if step % 100 == 0 or step == 1 or step == args.max_steps:
            record = {
                'step': step, 'phase': 'train', 'loss': rolling_loss / rolling_batches,
                'lr': scheduler.get_last_lr()[0], 'grad_norm': float(grad_norm),
                'elapsed_seconds': time.monotonic() - start_time,
                'max_cuda_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
                'optimizer_seconds': optimizer_seconds,
                'approx_epochs': step * args.batch_size * args.grad_accum / len(train_rows),
            }
            append_jsonl(output / 'metrics.jsonl', record)
            print(json.dumps(record), flush=True)
            rolling_loss, rolling_batches = 0., 0
        if args.mode == 'sanity' and step == max(1, args.max_steps // 2):
            witness_batch = next(iter(eval_loader))
            before = restore_witness(model, witness_batch, device)
            checkpoint_path = output / 'resume_checkpoint.pt'
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step, best_cer, contract)
            # Perturb a parameter to ensure the restore really reloads model weights.
            with torch.no_grad():
                next(model.parameters()).add_(0.01)
            restored_step, restored_best = restore_checkpoint(checkpoint_path, model, optimizer, scheduler, contract)
            after = restore_witness(model, witness_batch, device)
            difference = float((before - after).abs().max())
            checkpoint_witness = {
                'restored_step': restored_step, 'max_logit_difference': difference,
                'passed': restored_step == step and difference <= 1e-5,
                'optimizer_state_entries': len(optimizer.state),
                'scheduler_epoch': scheduler.last_epoch,
            }
            if not checkpoint_witness['passed']:
                raise RuntimeError(f'Checkpoint restore failed: {checkpoint_witness}')
            best_cer = restored_best
            write_json(output / 'checkpoint_restore_witness.json', checkpoint_witness)
        if step % interval == 0 or step == args.max_steps:
            metrics = evaluate(
                model, eval_loader, processor, prefix, generation_config, device,
                output / f'predictions_step_{step}.jsonl',
            )
            append_jsonl(output / 'metrics.jsonl', {'step': step, 'phase': 'eval', **metrics})
            print(json.dumps({'step': step, 'phase': 'eval', **metrics}), flush=True)
            if full_loader is None and metrics['cer'] < best_cer:
                best_cer = metrics['cer']
                model.save_pretrained(output / 'best')
                processor.save_pretrained(output / 'best')
                write_json(output / 'best_metrics.json', {'step': step, **metrics})
            write_json(output / 'status.json', {'state': 'running', 'step': step, 'max_steps': args.max_steps})
        selected_metrics = None
        if full_loader is not None and (step % args.full_eval_every == 0 or step == args.max_steps):
            write_json(output / 'status.json', {'state': 'evaluating_full_dev', 'step': step, 'max_steps': args.max_steps})
            selected_metrics = evaluate(
                model, full_loader, processor, prefix, generation_config, device,
                output / f'full_dev_predictions_step_{step}.jsonl',
            )
            append_jsonl(output / 'metrics.jsonl', {'step': step, 'phase': 'full_eval', **selected_metrics})
            print(json.dumps({'step': step, 'phase': 'full_eval', **selected_metrics}), flush=True)
        elif full_loader is None and (step % interval == 0 or step == args.max_steps):
            selected_metrics = metrics
        if selected_metrics is not None:
            improved = selected_metrics['cer'] < best_cer if full_loader is not None else selected_metrics['cer'] <= best_cer
            if improved:
                best_cer = selected_metrics['cer']
                stale_evaluations = 0
                if full_loader is not None:
                    model.save_pretrained(output / 'best')
                    processor.save_pretrained(output / 'best')
                    write_json(output / 'best_metrics.json', {'step': step, 'scope': 'complete_filtered_development', **selected_metrics})
            else:
                stale_evaluations += 1
            write_json(output / 'early_stopping.json', {'step': step, 'stale_evaluations': stale_evaluations, 'best_cer': best_cer})
            if args.early_stopping_patience and step >= args.min_steps and stale_evaluations >= args.early_stopping_patience:
                stopped_early = True
        if args.mode == 'train' and (selected_metrics is not None or step % interval == 0):
            save_checkpoint(output / 'resume_checkpoint.pt', model, optimizer, scheduler, step, best_cer, contract, {
                'stale_evaluations': stale_evaluations, 'optimizer_seconds': optimizer_seconds,
            })
        write_json(output / 'status.json', {
            'state': 'running', 'step': step, 'max_steps': args.max_steps,
            'approx_epochs': step * args.batch_size * args.grad_accum / len(train_rows),
            'elapsed_seconds': time.monotonic() - start_time,
            'max_cuda_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
        })
        if stopped_early:
            break
    model.save_pretrained(output / 'final')
    processor.save_pretrained(output / 'final')
    final = evaluate(model, full_loader or eval_loader, processor, prefix, generation_config, device, output / 'final_predictions.jsonl')
    write_json(output / 'final_metrics.json', {'step': step, **final})
    if full_loader is not None:
        best_model = WhisperForConditionalGeneration.from_pretrained(output / 'best', local_files_only=True)
        best_model.config.use_cache = False
        best_model.detect_language = forbid_language_detection
        del optimizer, scheduler, model
        torch.cuda.empty_cache()
        best_model.to(device)
        best_metrics = evaluate(best_model, full_loader, processor, prefix, generation_config, device, output / 'best_full_dev_predictions.jsonl')
        write_json(output / 'best_full_dev_metrics.json', best_metrics)
    passed = True
    if args.mode == 'sanity':
        criteria = {
            'checkpoint_restored': bool(checkpoint_witness and checkpoint_witness['passed']),
            'cer_below_threshold': final['cer'] <= args.m0_max_cer,
            'loss_below_one': final['teacher_forced_loss'] < 1.,
            'loss_reduced_by_half': final['teacher_forced_loss'] <= initial['teacher_forced_loss'] * .5,
        }
        passed = all(criteria.values())
        write_json(output / 'm0_gate.json', {
            'passed': passed, 'criteria': criteria, 'initial': initial, 'final': final,
            'cer_threshold': args.m0_max_cer, 'evaluation_scope': 'training_overfit_only',
            'language_detection_disabled': True, 'checkpoint_restore': checkpoint_witness,
        })
    write_json(output / 'status.json', {
        'state': 'completed' if passed else 'gate_failed', 'step': step,
        'elapsed_seconds': time.monotonic() - start_time, 'metrics': final,
        'stopped_early': stopped_early, 'optimizer_seconds': optimizer_seconds,
    })
    if not passed:
        raise RuntimeError('M0 gate failed; inspect measured loss/CER and checkpoint witness')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--dev-manifest')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['sanity', 'train'], default='sanity')
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-limit', type=int, default=128)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--full-dev-manifest', help='Full development is the only checkpoint selection source when provided')
    parser.add_argument('--full-eval-every', type=int, help='Defaults to approximately one training epoch')
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--skip-initial-eval', action='store_true', help='Train mode only; never skips the M0 initial witness')
    parser.add_argument('--early-stopping-patience', type=int, default=0)
    parser.add_argument('--min-steps', type=int, default=0)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--m0-max-cer', type=float, default=.15)
    parser.add_argument('--resume', help='Only resume checkpoints created by this script')
    args = parser.parse_args()
    if args.eval_limit < 1 or args.eval_every < 1 or args.lr <= 0 or args.max_grad_norm <= 0:
        parser.error('eval-limit/eval-every/lr/max-grad-norm must be positive')
    if args.workers < 0 or args.early_stopping_patience < 0 or args.min_steps < 0:
        parser.error('workers, patience and min-steps must be non-negative')
    if args.full_eval_every is not None and args.full_eval_every < 1:
        parser.error('full-eval-every must be positive')
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error('eval-batch-size must be positive')
    if args.early_stopping_patience and not args.full_dev_manifest:
        parser.error('Early stopping requires full development evaluation')
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'status.json', {'state': 'failed', 'error': str(error), 'mode': args.mode})
        if args.mode == 'sanity' and not (output / 'm0_gate.json').exists():
            write_json(output / 'm0_gate.json', {'passed': False, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
