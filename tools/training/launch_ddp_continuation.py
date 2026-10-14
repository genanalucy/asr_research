#!/usr/bin/env python3
"""Verified four-GPU Whisper continuation from an owned single-GPU checkpoint."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import write_json, test_metrics
from launch_full_baseline import digest, wait_dependencies, prepare_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--gpus', default='4,5,6,7')
    args = parser.parse_args()
    for name in [args.source_run, args.run_name]:
        if Path(name).name != name or name in ['.', '..']:
            parser.error('run names must be single directory names')
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        parser.error('Exactly four distinct GPUs required for global batch16')
    os.umask(0o077)
    root = Path(__file__).resolve().parents[2]
    run = root/'runs'/args.run_name
    source = root/'runs'/args.source_run
    run.mkdir(parents=True, exist_ok=False)
    state = {'state':'preparing', 'launcher_pid':os.getpid(), 'gpus':gpus,
             'world_size':4, 'source_run':args.source_run, 'test_read':False,
             'started_utc':datetime.now(timezone.utc).isoformat()}
    def update(**fields):
        state.update(**fields)
        write_json(run/'pipeline_status.json', state)
    update()
    try:
        source_pipeline = json.loads((source/'pipeline_status.json').read_text())
        if source_pipeline.get('state') not in ['stopped_for_ddp_reprioritization', 'stopped_for_ddp_retest', 'completed']:
            raise RuntimeError('Source checkpoint owner must be stopped/completed before migration')
        model = root/'data/model_cache/whisper-medium'
        wait_dependencies(root, run, state, model, root/'audit/private/whisper_medium_download.json', 60)
        with ExitStack() as stack:
            for gpu in sorted(gpus):
                lock = stack.enter_context((root/'audit/private'/f'full_gpu_{gpu}.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            used = {gpu:int(subprocess.check_output(['nvidia-smi',f'--id={gpu}', '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()) for gpu in gpus}
            if any(v >= 500 for v in used.values()):
                raise RuntimeError('Selected GPU occupied; no other process changed')
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)), HF_HUB_OFFLINE='1',
                     TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
                     ASR_AUDIO_CACHE=str(root/'data/audio_cache/16k_float_mono_ffmpeg42_v1'),
                     NCCL_SOCKET_IFNAME='lo', NCCL_SOCKET_FAMILY='AF_INET', NCCL_DEBUG='WARN',
                     TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
            os.environ.update({k:env[k] for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']})
            test_metrics()
            summary=prepare_inputs(root,run,model)
            for name in ['train','dev','dev_monitor','sanity']:
                if digest(run/'inputs'/f'{name}.jsonl') != digest(source/'inputs'/f'{name}.jsonl'):
                    raise RuntimeError('Source/input manifest mismatch: '+name)
            import torch
            ckpt=source/'baseline/resume_checkpoint.pt'
            saved=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
            contract=saved['contract'];step=int(saved['step'])
            if contract['mode']!='train' or not contract['model_dir'].endswith('/whisper-medium') or contract['lr']!=1e-5 or contract['seed']!=42:
                raise RuntimeError('Expected owned medium LR1e-5 seed42 checkpoint')
            if contract['batch_size']*contract['grad_accum'] !=16:
                raise RuntimeError('Source global batch differs')
            if step >= contract['max_steps']:
                raise RuntimeError('Source training already exhausted max steps')
            saved_metadata={'source_step':step, 'source_best_cer':saved['best_cer'], 'source_contract':contract,
                            'source_checkpoint_sha256':digest(ckpt),'source_state':saved.get('training_state',{}),
                            'migration':'single GPU to 4 ranks; same global batch, optimizer/scheduler/update count; shuffle path changes'}
            del saved
            baseline=run/'baseline';baseline.mkdir()
            shutil.copytree(source/'baseline/best',baseline/'best')
            shutil.copy2(source/'baseline/best_metrics.json',baseline/'best_metrics.json')
            shutil.copy2(source/'baseline/metrics.jsonl',run/'source_metrics.jsonl')
            write_json(run/'migration.json',saved_metadata)
            code=run/'code';code.mkdir()
            names=['train_whisper_ddp.py','common.py']
            for name in names:shutil.copy2(root/'tools/training'/name,code/name)
            update(source_step=step, global_effective_batch=16, max_steps=contract['max_steps'],
                   input_summary=summary, code_sha256={n:digest(code/n) for n in names})
            shared=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(code/'train_whisper_ddp.py'),
                    '--model-dir',str(model),'--seed','42','--batch-size','2','--grad-accum','2','--workers','1']
            m0=shared+['--mode','sanity','--train-manifest',str(run/'inputs/sanity.jsonl'),'--output',str(run/'m0'),
                       '--max-steps','400','--lr','0.00003','--grad-accum','2','--eval-every','100']
            train=shared+['--mode','train','--train-manifest',str(run/'inputs/train.jsonl'),
                          '--dev-manifest',str(run/'inputs/dev_monitor.jsonl'),'--full-dev-manifest',str(run/'inputs/dev.jsonl'),
                          '--output',str(baseline),'--max-steps',str(contract['max_steps']),'--lr','0.00001',
                          '--eval-limit','512','--eval-every','1000','--full-eval-every',str(contract['full_eval_every']),
                          '--early-stopping-patience',str(contract['early_stopping_patience']),
                          '--min-steps',str(contract['min_steps']),'--eval-batch-size','4','--skip-initial-eval',
                          '--resume',str(ckpt)]
            write_json(run/'launch_commands.json',{'m0':m0,'baseline':train,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','NCCL_SOCKET_IFNAME','NCCL_SOCKET_FAMILY','OMP_NUM_THREADS']},'test_read':False})
            for phase,cmd,timeout in [('m0',m0,3600),('baseline',train,24*3600)]:
                update(state=phase+'_running')
                start=time.monotonic()
                with (run/(phase+'.log')).open('w') as log:
                    proc=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    update(child_pid=proc.pid,child_pgid=proc.pid)
                    try:code=proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        raise TimeoutError('Owned '+phase+' process exceeded timeout')
                update(child_pid=None,**{phase+'_exit_code':code,phase+'_seconds':time.monotonic()-start})
                if code:raise RuntimeError(phase+' failed; evidence kept, no blind retry')
                if phase=='m0':
                    gate=json.loads((run/'m0/m0_gate.json').read_text())
                    if gate.get('passed') is not True:raise RuntimeError('Real DDP M0 gate failed')
                    update(m0_passed=True)
            done=json.loads((baseline/'status.json').read_text())
            if done.get('state')!='completed' or not(baseline/'best_full_dev_metrics.json').exists():
                raise RuntimeError('Incomplete baseline result')
            update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),baseline_step=done['step'])
    except Exception as error:
        update(state='failed',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
