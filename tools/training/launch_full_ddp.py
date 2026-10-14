#!/usr/bin/env python3
"""Launch one fresh Whisper full-data run across multiple GPUs with torchrun/DDP."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import write_json, test_metrics
from launch_full_baseline import (
    digest, model_record, prepare_inputs, wait_dependencies,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def wait_gpus(gpus, run, state, timeout):
    deadline = time.monotonic() + timeout
    while True:
        used = {}
        for gpu in gpus:
            used[gpu] = int(subprocess.check_output([
                'nvidia-smi', '--id=' + str(gpu), '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True).strip())
        if all(value < 500 for value in used.values()):
            return
        if time.monotonic() > deadline:
            raise TimeoutError('A selected GPU stayed busy; no other process changed')
        state.update(state='waiting_gpus', gpu_memory_mib=used, updated_utc=utc_now())
        write_json(run / 'pipeline_status.json', state)
        time.sleep(20)


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
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise TimeoutError('Only this launcher-owned torchrun child exceeded ' + phase + ' timeout')
    state.update(child_pid=None, **{phase + '_exit_code': code, phase + '_seconds': time.monotonic() - started})
    write_json(run / 'pipeline_status.json', state)
    if code != 0:
        raise RuntimeError(phase + ' failed; evidence saved, no automatic retry')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', required=True, help='Comma-separated physical GPU ids, e.g. 4,5,6,7')
    parser.add_argument('--model', choices=['whisper-medium'], default='whisper-medium')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',') if x.strip()]
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        parser.error('--gpus requires at least two distinct ids')
    if args.epochs < 2 or args.lr <= 0:
        parser.error('epochs >= 2 and positive lr required')
    os.umask(0o077)
    root = Path(__file__).resolve().parents[2]
    run = root / 'runs' / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    state = {'state':'preparing','model':args.model,'gpus':gpus,'lr':args.lr,'seed':args.seed,
             'world_size':len(gpus),'launcher_pid':os.getpid(),'started_utc':utc_now(),'test_read':False}
    write_json(run/'pipeline_status.json',state)
    locks=[]
    try:
        model_dir=root/'data/model_cache'/args.model
        audit=root/'audit/private/whisper_medium_download.json'
        wait_dependencies(root,run,state,model_dir,audit,8*3600)
        for gpu in gpus:
            lock_path=root/'audit/private'/('full_gpu_'+str(gpu)+'.lock')
            lock=lock_path.open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(lock)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(str(x) for x in gpus),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',ASR_AUDIO_CACHE=str(root/'data/audio_cache/16k_float_mono_ffmpeg42_v1'))
        test_metrics()
        summaries=prepare_inputs(root,run,model_dir)
        steps_per_epoch=(summaries['train']['rows']+15)//16
        state.update(input_summary=summaries,steps_per_epoch=steps_per_epoch,max_steps=steps_per_epoch*args.epochs)
        code_dir=run/'code';code_dir.mkdir()
        for filename in ['train_whisper_ddp.py','common.py','launch_full_baseline.py']:
            shutil.copy2(root/'tools/training'/filename,code_dir/filename)
        code_hashes={name:digest(code_dir/name) for name in ['train_whisper_ddp.py','common.py','launch_full_baseline.py']}
        state['code_sha256']=code_hashes
        script=str(code_dir/'train_whisper_ddp.py')
        torchrun=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node='+str(len(gpus)),script]
        shared=['--model-dir',str(model_dir),'--seed',str(args.seed)]
        m0=torchrun+shared+['--mode','sanity','--train-manifest',str(run/'inputs/sanity.jsonl'),'--output',str(run/'m0'),'--max-steps','400','--batch-size','2','--grad-accum','1','--lr','0.00003','--eval-every','100']
        baseline=torchrun+shared+['--mode','train','--train-manifest',str(run/'inputs/train.jsonl'),'--dev-manifest',str(run/'inputs/dev_monitor.jsonl'),'--full-dev-manifest',str(run/'inputs/dev.jsonl'),'--output',str(run/'baseline'),'--max-steps',str(steps_per_epoch*args.epochs),'--batch-size','2','--grad-accum','2','--lr',str(args.lr),'--eval-limit','512','--eval-every','1000','--full-eval-every',str(steps_per_epoch),'--eval-batch-size','2','--workers','2','--skip-initial-eval','--early-stopping-patience','3','--min-steps',str(steps_per_epoch*3)]
        write_json(run/'launch_commands.json',{'m0':m0,'baseline':baseline,'gpus':gpus,'world_size':len(gpus),'global_effective_batch':16,'test_read':False,'checkpoint_selection':'complete_filtered_development'})
        wait_gpus(gpus,run,state,8*3600)
        run_phase(m0,'m0',run,root,env,state,4*3600)
        gate=json.loads((run/'m0/m0_gate.json').read_text())
        if gate.get('passed') is not True: raise RuntimeError('DDP M0 gate did not pass; baseline forbidden')
        state['m0_passed']=True;write_json(run/'pipeline_status.json',state)
        wait_gpus(gpus,run,state,8*3600)
        run_phase(baseline,'baseline',run,root,env,state,72*3600)
        completion=json.loads((run/'baseline/status.json').read_text())
        if completion.get('state')!='completed' or not (run/'baseline/best_full_dev_metrics.json').exists(): raise RuntimeError('DDP baseline lacks full-dev result')
        state.update(state='completed',completed_utc=utc_now(),baseline_step=completion['step']);write_json(run/'pipeline_status.json',state)
    except Exception as error:
        state.update(state='failed',error_type=type(error).__name__,error=str(error),updated_utc=utc_now());write_json(run/'pipeline_status.json',state);raise
    finally:
        for lock in locks: lock.close()


if __name__=='__main__': main()
