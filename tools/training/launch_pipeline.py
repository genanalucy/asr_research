"""Run one real M0 gate followed by a fresh five-hour baseline, offline."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import write_json, test_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument("--family",choices=["ctc","whisper"],required=True)
    p.add_argument("--gpu",type=int,required=True);p.add_argument("--run-name",required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    run=root/"runs"/a.run_name
    run.mkdir(parents=True,exist_ok=False)
    state={"family":a.family,"gpu":a.gpu,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "state":"preflight","test_evaluated":False,"mode":"engineering_baseline_fixed_dev_subset"}
    write_json(run/"pipeline_status.json",state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
             HF_HUB_DISABLE_XET="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4",
             ASR_AUDIO_CACHE=str(root/"data/audio_cache/16k_float_mono_ffmpeg42_v1"))
    used=subprocess.check_output(["nvidia-smi","--id="+str(a.gpu),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip()
    if int(used)>500:
        state.update(state="blocked_gpu_busy",memory_used_mib=int(used));write_json(run/"pipeline_status.json",state)
        raise RuntimeError("Chosen GPU is occupied; no other process will be changed")
    test_metrics()
    manifests=root/"data/splits/baseline_v1"
    script=root/"tools/training"/("train_"+a.family+".py")
    model=root/"data/model_cache"/("wav2vec2-xls-r-300m" if a.family=="ctc" else "whisper-small")
    shared=[sys.executable,"-u",str(script),"--model-dir",str(model),"--seed","42"]
    if a.family=="ctc":
        shared += ["--vocab-manifest",str(manifests/"train.jsonl"),"--workers","2","--gradient-checkpointing"]
        sanity=shared+["--mode","sanity","--train-manifest",str(manifests/"sanity.jsonl"),
                       "--dev-manifest",str(manifests/"sanity.jsonl"),"--output",str(run/"m0"),
                       "--max-steps","1500","--batch-size","2","--grad-accum","1","--sanity-examples","16","--lr","0.0003"]
        training=shared+["--mode","train","--train-manifest",str(manifests/"train.jsonl"),
                        "--dev-manifest",str(manifests/"dev_monitor.jsonl"),"--output",str(run/"baseline"),
                        "--max-steps","1500","--batch-size","2","--grad-accum","8","--eval-limit","128", "--eval-every","500","--lr","0.0001"]
    else:
        sanity=shared+["--mode","sanity","--train-manifest",str(manifests/"sanity.jsonl"),"--output",str(run/"m0"),
                       "--max-steps","400","--batch-size","2","--grad-accum","1","--lr","0.00003","--eval-every","100"]
        training=shared+["--mode","train","--train-manifest",str(manifests/"train.jsonl"),
                        "--dev-manifest",str(manifests/"dev_monitor.jsonl"),"--output",str(run/"baseline"),
                        "--max-steps","1500","--batch-size","4","--grad-accum","4","--eval-limit","128","--eval-every","500","--lr","0.00001"]
    write_json(run/"launch_commands.json",{"sanity":sanity,"baseline":training,"environment":{k:env[k] for k in ["CUDA_VISIBLE_DEVICES","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","ASR_AUDIO_CACHE"]},
                                          "baseline_initialization":"Fresh process loads original local pretrained model; M0 weights are never inherited",
                                          "test_usage":"No test manifest passed; reported metrics are engineering dev-monitor metrics"})
    for phase,cmd in [("m0",sanity),("baseline",training)]:
        state.update(state=phase+"_running");write_json(run/"pipeline_status.json",state)
        begin=time.monotonic()
        with (run/(phase+".log")).open("w") as logfile:
            proc=subprocess.Popen(cmd,env=env,stdout=logfile,stderr=subprocess.STDOUT,cwd=root)
            state.update(child_pid=proc.pid);write_json(run/"pipeline_status.json",state)
            try: code=proc.wait(timeout=4*3600 if phase=="m0" else 24*3600)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                state.update(state=phase+"_timeout",child_pid=None);write_json(run/"pipeline_status.json",state)
                raise RuntimeError("Owned training process exceeded recorded timeout")
        state[phase+"_exit_code"]=code;state[phase+"_seconds"]=time.monotonic()-begin
        state.update(child_pid=None)
        if code:
            state.update(state=phase+"_failed");write_json(run/"pipeline_status.json",state);raise RuntimeError(f"{phase} failed; no blind retry")
        if phase=="m0":
            gate=json.loads((run/"m0/m0_gate.json").read_text())
            if not gate.get("passed"):
                state.update(state="m0_gate_failed");write_json(run/"pipeline_status.json",state);raise RuntimeError("M0 did not pass")
            state["m0_passed"]=True;write_json(run/"pipeline_status.json",state)
    state.update(state="completed",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(run/"pipeline_status.json",state)


if __name__=="__main__":main()
