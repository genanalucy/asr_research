"""Evaluate a saved final checkpoint on a fixed dev set, without training."""
import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from train_ctc import Rows, Collate, infer
from common import load_manifest, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dev-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=load_manifest(a.dev_manifest)
    vocab=json.loads((a.checkpoint/'vocab.json').read_text())
    model=Wav2Vec2ForCTC.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False).cuda().eval()
    extractor=Wav2Vec2FeatureExtractor.from_pretrained(a.checkpoint,local_files_only=True)
    loader=DataLoader(Rows(rows),batch_size=2,shuffle=False,collate_fn=Collate(extractor,vocab),num_workers=2)
    metrics=infer(model,loader,torch.device('cuda'),vocab,a.output/'final_predictions.jsonl')
    write_json(a.output/'final_metrics.json',metrics)
    write_json(a.output/'evaluation_scope.json',{'checkpoint':str(a.checkpoint),'dev_manifest':str(a.dev_manifest),
              'test_used':False,'training_updates':0,'reason':'Unify final-step dev evaluation after detecting original monitoring subset race',
              'checkpoint_selection':'Fixed final step 1500, not reselected using these scores',
              'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    print(json.dumps(metrics,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
