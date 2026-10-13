"""Deterministic stratified full-decoding spot check; raw audio is left unchanged."""
import argparse
import collections
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess


def inspect(row):
    cmd=["ffmpeg","-nostdin","-hide_banner","-i",row["audio"],"-vn","-af","astats=metadata=0:reset=0","-f","null","-"]
    try:
        out=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        rms=re.findall(r"RMS level dB:\s*([-+\w.]+)",out.stderr)
        peak=re.findall(r"Peak level dB:\s*([-+\w.]+)",out.stderr)
        return {"audio":row["audio"],"sha256":row["sha256"],"returncode":out.returncode,
                "rms_db":float(rms[-1]) if rms else None,"peak_db":float(peak[-1]) if peak else None,
                "decode_error":out.stderr[-1500:] if out.returncode else None}
    except Exception as e:return {"audio":row["audio"],"error":str(e)}


def main():
    p=argparse.ArgumentParser();p.add_argument("prepared",type=Path);p.add_argument("--samples",type=int,default=64)
    a=p.parse_args()
    rows=[json.loads(x) for x in (a.prepared/"candidates_deduplicated.jsonl").read_text().splitlines()]
    groups=collections.defaultdict(list)
    for r in sorted(rows,key=lambda x:x["sha256"]):
        groups[(Path(r["audio"]).suffix.lower(),r["sample_rate"],r["channels"])].append(r)
    selected=[]
    while len(selected)<min(a.samples,len(rows)):
        for key in sorted(groups):
            if groups[key] and len(selected)<a.samples:selected.append(groups[key].pop())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(inspect,selected))
    summary={"sample_count":len(results),"selection":"Deterministic stratification by container, sample rate and channels; full decode of sampled files only",
             "decode_failures":sum(r.get("returncode",1)!=0 for r in results),
             "quiet_rms_below_minus45_db":sum(r.get("rms_db") is not None and r["rms_db"] < -45 for r in results),
             "peak_near_full_scale":sum(r.get("peak_db") is not None and r["peak_db"] >= -.1 for r in results),
             "minimum_rms_db":min((r["rms_db"] for r in results if r.get("rms_db") is not None),default=None),
             "limitation":"No human transcript-audio accuracy audit, speaker identification, or certification of unsampled files."}
    (a.prepared/"decode_sample_private.json").write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=True)+"\n")
    (a.prepared/"decode_sample_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=True)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=True))


if __name__=="__main__":main()
