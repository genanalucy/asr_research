"""Create private baseline manifests and a small M0 training fixture."""
import argparse
import collections
from pathlib import Path
from common import load_manifest, write_json, append_jsonl, normalize_text


def main():
    p=argparse.ArgumentParser();p.add_argument("split_root",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    training=load_manifest(a.split_root/"fixed8speakers_05h.jsonl")
    development=load_manifest(a.split_root/"dev.jsonl")
    sets={"train":training,"dev":development}
    reports={}
    filtered={}
    for name,rows in sets.items():
        keep=[r for r in rows if 0<float(r["duration"])<=30 and normalize_text(r["text"])]
        filtered[name]=keep
        for r in keep:append_jsonl(a.output/(name+".jsonl"),r)
        reports[name]={"source_rows":len(rows),"rows":len(keep),"excluded_over_30_or_empty":len(rows)-len(keep),
                       "hours":sum(r["duration"] for r in keep)/3600,
                       "speakers":len({r["speaker_candidate"] for r in keep})}
    assert not ({r["sha256"] for r in filtered["train"]}&{r["sha256"] for r in filtered["dev"]})
    assert not ({r["speaker_candidate"] for r in filtered["train"]}&{r["speaker_candidate"] for r in filtered["dev"]})
    by_speaker=collections.defaultdict(list)
    for r in sorted(filtered["train"],key=lambda r:r["sha256"]):
        if 2<=float(r["duration"])<=8 and 3<=len(normalize_text(r["text"]))<=120:
            by_speaker[r["speaker_candidate"]].append(r)
    small=[]
    while len(small)<16:
        before=len(small)
        for speaker in sorted(by_speaker):
            if by_speaker[speaker] and len(small)<16:small.append(by_speaker[speaker].pop(0))
        if before==len(small):raise ValueError("Not enough suitable M0 records")
    for r in small:append_jsonl(a.output/"sanity.jsonl",r)
    dev_by=collections.defaultdict(list)
    for r in sorted(filtered["dev"],key=lambda r:r["sha256"]):dev_by[r["speaker_candidate"]].append(r)
    # Fixed development monitoring subset; final dev/test evaluation is separate.
    monitor=[]
    while len(monitor)<128:
        before=len(monitor)
        for speaker in sorted(dev_by):
            if dev_by[speaker] and len(monitor)<128:monitor.append(dev_by[speaker].pop(0))
        if before==len(monitor):break
    for r in monitor:append_jsonl(a.output/"dev_monitor.jsonl",r)
    reports.update(sanity={"rows":len(small),"origin":"train only; M0 weights never initialize a reported baseline"},
                   dev_monitor={"rows":len(monitor),"role":"checkpoint monitoring; not final test performance"},
                   test_touched=False,scope="Baseline manifests retain original audio; shared loader builds a versioned 16kHz mono float cache")
    write_json(a.output/"summary.json",reports)
    print(reports,flush=True)


if __name__=="__main__":main()
