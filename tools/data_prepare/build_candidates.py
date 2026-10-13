"""Deduplicate structurally paired examples and hold out inconsistent labels for review."""
import argparse
import collections
import json
from pathlib import Path
from prepare_zhuang import normalized


def main():
    p=argparse.ArgumentParser()
    p.add_argument("prepared",type=Path)
    a=p.parse_args()
    rows=[json.loads(x) for x in (a.prepared/"manifest.jsonl").read_text().splitlines() if x]
    groups=collections.defaultdict(list)
    for r in rows:
        if r.get("status")=="paired":groups[r["sha256"]].append(r)
    candidates=[];held=[];cross_speaker=[];case_only=0
    for group in groups.values():
        texts={normalized(r["text"]) for r in group}
        if len({r["speaker_candidate"] for r in group})>1:cross_speaker.append(group)
        if len(texts)>1:
            held.extend(group)
            if len({x.casefold() for x in texts})==1:case_only+=1
        else:
            r=dict(group[0]);r["verification_status"]="structural_candidate_not_human_verified"
            r["duplicate_provenance"]=[{"id":x["id"],"audio":x["audio"],"speaker_candidate":x["speaker_candidate"],
                                        "source_row":x["source_row"],"source_split_hint":x.get("source_split_hint")} for x in group]
            r["cross_speaker_identity_warning"]=len({x["speaker_candidate"] for x in group})>1
            candidates.append(r)
    hours=sum(r["duration"] for r in candidates)/3600
    distribution=collections.defaultdict(lambda:{"files":0,"hours":0.0})
    for r in candidates:
        distribution[r["speaker_candidate"]]["files"]+=1
        distribution[r["speaker_candidate"]]["hours"]+=r["duration"]/3600
    sorted_distribution=sorted(distribution.values(),key=lambda v:v["hours"],reverse=True)
    report={"paired_examples":sum(len(g) for g in groups.values()),"paired_unique_audio":len(groups),
            "inconsistent_label_groups":sum(len({normalized(x["text"]) for x in g})>1 for g in groups.values()),
            "inconsistent_label_examples":len(held),"case_only_inconsistent_groups":case_only,
            "cross_speaker_exact_audio_groups":len(cross_speaker),
            "structural_candidate_files":len(candidates),"structural_candidate_hours":hours,
            "candidate_speaker_counts_anonymized":sorted_distribution,
            "limitations":"No automatic train/dev/test split; source-domain, speaker aliases and label accuracy require review. Audio headers/hashes are not full decode or listening verification."}
    with (a.prepared/"candidates_deduplicated.jsonl").open("w") as out:
        for r in candidates:out.write(json.dumps(r,ensure_ascii=False)+"\n")
    (a.prepared/"inconsistent_labels_for_review.json").write_text(json.dumps(held,ensure_ascii=False,indent=2)+"\n")
    (a.prepared/"cross_speaker_duplicates_for_review.json").write_text(json.dumps(cross_speaker,ensure_ascii=False,indent=2)+"\n")
    (a.prepared/"candidate_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
