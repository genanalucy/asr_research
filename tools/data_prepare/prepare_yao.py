"""Join the two Yao annotation sources without assuming their speakers are distinct."""
import argparse
import collections
import json
from pathlib import Path
import re
from prepare_zhuang import read_rows, normalized, audio_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("inventory",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    audio=[json.loads(s) for s in a.inventory.read_text().splitlines() if s]
    by_path={r["path"]:r for r in audio}
    by_stem=collections.defaultdict(list)
    for r in audio: by_stem[audio_id(r["path"])].append(r)
    records=[]
    for source in sorted(a.root.glob("*/*.txt")):
        for lineno,line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(),1):
            if not line.strip(): continue
            fields=line.split("|",2)
            if len(fields)!=3:
                records.append({"source":"txt", "source_file":str(source),"source_row":lineno,"status":"malformed_label"})
                continue
            name,sid,text=fields
            relative_name=name.strip().replace("\\", "/")
            basename=relative_name.rsplit("/",1)[-1]
            candidate=by_path.get(str(source.parent/basename))
            records.append({"id":source.parent.name+"/"+basename,"source":"txt", "source_file":str(source),
                            "original_label_audio_path":name,
                            "source_split_hint":"test" if "test_data" in relative_name else "unspecified",
                            "source_row":lineno,"text":text,"text_nfc_whitespace":normalized(text),
                            "speaker_candidate":"txt_declared_"+sid.strip(),"speaker_verified":False,
                            "source_recording_candidate":re.sub(r"_\d+_\d+_\d+$","",Path(basename).stem) if source.parent.name=="0" else None,
                            "candidates":[candidate] if candidate else []})
    for source in a.root.rglob("*.xlsx"):
        rows=list(read_rows(source));header=rows.pop(0)
        if header.get("A")!="录音名称" or header.get("B")!="瑶语": raise ValueError("Unexpected Yao XLSX schema")
        for lineno,row in enumerate(rows,2):
            key=row.get("A","").strip()
            records.append({"id":"xlsx/"+key,"source":"xlsx","source_file":str(source),"source_row":lineno,
                            "text":row.get("B",""),"text_nfc_whitespace":normalized(row.get("B","")),
                            "translation_zh":row.get("C",""),"speaker_candidate":"xlsx_prefix_"+key.rsplit("-",1)[0],
                            "speaker_verified":False,"source_recording_candidate":None,"candidates":by_stem.get(audio_id(key),[])})
    ids=collections.Counter(r.get("id") for r in records)
    stats=collections.Counter(); source_stats=collections.defaultdict(collections.Counter)
    speaker_stats=collections.defaultdict(lambda:{"rows":0,"paired_hours":0.0})
    used=set(); hashes=collections.defaultdict(list)
    for r in records:
        if r.get("status")=="malformed_label":
            stats[r["status"]]+=1;source_stats[r["source"]][r["status"]]+=1;continue
        c=r.pop("candidates");r["audio_candidates"]=c
        status="paired"
        if not r["text_nfc_whitespace"]:status="empty_transcript"
        elif ids[r["id"]]>1:status="duplicate_label_id"
        elif not c:status="missing_audio"
        elif len(c)>1:status="multiple_audio_candidates"
        elif c[0]["status"]!="header_ok" or c[0].get("duration",0)<=0:status="invalid_audio"
        r["status"]=status;stats[status]+=1;source_stats[r["source"]][status]+=1
        speaker_stats[r["speaker_candidate"]]["rows"]+=1
        if status=="paired":
            item=c[0];used.add(item["path"])
            r.update(audio=item["path"],duration=item["duration"],sample_rate=item["sample_rate"],channels=item["channels"],sha256=item["sha256"])
            speaker_stats[r["speaker_candidate"]]["paired_hours"]+=item["duration"]/3600
            source_stats[r["source"]]["paired_hours"]+=item["duration"]/3600
            hashes[item["sha256"]].append(r)
    texts=collections.Counter(r.get("text_nfc_whitespace","") for r in records)
    mapping={k:"speaker_candidate_%03d"%(i+1) for i,k in enumerate(sorted(speaker_stats))}
    summary={"label_rows":len(records),"pairing_status":dict(stats),"source_statistics":{k:dict(v) for k,v in source_stats.items()},
             "paired_hours":sum(r.get("duration",0) for r in records if r["status"]=="paired")/3600,
             "unique_paired_file_hours":sum(v[0]["duration"] for v in hashes.values())/3600,
             "unreferenced_audio_files":sum(r["path"] not in used for r in audio),
             "speaker_candidate_count":len(speaker_stats),"speaker_verified":False,
             "speaker_distribution":{mapping[k]:v for k,v in speaker_stats.items()},
             "duplicate_text_groups":sum(v>1 and bool(k) for k,v in texts.items()),
             "duplicate_audio_groups":sum(len(v)>1 for v in hashes.values()),
             "audio_hash_transcript_conflicts":sum(len({r["text_nfc_whitespace"] for r in v})>1 for v in hashes.values()),
             "language_variant":"unconfirmed; Latin orthography is compatible with Iu Mien but not certified",
             "source_recording_groups_txt0":len({r["source_recording_candidate"] for r in records if r.get("source_recording_candidate")}),
             "source_test_hint_rows":sum(r.get("source_split_hint")=="test" for r in records),
             "scope":"TXT field 2 provides declared speaker IDs; XLSX uses filename prefixes. Identities across sources are unverified; their counts must not be presented as distinct people."}
    (a.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    with (a.output/"manifest.jsonl").open("w") as dst:
        for r in records: dst.write(json.dumps(r,ensure_ascii=False)+"\n")
    (a.output/"alignment_issues.json").write_text(json.dumps([r for r in records if r["status"]!="paired"],ensure_ascii=False,indent=2)+"\n")
    (a.output/"speaker_map_private.json").write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
