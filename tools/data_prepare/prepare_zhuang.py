"""Read source XLSX using stdlib and join labels to a completed audio inventory."""
import argparse
import collections
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".wma", ".ogg", ".aac", ".opus"}


def audio_id(name):
    name = unicodedata.normalize("NFC", Path(name).name.strip())
    while Path(name).suffix.lower() in AUDIO_EXTENSIONS:
        name = name[:-len(Path(name).suffix)]
    return name.casefold()


def read_rows(path):
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(el.itertext()) for el in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheets = sorted(n for n in archive.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
        for sheet in sheets:
            root = ET.fromstring(archive.read(sheet))
            for row in root.findall("s:sheetData/s:row", NS):
                cells = {}
                for cell in row:
                    col = re.sub(r"\d", "", cell.get("r", ""))
                    value = cell.find("s:v", NS)
                    text = value.text or "" if value is not None else ""
                    if cell.get("t") == "s":
                        text = strings[int(text)]
                    elif cell.get("t") == "inlineStr":
                        inline = cell.find("s:is", NS)
                        text = "".join(inline.itertext()) if inline is not None else ""
                    cells[col] = text
                yield cells


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speaker-prefix-verified", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels = list(read_rows(args.xlsx))
    header = labels.pop(0)
    if header.get("A") != "录音名称" or header.get("B") != "壮文":
        raise ValueError("Unexpected source schema; do not guess transcript columns")
    audio = [json.loads(s) for s in args.inventory.read_text().splitlines() if s]
    by_id = collections.defaultdict(list)
    for item in audio:
        by_id[audio_id(item["path"])].append(item)
    label_ids = collections.Counter(audio_id(r.get("A", "")) for r in labels)
    normalized_texts = collections.Counter(normalized(r.get("B", "")) for r in labels)
    stats = collections.Counter()
    stats_hours = collections.Counter()
    speakers = collections.defaultdict(lambda: {"rows": 0, "paired_hours": 0.0})
    used = set()
    manifest = []
    issues = []
    for index, row in enumerate(labels, 2):
        key = row.get("A", "").strip()
        text = row.get("B", "")
        canonical_id = audio_id(key)
        candidates = by_id.get(canonical_id, [])
        status = "paired"
        if not key:
            status = "missing_id"
        elif not normalized(text):
            status = "empty_transcript"
        elif label_ids[canonical_id] > 1:
            status = "duplicate_label_id"
        elif not candidates:
            status = "missing_audio"
        elif len(candidates) > 1:
            status = "multiple_audio_candidates"
        elif candidates[0]["status"] != "header_ok" or candidates[0].get("duration", 0) <= 0:
            status = "invalid_audio"
        prefix = key.rsplit("-", 1)[0] if "-" in key else "unknown"
        speakers[prefix]["rows"] += 1
        stats[status] += 1
        entry = {"id": key, "source_row": index, "text": text, "text_nfc_whitespace": normalized(text),
                 "canonical_audio_id": canonical_id,
                 "match_method": "casefold_and_strip_audio_extensions",
                 "translation_zh": row.get("C", ""), "speaker_candidate": prefix,
                 "speaker_verified": args.speaker_prefix_verified, "status": status, "audio_candidates": candidates}
        if status == "paired":
            item = candidates[0]
            used.add(item["path"])
            stats_hours[status] += item["duration"] / 3600
            speakers[prefix]["paired_hours"] += item["duration"] / 3600
            entry.update(audio=item["path"], duration=item["duration"], sample_rate=item["sample_rate"],
                         channels=item["channels"], sha256=item["sha256"])
        else:
            issues.append(entry)
        manifest.append(entry)
    hashes = collections.defaultdict(list)
    for r in manifest:
        if r["status"] == "paired":
            hashes[r["sha256"]].append(r)
    conflicts = [v for v in hashes.values() if len({normalized(r["text"]) for r in v}) > 1]
    unique_hours = sum(v[0]["duration"] for v in hashes.values()) / 3600
    sid_map = {key: "speaker_candidate_%03d" % (i+1) for i, key in enumerate(sorted(speakers))}
    report = {"source_label_rows": len(labels), "source_columns": header,
              "pairing_status": dict(stats), "paired_hours": stats_hours["paired"],
              "unique_paired_file_hours": unique_hours,
              "unreferenced_audio_files": sum(audio_id(r["path"]) not in label_ids for r in audio),
              "duplicate_label_ids": sum(v>1 for v in label_ids.values()),
              "duplicate_nonempty_text_groups": sum(v>1 and bool(k) for k, v in normalized_texts.items()),
              "unique_nonempty_texts": sum(bool(k) for k in normalized_texts),
              "audio_hash_transcript_conflicts": len(conflicts),
              "speaker_candidate_count": len(speakers), "speaker_verified": args.speaker_prefix_verified,
              "speaker_distribution": {sid_map[k]: v for k,v in speakers.items()},
              "text_contains_cjk_rows": sum(bool(re.search(r"[\u4e00-\u9fff]", r.get("B", ""))) for r in labels),
              "text_contains_digit_rows": sum(bool(re.search(r"\d",r.get("B", ""))) for r in labels),
              "text_nfc_changed_rows": sum(unicodedata.normalize("NFC",r.get("B", "")) != r.get("B", "") for r in labels),
              "scope": "Text is Zhuang column B; Chinese column C retained as translation, never used as ASR label. Speaker prefixes are verified only when the operator supplies --speaker-prefix-verified."}
    (args.output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    with (args.output/"manifest.jsonl").open("w") as dst:
        for row in manifest: dst.write(json.dumps(row,ensure_ascii=False)+"\n")
    (args.output/"alignment_issues.json").write_text(json.dumps(issues,ensure_ascii=False,indent=2)+"\n")
    (args.output/"hash_transcript_conflicts.json").write_text(json.dumps(conflicts,ensure_ascii=False,indent=2)+"\n")
    (args.output/"speaker_map_private.json").write_text(json.dumps(sid_map,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__": main()
