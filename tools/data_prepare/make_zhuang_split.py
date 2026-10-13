"""Make reproducible speaker-disjoint draft splits for the Zhuang candidates.

The input is the structural candidate JSONL produced by ``build_candidates.py``.
This script deliberately creates experiment manifests only; it does not copy or
modify audio and it does not start training.  Output directories should remain
on the private data server.
"""

import argparse
import collections
import itertools
import json
import os
import random
import re
import unicodedata
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".wma", ".ogg", ".aac", ".opus"}


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def basename_key(value):
    """Normalize a recording basename while preserving the original ID in output."""
    name = unicodedata.normalize("NFC", os.path.basename(str(value or "")).strip()).casefold()
    # An ID may appear with one or more audio suffixes in source inventories.
    while Path(name).suffix.lower() in AUDIO_EXTENSIONS:
        name = name[: -len(Path(name).suffix)]
    return name


def text_type(text):
    text = str(text or "")
    if not normalized(text):
        return "empty"
    flags = {
        "cjk": bool(re.search(r"[\u3400-\u9fff]", text)),
        "latin": bool(re.search(r"[A-Za-z]", text)),
        "digit": bool(re.search(r"\d", text)),
        "other": bool(re.search(r"[^\w\s\u3400-\u9fff]", text, re.UNICODE)),
    }
    present = [key for key, value in flags.items() if value]
    return present[0] if len(present) == 1 else "mixed"


def chars(text):
    return [char for char in normalized(text) if not char.isspace()]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def collect_conflict_keys(path):
    """Read nested conflict groups without retaining private transcript rows."""
    keys = {"sha256": set(), "id": set(), "basename": set()}
    if not path or not Path(path).exists():
        return keys

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if value.get("sha256"):
                keys["sha256"].add(value["sha256"])
            if value.get("id"):
                keys["id"].add(normalized(value["id"]).casefold())
            audio = value.get("audio") or value.get("path")
            if audio:
                keys["basename"].add(basename_key(audio))
            for item in value.values():
                if isinstance(item, (dict, list)):
                    visit(item)

    with Path(path).open(encoding="utf-8") as source:
        visit(json.load(source))
    return keys


def candidate_keys(row):
    original_id = normalized(row.get("id")).casefold()
    audio = row.get("audio") or row.get("path")
    return {
        "sha256": str(row.get("sha256") or ""),
        "id": original_id,
        "basename": basename_key(audio),
    }


def filter_candidates(rows, conflict_keys):
    valid = []
    excluded = []
    for index, row in enumerate(rows):
        reason = None
        if row.get("status") != "paired":
            reason = "status_not_paired"
        elif len(row.get("audio_candidates") or []) != 1:
            reason = "multiple_or_missing_audio_candidates"
        elif not row.get("sha256"):
            reason = "missing_audio_sha256"
        elif not row.get("audio"):
            reason = "missing_audio_path"
        elif float(row.get("duration") or 0) <= 0:
            reason = "invalid_duration"
        elif not normalized(row.get("text")):
            reason = "empty_transcript"
        else:
            keys = candidate_keys(row)
            if keys["sha256"] in conflict_keys["sha256"]:
                reason = "hash_transcript_conflict"
            elif keys["id"] in conflict_keys["id"]:
                reason = "conflict_original_recording_id"
            elif keys["basename"] in conflict_keys["basename"]:
                reason = "conflict_audio_basename"
        if reason:
            excluded.append({"source_index": index, "reason": reason, "id": row.get("id"),
                             "speaker": row.get("speaker_candidate"), "sha256": row.get("sha256")})
            continue
        valid.append(dict(row))
    return valid, excluded


def build_components(rows):
    """Union rows on SHA, original ID, or recording basename."""
    uf = UnionFind(len(rows))
    first = {"sha256": {}, "id": {}, "basename": {}}
    for index, row in enumerate(rows):
        for key, value in candidate_keys(row).items():
            if not value:
                continue
            previous = first[key].get(value)
            if previous is None:
                first[key][value] = index
            else:
                uf.union(index, previous)
    components = collections.defaultdict(list)
    for index, row in enumerate(rows):
        components[uf.find(index)].append(index)
    return list(components.values())


def make_speaker_units(rows, components):
    speaker_components = collections.defaultdict(set)
    for component in components:
        speakers = {rows[index].get("speaker_candidate") or "unknown" for index in component}
        for speaker in speakers:
            speaker_components[speaker].update(speakers)
    # A connected speaker component is kept together so duplicate keys can never
    # put the same recording in two speaker-disjoint collections.
    units = []
    pending = set(speaker_components)
    while pending:
        root = sorted(pending)[0]
        pending.remove(root)
        unit = set(speaker_components[root])
        changed = True
        while changed:
            changed = False
            for speaker in sorted(pending):
                if unit.intersection(speaker_components[speaker]):
                    unit.update(speaker_components[speaker])
                    pending.remove(speaker)
                    changed = True
        units.append(unit)
    return units


def hours_by_speaker(rows):
    result = collections.defaultdict(float)
    for row in rows:
        result[row.get("speaker_candidate") or "unknown"] += float(row["duration"]) / 3600.0
    return result


def choose_unit_subset(units, target_count, hours, target_hours, rng):
    """Choose units with an exact count where possible and closest available hours."""
    if target_count <= 0:
        return set(), "feasible"
    ordered = list(units)
    ordered.sort(key=lambda unit: tuple(sorted(unit)))
    rng.shuffle(ordered)
    unit_hours = [sum(hours.get(speaker, 0.0) for speaker in unit) for unit in ordered]
    states = {0: [(0.0, tuple())]}
    for index, unit in enumerate(ordered):
        additions = {}
        count = len(unit)
        for used_count, values in list(states.items()):
            new_count = used_count + count
            if new_count > target_count:
                continue
            entries = additions.setdefault(new_count, [])
            for current_hours, selected in values:
                entries.append((current_hours + unit_hours[index], selected + (index,)))
        for count_key, entries in additions.items():
            states.setdefault(count_key, []).extend(entries)
            # Keep deterministic diversity while avoiding combinatorial growth.
            if len(states[count_key]) > 600:
                states[count_key] = sorted(
                    states[count_key], key=lambda item: (abs(item[0] - target_hours), item[0])
                )[:600]
    candidates = states.get(target_count, [])
    if not candidates:
        all_candidates = [(count, item) for count, values in states.items() for item in values]
        if not all_candidates:
            return set(), "infeasible_no_speakers"
        count, (actual_hours, selected) = min(
            all_candidates, key=lambda item: (abs(item[0] - target_count), abs(item[1][0] - target_hours))
        )
        return set().union(*(ordered[index] for index in selected)), "infeasible_speaker_count"
    at_or_above = [item for item in candidates if item[0] >= target_hours]
    actual_hours, selected = min(
        at_or_above or candidates,
        key=lambda item: (abs(item[0] - target_hours), item[0]),
    )
    return set().union(*(ordered[index] for index in selected)), "feasible"


def choose_holdout_units(units, target_count, rng):
    """Randomly select holdout speaker units without favoring short recordings."""
    if target_count <= 0:
        return set(), "feasible"
    ordered = list(units)
    ordered.sort(key=lambda unit: tuple(sorted(unit)))
    rng.shuffle(ordered)
    selected = []
    count = 0
    for unit in ordered:
        if count + len(unit) <= target_count:
            selected.append(unit)
            count += len(unit)
        if count == target_count:
            break
    if count < target_count:
        return set().union(*selected) if selected else set(), "infeasible_speaker_count"
    return set().union(*selected), "feasible"


def choose_rows_for_target(rows, speakers, target_hours, rng):
    pool = [row for row in rows if (row.get("speaker_candidate") or "unknown") in speakers]
    if not pool:
        return []
    # Keep every selected speaker represented, then fill the target with the
    # largest row that still fits. This leaves only a short final-row rounding.
    by_speaker = collections.defaultdict(list)
    for row in pool:
        by_speaker[row.get("speaker_candidate") or "unknown"].append(row)
    selected = []
    selected_ids = set()
    for speaker in sorted(by_speaker):
        shortest = min(by_speaker[speaker], key=lambda row: (float(row["duration"]), str(row.get("id"))))
        selected.append(shortest)
        selected_ids.add(id(shortest))
    current = sum(float(row["duration"]) for row in selected) / 3600.0
    remaining = [row for row in pool if id(row) not in selected_ids]
    rng.shuffle(remaining)
    # One sorted pass avoids repeatedly scanning/removing a 60k-row pool. Since
    # clips are short, ascending accumulation reaches the hour target closely.
    remaining.sort(key=lambda row: (float(row["duration"]), str(row.get("id"))))
    best_over = None
    for row in remaining:
        if current >= target_hours:
            break
        proposed = current + float(row["duration"]) / 3600.0
        if proposed <= target_hours:
            selected.append(row)
            current = proposed
        elif best_over is None or abs(proposed - target_hours) < abs(best_over[0] - target_hours):
            best_over = (proposed, row)
    # A final over-budget row can be closer than the under-target result.
    if best_over is not None and abs(best_over[0] - target_hours) < abs(current - target_hours):
        selected.append(best_over[1])
    return sorted(selected, key=lambda row: (str(row.get("speaker_candidate")), str(row.get("id"))))


def annotate(row, collection, group):
    output = dict(row)
    output.pop("translation_zh", None)
    output["speaker"] = row.get("speaker_candidate") or "unknown"
    output["domain"] = row.get("domain") or "unknown"
    output["collection"] = collection
    output["group"] = group
    output["text_nfc_whitespace"] = normalized(row.get("text"))
    output["text_type"] = text_type(row.get("text"))
    output["char_count"] = len(chars(row.get("text")))
    output["audio_basename"] = os.path.basename(str(row.get("audio") or ""))
    return output


def statistics(rows):
    texts = [normalized(row.get("text")) for row in rows]
    char_lists = [chars(text) for text in texts]
    return {
        "rows": len(rows),
        "hours": sum(float(row.get("duration") or 0) for row in rows) / 3600.0,
        "actual_speakers": sorted({row.get("speaker_candidate") or "unknown" for row in rows}),
        "speaker_count": len({row.get("speaker_candidate") or "unknown" for row in rows}),
        "unique_texts": len(set(texts)),
        "text_type_counts": dict(collections.Counter(text_type(text) for text in texts)),
        "char_count_total": sum(len(chars_) for chars_ in char_lists),
        "unique_chars": len(set(char for chars_ in char_lists for char in chars_)),
        "domain_counts": dict(collections.Counter(row.get("domain") or "unknown" for row in rows)),
    }


def write_jsonl(path, rows, force):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError("Refusing to overwrite existing draft output: %s (use --force)" % path)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, value, force):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError("Refusing to overwrite existing draft output: %s (use --force)" % path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path, help="candidates_deduplicated.jsonl or its prepared directory")
    parser.add_argument("--conflicts", type=Path, help="hash_transcript_conflicts.json; defaults beside candidates")
    parser.add_argument("--output-root", type=Path, required=True, help="private data/splits/draft directory")
    parser.add_argument("--audit-root", type=Path, required=True, help="private audit/private/split_draft directory")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--test-speakers", type=int, default=2)
    parser.add_argument("--dev-speakers", type=int, default=2)
    parser.add_argument("--hour-tolerance", type=float, default=0.01,
                        help="small target-hour rounding tolerated for feasibility (default: 0.01 h)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    candidate_path = args.candidates / "candidates_deduplicated.jsonl" if args.candidates.is_dir() else args.candidates
    conflicts_path = args.conflicts or candidate_path.parent / "hash_transcript_conflicts.json"
    rows = read_jsonl(candidate_path)
    conflict_keys = collect_conflict_keys(conflicts_path)
    valid, excluded = filter_candidates(rows, conflict_keys)
    if not valid:
        raise SystemExit("No valid candidates remain after filtering")
    rng = random.Random(args.seed)
    components = build_components(valid)
    units = make_speaker_units(valid, components)
    hours = hours_by_speaker(valid)
    # Hold out speaker units first. Every subsequent group only sees train rows.
    test_speakers, test_status = choose_holdout_units(units, args.test_speakers, rng)
    remaining_units = [unit for unit in units if not unit.intersection(test_speakers)]
    dev_speakers, dev_status = choose_holdout_units(remaining_units, args.dev_speakers, rng)
    train_speakers = set().union(*(unit for unit in remaining_units if not unit.intersection(dev_speakers)))
    split_speakers = {"train": train_speakers, "dev": dev_speakers, "test": test_speakers}
    split_rows = {}
    for split, speakers in split_speakers.items():
        split_rows[split] = [row for row in valid if (row.get("speaker_candidate") or "unknown") in speakers]
        write_jsonl(args.output_root / (split + ".jsonl"),
                    [annotate(row, split, "base") for row in split_rows[split]], args.force)

    groups = {}
    # Fixed 5-hour groups by speaker count and fixed 8-speaker groups by hours.
    requests = [("fixed5h_%02dspeakers" % count, count, 5.0) for count in (1, 2, 4, 8)]
    train_units = [unit for unit in units if unit.issubset(train_speakers)]
    for group_name, target_speakers, target_hours in requests:
        selected_speakers, status = choose_unit_subset(train_units, target_speakers, hours, target_hours, rng)
        selected = choose_rows_for_target(split_rows["train"], selected_speakers, target_hours, rng)
        actual = statistics(selected)
        shortfall = max(0.0, target_hours - actual["hours"])
        feasible = (status == "feasible" and actual["speaker_count"] == target_speakers
                    and shortfall <= args.hour_tolerance)
        if feasible:
            feasibility = "feasible"
        elif status == "feasible" and actual["hours"] < target_hours:
            feasibility = "infeasible_insufficient_hours"
        else:
            feasibility = status
        groups[group_name] = {
            "target_hours": target_hours,
            "target_speakers": target_speakers,
            "feasibility": feasibility,
            "actual_hours": actual["hours"],
            "actual_speakers": actual["speaker_count"],
            "shortfall_hours": shortfall,
            "speaker_ids": sorted(actual["actual_speakers"]),
            "stats": actual,
        }
        write_jsonl(args.output_root / (group_name + ".jsonl"),
                    [annotate(row, "train", group_name) for row in selected], args.force)
    # The fixed-speaker hour sweep deliberately shares one 8-speaker pool so
    # that 5/10/20h comparisons change duration without changing identities.
    fixed8_speakers, fixed8_status = choose_unit_subset(train_units, 8, hours, 20.0, rng)
    for target_hours in (5.0, 10.0, 20.0):
        group_name = "fixed8speakers_%02dh" % int(target_hours)
        selected = choose_rows_for_target(split_rows["train"], fixed8_speakers, target_hours, rng)
        actual = statistics(selected)
        shortfall = max(0.0, target_hours - actual["hours"])
        feasible = (fixed8_status == "feasible" and actual["speaker_count"] == 8
                    and shortfall <= args.hour_tolerance)
        if feasible:
            feasibility = "feasible"
        elif fixed8_status != "feasible":
            feasibility = fixed8_status
        else:
            feasibility = "infeasible_insufficient_hours"
        groups[group_name] = {
            "target_hours": target_hours,
            "target_speakers": 8,
            "feasibility": feasibility,
            "actual_hours": actual["hours"],
            "actual_speakers": actual["speaker_count"],
            "shortfall_hours": shortfall,
            "speaker_ids": sorted(actual["actual_speakers"]),
            "stats": actual,
        }
        write_jsonl(args.output_root / (group_name + ".jsonl"),
                    [annotate(row, "train", group_name) for row in selected], args.force)
    excluded_path = args.audit_root / "excluded_rows.jsonl"
    write_jsonl(excluded_path, excluded, args.force)
    summary = {
        "schema": "zhuang_split_draft_v1",
        "seed": args.seed,
        "hour_tolerance": args.hour_tolerance,
        "input": str(candidate_path),
        "conflicts": str(conflicts_path),
        "filter_counts": {"input_rows": len(rows), "valid_rows": len(valid), "excluded_rows": len(excluded),
                           "excluded_reasons": dict(collections.Counter(item["reason"] for item in excluded))},
        "component_count": len(components),
        "split_status": {"test": test_status, "dev": dev_status},
        "splits": {name: statistics(rows_) for name, rows_ in split_rows.items()},
        "groups": groups,
        "constraints": {
            "speaker_disjoint": True,
            "audio_sha256_disjoint": True,
            "original_id_and_audio_basename_disjoint": True,
            "speaker_identity_policy": "use speaker_candidate as provided; do not merge unverified aliases",
            "group_manifests_are_alternative_training_subsets": True,
            "domain_default": "unknown",
            "training_started": False,
            "raw_manifests_published": False,
        },
    }
    write_json(args.audit_root / "selection_summary.json", summary, args.force)
    write_json(args.output_root / "selection_summary.json", summary, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
