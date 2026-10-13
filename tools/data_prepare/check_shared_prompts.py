#!/usr/bin/env python3
"""Audit repeated training prompts without exporting private labels or names."""

import argparse
import collections
import itertools
import json
import math
import os
import statistics
import sys
import unicodedata


def norm_text(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def qtile(values, q):
    if not values:
        return None
    data = sorted(values)
    return data[math.floor((len(data) - 1) * q)]


def summary(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "p90": round(qtile(values, 0.90), 6),
        "max": round(max(values), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="-", help="JSONL path, or stdin")
    parser.add_argument("--report", required=True, help="Anonymous Markdown output")
    parser.add_argument("--private-fd", type=int, help="Write private mapping JSON to this FD")
    parser.add_argument("--private-path", help="Write private mapping JSON to this path")
    args = parser.parse_args()

    source = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    rows = []
    speaker_names = set()
    missing = collections.Counter()
    for line_no, line in enumerate(source, 1):
        row = json.loads(line)
        for field in ("text", "duration", "speaker_candidate", "sha256"):
            if row.get(field) in (None, ""):
                missing[field] += 1
        text = norm_text(row["text"])
        duration = float(row["duration"])
        speaker = str(row["speaker_candidate"])
        sha = str(row["sha256"])
        if not text or duration <= 0 or not sha:
            continue
        speaker_names.add(speaker)
        rows.append({"line": line_no, "text": text, "duration": duration,
                     "speaker": speaker, "sha": sha})
    if source is not sys.stdin:
        source.close()

    speakers = sorted(speaker_names)
    speaker_index = {name: i for i, name in enumerate(speakers)}
    # Prompt index and per-speaker source records. SHA is the source identity
    # available in this manifest; repeated hashes are counted once.
    prompt_speakers = collections.defaultdict(lambda: collections.defaultdict(dict))
    prompt_index = {}
    for row in rows:
        text = row["text"]
        if text not in prompt_index:
            prompt_index[text] = len(prompt_index)
        by_sha = prompt_speakers[text][row["speaker"]]
        if row["sha"] not in by_sha:
            by_sha[row["sha"]] = row

    speaker_bits = [0] * len(speakers)
    prompt_mask = {}
    prompt_sources = {}
    sha_owner = collections.defaultdict(set)
    duplicate_sha_rows = 0
    for prompt, by_speaker in prompt_speakers.items():
        mask = 0
        eligible = {}
        all_sha = collections.defaultdict(list)
        for speaker, source_rows in by_speaker.items():
            for sha, row in source_rows.items():
                all_sha[sha].append((speaker, row))
        for sha, entries in all_sha.items():
            if len(entries) > 1:
                duplicate_sha_rows += len(entries)
                continue
            speaker, row = entries[0]
            eligible.setdefault(speaker, []).append(row)
            sha_owner[sha].add(speaker)
        for speaker, source_rows in eligible.items():
            if not source_rows:
                continue
            mask |= 1 << speaker_index[speaker]
            speaker_bits[speaker_index[speaker]] |= 1 << prompt_index[prompt]
        if mask:
            prompt_mask[prompt] = mask
            prompt_sources[prompt] = eligible

    unique_prompts = len(prompt_index)
    prompt_count_dist = collections.Counter(mask.bit_count() for mask in prompt_mask.values())
    thresholds = {str(k): sum(n >= k for n in prompt_count_dist.elements()) for k in (2, 3, 4, 8)}

    def top_groups(k, limit=3):
        ranked = []
        for indexes in itertools.combinations(range(len(speakers)), k):
            bits = (1 << unique_prompts) - 1
            for i in indexes:
                bits &= speaker_bits[i]
                if not bits:
                    break
            count = bits.bit_count()
            if count:
                ranked.append((count, indexes))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return ranked[:limit]

    def group_details(indexes, prompt_limit=100000):
        names = [speakers[i] for i in indexes]
        common = []
        for prompt, mask in prompt_mask.items():
            if all(mask & (1 << i) for i in indexes):
                common.append(prompt)
        rows_by_speaker = {name: [] for name in names}
        all_rows_by_speaker = {name: [] for name in names}
        for prompt in common:
            for name in names:
                sources = sorted(prompt_sources[prompt][name], key=lambda r: r["sha"])
                rows_by_speaker[name].append(sources[0])
                all_rows_by_speaker[name].extend(sources)
        person_hours = {name: sum(r["duration"] for r in rs) / 3600
                        for name, rs in rows_by_speaker.items()}
        multi_source_hours = {name: sum(r["duration"] for r in rs) / 3600
                              for name, rs in all_rows_by_speaker.items()}
        lengths = [len(prompt) for prompt in common]
        unique_word_counts = [len(set(prompt.split())) for prompt in common]
        return {
            "speaker_indexes": indexes,
            "shared_prompts": common,
            "count": len(common),
            "person_hours": person_hours,
            "multi_source_hours": multi_source_hours,
            "character_lengths": lengths,
            "unique_word_counts": unique_word_counts,
            "rows_by_speaker": rows_by_speaker,
            "all_rows_by_speaker": all_rows_by_speaker,
        }

    pair_ranked = top_groups(2, limit=max(1, math.comb(len(speakers), 2)))
    pair_details = [group_details(indexes) for _, indexes in pair_ranked]
    max_pair_count = pair_details[0]["count"] if pair_details else 0
    top_pair_details = [d for d in pair_details if d["count"] == max_pair_count][:10]

    for d in pair_details:
        indexes = d["speaker_indexes"]
        d["duration_matched"] = {}
        for tolerance in (0.10, 0.20):
            accepted = []
            for prompt in d["shared_prompts"]:
                per_speaker = [sorted(rs, key=lambda r: r["sha"])[0]["duration"]
                               for rs in (prompt_sources[prompt][speakers[i]] for i in indexes)]
                shortest = min(per_speaker)
                if shortest > 0 and max(per_speaker) / shortest - 1 <= tolerance:
                    accepted.append((prompt, per_speaker))
            hours = [sum(times[i] for _, times in accepted) / 3600
                     for i in range(len(indexes))]
            d["duration_matched"][str(int(tolerance * 100))] = {
                "prompts": [p for p, _ in accepted],
                "per_person_hours": hours,
            }

    duration_top_three = sorted(
        pair_details,
        key=lambda d: (-min(d["duration_matched"]["20"]["per_person_hours"]),
                       -len(d["duration_matched"]["20"]["prompts"]), d["speaker_indexes"]),
    )[:3]
    duration_matched_pairs = [{
        "shared_prompt_count": d["count"],
        "per_person_hours_on_one_source_per_prompt": sorted(d["person_hours"].values()),
        "within_prompt_duration_tolerance": {
            tolerance: {
                "prompt_count": len(d["duration_matched"][tolerance]["prompts"]),
                "per_person_hours": sorted(d["duration_matched"][tolerance]["per_person_hours"]),
                "meets_minimum_per_person_hours": {
                    str(target): min(d["duration_matched"][tolerance]["per_person_hours"]) >= target
                    for target in (0.5, 1.0, 5.0)
                },
            } for tolerance in ("10", "20")
        },
    } for d in duration_top_three]

    # Unique-word-count matches are a separate, weaker text control.
    pair_only_word_counts = []
    for d in top_pair_details:
        i, j = d["speaker_indexes"]
        left = collections.defaultdict(set)
        right = collections.defaultdict(set)
        for p, mask in prompt_mask.items():
            wc = len(set(p.split()))
            if mask & (1 << i):
                left[wc].add(p)
            if mask & (1 << j):
                right[wc].add(p)
        exact = set(d["shared_prompts"])
        overlapping_word_counts = left.keys() & right.keys()
        left_in_matching_bins = {p for wc in overlapping_word_counts for p in left[wc]} - exact
        right_in_matching_bins = {p for wc in overlapping_word_counts for p in right[wc]} - exact
        pair_only_word_counts.append({
            "same_prompt_count": len(exact),
            "shared_unique_word_count_bins": len(overlapping_word_counts),
            "left_distinct_text_prompts_in_shared_word_count_bins": len(left_in_matching_bins),
            "right_distinct_text_prompts_in_shared_word_count_bins": len(right_in_matching_bins),
            "limitation": "Counts prompts eligible for cardinality-only matching; their actual text differs and they are not exact-prompt samples.",
        })

    report = {
        "scope": "train.jsonl only",
        "normalization": "NFC plus collapsed whitespace; case and punctuation preserved",
        "rows_with_valid_required_fields": len(rows),
        "required_field_missing_counts": dict(missing),
        "speaker_candidate_count": len(speakers),
        "unique_normalized_prompts": unique_prompts,
        "duplicate_sha_rows_excluded_within_same_prompt": duplicate_sha_rows,
        "prompt_counts_by_independent_speaker_count": {
            str(n): count for n, count in sorted(prompt_count_dist.items())
        },
        "prompts_shared_by_at_least_n_speakers": thresholds,
        "top_three_pairs_by_20_percent_duration_matched_minimum_person_hours": duration_matched_pairs,
        "top_pair_exact_text_and_audio_stats": [
            {
                "shared_prompt_count": d["count"],
                "per_person_hours_one_source_per_prompt": sorted(d["person_hours"].values()),
                "per_person_hours_all_distinct_sources": sorted(d["multi_source_hours"].values()),
                "minimum_person_hours": min(d["person_hours"].values()),
                "prompt_character_length": summary(d["character_lengths"]),
                "unique_word_count": summary(d["unique_word_counts"]),
            } for d in top_pair_details
        ],
        "top_pair_unique_word_count_only_control": pair_only_word_counts,
        "capacity_notes": [
            "Primary exact-prompt hours use one source per speaker and prompt: the lexicographically first SHA256, consistently across speakers.",
            "All distinct recordings for these prompts are also summed separately as availability; they are not used as primary exact-prompt hours.",
            "A prompt passes duration matching when the longest/shortest selected-source duration ratio is within the stated tolerance.",
            "SHA256 detects byte-identical sources only; re-encoded or edited copies may remain.",
            "Unique word count is whitespace-token count after set deduplication; it is a weaker control than matching exact text.",
            "This is structural feasibility from train metadata, not transcript/audio verification or a frozen experiment split.",
        ],
    }

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# Shared Prompt Feasibility (Train Only)\n\n")
        f.write("Train-only structural audit using NFC + whitespace folding, preserving case and punctuation.\n\n")
        best_raw = top_pair_details[0]
        best_duration = duration_top_three[0]
        best_balanced_hours = min(best_duration["duration_matched"]["20"]["per_person_hours"])
        f.write(f"The largest exact-text pair shares {best_raw['count']:,} prompts and provides {min(best_raw['person_hours'].values()):.3f} h on the smaller side when using one source per prompt. With per-prompt duration ratios capped at 1.20, the best pair provides {best_balanced_hours:.3f} h per person across {len(best_duration['duration_matched']['20']['prompts']):,} prompts. A strict 0.5 h-per-person budget is {'feasible' if best_balanced_hours >= 0.5 else 'not feasible'} under that duration constraint. No prompt is common to eight speakers.\n\n")
        f.write("```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n")
        f.write("\nPrimary hours use one source per speaker and prompt, selected by sorted SHA256. Additional unique sources are reported as availability only. Duration matching requires the selected source durations for the same prompt to differ by no more than the stated ratio. Identical text does not imply verified audio correctness. Unique-word-count matches are reported separately and do not count as exact-text matches.\n")

    if args.private_fd is not None or args.private_path:
        private = {
            "scope": "train.jsonl only",
            "speaker_name_map": {f"speaker_{i+1:02d}": name for i, name in enumerate(speakers)},
            "top_pair_exact_shared_prompts": [
                {
                    "speakers": [f"speaker_{i+1:02d}" for i in d["speaker_indexes"]],
                    "per_speaker_hours": {f"speaker_{speaker_index[name]+1:02d}": hours
                                           for name, hours in d["person_hours"].items()},
                    "prompts": [
                        {"text": p, "sources_by_speaker": {
                            f"speaker_{speaker_index[name]+1:02d}": [
                                {"sha256": r["sha"], "duration": r["duration"], "line": r["line"]}
                                for r in d["all_rows_by_speaker"][name] if r["text"] == p
                            ] for name in [speakers[i] for i in d["speaker_indexes"]]
                        }} for p in d["shared_prompts"]
                    ],
                } for d in top_pair_details
            ],
        }
        payload = json.dumps(private, ensure_ascii=False) + "\n"
        if args.private_path:
            with open(args.private_path, "w", encoding="utf-8") as private_file:
                private_file.write(payload)
        if args.private_fd is not None:
            with os.fdopen(os.dup(args.private_fd), "wb") as private_file:
                private_file.write(payload.encode("utf-8"))

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
