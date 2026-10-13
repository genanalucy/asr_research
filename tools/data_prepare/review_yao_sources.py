#!/usr/bin/env python3
"""Read-only audit of Yao source aliases, label conflicts, and leakage.

The script writes only to the requested audit directory.  Speaker identities
remain provenance candidates: an identical complete audio hash is evidence of
an alias or duplicate source, not proof that other hashes are distinct people.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def without_category(text: str, prefix: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(prefix))


def without_punctuation(text: str) -> str:
    return without_category(text, "P")


def without_digits(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("N"))


def conflict_class(texts: Iterable[str]) -> tuple[str, list[str]]:
    values = sorted(set(texts))
    folded = {fold(x) for x in values}
    no_punct = {without_punctuation(fold(x)) for x in values}
    no_digits = {without_digits(fold(x)) for x in values}
    if len(folded) == 1:
        return "case_or_whitespace_only", ["case", "whitespace"]
    if len(no_punct) == 1:
        return "punctuation_only", ["punctuation"]
    if len(no_digits) == 1:
        return "digits_only", ["digits"]
    # Keep a separate mixed class so a substantive difference is not hidden
    # by a simultaneous punctuation or digit change.
    no_punct_digits = {without_digits(without_punctuation(fold(x))) for x in values}
    if len(no_punct_digits) == 1:
        return "mixed_format_and_digits", ["punctuation", "digits"]
    return "substantive_word_or_segment", ["substantive"]


def row_view(row: dict[str, Any], speaker_map: dict[str, str]) -> dict[str, Any]:
    # Keep full hashes and source provenance in the private server report.
    source_alias = row.get("speaker_candidate", "")
    return {
        "id": row.get("id"),
        "sha256": row.get("sha256"),
        "source": row.get("source"),
        "source_file": row.get("source_file"),
        "source_row": row.get("source_row"),
        "source_split_hint": row.get("source_split_hint"),
        "original_label_audio_path": row.get("original_label_audio_path"),
        "speaker_source_alias": source_alias,
        "speaker_candidate_id": speaker_map.get(source_alias),
        "text": row.get("text", ""),
        "duration": row.get("duration"),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path, help="data/prepared/yao directory")
    parser.add_argument("output", type=Path, help="private audit output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.prepared / "manifest.jsonl")
    speaker_map = load_json(args.prepared / "speaker_map_private.json")
    by_hash: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row.get("status") == "paired" and row.get("sha256"):
            by_hash[row["sha256"]].append(row)

    duplicate_groups = {h: group for h, group in by_hash.items() if len(group) > 1}
    # Match the existing review artifact: its 977 groups compare the prepared
    # NFC/whitespace-normalized labels, so trailing-space-only copies are not
    # counted as label conflicts.
    conflict_groups = {
        h: group
        for h, group in duplicate_groups.items()
        if len({r.get("text_nfc_whitespace", r.get("text", "")) for r in group}) > 1
    }

    identity_links = []
    for sha256, group in sorted(duplicate_groups.items()):
        aliases = sorted({r.get("speaker_candidate", "") for r in group})
        sources = sorted({r.get("source", "") for r in group})
        identity_links.append(
            {
                "sha256": sha256,
                "row_count": len(group),
                "source_aliases": aliases,
                "candidate_ids": sorted({speaker_map.get(a, "") for a in aliases}),
                "sources": sources,
                "cross_source": len(sources) > 1,
                "cross_candidate_alias": len(aliases) > 1,
                "rows": [row_view(r, speaker_map) for r in group],
                "interpretation": "candidate alias or duplicate-source link; speaker independence is unverified",
            }
        )
    write_json(args.output / "identity_hash_links_private.json", identity_links)

    category_groups = []
    category_counts: collections.Counter[str] = collections.Counter()
    for sha256, group in sorted(conflict_groups.items()):
        category, flags = conflict_class(r.get("text_nfc_whitespace", r.get("text", "")) for r in group)
        category_counts[category] += 1
        category_groups.append(
            {
                "sha256": sha256,
                "category": category,
                "flags": flags,
                "row_count": len(group),
                "candidate_aliases": sorted({r.get("speaker_candidate", "") for r in group}),
                "sources": sorted({r.get("source", "") for r in group}),
                "rows": [row_view(r, speaker_map) for r in group],
                "recommendation": "format-only categories may be reviewed for normalization; do not change labels automatically",
            }
        )
    write_json(
        args.output / "label_conflict_categories_private.json",
        {"group_count": len(category_groups), "category_counts": dict(category_counts), "groups": category_groups},
    )

    test_rows = [r for r in rows if r.get("source") == "txt" and r.get("source_split_hint") == "test"]
    test_hashes = {r.get("sha256") for r in test_rows if r.get("sha256")}
    leakage_groups = []
    for sha256 in sorted(test_hashes):
        group = by_hash.get(sha256, [])
        if not group:
            continue
        has_txt = any(r.get("source") == "txt" for r in group)
        has_xlsx = any(r.get("source") == "xlsx" for r in group)
        other_txt = any(r not in test_rows and r.get("source") == "txt" for r in group)
        if has_xlsx or other_txt or len({r.get("speaker_candidate", "") for r in group}) > 1:
            leakage_groups.append(
                {
                    "sha256": sha256,
                    "test_txt_rows": [row_view(r, speaker_map) for r in group if r in test_rows],
                    "all_rows": [row_view(r, speaker_map) for r in group],
                    "cross_source_txt_xlsx": has_txt and has_xlsx,
                    "test_to_other_txt": other_txt,
                    "candidate_alias_count": len({r.get("speaker_candidate", "") for r in group}),
                }
            )
    write_json(
        args.output / "txt_test_leakage_private.json",
        {
            "test_hint_rows": len(test_rows),
            "test_hint_unique_hashes": len(test_hashes),
            "test_hashes_with_duplicate_or_cross_source_rows": len(leakage_groups),
            "cross_source_txt_xlsx_groups": sum(x["cross_source_txt_xlsx"] for x in leakage_groups),
            "test_to_other_txt_groups": sum(x["test_to_other_txt"] for x in leakage_groups),
            "groups": leakage_groups,
            "interpretation": "old TXT test hints are provenance clues; hashes appearing in train candidates or XLSX are leakage risks until split is frozen",
        },
    )

    stats = {
        "manifest_rows": len(rows),
        "paired_rows": sum(r.get("status") == "paired" for r in rows),
        "unique_complete_sha256": len(by_hash),
        "duplicate_sha256_groups": len(duplicate_groups),
        "cross_candidate_alias_groups": sum(
            len({r.get("speaker_candidate", "") for r in group}) > 1 for group in duplicate_groups.values()
        ),
        "cross_source_groups": sum(len({r.get("source", "") for r in group}) > 1 for group in duplicate_groups.values()),
        "label_conflict_groups": len(conflict_groups),
        "label_conflict_category_counts": dict(category_counts),
        "txt_test_hint_rows": len(test_rows),
        "txt_test_hint_unique_hashes": len(test_hashes),
        "txt_test_leakage_groups": len(leakage_groups),
        "txt_test_cross_source_groups": sum(x["cross_source_txt_xlsx"] for x in leakage_groups),
        "scope": "complete audio hashes identify candidate source aliases or duplicates; no speaker identity is confirmed and no candidate IDs are treated as independent speakers",
    }
    write_json(args.output / "review_stats.json", stats)

    report = [
        "# Yao source follow-up audit (private)",
        "",
        "This is a read-only provenance audit. Complete audio-hash matches link source aliases or duplicate records; they do not confirm speaker identity or prove that unmatched candidate IDs are independent speakers.",
        "",
        "## Counts",
        "",
    ]
    for key, value in stats.items():
        report.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False)}`")
    report += [
        "",
        "## Conflict classification",
        "",
        "Categories are mutually exclusive in this order: case/whitespace only, punctuation only, digits only, mixed punctuation and digits, substantive word or segment difference. Format-only categories are review suggestions; labels are unchanged.",
        "",
        "## Leakage interpretation",
        "",
        "An old TXT `test` path hint does not create a valid split. Any complete hash shared with another TXT row, an XLSX row, or another candidate alias must be held out together when a split is frozen.",
        "",
        "Detailed rows and hashes are in the JSON files in this directory and intentionally remain on the server.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
