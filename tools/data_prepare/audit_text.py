"""Aggregate transcript text properties without exporting labels or speaker names."""
import argparse
import collections
import json
import statistics
import unicodedata
from pathlib import Path

from prepare_zhuang import read_rows, normalized


def is_han(char):
    code = ord(char)
    return (0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or
            0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x3134F)


def speaker_prefix(recording_name):
    name = recording_name.strip()
    return name.rsplit("-", 1)[0] if "-" in name else "unknown"


def length_summary(values):
    values = sorted(values)
    if not values:
        return {"min": 0, "median": 0, "p90": 0, "p95": 0, "max": 0}
    return {
        "min": values[0],
        "median": statistics.median(values),
        "p90": values[int((len(values) - 1) * 0.90)],
        "p95": values[int((len(values) - 1) * 0.95)],
        "max": values[-1],
    }


def audit(path):
    rows = list(read_rows(path))
    if not rows:
        raise ValueError("Workbook contains no rows")
    header, labels = rows[0], rows[1:]
    if header.get("A") != "录音名称" or header.get("B") != "壮文":
        raise ValueError("Unexpected source schema; refusing to guess transcript columns")

    text_rows = []
    repeated = collections.Counter()
    text_speakers = collections.defaultdict(set)
    speaker_rows = collections.Counter()
    blank_rows = 0
    empty_translation_rows = 0
    char_counts = collections.Counter()
    row_flags = collections.Counter()
    raw_changed_nfc = 0
    codepoint_lengths = []
    token_lengths = []
    token_total = 0
    whitespace_rows = 0
    all_single_tokens = 0
    candidate_explicit_tone_rows = 0
    candidate_trailing_digit_tokens = 0
    rows_with_trailing_digit_tokens = 0
    mixed_rows = collections.Counter()
    total_nonempty = 0

    for row in labels:
        raw_text = row.get("B", "")
        text = normalized(raw_text)
        speaker = speaker_prefix(row.get("A", ""))
        speaker_rows[speaker] += 1
        if not text:
            blank_rows += 1
            if not row.get("C", "").strip():
                empty_translation_rows += 1
            continue

        total_nonempty += 1
        raw_changed_nfc += unicodedata.normalize("NFC", raw_text) != raw_text
        text_rows.append(text)
        repeated[text] += 1
        text_speakers[text].add(speaker)
        codepoint_lengths.append(len(text))
        tokens = text.split()
        token_lengths.append(len(tokens))
        token_total += len(tokens)
        if len(tokens) > 1:
            whitespace_rows += 1
        else:
            all_single_tokens += 1

        row_has_ascii_letter = any("A" <= c <= "Z" or "a" <= c <= "z" for c in text)
        row_has_non_ascii_letter_or_mark = False
        row_has_han = False
        row_has_digit = False
        row_has_ascii_punctuation = False
        row_has_combining_mark = False
        row_has_precomposed_latin_diacritic = False
        for char in text:
            category = unicodedata.category(char)
            if char.isalpha():
                char_counts["alphabetic_codepoints"] += 1
            if "A" <= char <= "Z" or "a" <= char <= "z":
                char_counts["ascii_latin_letters"] += 1
            elif char.isalpha() and not is_han(char):
                char_counts["non_ascii_non_han_letters"] += 1
                row_has_non_ascii_letter_or_mark = True
                if "LATIN" in unicodedata.name(char, "") and "WITH" in unicodedata.name(char, ""):
                    row_has_precomposed_latin_diacritic = True
            if category.startswith("M"):
                char_counts["combining_marks"] += 1
                row_has_combining_mark = True
                row_has_non_ascii_letter_or_mark = True
            if is_han(char):
                char_counts["han_characters"] += 1
                row_has_han = True
            if char.isdecimal():
                char_counts["decimal_digits"] += 1
                row_has_digit = True
            if ord(char) < 128 and not char.isalnum() and not char.isspace():
                char_counts["ascii_punctuation_or_symbol"] += 1
                row_has_ascii_punctuation = True
            if char.isspace():
                char_counts["whitespace_characters"] += 1

        row_flags["rows_with_ascii_latin"] += row_has_ascii_letter
        row_flags["rows_with_non_ascii_letters_or_marks"] += row_has_non_ascii_letter_or_mark
        row_flags["rows_with_han"] += row_has_han
        row_flags["rows_with_decimal_digits"] += row_has_digit
        row_flags["rows_with_ascii_punctuation_or_symbol"] += row_has_ascii_punctuation
        row_flags["rows_with_combining_marks"] += row_has_combining_mark
        row_flags["rows_with_precomposed_latin_diacritic_candidate"] += row_has_precomposed_latin_diacritic

        if row_has_combining_mark or row_has_precomposed_latin_diacritic:
            candidate_explicit_tone_rows += 1
        row_has_trailing_digit_token = False
        for token in tokens:
            if token and token[-1].isdigit():
                candidate_trailing_digit_tokens += 1
                row_has_trailing_digit_token = True
        rows_with_trailing_digit_tokens += row_has_trailing_digit_token
        mixed_rows["ascii_latin_and_digits"] += row_has_ascii_letter and row_has_digit
        mixed_rows["ascii_latin_and_han"] += row_has_ascii_letter and row_has_han
        mixed_rows["ascii_latin_and_diacritic_candidate"] += row_has_ascii_letter and (
            row_has_combining_mark or row_has_precomposed_latin_diacritic
        )
        mixed_rows["digits_and_han"] += row_has_digit and row_has_han
        mixed_rows["ascii_latin_digits_and_han"] += row_has_ascii_letter and row_has_digit and row_has_han

    duplicate_groups = [count for count in repeated.values() if count > 1]
    cross_speaker_groups = [
        text for text, speakers in text_speakers.items() if len(speakers) > 1
    ]
    cross_speaker_rows = sum(repeated[text] for text in cross_speaker_groups)
    return {
        "input_rows": len(labels),
        "nonempty_transcript_rows": total_nonempty,
        "suspected_blank_transcript_rows": blank_rows,
        "blank_transcript_and_translation_rows": empty_translation_rows,
        "speaker_prefix_count": len(speaker_rows),
        "speaker_row_count_distribution": length_summary(speaker_rows.values()),
        "repeated_transcript_groups": len(duplicate_groups),
        "rows_in_repeated_transcript_groups": sum(duplicate_groups),
        "repeated_transcript_extra_rows": sum(count - 1 for count in duplicate_groups),
        "unique_nonempty_transcripts": len(repeated),
        "transcript_groups_repeated_across_speakers": len(cross_speaker_groups),
        "rows_in_transcripts_repeated_across_speakers": cross_speaker_rows,
        "transcript_codepoint_length": length_summary(codepoint_lengths),
        "whitespace_token_count_per_row": length_summary(token_lengths),
        "mean_whitespace_tokens_per_nonempty_row": token_total / total_nonempty if total_nonempty else 0,
        "rows_with_internal_whitespace": whitespace_rows,
        "rows_without_internal_whitespace": all_single_tokens,
        "character_counts": dict(char_counts),
        "row_composition_counts": dict(row_flags),
        "mixed_component_row_counts": dict(mixed_rows),
        "distinct_ascii_punctuation_or_symbols": len(set(
            char for text in text_rows for char in text
            if ord(char) < 128 and not char.isalnum() and not char.isspace()
        )),
        "nfc_changed_rows": raw_changed_nfc,
        "candidate_explicit_tone_mark_rows": candidate_explicit_tone_rows,
        "candidate_tokens_ending_in_digit": candidate_trailing_digit_tokens,
        "rows_with_candidate_token_ending_in_digit": rows_with_trailing_digit_tokens,
        "interpretation_limits": [
            "Transcript equality uses NFC normalization and collapsed whitespace; it is a text duplicate check, not a verified linguistic equivalence judgment.",
            "Han, digits, ASCII letters, combining marks, and precomposed Latin letters are counted as character-pattern candidates only.",
            "Whitespace tokens and final-digit tokens are candidate written units; they do not establish phonological syllable or tone boundaries.",
            "Blank B cells are suspected missing labels; this audit does not judge whether a blank is intentional.",
            "Speaker prefixes are used only as grouping keys; no names, recording identifiers, or transcript strings are written to output.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.xlsx)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "text_summary.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
