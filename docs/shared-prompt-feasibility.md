# Shared Prompt Feasibility (Train Only)

Train-only structural audit using NFC + whitespace folding, preserving case and punctuation.

```json
{
  "scope": "train.jsonl only",
  "normalization": "NFC plus collapsed whitespace; case and punctuation preserved",
  "rows_with_valid_required_fields": 55805,
  "required_field_missing_counts": {},
  "speaker_candidate_count": 22,
  "unique_normalized_prompts": 48308,
  "duplicate_sha_rows_excluded_within_same_prompt": 0,
  "prompt_counts_by_independent_speaker_count": {
    "1": 41358,
    "2": 6915,
    "3": 33,
    "4": 1,
    "6": 1
  },
  "prompts_shared_by_at_least_n_speakers": {
    "2": 6950,
    "3": 35,
    "4": 2,
    "8": 0
  },
  "top_three_pairs_by_20_percent_duration_matched_prompt_count": [
    {
      "shared_prompt_count": 152,
      "per_person_hours_on_one_source_per_prompt": [
        0.32538073999999995,
        0.33132295288485764
      ],
      "within_prompt_duration_tolerance": {
        "10": {
          "prompt_count": 91,
          "per_person_hours": [
            0.19288296222222215,
            0.1953185185185185
          ],
          "meets_minimum_per_person_hours": {
            "0.5": false,
            "1.0": false,
            "5.0": false
          }
        },
        "20": {
          "prompt_count": 133,
          "per_person_hours": [
            0.29028740666666664,
            0.2986344167296549
          ],
          "meets_minimum_per_person_hours": {
            "0.5": false,
            "1.0": false,
            "5.0": false
          }
        }
      }
    },
    {
      "shared_prompt_count": 242,
      "per_person_hours_on_one_source_per_prompt": [
        0.4354793640047873,
        0.4776177777777777
      ],
      "within_prompt_duration_tolerance": {
        "10": {
          "prompt_count": 77,
          "per_person_hours": [
            0.15497098900667677,
            0.1557333333333333
          ],
          "meets_minimum_per_person_hours": {
            "0.5": false,
            "1.0": false,
            "5.0": false
          }
        },
        "20": {
          "prompt_count": 128,
          "per_person_hours": [
            0.2600942946252206,
            0.26192
          ],
          "meets_minimum_per_person_hours": {
            "0.5": false,
            "1.0": false,
            "5.0": false
          }
        }
      }
    },
    {
      "shared_prompt_count": 819,
      "per_person_hours_on_one_source_per_prompt": [
        1.0534119425548003,
        1.5941047653703704
      ],
      "within_prompt_duration_tolerance": {
        "10": {
          "prompt_count": 30,
          "per_person_hours": [
            0.05047760141093474,
            0.05255111027777778
          ],
          "meets_minimum_per_person_hours": {
            "0.5": false,
            "1.0": false,
            "5.0": false
          }
        },
        "20": {
          "prompt_count": 88,
          "per_person_hours": [
            0.1503879062736206,
            0.16697481361111108
          ],
          "meets_minimum_per_person_hours": {
            "0.5": false,
            "1.0": false,
            "5.0": false
          }
        }
      }
    }
  ],
  "top_pair_exact_text_and_audio_stats": [
    {
      "shared_prompt_count": 1088,
      "per_person_hours_one_source_per_prompt": [
        1.1369005794910556,
        2.979362516492818
      ],
      "per_person_hours_all_distinct_sources": [
        1.1473108591584786,
        2.9916765906594858
      ],
      "minimum_person_hours": 1.1369005794910556,
      "prompt_character_length": {
        "n": 1088,
        "min": 2,
        "median": 50.0,
        "p90": 86,
        "max": 170
      },
      "unique_word_count": {
        "n": 1088,
        "min": 1,
        "median": 7.0,
        "p90": 13,
        "max": 22
      }
    }
  ],
  "top_pair_unique_word_count_only_control": [
    {
      "same_prompt_count": 1088,
      "shared_unique_word_count_bins": 24,
      "left_distinct_text_prompts_in_shared_word_count_bins": 4175,
      "right_distinct_text_prompts_in_shared_word_count_bins": 8284,
      "limitation": "Counts prompts eligible for cardinality-only matching; their actual text differs and they are not exact-prompt samples."
    }
  ],
  "capacity_notes": [
    "Primary exact-prompt hours use one source per speaker and prompt: the lexicographically first SHA256, consistently across speakers.",
    "All distinct recordings for these prompts are also summed separately as availability; they are not used as primary exact-prompt hours.",
    "A prompt passes duration matching when the longest/shortest selected-source duration ratio is within the stated tolerance.",
    "SHA256 detects byte-identical sources only; re-encoded or edited copies may remain.",
    "Unique word count is whitespace-token count after set deduplication; it is a weaker control than matching exact text.",
    "This is structural feasibility from train metadata, not transcript/audio verification or a frozen experiment split."
  ]
}
```

Primary hours use one source per speaker and prompt, selected by sorted SHA256. Additional unique sources are reported as availability only. Duration matching requires the selected source durations for the same prompt to differ by no more than the stated ratio. Identical text does not imply verified audio correctness. Unique-word-count matches are reported separately and do not count as exact-text matches.
