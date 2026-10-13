"""Cache and validate audio from the frozen full-train and dev manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from common import load_audio_16k, load_manifest, write_json


MANIFEST_NAMES = ("train", "dev")
CACHE_DIR = "/home/hzh/asr_research/data/audio_cache/16k_float_mono_ffmpeg42_v1"
DEFAULT_AUDIT = Path("audit/private/full_audio_cache.json")
EXPECTED = {
    "train": {"rows": 55805, "hours": 107.407, "speakers": 22},
    "dev": {"rows": 4120, "hours": 7.844, "speakers": 2},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def inspect_manifest(path, rows):
    shas = [row.get("sha256") for row in rows]
    return {
        "rows": len(rows),
        "hours": sum(float(row.get("duration", 0) or 0) for row in rows) / 3600,
        "speaker_count": len({row.get("speaker_candidate") for row in rows if row.get("speaker_candidate")}),
        "unique_sha256": len(set(shas)),
        "missing_sha256": sum(not sha for sha in shas),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/draft"))
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers != 8:
        parser.error("This cache job is fixed at 8 CPU workers")

    os.environ["ASR_AUDIO_CACHE"] = CACHE_DIR
    started = utc_now()
    rows_by_sha = {}
    manifest_rows = {}
    stats = {}
    for name in MANIFEST_NAMES:
        path = args.split_dir / f"{name}.jsonl"
        rows = load_manifest(path)
        manifest_rows[name] = rows
        stats[name] = inspect_manifest(path, rows)
        for row in rows:
            sha = row.get("sha256")
            try:
                valid_sha = isinstance(sha, str) and len(sha) == 64 and int(sha, 16) >= 0
            except ValueError:
                valid_sha = False
            if not valid_sha:
                stats[name]["invalid_sha256"] = stats[name].get("invalid_sha256", 0) + 1
                continue
            rows_by_sha.setdefault(sha, row)

    train_sha = {row.get("sha256") for row in manifest_rows["train"] if row.get("sha256")}
    dev_sha = {row.get("sha256") for row in manifest_rows["dev"] if row.get("sha256")}
    overlap = len(train_sha & dev_sha)
    mismatches = {}
    for name in MANIFEST_NAMES:
        actual = stats[name]
        expected = EXPECTED[name]
        delta = {
            key: {"expected": expected[key], "actual": actual["rows"] if key == "rows" else
                  actual["speaker_count"] if key == "speakers" else actual["hours"]}
            for key in expected
            if (actual["rows"] != expected[key] if key == "rows" else
                actual["speaker_count"] != expected[key] if key == "speakers" else
                abs(actual["hours"] - expected[key]) > 0.001)
        }
        if delta or actual["missing_sha256"] or actual["unique_sha256"] != actual["rows"] or actual.get("invalid_sha256", 0):
            mismatches[name] = delta or {"sha256_integrity": "failed"}
    input_ok = not mismatches and overlap == 0
    audit = {
        "started_at_utc": started,
        "updated_at_utc": utc_now(),
        "status": "preflight",
        "manifests": stats,
        "train_dev_sha256_overlap": overlap,
        "unique_sha256_total": len(rows_by_sha),
        "duplicate_rows_removed": sum(len(rows) for rows in manifest_rows.values()) - len(rows_by_sha),
        "cache_dir": CACHE_DIR,
        "workers": 8,
        "expected_input_match": input_ok,
        "input_mismatches": mismatches,
        "test_manifest_read": False,
        "labels_recorded": False,
        "errors": [],
    }
    write_json(args.audit, audit)
    if not input_ok:
        audit.update(status="stopped_input_mismatch", updated_at_utc=utc_now())
        write_json(args.audit, audit)
        print({"status": audit["status"], "manifests": stats, "overlap": overlap}, flush=True)
        raise SystemExit(2)

    cache_root = Path(CACHE_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)
    audit.update(status="running", total=len(rows_by_sha), processed=0, succeeded=0, failed=0)
    write_json(args.audit, audit)

    def process(item):
        sha, row = item
        try:
            waveform = np.asarray(load_audio_16k(row), dtype=np.float32)
            if waveform.ndim != 1 or waveform.size == 0:
                raise ValueError("invalid waveform shape")
            if not np.isfinite(waveform).all():
                raise ValueError("non-finite waveform")
            if float(np.max(np.abs(waveform))) == 0:
                raise ValueError("silent waveform")
            cached = cache_root / f"{sha}.wav"
            if not cached.is_file():
                raise FileNotFoundError("cache file was not created")
            info = sf.info(cached)
            if info.samplerate != 16000 or info.channels != 1 or info.subtype != "FLOAT":
                raise ValueError("cached WAV is not 16 kHz mono FLOAT")
            return sha, None
        except Exception as exc:
            return sha, type(exc).__name__

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(process, item) for item in rows_by_sha.items()]
        for future in as_completed(futures):
            sha, error_type = future.result()
            audit["processed"] += 1
            if error_type:
                audit["failed"] += 1
                audit["errors"].append({"sha256": sha, "error_type": error_type})
            else:
                audit["succeeded"] += 1
            if audit["processed"] % 500 == 0:
                audit["updated_at_utc"] = utc_now()
                write_json(args.audit, audit)
                print({"processed": audit["processed"], "total": audit["total"],
                       "succeeded": audit["succeeded"], "failed": audit["failed"]}, flush=True)

    audit.update(
        status="completed" if audit["failed"] == 0 else "completed_with_errors",
        updated_at_utc=utc_now(),
        validation="non-empty, finite, non-silent waveform; 16 kHz mono FLOAT cache",
    )
    audit["errors"].sort(key=lambda item: item["sha256"])
    write_json(args.audit, audit)
    print({key: audit[key] for key in ("status", "processed", "succeeded", "failed", "updated_at_utc")}, flush=True)
    if audit["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
