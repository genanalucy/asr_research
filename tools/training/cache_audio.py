"""Build the versioned 16 kHz mono FLOAT cache for baseline manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from common import load_audio_16k, load_manifest, write_json


MANIFEST_NAMES = ("train", "dev_monitor", "sanity")
DEFAULT_CACHE = "/home/hzh/asr_research/data/audio_cache/16k_float_mono_ffmpeg42_v1"
DEFAULT_AUDIT = "audit/private/baseline_cache.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/baseline_v1"))
    parser.add_argument("--audit", type=Path, default=Path(DEFAULT_AUDIT))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers != 8:
        parser.error("This baseline cache job is fixed at 8 workers")

    os.environ["ASR_AUDIO_CACHE"] = DEFAULT_CACHE
    started = utc_now()
    rows_by_sha = {}
    manifest_counts = {}
    for name in MANIFEST_NAMES:
        rows = load_manifest(args.split_dir / f"{name}.jsonl")
        manifest_counts[name] = len(rows)
        for row in rows:
            sha = row.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                raise ValueError(f"Invalid or missing SHA256 in {name} manifest")
            rows_by_sha.setdefault(sha, row)

    cache_root = Path(DEFAULT_CACHE)
    cache_root.mkdir(parents=True, exist_ok=True)
    errors = []
    succeeded = 0

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
            if error_type:
                errors.append({"sha256": sha, "error_type": error_type})
            else:
                succeeded += 1

    result = {
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "manifests": manifest_counts,
        "manifest_rows_total": sum(manifest_counts.values()),
        "unique_sha256": len(rows_by_sha),
        "duplicate_rows_removed": sum(manifest_counts.values()) - len(rows_by_sha),
        "cache_dir": str(cache_root),
        "format": "16000 Hz, mono, WAV FLOAT",
        "workers": 8,
        "succeeded": succeeded,
        "failed": len(errors),
        "validation": "non-empty, finite, non-silent mono waveform",
        "errors": sorted(errors, key=lambda item: item["sha256"]),
        "test_manifest_read": False,
        "labels_recorded": False,
    }
    write_json(args.audit, result)
    print({key: result[key] for key in ("manifest_rows_total", "unique_sha256", "succeeded", "failed", "finished_at_utc")}, flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
