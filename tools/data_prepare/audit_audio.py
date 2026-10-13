"""Inventory audio headers and exact file duplicates without modifying raw data."""
import argparse
import concurrent.futures
import collections
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import wave


def inspect(path):
    row = {"path": str(path), "bytes": path.stat().st_size}
    try:
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as src:
                    row.update(sample_rate=src.getframerate(), channels=src.getnchannels(),
                               frames=src.getnframes(), sample_width=src.getsampwidth())
                    row["duration"] = row["frames"] / row["sample_rate"]
            except (wave.Error, EOFError):
                pass
        if "duration" not in row:
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                  "-show_entries", "stream=sample_rate,channels,duration:format=duration",
                                  "-of", "json", str(path)], capture_output=True, text=True,
                                 check=True, timeout=30)
            data = json.loads(out.stdout)
            stream = data["streams"][0]
            row.update(sample_rate=int(stream["sample_rate"]), channels=int(stream["channels"]),
                       duration=float(stream.get("duration", data.get("format", {}).get("duration"))))
        h = hashlib.sha256()
        with path.open("rb") as src:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                h.update(block)
        row["sha256"] = h.hexdigest()
        row["status"] = "header_ok"
    except Exception as exc:
        row.update(status="error", error=str(exc))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.root.rglob("*") if p.is_file())
    audio = [p for p in files if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".wma", ".aac", ".opus"}]
    rows = []
    with (args.output / "audio_inventory.jsonl").open("w") as dst:
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            for i, row in enumerate(pool.map(inspect, audio), 1):
                rows.append(row)
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                if i % 2000 == 0:
                    dst.flush()
                    print(f"inspected {i}/{len(audio)}", flush=True)
    ok = [r for r in rows if r["status"] == "header_ok"]
    durations = sorted(r["duration"] for r in ok)
    groups = collections.defaultdict(list)
    unique_durations = {}
    for r in ok:
        groups[r["sha256"]].append(r["path"])
        unique_durations.setdefault(r["sha256"], r["duration"])
    duplicates = [v for v in groups.values() if len(v) > 1]
    report = {
        "root": str(args.root), "files": len(files), "audio_files": len(audio),
        "extensions": dict(collections.Counter(p.suffix.lower() for p in files)),
        "header_ok": len(ok), "header_errors": len(rows) - len(ok),
        "hours": sum(durations) / 3600,
        "unique_file_hours": sum(unique_durations.values()) / 3600,
        "sample_rates": dict(collections.Counter(r["sample_rate"] for r in ok)),
        "channels": dict(collections.Counter(r["channels"] for r in ok)),
        "duration_seconds": {"min": min(durations, default=0), "median": statistics.median(durations) if durations else 0,
                             "p95": durations[int((len(durations)-1)*.95)] if durations else 0,
                             "max": max(durations, default=0)},
        "zero_duration": sum(d == 0 for d in durations),
        "over_30_seconds": sum(d > 30 for d in durations),
        "duplicate_groups": len(duplicates),
        "duplicate_extra_files": sum(len(v)-1 for v in duplicates),
        "verification_scope": "Headers and full-file hashes; no full decoding or transcript accuracy certification."
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "exact_duplicates.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2) + "\n")
    (args.output / "audio_errors.json").write_text(json.dumps([r for r in rows if r["status"] != "header_ok"], ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
