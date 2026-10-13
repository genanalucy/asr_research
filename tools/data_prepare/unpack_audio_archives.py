#!/usr/bin/env python3
"""Safely stream two gzip-compressed tar datasets into separate directories."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
import traceback


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def log_line(log, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def safe_relative(member_name: str) -> Path:
    if "\x00" in member_name or "\\" in member_name:
        raise ValueError(f"unsafe member path: {member_name!r}")
    raw = PurePosixPath(member_name)
    if raw.is_absolute() or any(part in ("..", "") for part in raw.parts):
        raise ValueError(f"unsafe member path: {member_name!r}")
    parts = [part for part in raw.parts if part not in (".", "")]
    if not parts:
        return Path(".")
    return Path(*parts)


def archive_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def ensure_stable(path: Path, interval: float) -> dict:
    before = archive_fingerprint(path)
    time.sleep(interval)
    after = archive_fingerprint(path)
    if before != after:
        raise RuntimeError(f"archive changed during stability check: before={before}, after={after}")
    return after


def extract_one(label: str, archive: Path, target: Path, audit: Path,
                manifest: dict, stable_seconds: float, sample_limit: int) -> dict:
    log_path = audit / f"unpack_{label}.log"
    record = {
        "label": label,
        "archive": str(archive),
        "target": str(target),
        "status": "checking",
        "started_at": utc_now(),
    }
    manifest["archives"][label] = record
    write_json(audit / "data_unpack_manifest.json", manifest)

    with log_path.open("a", encoding="utf-8") as log:
        try:
            if target.exists():
                raise FileExistsError(f"refusing to overwrite existing target: {target}")
            if not archive.is_file():
                raise FileNotFoundError(archive)
            fingerprint = ensure_stable(archive, stable_seconds)
            record["archive_fingerprint_before"] = fingerprint
            log_line(log, f"START archive={archive} size={fingerprint['size_bytes']} target={target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = target.with_name(target.name + ".extracting")
            if stage.exists():
                raise FileExistsError(f"refusing to overwrite existing staging directory: {stage}")
            stage.mkdir(parents=True)

            counts = collections.Counter()
            extensions = collections.Counter()
            top_level = collections.Counter()
            speakers = collections.Counter()
            split_hints = []
            first_members = []
            annotation_candidates = []
            extracted_bytes = 0
            member_count = 0
            with tarfile.open(archive, mode="r|gz") as tf:
                for member in tf:
                    member_count += 1
                    rel = safe_relative(member.name)
                    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                        raise ValueError(f"refusing link or special tar member: {member.name!r}")
                    out = stage / rel
                    resolved_out = os.path.realpath(str(out))
                    resolved_stage = os.path.realpath(str(stage))
                    if os.path.commonpath([resolved_out, resolved_stage]) != resolved_stage:
                        raise ValueError(f"member resolves outside target: {member.name!r}")
                    if member.name not in ("", ".") and len(first_members) < 40:
                        first_members.append({"name": member.name, "type": "dir" if member.isdir() else "file", "size": member.size})
                        log_line(log, f"MEMBER[{len(first_members):02d}] type={'dir' if member.isdir() else 'file'} size={member.size} path={member.name[:240]}")

                    if member.isdir():
                        out.mkdir(parents=True, exist_ok=True)
                        counts["directories"] += 1
                        continue
                    if not member.isfile():
                        raise ValueError(f"unsupported tar member type: {member.name!r}")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if out.exists():
                        raise FileExistsError(f"archive contains a duplicate path: {member.name!r}")
                    source = tf.extractfile(member)
                    if source is None:
                        raise IOError(f"could not read regular member: {member.name!r}")
                    with source, out.open("xb") as dest:
                        shutil.copyfileobj(source, dest, length=1024 * 1024)
                    try:
                        os.chmod(out, member.mode & 0o777)
                    except OSError:
                        pass
                    counts["files"] += 1
                    counts["regular_file_bytes"] += member.size
                    extracted_bytes += member.size
                    suffix = out.suffix.lower() or "[no extension]"
                    extensions[suffix] += 1
                    top_level[rel.parts[0] if rel.parts else "."] += 1
                    name = out.name
                    match = re.search(r"(?:^|[_-])(spk|speaker|spkr|s)([_-]?\d+[a-z]?)", name, flags=re.I)
                    if match:
                        speakers[match.group(0).strip("_- ")] += 1
                    lowered = "/".join(rel.parts).lower()
                    if any(token in lowered for token in ("train", "dev", "valid", "test", "eval")):
                        split_hints.append(member.name[:240])
                    if suffix in {".txt", ".tsv", ".csv", ".json", ".jsonl", ".lab", ".stm", ".trn", ".trans", ".trans.txt"}:
                        if len(annotation_candidates) < 100:
                            annotation_candidates.append(member.name)

            fingerprint_after = archive_fingerprint(archive)
            if fingerprint_after != fingerprint:
                raise RuntimeError(f"archive changed during extraction: before={fingerprint}, after={fingerprint_after}")
            os.replace(stage, target)
            record.update({
                "status": "completed",
                "completed_at": utc_now(),
                "archive_fingerprint_after": fingerprint_after,
                "member_count": member_count,
                "counts": dict(counts),
                "file_extensions": extensions.most_common(),
                "top_level_entry_counts": top_level.most_common(100),
                "filename_speaker_pattern_counts": speakers.most_common(100),
                "split_path_hints": split_hints[:200],
                "annotation_candidates": annotation_candidates,
                "first_members": first_members,
                "extracted_bytes": extracted_bytes,
            })
            log_line(log, f"DONE files={counts['files']} dirs={counts['directories']} bytes={extracted_bytes} extensions={extensions.most_common(20)}")
        except Exception as exc:
            record.update({"status": "failed", "failed_at": utc_now(), "error": repr(exc), "traceback": traceback.format_exc()})
            log_line(log, f"FAILED {type(exc).__name__}: {exc}")
            log.write(record["traceback"] + "\n")
            log.flush()
            write_json(audit / "data_unpack_manifest.json", manifest)
            raise
        finally:
            record["updated_at"] = utc_now()
            write_json(audit / "data_unpack_manifest.json", manifest)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/hzh/asr_research"))
    parser.add_argument("--stable-seconds", type=float, default=5)
    parser.add_argument("--sample-limit", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    manifest_path = audit / "data_unpack_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"created_at": utc_now(), "root": str(root), "archives": {}}
    jobs = [
        ("zhuang", root / "zhuang_dataset.tar.gz", root / "data/raw/zhuang"),
        ("yao", root / "yaoyu.tar.gz", root / "data/raw/yao"),
    ]
    for label, archive, target in jobs:
        extract_one(label, archive, target, audit, manifest, args.stable_seconds, args.sample_limit)
    manifest["overall_status"] = "completed"
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(json.dumps({k: {"status": v["status"], "target": v["target"], "counts": v.get("counts"), "file_extensions": v.get("file_extensions"), "split_path_hints": v.get("split_path_hints", [])[:20], "first_members": v.get("first_members", [])} for k, v in manifest["archives"].items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
