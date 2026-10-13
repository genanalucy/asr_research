#!/usr/bin/env python3
"""Prepare and verify one pinned public Whisper-medium checkpoint."""

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor


ROOT = Path("/home/hzh/asr_research")
CACHE = ROOT / "data/model_cache/whisper-medium"
AUDIT = ROOT / "audit/private/whisper_medium_download.json"
REPO = "openai/whisper-medium"
REVISION = "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
ENDPOINT = "https://hf-mirror.com"
WEIGHT = "model.safetensors"

# File sizes and Git blob IDs are from the public Hugging Face API at REVISION.
# The model weight is an LFS object; its SHA-256 is pinned independently here.
EXPECTED = {
    "added_tokens.json": (34604, "e3d256c988462aa153dcabe2aa38b8e9b436c06f", None),
    "config.json": (1991, "eda91ab1edc34a7c54512ed7a53e643b202387af", None),
    "generation_config.json": (3755, "dda8239f83dec12c4a47e4b4c82af3e7f3391855", None),
    "merges.txt": (493869, "6038932a2a1f09a66991b1c2adae0d14066fa29e", None),
    WEIGHT: (
        3055544304,
        "4447185cd32dd9cdc17cfdfc5bd3c926acef1928",
        "62f73550fa6db24b0c6f6c5962bd0dae80fa644e93cde9cd9c3792971b47fd28",
    ),
    "normalizer.json": (52666, "dd6ae819ad738ac1a546e9f9282ef325c33b9ea0", None),
    "preprocessor_config.json": (184990, "c2048dfa9fd94a052e62e908d2c4dfb18534b4d2", None),
    "special_tokens_map.json": (2194, "bf69932dca4b3719b59fdd8f6cc1978109509f6c", None),
    "tokenizer.json": (2480466, "1e95340ff836fad1b5932e800fb7b8c5e6d78a74", None),
    "tokenizer_config.json": (282683, "d13b786c04765fb1a06492b53587752cd67665ea", None),
    "vocab.json": (835550, "d7016e21da8776c8a9d577d0f559600f09a240eb", None),
}
FILES = list(EXPECTED)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_existing_audit() -> dict | None:
    if not AUDIT.exists():
        return None
    record = json.loads(AUDIT.read_text())
    identity = (record.get("repo"), record.get("revision"), record.get("target_directory"))
    expected = (REPO, REVISION, str(CACHE))
    if identity != expected:
        raise RuntimeError(f"refusing to overwrite unrelated audit record: {AUDIT}")
    return record


def write_audit(record: dict) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    record["updated_at_unix"] = time.time()
    temporary = AUDIT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(AUDIT)


def expected_manifest() -> dict:
    return {
        name: {
            "size": size,
            "git_blob_id": blob_id,
            "lfs_sha256": lfs_sha256,
        }
        for name, (size, blob_id, lfs_sha256) in EXPECTED.items()
    }


def verify_local_files() -> dict:
    if not CACHE.is_dir():
        raise RuntimeError(f"checkpoint directory does not exist: {CACHE}")
    weight_files = {
        path.name for path in CACHE.iterdir()
        if path.is_file() and path.name in {"model.safetensors", "pytorch_model.bin"}
    }
    if weight_files != {WEIGHT}:
        raise RuntimeError(f"expected only {WEIGHT}; found weight files: {sorted(weight_files)}")
    verified = {}
    for name, (size, blob_id, lfs_sha256) in EXPECTED.items():
        path = CACHE / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"file missing or wrong size: {path}")
        local_sha256 = sha256(path)
        if lfs_sha256 and local_sha256 != lfs_sha256:
            raise RuntimeError(f"official LFS SHA-256 mismatch: {path}")
        if not lfs_sha256 and git_blob_sha1(path) != blob_id:
            raise RuntimeError(f"official Git blob ID mismatch: {path}")
        verified[name] = {
            "size": size,
            "git_blob_id": blob_id,
            "lfs_sha256": lfs_sha256,
            "local_sha256": local_sha256,
        }
    return verified


def verify_mirror_metadata() -> None:
    info = HfApi(endpoint=ENDPOINT).model_info(
        REPO, revision=REVISION, files_metadata=True
    )
    if info.sha != REVISION:
        raise RuntimeError(f"mirror resolved unexpected revision: {info.sha}")
    siblings = {item.rfilename: item for item in info.siblings}
    for name, (size, blob_id, lfs_sha256) in EXPECTED.items():
        item = siblings.get(name)
        if item is None or item.size != size or item.blob_id != blob_id:
            raise RuntimeError(f"mirror file metadata differs from pinned public metadata: {name}")
        mirror_lfs = item.lfs.sha256 if item.lfs else None
        if mirror_lfs != lfs_sha256:
            raise RuntimeError(f"mirror LFS metadata differs from pinned public metadata: {name}")


def verify_offline_cpu_load(record: dict) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    processor = WhisperProcessor.from_pretrained(
        CACHE, local_files_only=True, trust_remote_code=False
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        CACHE, local_files_only=True, trust_remote_code=False, use_safetensors=True
    )
    device = str(next(model.parameters()).device)
    if device != "cpu" or torch.cuda.is_initialized():
        raise RuntimeError(f"expected offline CPU-only load; device={device}")
    record["verification"] = {
        "processor_loaded_offline": True,
        "processor_type": type(processor).__name__,
        "model_loaded_offline": True,
        "model_type": model.config.model_type,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "first_parameter_device": device,
        "forward_pass_run": False,
        "gpu_used": False,
    }


def main() -> None:
    record = check_existing_audit() or {}
    record.update({
        "repo": REPO,
        "revision": REVISION,
        "target_directory": str(CACHE),
        "download_endpoint": ENDPOINT,
        "read_source": "Pinned public Hugging Face model metadata for openai/whisper-medium at the fixed revision; weight LFS SHA-256 checked against both the public pin and mirror metadata.",
        "weight_format": "safetensors only",
        "expected_files": expected_manifest(),
        "gpu_used": False,
    })

    if CACHE.exists() and any(CACHE.iterdir()):
        record["status"] = "verifying_existing_files"
        write_audit(record)
    else:
        verify_mirror_metadata()
        record["status"] = "downloading"
        write_audit(record)
        snapshot_download(
            repo_id=REPO,
            revision=REVISION,
            endpoint=ENDPOINT,
            allow_patterns=FILES,
            local_dir=str(CACHE),
            max_workers=4,
        )

    record["expected_files"] = verify_local_files()
    record["status"] = "offline_loading"
    write_audit(record)
    verify_offline_cpu_load(record)
    record["status"] = "offline_cpu_load_verified"
    write_audit(record)
    print(json.dumps({
        "status": record["status"],
        "audit": str(AUDIT),
        "revision": REVISION,
        "weight_bytes": EXPECTED[WEIGHT][0],
        "weight_sha256": EXPECTED[WEIGHT][2],
        "parameters": record["verification"]["parameters"],
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Preserve an unrelated audit; our own failed preparation must be visible
        # to the dependent launcher instead of looking like an endless download.
        try:
            record = check_existing_audit() or {
                "repo": REPO, "revision": REVISION, "target_directory": str(CACHE),
            }
        except Exception:
            record = None
        if record is not None:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            write_audit(record)
        raise
