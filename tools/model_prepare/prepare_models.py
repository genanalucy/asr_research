#!/usr/bin/env python3
"""Download two pinned public ASR checkpoints and verify offline CPU loading."""

import hashlib
import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/model_cache"
AUDIT = ROOT / "audit/private/model_download"
SPECS = [
    {
        "repo": "openai/whisper-small",
        "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
        "directory": "whisper-small",
        "weights": "model.safetensors",
        "patterns": [
            "config.json", "generation_config.json", "preprocessor_config.json",
            "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
            "normalizer.json", "tokenizer.json", "merges.txt", "vocab.json",
            "model.safetensors",
        ],
    },
    {
        "repo": "facebook/wav2vec2-xls-r-300m",
        "revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
        "directory": "wav2vec2-xls-r-300m",
        "weights": "pytorch_model.bin",
        "patterns": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    },
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "endpoint": os.environ["HF_ENDPOINT"],
        "source": "mirror model_info metadata; LFS SHA-256 is not independently verified against official metadata",
        "models": [],
    }
    manifest_path = AUDIT / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
    for spec in SPECS:
        info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
        if info.sha != spec["revision"]:
            raise RuntimeError(f"revision mismatch for {spec['repo']}: {info.sha}")
        metadata = {f.rfilename: f for f in info.siblings}
        expected = {}
        for name in spec["patterns"]:
            item = metadata.get(name)
            if item is None:
                raise RuntimeError(f"missing mirror metadata: {spec['repo']}:{name}")
            expected[name] = {
                "size": item.size,
                "lfs_sha256": item.lfs.sha256 if item.lfs else None,
            }
        record = {
            "repo": spec["repo"], "revision": info.sha,
            "directory": spec["directory"], "expected_files": expected,
            "status": "downloading",
        }
        manifest["models"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        local = CACHE / spec["directory"]
        snapshot_download(
            repo_id=spec["repo"], revision=spec["revision"],
            allow_patterns=spec["patterns"], local_dir=str(local), max_workers=4,
        )
        for name, want in expected.items():
            path = local / name
            if not path.is_file() or path.stat().st_size != want["size"]:
                raise RuntimeError(f"size/missing-file mismatch: {path}")
            got = file_sha256(path)
            if want["lfs_sha256"] and got != want["lfs_sha256"]:
                raise RuntimeError(f"LFS SHA-256 mismatch: {path}")
            want["local_sha256"] = got
        record["status"] = "downloaded_and_hashed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    whisper_dir = CACHE / "whisper-small"
    w2v_dir = CACHE / "wav2vec2-xls-r-300m"
    WhisperProcessor.from_pretrained(whisper_dir, local_files_only=True, trust_remote_code=False)
    whisper = WhisperForConditionalGeneration.from_pretrained(
        whisper_dir, local_files_only=True, trust_remote_code=False
    )
    del whisper
    Wav2Vec2FeatureExtractor.from_pretrained(
        w2v_dir, local_files_only=True, trust_remote_code=False
    )
    state = torch.load(w2v_dir / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("XLS-R weights_only load did not produce a state dict")
    del state
    wav2vec = Wav2Vec2Model.from_pretrained(
        w2v_dir, local_files_only=True, trust_remote_code=False,
        use_safetensors=False,
    )
    del wav2vec
    for record in manifest["models"]:
        record["status"] = "offline_cpu_load_verified"
    manifest["verification"] = "processors loaded locally; both models loaded on CPU; no forward pass"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"VERIFIED manifest={manifest_path}")


if __name__ == "__main__":
    main()
