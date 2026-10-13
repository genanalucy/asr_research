# Whisper-medium preparation record

Date: 2026-10-14. This records preparation only; no GPU training was run by the download task.

## Pinned public artifact

- Repository: [openai/whisper-medium](https://huggingface.co/openai/whisper-medium)
- Fixed revision: `abdf7c39ab9d0397620ccaea8974cc764cd0953e`
- Read source: public Hugging Face model API at that commit, queried from the local workstation because the server's direct `huggingface.co` API request timed out. The server's `hf-mirror.com` model metadata resolved to the same revision and matching file sizes, Git blob IDs, and LFS metadata.
- Download endpoint: `https://hf-mirror.com`
- Weight format: `model.safetensors` only; no `pytorch_model.bin` copy.
- Weight size: 3,055,544,304 bytes.
- Official LFS SHA-256 and downloaded file SHA-256: `62f73550fa6db24b0c6f6c5962bd0dae80fa644e93cde9cd9c3792971b47fd28`.
- Model parameters: 763,857,920.

The cache contains that weight plus the pinned config, generation config, preprocessor config, tokenizer files, normalizer and special-token metadata required by Transformers' Whisper processor/model loaders.

## Server verification

Target directory: `/home/hzh/asr_research/data/model_cache/whisper-medium`.

Audit record: `/home/hzh/asr_research/audit/private/whisper_medium_download.json`. It retains the revision, file sizes, Git blob IDs, LFS hash and each local SHA-256. Final status is `offline_cpu_load_verified`.

The actual run took 16 minutes 29 seconds for the 11 selected files. It then checked file sizes and hashes, loaded `WhisperProcessor` and `WhisperForConditionalGeneration` with `local_files_only=True`, and confirmed the first parameter was on CPU. It did not run a forward pass, initialize CUDA, use a GPU, read project audio or transcripts, or alter any other model cache/audit. The model load confirms the files and processor are structurally usable; it does not validate recognition quality.

## Reproducible entry point

[`prepare_whisper_medium.py`](../tools/model_prepare/prepare_whisper_medium.py) pins the same public revision and metadata. Re-running it checks and reuses a complete valid cache; it stops if an existing directory is partial/corrupt or if the audit path belongs to another model revision. A missing cache is downloaded from the mirror and compared with pinned size/blob/LFS metadata before offline CPU loading.
