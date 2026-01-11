# NVIDIA Arabic Conformer Baseline (NeMo)

Baseline pipeline to preprocess Quran recitation audio, run NVIDIA NeMo Conformer-CTC ASR, and score results with Quran-specific evaluation modes. Use this as a reproducible starting point for experiments and fine-tuning.

## What’s here
- CLI pipeline: `python -m ozzie_asr.run …` (download, eval, list-models)
- Core modules: preprocessing (`scripts/preprocess_audio.py`), model loading (`ozzie_asr/model_loader.py`), transcription (`ozzie_asr/transcriber.py`), evaluation (`ozzie_asr/evaluator.py`, `ozzie_asr/alignment.py`), normalization (`ozzie_asr/text_normalize.py`)
- Canonical text config: `configs/canonical_texts.json` (Al-Fatiha, Al-Ikhlas)
- Ops helpers: `Makefile`, `RUNBOOK.md`, `upload_to_runpod.sh`
- Reports/outputs: `outputs/…` (run artifacts), `outputs/problematic_files_analysis.md` (bad audio list)

## Quick start (GPU recommended)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# (Optional) install ffmpeg for audio conversion
sudo apt-get update && sudo apt-get install -y ffmpeg

# Download model from NGC (needs ngc CLI configured)
make download-model

# Run evaluation on your audio
make run AUDIO_DIR=./data/audio OUT_DIR=./outputs MODEL_PATH=./models/Conformer-CTC-L_spe128_ar-AR_3.0.nemo

# View summary
cat outputs/<timestamp>/summary.md
```

## Pipeline overview
1) **Preprocess audio** (`scripts/preprocess_audio.py`): converts to 16 kHz mono WAV; caches in run output dir.
2) **Load model** (`ozzie_asr/model_loader.py`): uses provided `.nemo` or downloads from NGC.
3) **Transcribe** (`ozzie_asr/transcriber.py`): batch transcribe with progress; collects durations.
4) **Evaluate** (`ozzie_asr/evaluator.py`): Quran-aware classification + CER/WER in two modes:
   - `SNAP_TO_CANONICAL` (default): align to best substring of canonical text.
   - `FULL_SURAH`: compare to full surah text.
5) **Emit artifacts**: `predictions.csv`, `metrics.csv`, `summary.md`, `run_info.json`, `errors.log` in `outputs/<timestamp>/`.

## CLI usage
- Download: `python -m ozzie_asr.run download --output_dir ./models`
- List models: `python -m ozzie_asr.run list-models [--all]`
- Evaluate: `python -m ozzie_asr.run eval --audio_dir <dir> --nemo_model_path <file> [--assume_full_surah] [--normalize_taa_marbuta] [--batch_size N] [--max_files N] [--cpu]`

`make run-full-surah` switches to full-surah mode. `make sanity` transcribes a single file for smoke testing.

## Data prep essentials
- Organize audio under `data/<surah_folder>/file.wav` (current canonical texts cover Al-Fatiha and Al-Ikhlas).
- Use `scripts/prefilter_audio.py` to filter silence/too-short files before upload.
- Use `scripts/preprocess_audio.py` if you need bulk conversion outside the main pipeline.
- Known bad files are listed in `outputs/problematic_files_analysis.md` (move to `data/excluded/` before eval).

## Outputs
- `predictions.csv`: filename, duration, surah prediction, raw/normalized text.
- `metrics.csv`: per-file CER/WER, snapped reference, alignment spans.
- `summary.md`: aggregated metrics, worst files, go/no-go guidance.
- `run_info.json`: reproducibility info (args, model, git hash, environment).
- `errors.log`: preprocessing/transcription/evaluation issues.

## Troubleshooting
- Missing model: run `make download-model` or point `--nemo_model_path` to an existing `.nemo`.
- ffmpeg missing: install via `apt-get install -y ffmpeg`.
- CPU-only: add `--cpu` (slower) and consider `--batch_size 1`.
- Silence/garbage audio: pre-filter to avoid skewed CER/WER (see docs/03-audio-prep.md).

## Extending / Fine-tuning (placeholder)
- TODO: Add guidance on fine-tuning with NeMo, data augmentation, and adding new surahs/models.

## More detail
See `docs/` for deeper dives: architecture, evaluation modes, audio prep, and test coverage.
