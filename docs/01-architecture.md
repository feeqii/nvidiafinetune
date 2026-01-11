# Architecture & Workflow

## Purpose
Single-node batch pipeline to preprocess Quran recitations, run NeMo Conformer-CTC ASR, and evaluate against canonical texts (Al-Fatiha, Al-Ikhlas) with Quran-aware scoring.

## System view
- **Entry**: `ozzie_asr/run.py` CLI (`download`, `list-models`, `eval`)
- **Preprocess**: `scripts/preprocess_audio.py` (ffmpeg -> 16 kHz mono WAV; caches)
- **Model**: `ozzie_asr/model_loader.py` (load local .nemo or NGC download)
- **Transcribe**: `ozzie_asr/transcriber.py` (batch, tqdm)
- **Evaluate**: `ozzie_asr/evaluator.py` + `ozzie_asr/alignment.py`
- **Canonical data**: `configs/canonical_texts.json` via `ozzie_asr/canonical_texts.py`
- **Normalization**: `ozzie_asr/text_normalize.py` (diacritics stripped; taa marbuta optional)
- **Outputs**: `outputs/<timestamp>/` artifacts; logs + run metadata

## Data flow (eval)
1) Input: `--audio_dir` (expects WAV/MP3/M4A/FLAC/OGG/WEBM).
2) Preprocess → `run_dir/preprocessed/*.wav` + `file_infos` (duration, original path, format).
3) Model load → uses `--nemo_model_path` else cached/downloaded NGC model (`speechtotext_ar_ar_conformer`, `trainable_v3.0`).
4) Transcribe batch → `(path, text, error)` tuples; durations merged from preprocess metadata.
5) Normalize text → `ArabicNormalizer` (diacritics, tatweel, alef variants; optional taa marbuta).
6) Classify surah → similarity vs canonical texts (normalized) using `find_best_alignment`; threshold fallback to `unknown`.
7) Evaluate:
   - `SNAP_TO_CANONICAL` (default): best matching substring; records alignment window.
   - `FULL_SURAH`: compare against full surah text.
   CER/WER via `jiwer`.
8) Persist artifacts (CSV/MD/JSON/log) to `run_dir`.

## Key configs and switches
- `--assume_full_surah` → forces full-surah evaluation.
- `--normalize_taa_marbuta` → converts ة→ه in both hypothesis and canonical.
- `--batch_size`, `--cpu` → performance knobs for inference.
- `--max_files` → cap for quick checks.
- Model cache dir: `--model_cache_dir` (default `./models`).

## Error handling
- Preprocess errors collected in `errors` list and `errors.log`.
- Transcription failures recorded per file (error message preserved).
- Evaluation falls back to `SURAH_UNKNOWN` → default evaluated against Fatiha with note.
- Any exception bubbles to CLI with non-zero exit.

## Outputs (per run)
- `predictions.csv`: filename, duration, surah_pred, raw/normalized text.
- `metrics.csv`: per-file CER/WER, snapped reference, alignment span, notes.
- `summary.md`: aggregate metrics, worst files, go/no-go guidance.
- `run_info.json`: args, model info, git hash, environment, preprocessing/eval config.
- `errors.log`: warnings/errors encountered.

## Assumptions/constraints
- Canonical texts limited to Fatiha and Ikhlas (add more via `configs/canonical_texts.json`).
- Audio expected to be mostly Quran recitation; silence/gibberish degrades metrics (see `outputs/problematic_files_analysis.md`).
- GPU strongly recommended; CPU path exists but slow for batches.
