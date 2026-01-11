# Audio Preparation

## Expected layout
- Place audio under `data/<surah_folder>/file.wav` (current canonical coverage: `001_Al-Fatiha`, `112_Al-Ikhlas`).
- Output artifacts land in `outputs/<timestamp>/`.

## Quality checklist
- Duration: 1–120s recommended.
- Format: WAV/MP3/M4A/FLAC/OGG/WEBM acceptable; pipeline will convert to 16 kHz mono WAV.
- Content: Quran recitation only (no chatter, noise, music).
- Loudness: avoid near-silent clips; ensure speech is present.
- Folder correctness: ensure file lives in the correct surah folder.

## Pre-filter before upload
Script: `scripts/prefilter_audio.py`
- Filters too-short/too-long clips, silence-heavy clips (ffmpeg silencedetect), and unreadable files.
- Example:
```bash
python scripts/prefilter_audio.py data \
  --min-duration 1.0 \
  --max-duration 120 \
  --min-non-silence 0.3 \
  --output prefilter_results.json \
  --output-list prefilter_passed.txt
```

## Preprocess/convert
Script: `scripts/preprocess_audio.py`
- Converts to WAV 16 kHz mono (ffmpeg), caches outputs.
- Used automatically by the main pipeline; can be run standalone:
```bash
python scripts/preprocess_audio.py data/001_Al-Fatiha --output preprocessed/
```

## Known problematic files
- See `outputs/problematic_files_analysis.md` for a curated list of bad clips (wrong surah, gibberish, silence). Move them to `data/excluded/` before running evaluation.

## Run-time preprocessing (pipeline)
- During `eval`, files are converted into `outputs/<timestamp>/preprocessed/`.
- Durations and original paths are tracked to map metrics back to originals.
