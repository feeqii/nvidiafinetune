# Evaluation Details

## Modes
- `SNAP_TO_CANONICAL` (default): Aligns normalized hypothesis to best-matching substring of canonical text using word-level similarity. Stores snapped text and alignment indices.
- `FULL_SURAH`: Compares to full surah text (strict, best for full recitations).

Switch with `--assume_full_surah` or `EvalMode.FULL_SURAH`.

## Normalization pipeline (Arabic)
Applied to hypothesis (and canonical when needed):
1) Remove diacritics (harakat, shadda, sukun, dagger alef, etc.)
2) Remove tatweel
3) Normalize alef variants (آأإٱ → ا)
4) Normalize alef maqsurah (ى → ي)
5) Optional: normalize taa marbuta (ة → ه) when `--normalize_taa_marbuta`
6) Remove punctuation (Arabic + general)
7) Collapse whitespace

Implementation: `ozzie_asr/text_normalize.py` (`ArabicNormalizer`). Canonical texts in `configs/canonical_texts.json` are already diacritic-free and alef-normalized; taa marbuta is preserved unless the flag is set.

## Surah classification
- Normalized hypothesis is compared to each canonical text via `find_best_alignment` (word-level SequenceMatcher).
- Highest similarity → `predicted_surah`; if below `unknown_threshold` (default 0.2) → `unknown` (evaluated against Fatiha with a note).
- Scores per surah retained in `ClassificationResult`.

## Metrics
- CER/WER via `jiwer` on normalized hypothesis vs reference (full or snapped).
- Per-file outputs stored in `metrics.csv` with alignment window and notes.
- Summary metrics: mean/median/std/min/max CER/WER, surah distribution, total duration (see `summary.md` and `run_info.json`).

## Outputs and how to read them
- `metrics.csv`: Inspect `notes` for low alignment warnings; `alignment_start/end` show matched window (SNAP mode).
- `predictions.csv`: Includes `surah_pred`, raw text, normalized text (handy for spot checks).
- `summary.md`: Lists worst 5 files by CER; includes go/no-go guidance:
  - CER < 15%: GO
  - 15–30%: CONDITIONAL GO (fine-tune recommended)
  - >30%: NO-GO (needs fine-tuning)

## Adding more surahs
1) Extend `configs/canonical_texts.json` with normalized text and ayat list.
2) Ensure text is already normalized (no diacritics, alef unified, taa marbuta preserved unless you intend to normalize it).
3) No code changes required unless you want custom thresholds per surah.

## Known caveats
- Garbage/silent clips pull CER/WER up; pre-filter audio first.
- `--normalize_taa_marbuta` can hurt matches where ة/ه distinction matters; use only if your transcripts lack taa marbuta.
- Classification uses similarity, not acoustic cues; wrong-surah clips with high textual similarity can be misclassified.
