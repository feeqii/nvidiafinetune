# Tests Overview

## Current suite
- `scripts/test_normalize.py` (pytest)
  - Covers Arabic text normalization functions (`remove_diacritics`, `remove_tatweel`, `normalize_alef`, `normalize_alef_maqsurah`, `normalize_taa_marbuta`, `remove_punctuation`, `collapse_whitespace`, `normalize_arabic` end-to-end).
  - Verifies taa marbuta handling defaults (off) and optional conversion (on).
  - Edge cases: empty strings, already-normalized text, mixed Arabic/English, numbers, only diacritics, Unicode normalization consistency.
  - Ensures `ArabicNormalizer.get_config()` exposes settings.

## How to run
```bash
pip install -r requirements.txt
pip install pytest
make test
# or: python -m pytest scripts/test_normalize.py -v
```

## Status
- Tests were not executed in this documentation pass (no runtime validation performed).

## Gaps / next steps
- No automated coverage for preprocessing, model loading, transcription, or evaluation modes.
- Add smoke tests for `ozzie_asr.run eval` with a tiny fixture audio + stub model.
- Add regression tests for alignment edge cases (short clips, wrong-surah clips, silence).
- Add CLI contract tests for `download` and `list-models` (with mocking).
