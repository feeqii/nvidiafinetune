# Repository Guidelines

## Project Structure & Modules
- Core code lives in `ozzie_asr/`: CLI entrypoint `run.py`, model loader, transcriber, evaluator/alignment, canonical text helpers, and normalization (`text_normalize.py`).
- Data and artifacts are separated: place input audio under `data/`, generated run outputs under `outputs/<timestamp>/`, and keep large models in `models/` (all git-ignored).
- Support files: `scripts/` (preprocess/prefilter helpers), `configs/` (canonical texts), `docs/` (architecture, evaluation, tests), `RUNBOOK.md` (RunPod recipe), and `Makefile` for common workflows.

## Setup, Build, and Run
- Create a virtual env and install deps: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`; add `pip install pytest` or `make install-dev` for tests.
- Download the NeMo model: `make download-model` (uses `ozzie_asr.run download` with NGC).
- Smoke check: `make sanity AUDIO_DIR=./data/audio MODEL_PATH=./models/<model>.nemo` (transcribes one file).
- Full eval: `make run AUDIO_DIR=./data/audio OUT_DIR=./outputs MODEL_PATH=./models/<model>.nemo`; use `make run-full-surah` to force full-surah scoring.
- Direct CLI option: `python -m ozzie_asr.run eval --audio_dir <dir> --out_dir <dir> --nemo_model_path <file> [--assume_full_surah] [--normalize_taa_marbuta] [--batch_size N] [--cpu]`.

## Coding Style & Naming Conventions
- Python 3 with 4-space indentation; favor small, typed functions and module-level logging via `logging.getLogger(__name__)` over `print`.
- Modules and files use `snake_case`; keep CLI flags aligned with existing ones (`--assume_full_surah`, `--normalize_taa_marbuta`).
- Preserve default normalization behavior (`normalize_taa_marbuta=False` by default) unless explicitly toggled; document new flags in `README.md` and `RUNBOOK.md`.
- When adding modules, mirror current patterns: docstring at top, type hints, and pure functions where possible to ease testing.

## Testing Guidelines
- Tests use pytest; current suite lives in `scripts/test_normalize.py` and can be run with `make test` or `python -m pytest scripts/test_normalize.py -v`.
- Name new tests `test_*.py` and co-locate near the logic (e.g., under `scripts/` or alongside `ozzie_asr` modules). Cover both default and optional code paths, especially normalization flags and CLI argument parsing.
- For pipeline changes, add small fixtures and smoke tests (even CPU-only) to verify `ozzie_asr.run eval` behavior without relying on large models where possible.

## Commit & Pull Request Guidelines
- Keep commits small with imperative, present-tense subjects (e.g., “Add documentation”, “Fix taa marbuta toggle”); existing history favors concise summaries.
- In PRs, include: purpose and scope, key commands run (`make test`, `make run` if applicable), notable outputs (link to `outputs/<timestamp>/summary.md`), and any data/model requirements.
- Do not commit large artifacts (`outputs/`, `data/`, `models/`, `.nemo` files) or secrets (`.ngc_api_key`); rely on the gitignore defaults and share paths instead of payloads.
