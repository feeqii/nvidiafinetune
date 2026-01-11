.PHONY: help install install-dev download-model run sanity test clean

AUDIO_DIR ?= ./data/audio
OUT_DIR ?= ./outputs
MODEL_PATH ?= ./models/Conformer-CTC-L_spe128_ar-AR_3.0.nemo

help:
	@echo "Ozzie Arabic ASR Baseline Evaluation"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install dev dependencies (includes test tools)"
	@echo "  make download-model Download the NeMo model from NGC"
	@echo "  make sanity         Run sanity test (transcribe one file)"
	@echo "  make run            Run full batch evaluation"
	@echo "  make test           Run unit tests"
	@echo "  make clean          Remove outputs and cache"
	@echo ""
	@echo "Variables:"
	@echo "  AUDIO_DIR=$(AUDIO_DIR)"
	@echo "  OUT_DIR=$(OUT_DIR)"
	@echo "  MODEL_PATH=$(MODEL_PATH)"

install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest

download-model:
	python -m ozzie_asr.run download \
		--model_id speechtotext_ar_ar_conformer \
		--model_version trainable_v3.0 \
		--output_dir ./models

sanity:
	@echo "Running sanity test..."
	@FIRST_FILE=$$(ls $(AUDIO_DIR)/*.wav $(AUDIO_DIR)/*.mp3 $(AUDIO_DIR)/*.m4a 2>/dev/null | head -1); \
	if [ -z "$$FIRST_FILE" ]; then \
		echo "Error: No audio files found in $(AUDIO_DIR)"; \
		exit 1; \
	fi; \
	python -m ozzie_asr.run eval \
		--audio_dir $(AUDIO_DIR) \
		--out_dir $(OUT_DIR)/sanity \
		--nemo_model_path $(MODEL_PATH) \
		--max_files 1

run:
	python -m ozzie_asr.run eval \
		--audio_dir $(AUDIO_DIR) \
		--out_dir $(OUT_DIR) \
		--nemo_model_path $(MODEL_PATH)

run-full-surah:
	python -m ozzie_asr.run eval \
		--audio_dir $(AUDIO_DIR) \
		--out_dir $(OUT_DIR) \
		--nemo_model_path $(MODEL_PATH) \
		--assume_full_surah

test:
	python -m pytest scripts/test_normalize.py -v

clean:
	rm -rf outputs/*
	rm -rf __pycache__
	rm -rf ozzie_asr/__pycache__
	rm -rf scripts/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

