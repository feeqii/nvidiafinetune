# Agent Handoff: Ozzie Arabic ASR Baseline Evaluation

## 🎯 PROJECT GOAL
Build a minimal, reproducible baseline evaluation pipeline for Arabic Quran recitation (children's voices) using NVIDIA's Arabic Conformer-CTC model to answer: **"Is this model close enough on kids' recitation for Al-Fatiha and Al-Ikhlas (no diacritics) before fine-tuning?"**

---

## ✅ WHAT HAS BEEN COMPLETED

### 1. RunPod Environment Setup
- **Pod ID**: `5e3317hkao5rx7`
- **Pod Name**: `ozzie-asr-baseline`
- **GPU**: RTX A4000 (16GB) - Community Cloud
- **Status**: STOPPED (to save costs) - needs to be restarted
- **Cost**: ~$0.17/hour when running

### 2. Model Downloaded on RunPod
- **Model**: `nvidia/riva/speechtotext_ar_ar_conformer:trainable_v3.0`
- **Local Path**: `/workspace/models/speechtotext_ar_ar_conformer_vtrainable_v3.0/Conformer-CTC-L_spe128_ar-AR_3.0.nemo`
- **Size**: 451 MB
- **Model Type**: `EncDecCTCModelBPE` (Conformer-CTC Large, Arabic)

### 3. Virtual Environment Created on RunPod
- **Path**: `/workspace/nemo_env`
- **Activation**: `source /workspace/nemo_env/bin/activate`
- **Key packages installed**:
  - `nemo_toolkit==1.23.0`
  - `pytorch-lightning==2.0.7`
  - `torch==2.9.1`
  - `huggingface_hub==0.21.0`
  - `transformers==4.36.0`
  - `datasets==2.16.0`
  - `jiwer`, `pandas`, `editdistance`, `pyyaml`

### 4. Model Loading VERIFIED
The model loads successfully in the virtual environment. Test command:
```bash
source /workspace/nemo_env/bin/activate
python3 << 'EOF'
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecCTCModel.restore_from("./models/speechtotext_ar_ar_conformer_vtrainable_v3.0/Conformer-CTC-L_spe128_ar-AR_3.0.nemo")
print("Model loaded successfully")
EOF
```

### 5. Local Codebase Structure (NOT YET UPLOADED TO RUNPOD)
```
/Users/feeq/nvidiaconformerfinetune/
├── ozzie_asr/                    # Main Python package
│   ├── __init__.py
│   ├── run.py                    # CLI entrypoint (needs review)
│   ├── text_normalize.py         # Arabic text normalization ✅
│   ├── canonical_texts.py        # Loads canonical surah texts ✅
│   ├── alignment.py              # Word-level alignment for SNAP_TO_CANONICAL ✅
│   ├── evaluator.py              # CER/WER evaluation logic ✅
│   ├── model_loader.py           # NeMo model loading ✅
│   └── transcriber.py            # Batch transcription wrapper ✅
├── scripts/
│   ├── preprocess_audio.py       # Audio conversion (ffmpeg) ✅
│   └── test_normalize.py         # Unit tests for normalization ✅
├── configs/
│   └── canonical_texts.json      # Al-Fatiha & Al-Ikhlas (no diacritics) ✅
├── data/
│   ├── 001_Al-Fatiha/            # 75 WAV recordings
│   └── 112_Al-Ikhlas/            # 42 WAV recordings
├── outputs/                      # gitignored, for results
├── requirements.txt              # Python dependencies
├── Makefile                      # Convenience commands
├── RUNBOOK.md                    # Setup & usage instructions
└── upload_to_runpod.sh           # SCP upload script (SSH blocked)
```

### 6. Audio Data Available Locally
- **75 Al-Fatiha recordings** in `/Users/feeq/nvidiaconformerfinetune/data/001_Al-Fatiha/`
- **42 Al-Ikhlas recordings** in `/Users/feeq/nvidiaconformerfinetune/data/112_Al-Ikhlas/`
- All files are `.wav` format

---

## ❌ WHAT STILL NEEDS TO BE DONE

### 1. Upload Code to RunPod
The Python code (`ozzie_asr/`, `configs/`, `scripts/`) needs to be transferred to `/workspace/` on RunPod.

**Options**:
- Use RunPod's web file upload feature
- Create files via web terminal using heredocs/cat
- Set up SSH properly (port 1753 was blocked previously)

### 2. Upload Audio Files to RunPod
The 117 audio files need to be transferred to `/workspace/data/` on RunPod.

### 3. Test the Full Pipeline
Run the evaluation pipeline end-to-end:
```bash
source /workspace/nemo_env/bin/activate
python -m ozzie_asr.run --audio_dir ./data --out_dir ./outputs --mode snap
```

### 4. User Mentioned Adding Pre-Filter for Audio
The user wants a pre-filter step to check:
- Minimum duration (skip very short clips)
- Audio quality/silence detection
- File integrity validation

---

## 📋 REQUIRED OUTPUTS (per original spec)

1. **`outputs/predictions.csv`**: file, duration_sec, surah_pred, surah_confidence, predicted_text_raw, predicted_text_normalized
2. **`outputs/metrics.csv`**: file, surah_ref_used, mode_used, CER, WER, snapped_text, notes
3. **`outputs/summary.md`**: Overall metrics, worst 5 files, common error patterns, Go/No-Go notes
4. **`outputs/run_info.json`**: Model ID, git commit, package versions, GPU name, date

---

## 🔧 KEY TECHNICAL DETAILS

### Evaluation Modes
1. **SNAP_TO_CANONICAL (default)**: Aligns model output to canonical text, scores snapped segment. Works for partial clips.
2. **FULL_SURAH**: Compares against full surah text. Only valid with `--assume_full_surah` flag.

### Text Normalization Rules
- Remove diacritics (harakat) and tatweel
- Normalize alef forms (إأآا → ا)
- Normalize yaa/alef maqsurah (ى → ي)
- **Taa marbuta (ة→ه) is OFF by default** (optional flag to enable)
- Remove punctuation, collapse whitespace

### Canonical Texts (no diacritics)
- **Al-Fatiha** (7 ayat, includes Bismillah): `بسم الله الرحمن الرحيم الحمد لله رب العالمين...`
- **Al-Ikhlas** (4 ayat): `قل هو الله احد الله الصمد...`

---

## 🚀 TO RESUME WORK

1. **Start the RunPod pod**:
   ```
   Use MCP tool: mcp_runpod_start-pod with podId: 5e3317hkao5rx7
   ```

2. **Connect to web terminal** at:
   `https://5e3317hkao5rx7-19123.proxy.runpod.net/` (URL may change after restart)

3. **Activate the virtual environment**:
   ```bash
   cd /workspace
   source nemo_env/bin/activate
   ```

4. **Upload the code and audio files** (see options above)

5. **Run the pipeline**

---

## ⚠️ KNOWN ISSUES & WORKAROUNDS

1. **SSH Connection Refused**: Port 1753 was blocked. Use web terminal instead.
2. **Dependency Conflicts**: NeMo 1.23.0 requires specific versions of huggingface_hub (0.21.0), transformers (4.36.0), datasets (2.16.0). These are already installed in the venv.
3. **NGC CLI**: Configured with API key, org set to "Ozzie (0873240708406460)".

---

## 📁 IMPORTANT FILE CONTENTS

### configs/canonical_texts.json
```json
{
  "surahs": {
    "fatiha": {
      "name": "Al-Fatiha",
      "surah_number": 1,
      "ayat_count": 7,
      "includes_bismillah": true,
      "text_normalized": "بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب عليهم ولا الضالين"
    },
    "ikhlas": {
      "name": "Al-Ikhlas",
      "surah_number": 112,
      "ayat_count": 4,
      "includes_bismillah": false,
      "text_normalized": "قل هو الله احد الله الصمد لم يلد ولم يولد ولم يكن له كفوا احد"
    }
  }
}
```

---

## 🔑 CREDENTIALS (already configured on RunPod)

- **NGC API Key**: `b2traWhydWQ5MGJ1bWRuOWVkczhvYjRwNWE6MDI5MmEwZjUtOWRkYy00NzU5LTk3MTYtZjBhMjIwMzBiYjBk`
- **NGC Org**: Ozzie (0873240708406460)
- **RunPod API Key**: Configured in Cursor MCP

---

## 📝 USER PREFERENCES

1. Always review entire codebase before making changes
2. Ask clarifying questions when necessary
3. Break problems into smaller tasks
4. Focus on issues at hand with strict organization
5. Don't create unnecessary complexity



