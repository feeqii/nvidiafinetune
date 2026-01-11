# Ozzie Arabic ASR Baseline Evaluation - RUNBOOK

This runbook provides copy-paste commands for running the Arabic ASR baseline evaluation on a RunPod L4 (24GB) instance.

## Table of Contents

1. [RunPod Setup](#1-runpod-setup)
2. [Environment Setup](#2-environment-setup)
3. [NGC CLI Setup](#3-ngc-cli-setup)
4. [Download Model](#4-download-model)
5. [Prepare Audio Files](#5-prepare-audio-files)
6. [Run Evaluation](#6-run-evaluation)
7. [Interpret Results](#7-interpret-results)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. RunPod Setup

### 1.1 Create Pod

1. Go to [RunPod](https://www.runpod.io/)
2. Create a new GPU Pod:
   - GPU: **NVIDIA L4** (24GB VRAM)
   - Template: **RunPod Pytorch 2.1** (or similar with CUDA)
   - Container Disk: 50GB minimum
   - Volume: 50GB (for model storage)

### 1.2 Connect via SSH

Get your SSH connection string from RunPod dashboard:

```bash
# Example (replace with your actual connection)
ssh root@<your-pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

### 1.3 Connect Cursor/VSCode via Remote SSH

1. Install "Remote - SSH" extension
2. Add SSH host in `~/.ssh/config`:

```
Host runpod-l4
    HostName <your-pod-ip>
    User root
    Port <port>
    IdentityFile ~/.ssh/id_ed25519
```

3. Connect: `Cmd/Ctrl + Shift + P` → "Remote-SSH: Connect to Host" → select `runpod-l4`

---

## 2. Environment Setup

### 2.1 Clone Repository

```bash
cd /workspace
git clone <your-repo-url> nvidiaconformerfinetune
cd nvidiaconformerfinetune
```

Or if already cloned:

```bash
cd /workspace/nvidiaconformerfinetune
git pull
```

### 2.2 Install System Dependencies

```bash
# Install ffmpeg for audio processing
apt-get update && apt-get install -y ffmpeg

# Verify ffmpeg
ffmpeg -version
```

### 2.3 Install Python Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify NeMo installation
python -c "import nemo; print(f'NeMo version: {nemo.__version__}')"
```

### 2.4 Verify GPU

```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 3. NGC CLI Setup

### 3.1 Install NGC CLI

```bash
# Download NGC CLI
wget -O ngc-cli.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.31.0/files/ngccli_linux.zip

# Extract
unzip ngc-cli.zip -d ngc-cli
chmod +x ngc-cli/ngc-cli/ngc

# Add to PATH
export PATH=$PATH:$(pwd)/ngc-cli/ngc-cli
echo 'export PATH=$PATH:/workspace/nvidiaconformerfinetune/ngc-cli/ngc-cli' >> ~/.bashrc

# Verify
ngc --version
```

### 3.2 Configure NGC API Key

1. Get your API key from [NGC](https://ngc.nvidia.com/setup/api-key)
2. Configure:

```bash
ngc config set
# Enter your API key when prompted
# Accept defaults for other options
```

### 3.3 Verify NGC Access

```bash
ngc registry model list nvidia/riva/speechtotext_ar_ar_conformer
```

---

## 4. Download Model

### 4.1 Download via CLI Tool

```bash
cd /workspace/nvidiaconformerfinetune

# Download the trainable NeMo model
python -m ozzie_asr.run download \
    --model_id speechtotext_ar_ar_conformer \
    --model_version trainable_v3.0 \
    --output_dir ./models
```

### 4.2 Download via NGC CLI (Alternative)

```bash
mkdir -p models
ngc registry model download-version \
    nvidia/riva/speechtotext_ar_ar_conformer:trainable_v3.0 \
    --dest ./models
```

### 4.3 Verify Download

```bash
ls -la models/
# Should see: Conformer-CTC-L_spe128_ar-AR_3.0.nemo (~430 MB)

# Find exact path
find ./models -name "*.nemo"
```

---

## 5. Prepare Audio Files

### 5.1 Upload Audio Files

Upload your audio files to the `data/audio` directory:

```bash
mkdir -p data/audio

# Option 1: SCP from local machine
# (Run this on your LOCAL machine, not the pod)
scp -P <port> -i ~/.ssh/id_ed25519 \
    /path/to/your/audio/*.wav \
    root@<pod-ip>:/workspace/nvidiaconformerfinetune/data/audio/

# Option 2: Download from cloud storage
# wget or curl your files

# Option 3: Use rclone for Google Drive, S3, etc.
```

### 5.2 Verify Audio Files

```bash
ls -la data/audio/
# Should show your .wav, .mp3, or .m4a files

# Check file count
find data/audio -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.m4a" \) | wc -l
```

---

## 6. Run Evaluation

### 6.1 Sanity Test (Single File)

Test with one file first to verify everything works:

```bash
cd /workspace/nvidiaconformerfinetune

# Find the model path
MODEL_PATH=$(find ./models -name "*.nemo" | head -1)
echo "Model path: $MODEL_PATH"

# Run sanity test
python -m ozzie_asr.run eval \
    --audio_dir ./data/audio \
    --out_dir ./outputs/sanity \
    --nemo_model_path "$MODEL_PATH" \
    --max_files 1
```

### 6.2 Full Batch Evaluation (Default: SNAP_TO_CANONICAL)

```bash
MODEL_PATH=$(find ./models -name "*.nemo" | head -1)

python -m ozzie_asr.run eval \
    --audio_dir ./data/audio \
    --out_dir ./outputs \
    --nemo_model_path "$MODEL_PATH"
```

### 6.3 Full Surah Mode

If you know all clips are full surah recitations:

```bash
MODEL_PATH=$(find ./models -name "*.nemo" | head -1)

python -m ozzie_asr.run eval \
    --audio_dir ./data/audio \
    --out_dir ./outputs \
    --nemo_model_path "$MODEL_PATH" \
    --assume_full_surah
```

### 6.4 With Taa Marbuta Normalization

```bash
MODEL_PATH=$(find ./models -name "*.nemo" | head -1)

python -m ozzie_asr.run eval \
    --audio_dir ./data/audio \
    --out_dir ./outputs \
    --nemo_model_path "$MODEL_PATH" \
    --normalize_taa_marbuta
```

### 6.5 Using Makefile

```bash
# Set model path
export MODEL_PATH=$(find ./models -name "*.nemo" | head -1)

# Run sanity test
make sanity MODEL_PATH=$MODEL_PATH

# Run full evaluation
make run MODEL_PATH=$MODEL_PATH

# Run with full surah mode
make run-full-surah MODEL_PATH=$MODEL_PATH
```

---

## 7. Interpret Results

### 7.1 Output Files

After evaluation, check the timestamped output directory:

```bash
ls -la outputs/
# Find latest run
LATEST=$(ls -t outputs/ | head -1)
ls -la outputs/$LATEST/
```

Files generated:
- `predictions.csv` - Raw and normalized transcriptions
- `metrics.csv` - Per-file CER/WER metrics
- `summary.md` - Overall statistics and Go/No-Go assessment
- `run_info.json` - Reproducibility metadata
- `errors.log` - Any processing errors

### 7.2 View Summary

```bash
cat outputs/$LATEST/summary.md
```

### 7.3 Check Metrics

```bash
# View metrics CSV
head -20 outputs/$LATEST/metrics.csv

# Quick stats with Python
python -c "
import pandas as pd
df = pd.read_csv('outputs/$LATEST/metrics.csv')
print(f'Mean CER: {df.cer.mean():.4f} ({df.cer.mean()*100:.2f}%)')
print(f'Mean WER: {df.wer.mean():.4f} ({df.wer.mean()*100:.2f}%)')
print(f'Worst file: {df.loc[df.cer.idxmax(), \"file\"]} (CER: {df.cer.max():.4f})')
"
```

### 7.4 Go/No-Go Interpretation

| CER Range | Assessment | Action |
|-----------|------------|--------|
| < 10% | Excellent | Ready for production |
| 10-20% | Good | Minor fine-tuning optional |
| 20-30% | Acceptable | Fine-tuning recommended |
| > 30% | Poor | Fine-tuning required |

---

## 8. Troubleshooting

### 8.1 CUDA Out of Memory

```bash
# Reduce batch size
python -m ozzie_asr.run eval \
    --audio_dir ./data/audio \
    --nemo_model_path "$MODEL_PATH" \
    --batch_size 4

# Or use CPU (slower)
python -m ozzie_asr.run eval \
    --audio_dir ./data/audio \
    --nemo_model_path "$MODEL_PATH" \
    --cpu
```

### 8.2 NGC Download Fails

```bash
# Check NGC config
ngc config current

# Re-authenticate
ngc config set

# Manual download via browser:
# 1. Go to: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/speechtotext_ar_ar_conformer
# 2. Select version: trainable_v3.0
# 3. Download the .nemo file
# 4. Upload to pod
```

### 8.3 Audio Conversion Fails

```bash
# Check ffmpeg
ffmpeg -version

# Test single file conversion
ffmpeg -i data/audio/test.mp3 -ar 16000 -ac 1 test.wav

# Check file format
file data/audio/*
```

### 8.4 Model Loading Fails

```bash
# Check model file
ls -la models/*.nemo
file models/*.nemo

# Test model loading
python -c "
from nemo.collections.asr.models import EncDecCTCModel
model = EncDecCTCModel.restore_from('$(find ./models -name \"*.nemo\" | head -1)')
print('Model loaded successfully!')
print(f'Model type: {type(model).__name__}')
"
```

### 8.5 Import Errors

```bash
# Reinstall NeMo
pip uninstall nemo_toolkit -y
pip install nemo_toolkit[asr]>=1.20.0

# Check installation
pip show nemo_toolkit
```

### 8.6 Run Unit Tests

```bash
# Test normalization
python -m pytest scripts/test_normalize.py -v

# Or with make
make test
```

---

## Quick Reference

```bash
# One-liner: Full setup and run
cd /workspace/nvidiaconformerfinetune && \
pip install -r requirements.txt && \
apt-get update && apt-get install -y ffmpeg && \
python -m ozzie_asr.run download --output_dir ./models && \
python -m ozzie_asr.run eval \
    --audio_dir ./data/audio \
    --nemo_model_path $(find ./models -name "*.nemo" | head -1)
```

---

## Links

- NGC Model: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/speechtotext_ar_ar_conformer
- NeMo Docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html
- NGC CLI: https://org.ngc.nvidia.com/setup/installers/cli
- RunPod Docs: https://docs.runpod.io/pods/connect-to-a-pod

