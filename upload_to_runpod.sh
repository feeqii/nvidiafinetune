#!/bin/bash
# Upload project files to RunPod
# Run this script from your local machine

RUNPOD_HOST="193.183.22.59"
RUNPOD_PORT="1753"
RUNPOD_USER="root"
LOCAL_PROJECT_DIR="/Users/feeq/nvidiaconformerfinetune"
REMOTE_DIR="/workspace"

echo "=== Uploading Ozzie ASR Baseline to RunPod ==="
echo "Host: $RUNPOD_HOST:$RUNPOD_PORT"
echo ""

# Upload Python code (ozzie_asr package)
echo "Uploading ozzie_asr package..."
scp -P $RUNPOD_PORT -r "$LOCAL_PROJECT_DIR/ozzie_asr" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/"

# Upload configs
echo "Uploading configs..."
scp -P $RUNPOD_PORT -r "$LOCAL_PROJECT_DIR/configs" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/"

# Upload scripts
echo "Uploading scripts..."
scp -P $RUNPOD_PORT -r "$LOCAL_PROJECT_DIR/scripts" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/"

# Upload requirements.txt and other files
echo "Uploading requirements.txt, Makefile, RUNBOOK.md..."
scp -P $RUNPOD_PORT "$LOCAL_PROJECT_DIR/requirements.txt" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/"
scp -P $RUNPOD_PORT "$LOCAL_PROJECT_DIR/Makefile" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/"
scp -P $RUNPOD_PORT "$LOCAL_PROJECT_DIR/RUNBOOK.md" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/"

# Upload audio data
echo "Uploading audio data (this may take a while)..."
scp -P $RUNPOD_PORT -r "$LOCAL_PROJECT_DIR/data/001_Al-Fatiha" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/data/"
scp -P $RUNPOD_PORT -r "$LOCAL_PROJECT_DIR/data/112_Al-Ikhlas" "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/data/"

echo ""
echo "=== Upload complete! ==="
echo "You can now SSH into RunPod and run the pipeline:"
echo "  ssh -p $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST"
echo "  cd /workspace && python -m ozzie_asr.run --help"



