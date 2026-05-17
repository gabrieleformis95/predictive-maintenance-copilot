#!/bin/bash
set -e

DATA_DIR="data/raw/CMAPSSData"
if [ ! -f "${DATA_DIR}/train_FD001.txt" ]; then
    echo "Downloading C-MAPSS dataset from HF Hub..."
    python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="gabrieleformis/cmapss-dataset",
    repo_type="dataset",
    local_dir="data/raw/CMAPSSData",
)
EOF
fi

CKPT="checkpoints/autoencoder_FD001.pt"
if [ ! -f "$CKPT" ]; then
    echo "Downloading checkpoint from HF Hub..."
    python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id="gabrieleformis/pmcopilot-checkpoint",
    filename="autoencoder_FD001.pt",
    repo_type="model",
)
os.makedirs("checkpoints", exist_ok=True)
shutil.copy(path, "checkpoints/autoencoder_FD001.pt")
EOF
fi

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
exec streamlit run src/ui/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true
