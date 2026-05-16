#!/bin/bash
set -e

CKPT="checkpoints/autoencoder_FD001.pt"
if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found. Training model..."
    python scripts/train.py --epochs 50
fi

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
exec streamlit run src/ui/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true
