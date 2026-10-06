#!/usr/bin/env bash
cd "$(dirname "$0")"
WEIGHTS="${IDENTISKIN_WEIGHTS:-../weights/ablation_yolov26/best_ModelD_proposed.pt}"
if [ ! -f "$WEIGHTS" ]; then
  echo "Model D weights not found yet: $WEIGHTS (wait for training to finish)"
  exit 1
fi
echo "Laptop IP address(es): $(hostname -I)"
echo "Phone app server address: http://$(hostname -I | awk '{print $1}'):8000"
IDENTISKIN_WEIGHTS="$WEIGHTS" exec ../.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
