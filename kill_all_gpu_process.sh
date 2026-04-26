#!/bin/bash
# Kill all processes occupying GPU memory and free VRAM

echo "========== Current GPU status =========="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

echo ""
echo "========== GPU processes =========="
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader

PIDS=$(fuser /dev/nvidia* 2>/dev/null | tr -s ' ' '\n' | sort -u)

if [ -z "$PIDS" ]; then
    echo "No GPU processes found."
    exit 0
fi

echo ""
echo "Killing PIDs: $PIDS"
echo "$PIDS" | xargs kill -9 2>/dev/null

sleep 2

echo ""
echo "========== GPU status after cleanup =========="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
