#!/bin/bash
# Quick start script for YOLO11 training

echo "=================================="
echo "YOLO11 UBL Annotator Quick Start"
echo "=================================="
echo ""

# Check if dataset exists
if [ ! -d "MTSOS" ]; then
    echo "Error: MTSOS dataset not found in current directory"
    echo "Please ensure the dataset is in the correct location"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting training with default settings:"
echo "  - Model: yolo11n.pt (nano)"
echo "  - Epochs: 100"
echo "  - Batch size: 16"
echo "  - Image size: 640"
echo ""
echo "You can modify these settings by editing this script or running run.py directly"
echo ""

# Start training
python3 run.py \
    --dataset ./MTSOS \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --model yolo11n.pt \
    --name ubl_annotator_quickstart \
    --patience 50

echo ""
echo "Training complete!"
echo "Check results in: runs/train/ubl_annotator_quickstart/"
