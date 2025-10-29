#!/usr/bin/env python3
"""
Train YOLO model on SKU-110K dataset for generic product detection.
This model will be used as Stage 1 in the two-stage detection pipeline.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO on SKU-110K dataset')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to dataset YAML file (default: SKU-110K.yaml in script directory)')
    parser.add_argument('--model', type=str, default='yolo11x.pt',
                       help='YOLO model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (0, 1, cpu, or empty for auto)')
    parser.add_argument('--name', type=str, default='sku110k',
                       help='Experiment name')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from the last saved checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine YAML path
    if args.data:
        yaml_path = Path(args.data)
    else:
        yaml_path = Path(__file__).parent / 'SKU-110K.yaml'

    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found!")
        return

    print("="*60)
    print("SKU-110K Training Configuration")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Resume: {args.resume}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Experiment name: {args.name}")
    print("="*60)
    print()

    # Note: The first time you run this, it will download the dataset (~13.6 GB)
    # This may take a while depending on your internet connection
    print("Note: If this is your first time, the dataset will be downloaded.")
    print("This is a ~13.6 GB download and may take some time.")
    print()

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Train model
    print("\nStarting training...")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device if args.device else None,
        project='runs/train',
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        verbose=True,
        pretrained=True,
        val=True,
        plots=True
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: runs/train/{args.name}/weights/best.pt")
    print(f"This model detects generic 'object' class (any product on shelf)")
    print("Use this as Stage 1 in your two-stage detection pipeline")
    print("="*60)


if __name__ == '__main__':
    main()
