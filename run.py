#!/usr/bin/env python3
"""
YOLO11 Training Pipeline for Unilever Bangladesh Product Annotation
This script trains a YOLO11 model to automatically annotate product images.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import shutil
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install it using:")
    print("pip install ultralytics")
    sys.exit(1)

try:
    import wandb
    from ultralytics.utils.callbacks.wb import callbacks as wb_callbacks
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wb_callbacks = None
    print("Warning: wandb not found. Install with: pip install wandb")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLO11 model for Unilever Bangladesh product annotation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the dataset directory (e.g., /path/to/MTSOS or ./MTSOS for Kaggle)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Image size for training (will be resized to img-size x img-size)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n.pt',
        help='YOLO11 model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt) or path to checkpoint for resuming'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use for training (e.g., 0 for GPU 0, cpu for CPU, or empty for auto)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads for data loading'
    )

    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Project directory to save results'
    )

    parser.add_argument(
        '--name',
        type=str,
        default='ubl_annotator',
        help='Experiment name'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (epochs without improvement)'
    )

    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['SGD', 'Adam', 'AdamW', 'auto'],
        help='Optimizer to use for training'
    )

    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )

    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='Final learning rate (lr0 * lrf)'
    )

    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help='SGD momentum/Adam beta1'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='Optimizer weight decay'
    )

    parser.add_argument(
        '--warmup-epochs',
        type=float,
        default=3.0,
        help='Warmup epochs'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights'
    )

    parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache images for faster training'
    )

    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio (only used if no separate validation set exists)'
    )

    return parser.parse_args()


def create_dataset_yaml(dataset_path, output_path='configs/dataset_generated.yaml'):
    """
    Create or update dataset.yaml file with absolute paths.

    Args:
        dataset_path: Path to the dataset directory
        output_path: Path to save the generated yaml file
    """
    dataset_path = Path(dataset_path).resolve()

    # Check if original data.yaml exists
    original_yaml = dataset_path / 'data.yaml'
    if not original_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")

    # Read original yaml
    with open(original_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # Update paths to absolute paths
    train_path = dataset_path / 'train' / 'images'

    # Check for valid and test directories
    valid_path = dataset_path / 'valid' / 'images'
    test_path = dataset_path / 'test' / 'images'

    if not valid_path.exists():
        print(f"Warning: Validation set not found at {valid_path}")
        print("Using training set for validation. Consider splitting your data.")
        valid_path = train_path

    # Update paths
    data['path'] = str(dataset_path)
    data['train'] = str(train_path)
    data['val'] = str(valid_path)

    # Only include test path if it exists
    if test_path.exists():
        data['test'] = str(test_path)
    elif 'test' in data:
        # Remove test key if test folder doesn't exist
        del data['test']

    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save updated yaml
    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\nDataset configuration saved to: {output_file}")
    print(f"Dataset path: {dataset_path}")
    print(f"Training images: {train_path}")
    print(f"Validation images: {valid_path}")
    if 'test' in data:
        print(f"Test images: {data['test']}")
    else:
        print(f"Test images: None (test folder not found)")
    print(f"Number of classes: {data['nc']}")

    return str(output_file), data


def print_training_info(args, yaml_path, data_config):
    """Print training configuration information."""
    print("\n" + "="*60)
    print("YOLO11 Training Configuration")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {yaml_path}")
    print(f"Classes: {data_config['nc']}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Workers: {args.workers}")
    print(f"Learning rate: {args.lr0}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")


def main():
    """Main training function."""
    args = parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Create dataset yaml with absolute paths
    print("\nPreparing dataset configuration...")
    yaml_path, data_config = create_dataset_yaml(args.dataset)

    # Initialize Weights & Biases with image logging
    wandb_project = os.environ.get('WANDB_PROJECT', 'UBL-Annotation')
    wandb_run = None
    if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY'):
        try:
            wandb.login(key=os.environ.get('WANDB_API_KEY'))
            wandb_run = wandb.init(
                project=wandb_project,
                name=args.name,
                config={
                    'model': args.model,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'img_size': args.img_size,
                    'optimizer': args.optimizer,
                    'lr0': args.lr0,
                    'classes': data_config['nc'],
                    'dataset': args.dataset,
                },
                settings=wandb.Settings(
                    # Log validation images every epoch
                    _disable_stats=False,
                    _disable_meta=False,
                )
            )
            print(f"\n{'='*60}")
            print(f"Weights & Biases initialized")
            print(f"Project: {wandb_project}")
            print(f"Run: {args.name}")
            print(f"Dashboard: {wandb_run.url}")
            print(f"Validation images will be logged every epoch")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            wandb_run = None

    # Print training info
    print_training_info(args, yaml_path, data_config)

    # Initialize model
    print(f"Loading YOLO11 model: {args.model}")
    model = YOLO(args.model)

    # Prepare training arguments
    train_args = {
        'data': yaml_path,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'device': args.device if args.device else None,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'save_period': args.save_period,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'seed': args.seed,
        'verbose': args.verbose,
        'pretrained': args.pretrained,
        'resume': args.resume,
    }

    # Add cache if specified
    if args.cache:
        train_args['cache'] = True

    # Start training
    print("\nStarting training...")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        results = model.train(**train_args)

        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print(f"Results saved to: {args.project}/{args.name}")
        print(f"Best model: {args.project}/{args.name}/weights/best.pt")
        print(f"Last model: {args.project}/{args.name}/weights/last.pt")
        print("="*60 + "\n")

        # Print best metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\nBest Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

        # Finish wandb run
        if wandb_run is not None:
            wandb.finish()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Checkpoints saved to: {args.project}/{args.name}/weights/")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
