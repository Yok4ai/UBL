#!/usr/bin/env python3
"""
Annotation script for using trained YOLO11 model to annotate new images.
This script loads the trained model and generates annotations for new product images.
"""

import argparse
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install it using:")
    print("pip install ultralytics")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Use trained YOLO11 model to annotate product images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (e.g., runs/train/ubl_annotator/weights/best.pt)'
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to images or directory of images to annotate'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='runs/annotate',
        help='Output directory for annotations'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Image size for inference'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (0 for GPU, cpu for CPU, empty for auto)'
    )

    parser.add_argument(
        '--save-txt',
        action='store_true',
        default=True,
        help='Save annotations as .txt files in YOLO format'
    )

    parser.add_argument(
        '--save-conf',
        action='store_true',
        default=True,
        help='Save confidence scores in annotations'
    )

    parser.add_argument(
        '--save-img',
        action='store_true',
        default=True,
        help='Save annotated images'
    )

    parser.add_argument(
        '--line-width',
        type=int,
        default=2,
        help='Bounding box line width'
    )

    parser.add_argument(
        '--show-labels',
        action='store_true',
        default=True,
        help='Show labels on annotated images'
    )

    parser.add_argument(
        '--show-conf',
        action='store_true',
        default=True,
        help='Show confidence scores on annotated images'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def main():
    """Main annotation function."""
    args = parse_args()

    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source path not found: {source_path}")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = YOLO(str(model_path))
    print(f"Model loaded successfully!")
    print(f"Number of classes: {len(model.names)}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Annotation Configuration")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IOU threshold: {args.iou}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Save annotations: {args.save_txt}")
    print(f"Save images: {args.save_img}")
    print("="*60 + "\n")

    # Run inference
    print("Starting annotation...")
    try:
        results = model.predict(
            source=str(source_path),
            save=args.save_img,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.img_size,
            device=args.device if args.device else None,
            project=str(output_path),
            name='predictions',
            line_width=args.line_width,
            show_labels=args.show_labels,
            show_conf=args.show_conf,
            verbose=args.verbose
        )

        print("\n" + "="*60)
        print("Annotation Complete!")
        print("="*60)
        print(f"Processed {len(results)} images")

        # Count total detections
        total_detections = sum(len(r.boxes) for r in results)
        print(f"Total detections: {total_detections}")

        if args.save_img:
            print(f"Annotated images saved to: {output_path}/predictions/")
        if args.save_txt:
            print(f"Annotation files saved to: {output_path}/predictions/labels/")
        print("="*60 + "\n")

        # Print detection summary
        if args.verbose:
            print("\nDetection Summary:")
            class_counts = {}
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls_name}: {count}")

    except Exception as e:
        print(f"\nError during annotation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
