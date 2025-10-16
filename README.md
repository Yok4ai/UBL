# YOLO11 Product Annotator - Unilever Bangladesh

This project trains a YOLO11 model to automatically annotate Unilever Bangladesh product images. The trained model can be used as an automatic annotation tool for new product images.

## Dataset

The project uses the MTSOS dataset containing 60 different Unilever Bangladesh products including:
- Cleaning products (Domex, Vim, Surf Excel, Rin, Wheel)
- Personal care (Dove, Lux, Ponds, Vaseline)
- Food products (Knorr, Horlicks)
- And many more

## Project Structure

```
UBL/
├── MTSOS/                    # Dataset directory
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── labels/          # YOLO format annotations
│   └── data.yaml            # Original dataset config
├── configs/                  # Configuration files
│   ├── dataset.yaml         # Base dataset config
│   └── dataset_generated.yaml  # Auto-generated config (created during training)
├── models/                   # Directory for saved models
├── runs/                     # Training runs and results
├── utils/                    # Utility scripts
├── run.py                   # Main training script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Local Setup

1. Clone or navigate to the project directory:
```bash
cd /home/mkultra/Documents/UBL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Kaggle Setup

1. Upload the project files to Kaggle
2. Install dependencies in a Kaggle notebook:
```python
!pip install -r requirements.txt
```

## Usage

### Basic Training

Train with default settings (100 epochs, batch size 16):

```bash
python run.py --dataset ./MTSOS
```

### Custom Training Configuration

```bash
python run.py \
    --dataset /path/to/MTSOS \
    --epochs 200 \
    --batch-size 32 \
    --img-size 640 \
    --model yolo11s.pt \
    --device 0 \
    --name ubl_annotator_v1
```

### Kaggle Training

```python
!python run.py \
    --dataset /kaggle/input/mtsos-dataset/MTSOS \
    --epochs 150 \
    --batch-size 16 \
    --img-size 640 \
    --model yolo11n.pt \
    --device 0 \
    --project /kaggle/working/runs/train
```

## Command Line Arguments

### Dataset Arguments
- `--dataset`: Path to dataset directory (required)
- `--val-split`: Validation split ratio if no separate validation set (default: 0.2)

### Model Arguments
- `--model`: YOLO11 model variant (default: yolo11n.pt)
  - `yolo11n.pt`: Nano (fastest, smallest)
  - `yolo11s.pt`: Small
  - `yolo11m.pt`: Medium
  - `yolo11l.pt`: Large
  - `yolo11x.pt`: XLarge (slowest, most accurate)

### Training Arguments
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Image size for training (default: 640)
- `--device`: Device for training (default: auto)
  - `0`: GPU 0
  - `cpu`: CPU only
  - `0,1`: Multiple GPUs
  - ``: Auto-detect
- `--workers`: Number of data loading workers (default: 8)

### Optimization Arguments
- `--optimizer`: Optimizer choice (SGD, Adam, AdamW, auto) (default: auto)
- `--lr0`: Initial learning rate (default: 0.01)
- `--lrf`: Final learning rate factor (default: 0.01)
- `--momentum`: SGD momentum/Adam beta1 (default: 0.937)
- `--weight-decay`: Weight decay (default: 0.0005)
- `--warmup-epochs`: Warmup epochs (default: 3.0)

### Output Arguments
- `--project`: Project directory (default: runs/train)
- `--name`: Experiment name (default: ubl_annotator)
- `--save-period`: Save checkpoint every N epochs (default: 10)

### Other Arguments
- `--patience`: Early stopping patience (default: 50)
- `--seed`: Random seed for reproducibility (default: 0)
- `--cache`: Cache images for faster training
- `--resume`: Resume from last checkpoint
- `--verbose`: Verbose output
- `--pretrained`: Use pretrained weights (default: True)

## Training Examples

### Quick Test Run (Fast training for testing)
```bash
python run.py \
    --dataset ./MTSOS \
    --epochs 10 \
    --batch-size 8 \
    --model yolo11n.pt
```

### Production Training (High accuracy)
```bash
python run.py \
    --dataset ./MTSOS \
    --epochs 300 \
    --batch-size 32 \
    --img-size 640 \
    --model yolo11m.pt \
    --device 0 \
    --patience 100 \
    --cache \
    --name ubl_annotator_production
```

### Resume Training
```bash
python run.py \
    --dataset ./MTSOS \
    --resume \
    --name ubl_annotator_v1
```

## After Training

### Model Files

After training completes, you'll find:
- `runs/train/ubl_annotator/weights/best.pt`: Best model checkpoint
- `runs/train/ubl_annotator/weights/last.pt`: Last epoch checkpoint
- `runs/train/ubl_annotator/results.png`: Training metrics plots
- `runs/train/ubl_annotator/confusion_matrix.png`: Confusion matrix

### Using the Trained Model for Annotation

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/train/ubl_annotator/weights/best.pt')

# Annotate new images
results = model.predict(
    source='path/to/new/images',
    save=True,
    save_txt=True,  # Save annotations in YOLO format
    conf=0.25,      # Confidence threshold
    project='runs/predict',
    name='annotations'
)

# Access annotations
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        cls = int(box.cls[0])  # Class ID
        conf = float(box.conf[0])  # Confidence
        xyxy = box.xyxy[0].tolist()  # Coordinates
        print(f"Class: {model.names[cls]}, Confidence: {conf:.2f}")
```

### Batch Annotation Script

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO('runs/train/ubl_annotator/weights/best.pt')

# Annotate all images in a directory
image_dir = Path('path/to/images')
output_dir = Path('path/to/annotations')
output_dir.mkdir(exist_ok=True)

results = model.predict(
    source=str(image_dir),
    save_txt=True,
    save_conf=True,
    project=str(output_dir),
    name='labels'
)

print(f"Annotated {len(results)} images")
print(f"Annotations saved to: {output_dir}/labels/labels")
```

## Performance Tips

### For Faster Training
1. Use smaller model variant (`yolo11n.pt` or `yolo11s.pt`)
2. Reduce image size (`--img-size 320` or `--img-size 416`)
3. Increase batch size if you have enough GPU memory
4. Enable image caching (`--cache`)
5. Reduce number of workers if CPU is bottleneck

### For Better Accuracy
1. Use larger model variant (`yolo11l.pt` or `yolo11x.pt`)
2. Train for more epochs (`--epochs 300`)
3. Use higher image resolution (`--img-size 1280`)
4. Adjust learning rate (`--lr0 0.001`)
5. Increase patience for early stopping (`--patience 100`)

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `--batch-size 8`
- Reduce image size: `--img-size 320`
- Use smaller model: `--model yolo11n.pt`

### Training Too Slow
- Enable caching: `--cache`
- Reduce workers: `--workers 4`
- Use smaller image size: `--img-size 416`

### Poor Performance
- Train longer: `--epochs 200`
- Use larger model: `--model yolo11m.pt`
- Adjust learning rate: `--lr0 0.001`

## Model Information

### YOLO11 Variants Comparison

| Model | Size (MB) | mAPval 50-95 | Speed (ms) | Params (M) |
|-------|-----------|--------------|------------|------------|
| YOLOv11n | 5.7 | 39.5 | 1.5 | 2.6 |
| YOLOv11s | 19.4 | 47.0 | 2.5 | 9.4 |
| YOLOv11m | 40.8 | 51.5 | 4.7 | 20.1 |
| YOLOv11l | 53.9 | 53.4 | 6.2 | 25.3 |
| YOLOv11x | 110.2 | 54.7 | 11.3 | 56.9 |

## Dataset Classes

The model is trained to detect 60 Unilever Bangladesh products:

```
clinic_strong, domex_lime, domex_ocen, dove_hijab, dove_oxy, dove_pink, dove_white,
gl_men_crm, gl_men_fw, gl_wntr_crm, glucomax_d, horlicks_jr_bib_2, horlicks_pouch,
knorr_corn, knorr_frd_chicken, knorr_hot_sour, knorr_thai, lb_blue_handwash,
lb_blue_packet, lb_lmn_bottle_1l, lb_lmn_handwash, lb_lmn_packet, lb_red_bottle_1l,
lb_red_gallon_5l, lb_red_handwash, lb_red_packet, lux_almond, lux_aloe, lux_orchd,
lux_soft, pepsodent_pwdr, ponds_antibac, ponds_cold_crm, ponds_daily_care_fw,
ponds_lgt_oil_men_fw, ponds_moist_body, ponds_oil_cntr_men_fw, ponds_super_lgt,
ponds_van_crm, rin_lqd, rin_pwdr, sunsilk_perfect, surfxl_lqd, surfxl_pwdr,
tresemme_hd, tresemme_ks_wt, tresemme_nr, vaseline_ln_white_lc, vaseline_mosq,
vaseline_petro, vaseline_petro_aloe, vaseline_petro_cocoa, vaseline_total_moist,
vim_bar, vim_bottle, vim_gallon, vim_packet, vim_pwdr, wheel_pwdr, wheel_soap
```

## License

This project uses the MTSOS dataset which is licensed under CC BY 4.0.

## Support

For issues or questions, please check:
1. Ultralytics documentation: https://docs.ultralytics.com/
2. YOLO11 documentation: https://docs.ultralytics.com/models/yolo11/

## Citation

If you use this project, please cite:

```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}
```
