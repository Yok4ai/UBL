# Grounding DINO Zero-Shot Annotator Guide

## Overview

This tool provides a Roboflow-like annotation experience using Grounding DINO for zero-shot object detection. You can:

1. **Manually annotate** objects by drawing bounding boxes
2. **Auto-detect objects** by providing text descriptions
3. **Work with existing annotations** - add new labels on top of current YOLO annotations
4. **Export in YOLO format** - compatible with your existing YOLO11 training pipeline

## Quick Start

### 1. Install Dependencies

```bash
cd /home/mkultra/Documents/UBL
pip install -r requirements.txt
```

### 2. Start the Annotator

Option A: Use the startup script (recommended):
```bash
./start_annotator.sh
```

Option B: Start manually:

Terminal 1 (Backend):
```bash
cd annotator_backend
python app.py
```

Terminal 2 (Frontend):
```bash
cd annotator
npm run dev
```

### 3. Access the Interface

Open your browser and navigate to: `http://localhost:3000`

The backend API will be running at: `http://localhost:8000`

## Workflow Examples

### Scenario 1: Annotating from Scratch

1. Select an image from the sidebar
2. Click and drag to draw a bounding box around an object
3. Enter a label name (e.g., "bottle")
4. Repeat for other objects
5. Click "Save Annotations"

### Scenario 2: Zero-Shot Auto-Detection (Like Roboflow)

This is the main feature - detecting objects automatically!

1. Select an image
2. In the "Enter labels" field, type the objects you want to detect:
   ```
   bottle, can, package, carton, sachet
   ```
3. Click "Auto-Detect"
4. Grounding DINO will automatically find and annotate all matching objects
5. Review the detections - you can:
   - Click boxes to select them
   - Delete incorrect detections
   - Add more boxes manually
6. Click "Save Annotations"

### Scenario 3: Adding to Existing Annotations

Perfect for when you already have some annotations and want to add more!

1. Select an image (existing annotations will load automatically)
2. Either:
   - Draw new boxes manually for new object types
   - Use Auto-Detect to find additional objects
3. All annotations (old + new) will be combined
4. Click "Save Annotations"

### Scenario 4: Iterative Labeling

1. Start with auto-detection for common objects:
   ```
   bottle, can
   ```
2. Review and save
3. Then add more specific labels:
   ```
   damaged bottle, crushed can
   ```
4. Keep adding labels iteratively

## Features Explained

### Auto-Detection Settings

- **Threshold (0.4)**: How confident the model needs to be to detect an object
  - Lower = more detections (may include false positives)
  - Higher = fewer detections (more precise)

- **Text Threshold (0.3)**: How well the text description must match
  - Adjust in the code if needed for better results

### Label Management

- **Color-coded labels**: Each class gets a unique color
- **Label statistics**: See count of boxes per class
- **Class persistence**: Classes are saved to `classes.txt`

### Annotation Controls

- **Delete Selected**: Remove a selected box (click a box first)
- **Clear All**: Remove all annotations from current image
- **Save Annotations**: Export to YOLO format

### Canvas Controls

- **Click and drag**: Draw new bounding box
- **Click on box**: Select existing box
- **Crosshair cursor**: Shows you're in drawing mode

## File Structure

```
/home/mkultra/Documents/UBL/
├── runs/annotate/predictions3/
│   ├── *.jpg                    # Your images
│   ├── labels/
│   │   ├── *.txt               # YOLO format annotations
│   │   └── classes.txt         # List of class names
│   └── classes.txt             # Alternative location
├── annotator_backend/
│   ├── app.py                  # FastAPI server
│   └── requirements.txt
└── annotator/
    └── src/
        ├── app/page.tsx        # Main UI
        └── components/
            └── ImageAnnotator.tsx
```

## YOLO Format Output

Annotations are saved in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1) relative to image dimensions.

Example `000001.txt`:
```
0 0.5 0.5 0.2 0.3
1 0.7 0.4 0.15 0.25
```

With `classes.txt`:
```
bottle
can
```

## Tips for Best Results

### 1. Descriptive Labels

Good:
- "plastic bottle"
- "aluminum can"
- "cardboard box"

Less effective:
- "object"
- "thing"

### 2. Multiple Variations

Try different phrasings:
- "bottle, plastic bottle, water bottle"
- "can, soda can, beverage can"

### 3. Comma-Separated Lists

Enter multiple objects at once:
```
bottle, can, package, carton, box, sachet, pouch
```

### 4. Review Before Saving

Always review auto-detections:
- Check for missed objects (add manually)
- Remove false positives (Delete Selected)
- Verify label correctness

### 5. Batch Similar Images

Group similar images together and use the same labels for efficiency.

## Common Use Cases

### Product Annotation for Retail

```
Labels: bottle, can, package, box, sachet, pouch, carton
```

### Quality Control

```
Labels: defect, damage, crack, dent, scratch
```

### Inventory Management

```
Labels: full shelf, empty shelf, misplaced item, out of stock
```

## Troubleshooting

### Backend won't start
- Check if port 8000 is in use: `lsof -i :8000`
- Ensure dependencies are installed: `pip install -r requirements.txt`

### Frontend won't connect
- Verify backend is running: `curl http://localhost:8000`
- Check CORS settings in `app.py`

### Images not loading
- Verify path in `app.py`: `IMAGES_PATH`
- Check file permissions
- Supported formats: JPG, JPEG, PNG

### Auto-detect not working
- Try lowering thresholds
- Use more descriptive labels
- Check GPU availability: model runs faster on GPU

### Annotations not saving
- Check write permissions on labels directory
- Verify `LABELS_PATH` in `app.py`

## Advanced Configuration

### Change Image/Label Paths

Edit `annotator_backend/app.py`:

```python
IMAGES_PATH = "/your/custom/images/path"
LABELS_PATH = "/your/custom/labels/path"
```

### Adjust Detection Thresholds

In the frontend component or backend:

```python
threshold=0.3,        # Lower for more detections
text_threshold=0.2    # Lower for broader matching
```

### Use Different Model

Edit `app.py`:

```python
model_id = "IDEA-Research/grounding-dino-base"  # Larger, more accurate
# or
model_id = "IDEA-Research/grounding-dino-tiny"  # Faster, smaller
```

## Integration with YOLO11 Training

Your annotations are already compatible! The tool outputs:

1. **Annotations**: `labels/*.txt` in YOLO format
2. **Class names**: `classes.txt` with class list
3. **Images**: Already in `predictions3/`

Just update your `data.yaml`:

```yaml
path: /home/mkultra/Documents/UBL/runs/annotate/predictions3
train: images
val: images

names:
  0: bottle
  1: can
  # ... from classes.txt
```

## API Reference

### GET /images
List all images in the directory

### GET /image/{name}
Get base64-encoded image data

### GET /annotations/{name}
Load existing YOLO annotations

### POST /predict
```json
{
  "image_path": "image.jpg",
  "text_labels": ["bottle", "can"],
  "threshold": 0.4,
  "text_threshold": 0.3
}
```

### POST /save_annotations
```json
{
  "image_path": "image.jpg",
  "boxes": [...],
  "class_names": ["bottle", "can"]
}
```

## Comparison with Roboflow

| Feature | This Tool | Roboflow |
|---------|-----------|----------|
| Zero-shot detection | ✅ Grounding DINO | ❌ |
| Manual annotation | ✅ | ✅ |
| YOLO export | ✅ | ✅ |
| Local/offline | ✅ | ❌ |
| Free | ✅ | Limited |
| Auto-labeling | ✅ | ✅ (paid) |

## Next Steps

1. Start with a small batch (10-20 images)
2. Use auto-detection for common objects
3. Manually refine the annotations
4. Save and verify YOLO format
5. Train your YOLO11 model
6. Iterate based on results

## Support

For issues or questions:
- Check the README in `annotator_backend/`
- Review the code comments
- Check backend logs for errors
