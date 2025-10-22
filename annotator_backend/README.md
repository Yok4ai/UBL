# OWLv2 Annotator

A zero-shot object detection annotation tool using OWLv2 and Next.js. This tool allows you to annotate images by selecting objects and automatically detecting similar objects, similar to Roboflow.

## Features

- Zero-shot object detection using OWLv2 (both text and image queries)
- Manual annotation with bounding box drawing
- Automatic detection of objects based on text prompts
- YOLO format label import/export
- Web-based UI for easy annotation
- Support for existing annotations (can add on top of current annotations)

## Project Structure

```
.
├── annotator_backend/          # FastAPI backend
│   ├── app.py                  # Main API server
│   └── requirements.txt        # Python dependencies
└── annotator/                  # Next.js frontend
    └── src/
        ├── app/
        │   └── page.tsx        # Main page
        └── components/
            └── ImageAnnotator.tsx  # Annotation interface
```

## Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd /home/mkultra/Documents/UBL/annotator_backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backend server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd /home/mkultra/Documents/UBL/annotator
```

2. Install dependencies (if not already installed):
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

### Manual Annotation

1. Select an image from the sidebar
2. Click and drag on the image to draw a bounding box
3. Enter a label name when prompted
4. Click "Save Annotations" to save in YOLO format

### Auto-Detection (Zero-Shot)

1. Select an image from the sidebar
2. Enter object labels in the input field (comma-separated), e.g., "bottle, can, package"
3. Click "Auto-Detect" to run OWLv2 inference
4. The model will automatically detect and annotate matching objects
5. You can manually adjust or add more boxes
6. Click "Save Annotations" to save

### Working with Existing Annotations

- The tool automatically loads existing annotations from `/home/mkultra/Documents/UBL/runs/annotate/predictions3/labels`
- You can add new annotations on top of existing ones
- All annotations are saved in YOLO format

## Configuration

### Image and Label Paths

The backend is configured to use:
- Images: `/home/mkultra/Documents/UBL/runs/annotate/predictions3`
- Labels: `/home/mkultra/Documents/UBL/runs/annotate/predictions3/labels`

You can modify these paths in `annotator_backend/app.py`:

```python
IMAGES_PATH = "/your/custom/path/to/images"
LABELS_PATH = "/your/custom/path/to/labels"
```

### Model Configuration

The default model is `google/owlv2-large-patch14-ensemble`. You can change it in `app.py`:

```python
owlv2_model_id = "google/owlv2-large-patch14-ensemble"  # or other OWLv2 variants
```

### Detection Thresholds

You can adjust detection thresholds when making predictions:
- `threshold`: Object detection confidence threshold (default: 0.4)
- `text_threshold`: Text-to-object matching threshold (default: 0.3)

## API Endpoints

- `GET /images` - List all images
- `GET /image/{image_name}` - Get image data
- `GET /annotations/{image_name}` - Get existing annotations
- `POST /predict` - Run zero-shot text-based detection (OWLv2)
- `POST /predict_visual_similarity` - Run visual similarity detection (OWLv2 image queries)
- `POST /save_annotations` - Save annotations in YOLO format
- `DELETE /annotations/{image_name}` - Delete annotations

## Keyboard Shortcuts

- Click and drag: Draw new bounding box
- Click on box: Select box
- Delete Selected: Remove selected box
- Clear All: Remove all annotations

## Notes

- Annotations are automatically saved in YOLO format
- Class names are saved in `classes.txt` in the labels directory
- The tool supports JPG, JPEG, and PNG image formats
- GPU acceleration is automatically used if available
