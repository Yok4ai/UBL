# Quick Start - Grounding DINO Annotator

## Installation

```bash
cd /home/mkultra/Documents/UBL
pip install -r requirements.txt
cd annotator
npm install
```

## Start Servers

```bash
cd /home/mkultra/Documents/UBL
./start_annotator.sh
```

Or manually:

**Terminal 1 - Backend:**
```bash
cd /home/mkultra/Documents/UBL/annotator_backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd /home/mkultra/Documents/UBL/annotator
npm run dev
```

## Access

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Basic Usage

### Zero-Shot Auto-Detection (Main Feature!)

1. Open http://localhost:3000
2. Select an image from the left sidebar
3. Enter labels: `bottle, can, package, box`
4. Click **Auto-Detect**
5. Review detections
6. Click **Save Annotations**

### Manual Annotation

1. Select image
2. Click and drag to draw box
3. Enter label when prompted
4. Click **Save Annotations**

### Edit Existing Annotations

1. Select image (existing annotations load automatically)
2. Add new boxes or use auto-detect
3. Delete unwanted boxes (click to select, then "Delete Selected")
4. Click **Save Annotations**

## Output Location

- **Labels**: `/home/mkultra/Documents/UBL/runs/annotate/predictions3/labels/*.txt`
- **Classes**: `/home/mkultra/Documents/UBL/runs/annotate/predictions3/labels/classes.txt`
- **Format**: YOLO (ready for training!)

## Common Commands

```bash
# View saved annotations
cat /home/mkultra/Documents/UBL/runs/annotate/predictions3/labels/classes.txt

# Count annotated images
ls /home/mkultra/Documents/UBL/runs/annotate/predictions3/labels/*.txt | wc -l

# Check backend logs
cd /home/mkultra/Documents/UBL/annotator_backend
python app.py

# Check if backend is running
curl http://localhost:8000
```

## Tips

- Use **comma-separated labels** for multiple objects: `bottle, can, box`
- **Click boxes** to select them before deleting
- **Save frequently** to avoid losing work
- Try **descriptive labels** for better detection: "plastic bottle" vs "bottle"

## Need Help?

- Full guide: `ANNOTATOR_GUIDE.md`
- Backend README: `annotator_backend/README.md`
- API docs: http://localhost:8000/docs
