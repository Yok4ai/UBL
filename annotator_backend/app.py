"""
LLMDet Zero-Shot Annotator API

A simplified backend for zero-shot object detection and annotation using LLMDet.

Features:
- Text-based detection: Describe what you want to find, LLMDet finds it
- Reference-based detection: Draw one example, find all similar objects
- Automatic filtering by size, color, and geometry
- YOLO format output for training

Workflow:
1. Load images from directory
2. Use text prompts OR draw reference boxes to detect objects
3. LLMDet finds matching objects zero-shot
4. Save annotations in YOLO format
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from PIL import Image, ImageStat
import colorsys
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
from pathlib import Path
import base64
import tempfile

app = FastAPI(title="LLMDet Zero-Shot Annotator API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and processors
llmdet_model_id = "iSEE-Laboratory/llmdet_large"
device = "cuda" if torch.cuda.is_available() else "cpu"

# LLMDet for zero-shot object detection
llmdet_processor = None
llmdet_model = None

# Base paths - can be configured via API
CONFIG = {
    "images_path": "/home/mkultra/Documents/UBL/runs/annotate/predictions3",
    "labels_path": "/home/mkultra/Documents/UBL/runs/annotate/predictions3/labels",
    "dataset_yaml": "/home/mkultra/Documents/UBL/configs/dataset_generated.yaml"
}

IMAGES_PATH = CONFIG["images_path"]
LABELS_PATH = CONFIG["labels_path"]

PACKAGING_HINTS = {
    "bottle": "single bottle-shaped container",
    "bar": "rectangular soap bar",
    "sachet": "small sachet-sized packet",
    "powder": "powder pouch or can",
    "detergent": "detergent packaging",
    "liquid": "liquid cleaner bottle",
    "cleaner": "cleaning product bottle",
    "dish": "dishwashing product",
    "shampoo": "shampoo sachet or bottle",
    "soap": "soap bar package",
    "toothpaste": "toothpaste carton",
    "tube": "tube-shaped package",
    "pack": "single consumer pack",
}

COLOR_KEYWORDS = {
    "red": "red",
    "green": "green",
    "blue": "blue",
    "yellow": "yellow",
    "orange": "orange",
    "purple": "purple",
    "violet": "purple",
    "pink": "pink",
    "cyan": "cyan",
    "teal": "teal",
    "white": "white",
    "black": "black",
    "gray": "gray",
    "grey": "gray",
    "silver": "gray",
    "gold": "gold",
    "brown": "brown",
}

NEGATIVE_PROMPT = "Avoid decorative items, signs, shelves, or background objects. Focus only on the product package."


def parse_label_metadata(raw_label: str) -> dict:
    cleaned = raw_label.replace("_", " ").replace("-", " ").strip() if raw_label else ""
    tokens = [token for token in cleaned.split() if token]

    brand_tokens = []
    packaging_hints = []
    color_hints: list[str] = []

    for token in tokens:
        lower = token.lower()
        if lower in PACKAGING_HINTS:
            packaging_hints.append(PACKAGING_HINTS[lower])
        elif lower in COLOR_KEYWORDS:
            color_hints.append(COLOR_KEYWORDS[lower])
        else:
            brand_tokens.append(token)

    def _dedupe(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in seq:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    return {
        "cleaned": cleaned or "product",
        "brand_tokens": _dedupe(brand_tokens),
        "packaging_hints": _dedupe(packaging_hints),
        "color_hints": _dedupe(color_hints),
    }


def build_dino_prompt(
    raw_label: str,
    color_hints: Optional[List[str]] = None,
    user_context: Optional[str] = None,
) -> str:
    """Build a more specific text prompt for Grounding DINO based on the label."""
    meta = parse_label_metadata(raw_label)

    parts: list[str] = []
    parts.append(meta["cleaned"] or "product")

    if meta["packaging_hints"]:
        parts.extend(meta["packaging_hints"])
    else:
        parts.append("single retail product package")

    combined_color_hints = (color_hints or []) + meta.get("color_hints", [])
    if combined_color_hints:
        unique_colors: list[str] = []
        for hint in combined_color_hints:
            if hint and hint not in unique_colors:
                unique_colors.append(hint)
        if unique_colors:
            parts.append("expected colors: " + " and ".join(unique_colors[:3]))

    if meta["brand_tokens"]:
        parts.append("brand keywords: " + " ".join(meta["brand_tokens"]))

    if user_context:
        context_line = user_context.strip()
        if context_line:
            parts.append("context cues: " + context_line)

    parts.append(NEGATIVE_PROMPT)

    unique_parts = []
    for entry in parts:
        if entry and entry not in unique_parts:
            unique_parts.append(entry)

    return ". ".join(unique_parts) + "."


def merge_unique(base: List[str], extra: List[str]) -> List[str]:
    merged: List[str] = []
    for item in base + extra:
        if item and item not in merged:
            merged.append(item)
    return merged


def compute_iou(box_a: dict, box_b: dict) -> float:
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["width"], ay1 + box_a["height"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["width"], by1 + box_b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def deduplicate_by_iou(boxes: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)
    kept: list[dict] = []

    for box in sorted_boxes:
        if any(compute_iou(box, existing) > iou_threshold for existing in kept):
            continue
        kept.append(box)

    return kept


def generate_reference_slices(
    boxes: list[dict],
    reference_box: "BoundingBox",
    image_dims: tuple[int, int],
    top_k: int = 4,
    max_segments: int = 6,
    stride_factor: float = 0.95,
) -> list[dict]:
    if not boxes:
        return []

    img_width, img_height = image_dims
    ref_w = max(1.0, reference_box.width)
    ref_h = max(1.0, reference_box.height)

    candidates: list[dict] = []
    sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)[:top_k]

    for box in sorted_boxes:
        box_w = max(1.0, box.get("width", 0.0))
        box_h = max(1.0, box.get("height", 0.0))

        segments = max(1, min(max_segments, int(round(box_w / (ref_w * stride_factor)))))
        stride = ref_w * stride_factor

        for segment_idx in range(segments):
            x = box["x"] + segment_idx * stride
            x = min(max(0.0, x), img_width - ref_w)

            y = min(max(0.0, box["y"]), img_height - ref_h)

            candidates.append({
                "x": float(x),
                "y": float(y),
                "width": float(ref_w),
                "height": float(ref_h),
                "confidence": float(box.get("confidence", 0.0) * 0.75),
            })

    return candidates


def filter_boxes_by_geometry(
    boxes: list[dict],
    reference_box: "BoundingBox",
    image_dims: tuple[int, int],
    image_area_ratio: float = 0.25,  # Increased from 0.18 - allow larger boxes
    area_ratio_bounds: tuple[float, float] = (0.3, 2.2),  # More lenient: 30-220% of reference
    aspect_slack: float = 0.5,  # Increased from 0.35 - more aspect tolerance
    dimension_slack: float = 0.7,  # Increased from 0.55 - more size tolerance
    min_candidates: int = 2,  # Reduced from 3 - don't fallback too aggressively
    allow_fallback: bool = True,
) -> list[dict]:
    img_width, img_height = image_dims
    img_area = max(1.0, img_width * img_height)
    max_image_area = img_area * image_area_ratio

    ref_area = max(1.0, reference_box.width * reference_box.height)
    ref_aspect = reference_box.width / reference_box.height if reference_box.height else None

    def _filter(
        bounds: tuple[float, float],
        slack: float,
        aspect_tolerance: float,
    ) -> list[dict]:
        min_area = ref_area * bounds[0]
        max_area = ref_area * bounds[1]
        results: list[dict] = []

        for box in boxes:
            width = max(0.0, box.get("width", 0.0))
            height = max(0.0, box.get("height", 0.0))
            area = width * height

            if area <= 0:
                continue
            if area > max_image_area:
                continue
            if area < min_area or area > max_area:
                continue

            aspect = (width / height) if height else None
            if ref_aspect and aspect and ref_aspect > 0:
                if abs(aspect - ref_aspect) / ref_aspect > aspect_tolerance:
                    continue

            if width > reference_box.width * (1 + slack):
                continue
            if height > reference_box.height * (1 + slack):
                continue
            if width < reference_box.width * (1 - slack * 0.75):
                continue
            if height < reference_box.height * (1 - slack * 0.75):
                continue

            results.append(box)

        return results

    primary = _filter(area_ratio_bounds, dimension_slack, aspect_slack)

    if allow_fallback and len(primary) < min_candidates:
        # Very lenient fallback - only filter obviously wrong sizes
        relaxed_bounds = (0.15, 3.0)  # 15-300% of reference size
        relaxed_slack = 1.0  # 100% tolerance on dimensions
        relaxed_aspect = 0.8  # 80% aspect ratio tolerance
        fallback_results = _filter(relaxed_bounds, relaxed_slack, relaxed_aspect)

        if fallback_results:
            print(
                f"Geometry filter fallback applied (kept {len(fallback_results)} vs {len(primary)})."
            )
            return fallback_results

    return primary


def compute_average_color(image: Image.Image) -> tuple[float, float, float]:
    stat = ImageStat.Stat(image.convert("RGB"))
    return tuple(stat.mean)


def color_distance(color_a: tuple[float, float, float], color_b: tuple[float, float, float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(color_a, color_b)) ** 0.5


def rgb_to_basic_color(rgb: tuple[int, int, int]) -> str:
    r, g, b = [channel / 255.0 for channel in rgb]
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    v = max_c
    delta = max_c - min_c
    s = 0 if max_c == 0 else delta / max_c

    if v < 0.15:
        return "black"
    if v > 0.9 and s < 0.2:
        return "white"
    if s < 0.15:
        return "gray"

    h = 0.0
    if delta != 0:
        if max_c == r:
            h = (g - b) / delta % 6
        elif max_c == g:
            h = (b - r) / delta + 2
        else:
            h = (r - g) / delta + 4
        h *= 60
    else:
        h = 0.0

    if v < 0.5 and 15 <= h <= 60:
        return "brown"

    if 0 <= h < 25:
        return "red"
    if 25 <= h < 50:
        return "orange"
    if 50 <= h < 70:
        return "yellow"
    if 70 <= h < 170:
        return "green"
    if 170 <= h < 200:
        return "cyan"
    if 200 <= h < 255:
        return "blue"
    if 255 <= h < 290:
        return "purple"
    if 290 <= h < 330:
        return "magenta"

    return "red"


def extract_color_hints(image: Image.Image, max_colors: int = 3) -> List[str]:
    palette_source = image.resize((128, 128))
    adaptive = palette_source.convert("P", palette=Image.ADAPTIVE, colors=max_colors * 4)
    palette = adaptive.getpalette()
    color_counts = adaptive.getcolors()
    if not color_counts:
        return []

    hints: List[str] = []
    for count, palette_index in sorted(color_counts, reverse=True):
        rgb = tuple(palette[palette_index * 3: palette_index * 3 + 3])
        color_name = rgb_to_basic_color(rgb)
        if color_name not in hints:
            hints.append(color_name)
        if len(hints) >= max_colors:
            break

    return hints

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: Optional[float] = None

class Annotation(BaseModel):
    image_path: str
    boxes: List[BoundingBox]

class PredictRequest(BaseModel):
    image_path: str
    text_labels: List[str]
    threshold: float = 0.4
    text_threshold: float = 0.3

class SaveAnnotationRequest(BaseModel):
    image_path: str
    boxes: List[BoundingBox]
    class_names: List[str]

class VisualSimilarityRequest(BaseModel):
    image_path: str
    crop_box: BoundingBox  # The box user drew (label field contains AI description)
    similarity_threshold: float = 0.25  # DINO confidence threshold
    dino_threshold: float = 0.25  # DINO confidence threshold (explicit parameter)
    max_aspect_ratio: float = 3.0  # Legacy parameter (kept for backwards compatibility)
    label_context: Optional[str] = None  # Legacy - no longer used
    output_label: Optional[str] = None  # Class name for labeling detected boxes

@app.on_event("startup")
async def startup_event():
    global llmdet_processor, llmdet_model
    print(f"Loading models on device: {device}")

    # Load LLMDet for zero-shot object detection
    print("Loading LLMDet (this may take a moment)...")
    llmdet_processor = AutoProcessor.from_pretrained(llmdet_model_id)
    llmdet_model = AutoModelForZeroShotObjectDetection.from_pretrained(llmdet_model_id).to(device)
    llmdet_model.eval()
    print("LLMDet loaded successfully!")

    print("Model loaded and ready for zero-shot annotation!")
    # Ensure directories exist
    os.makedirs(LABELS_PATH, exist_ok=True)

@app.get("/")
async def root():
    return {
        "message": "LLMDet Annotator API",
        "device": device,
        "model": llmdet_model_id
    }

@app.get("/images")
async def list_images():
    """List all images in the predictions directory"""
    try:
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend([str(p.name) for p in Path(IMAGES_PATH).glob(ext)])
        return {"images": sorted(images)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/{image_name}")
async def get_image(image_name: str):
    """Get image as base64"""
    try:
        image_path = Path(IMAGES_PATH) / image_name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Get image dimensions
        img = Image.open(image_path)
        width, height = img.size

        return {
            "name": image_name,
            "data": f"data:image/jpeg;base64,{image_data}",
            "width": width,
            "height": height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/annotations/{image_name}")
async def get_annotations(image_name: str):
    """Get existing annotations for an image in YOLO format"""
    try:
        label_file = Path(LABELS_PATH) / f"{Path(image_name).stem}.txt"

        if not label_file.exists():
            return {"boxes": [], "class_names": []}

        # Read class names if classes.txt exists
        classes_file = Path(LABELS_PATH).parent / "classes.txt"
        class_names = []
        if classes_file.exists():
            with open(classes_file, "r") as f:
                class_names = [line.strip() for line in f.readlines()]

        # Get image dimensions
        image_path = Path(IMAGES_PATH) / image_name
        img = Image.open(image_path)
        img_width, img_height = img.size

        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    
                    height = float(parts[4])

                    # Convert from YOLO format (normalized center) to absolute coordinates
                    x = (x_center - width / 2) * img_width
                    y = (y_center - height / 2) * img_height
                    w = width * img_width
                    h = height * img_height

                    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    boxes.append({
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "label": label,
                        "confidence": None
                    })

        return {"boxes": boxes, "class_names": class_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def detect_with_llmdet(
    full_image: Image.Image,
    label: str,
    confidence_threshold: float = 0.35,
    color_hints: Optional[List[str]] = None,
    user_context: Optional[str] = None,
) -> list[dict]:
    """
    LLMDet zero-shot detection - powerful and accurate!
    Detects all objects matching the label using LLMDet's confidence threshold.
    """
    img_width, img_height = full_image.size

    # Build a label-aware prompt so LLMDet narrows the candidates up front.
    prompt = build_dino_prompt(label, color_hints=color_hints, user_context=user_context)

    print(f"\n{'='*60}")
    print(f"LLMDet Detection - Label: '{label}' | Prompt: '{prompt}' | Threshold: {confidence_threshold}")
    print(f"Image size: {img_width}x{img_height}")
    print(f"{'='*60}")

    inputs = llmdet_processor(images=full_image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llmdet_model(**inputs)

    # Debug raw outputs
    print(f"Raw LLMDet outputs - logits shape: {outputs.logits.shape}, pred_boxes shape: {outputs.pred_boxes.shape}")
    logits_sigmoid = outputs.logits.sigmoid()
    max_scores_per_query = logits_sigmoid[0].max(dim=-1)
    print(f"Max score across all queries: {max_scores_per_query.values.max().item():.4f}")
    print(f"Number of queries with score > 0.3: {(max_scores_per_query.values > 0.3).sum().item()}")
    print(f"Number of queries with score > 0.2: {(max_scores_per_query.values > 0.2).sum().item()}")
    print(f"Number of queries with score > 0.1: {(max_scores_per_query.values > 0.1).sum().item()}")

    # Post-process
    results = llmdet_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[(img_height, img_width)],
        threshold=confidence_threshold
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    print(f"After post-processing with threshold {confidence_threshold}: Detected {len(boxes)} objects")

    detected_boxes = []
    for i, (box, conf) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        print(f"  Box {i+1}: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] confidence: {conf:.3f}")
        detected_boxes.append({
            "x": float(x1),
            "y": float(y1),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
            "confidence": float(conf)
        })

    print(f"{'='*60}\n")
    return detected_boxes

@app.post("/predict")
async def predict(request: PredictRequest):
    """Simple text-based LLMDet detection"""
    try:
        image_path = Path(IMAGES_PATH) / request.image_path
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # Process labels
        labels = [label.strip() for label in request.text_labels]
        text_prompt = ". ".join(labels) + "."

        print(f"\n{'='*60}")
        print(f"Text-based LLMDet Detection")
        print(f"Prompt: '{text_prompt}'")
        print(f"Threshold: {request.threshold}")
        print(f"{'='*60}")

        inputs = llmdet_processor(images=image, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = llmdet_model(**inputs)

        results = llmdet_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[(img_height, img_width)],
            threshold=request.threshold
        )[0]

        boxes_data = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels_list = results["labels"]  # This might be strings or ints depending on version

        print(f"Detected {len(boxes_data)} objects")
        print(f"Debug - labels type: {type(labels_list)}, first element: {labels_list[0] if labels_list else 'empty'}, type: {type(labels_list[0]) if labels_list else 'N/A'}")

        boxes = []
        for box, score, label_item in zip(boxes_data, scores, labels_list):
            x1, y1, x2, y2 = box

            # Handle both integer indices and string labels
            if isinstance(label_item, str):
                label = label_item
            elif isinstance(label_item, int):
                label = labels[label_item] if label_item < len(labels) else "unknown"
            else:
                label = "unknown"

            boxes.append({
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "label": label,
                "confidence": float(score)
            })
            print(f"  {label}: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] conf={score:.3f}")

        print(f"{'='*60}\n")
        return {"boxes": boxes}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_visual_similarity")
async def predict_visual_similarity(request: VisualSimilarityRequest):
    """
    LLMDet zero-shot detection with reference-based filtering:
    1. User draws a reference box and provides description
    2. Extract visual features (colors, size) from reference
    3. Use LLMDet to detect similar objects with enhanced prompting
    4. Filter by geometry and color to match reference
    """
    try:
        image_path = Path(IMAGES_PATH) / request.image_path
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        image = Image.open(image_path).convert("RGB")
        crop_box = request.crop_box
        ai_description = crop_box.label  # User's description (e.g., "green bottle, red logo")
        output_label = request.output_label or crop_box.label  # Class name for dataset

        # Extract reference features from user's drawing
        reference_crop = image.crop((
            int(crop_box.x),
            int(crop_box.y),
            int(crop_box.x + crop_box.width),
            int(crop_box.y + crop_box.height)
        ))
        color_hints = extract_color_hints(reference_crop, max_colors=3)

        print(f"\n{'='*60}")
        print(f"Zero-Shot Detection with Reference")
        print(f"Description: '{ai_description}' | Output Label: '{output_label}'")
        print(f"Reference size: {reference_crop.size}")
        print(f"Detected colors: {color_hints}")
        print(f"Threshold: {request.dino_threshold}")
        print(f"{'='*60}")

        # Use LLMDet to detect all matching objects
        raw_boxes = detect_with_llmdet(
            image,
            ai_description,
            confidence_threshold=request.dino_threshold,
            color_hints=color_hints,
            user_context=None
        )

        img_width, img_height = image.size

        # Filter by geometry (size/aspect ratio similar to reference)
        filtered_boxes = filter_boxes_by_geometry(
            raw_boxes,
            crop_box,
            (img_width, img_height)
        )

        # Deduplicate overlapping boxes
        final_boxes = deduplicate_by_iou(filtered_boxes, iou_threshold=0.4)

        # Handle case where no matches found - try fallback slicing
        if not final_boxes:
            print("No matches found with strict filtering, trying fallback...")
            slice_candidates = generate_reference_slices(
                raw_boxes,
                crop_box,
                (img_width, img_height)
            )
            if slice_candidates:
                print(f"Generated {len(slice_candidates)} fallback candidates")
                final_boxes = deduplicate_by_iou(slice_candidates, iou_threshold=0.4)

        # Format results with the output label
        result_boxes = []
        for box in final_boxes:
            # Skip the original reference box user drew
            if abs(box["x"] - crop_box.x) < 20 and abs(box["y"] - crop_box.y) < 20:
                continue

            result_boxes.append({
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
                "label": output_label,
                "confidence": box["confidence"]
            })

        print(f"\nFound {len(result_boxes)} matching objects")
        print(f"{'='*60}\n")
        return {"boxes": result_boxes}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_annotations")
async def save_annotations(request: SaveAnnotationRequest):
    """Save annotations in YOLO format, merging with existing dataset.yaml"""
    try:
        # Get image dimensions
        image_path = Path(IMAGES_PATH) / request.image_path
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        img = Image.open(image_path)
        img_width, img_height = img.size

        # Load existing classes from dataset.yaml if it exists
        yaml_path = Path("/home/mkultra/Documents/UBL/configs/dataset_generated.yaml")
        existing_classes = []

        if yaml_path.exists():
            import yaml
            with open(yaml_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
                existing_classes = dataset_config.get('names', [])

        # Merge new classes with existing ones
        all_classes = existing_classes.copy()
        for class_name in request.class_names:
            if class_name not in all_classes:
                all_classes.append(class_name)

        # Save merged class names to classes.txt
        classes_file = Path(LABELS_PATH).parent / "classes.txt"
        with open(classes_file, "w") as f:
            for class_name in all_classes:
                f.write(f"{class_name}\n")

        # Convert boxes to YOLO format and save
        label_file = Path(LABELS_PATH) / f"{Path(request.image_path).stem}.txt"

        with open(label_file, "w") as f:
            for box in request.boxes:
                # Find class id from merged class list
                try:
                    class_id = all_classes.index(box.label)
                except ValueError:
                    continue

                # Convert from absolute coordinates to YOLO format (normalized center)
                x_center = (box.x + box.width / 2) / img_width
                y_center = (box.y + box.height / 2) / img_height
                width = box.width / img_width
                height = box.height / img_height

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # Update dataset.yaml with new classes
        if yaml_path.exists():
            import yaml
            with open(yaml_path, 'r') as f:
                dataset_config = yaml.safe_load(f)

            dataset_config['names'] = all_classes
            dataset_config['nc'] = len(all_classes)

            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

        return {"message": "Annotations saved successfully", "total_classes": len(all_classes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/annotations/{image_name}")
async def delete_annotations(image_name: str):
    """Delete annotations for an image"""
    try:
        label_file = Path(LABELS_PATH) / f"{Path(image_name).stem}.txt"
        if label_file.exists():
            label_file.unlink()
            return {"message": "Annotations deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Annotations not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        import yaml
        yaml_path = Path(CONFIG["dataset_yaml"])
        classes = []

        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
                classes = dataset_config.get('names', [])

        return {
            "images_path": CONFIG["images_path"],
            "labels_path": CONFIG["labels_path"],
            "dataset_yaml": CONFIG["dataset_yaml"],
            "classes": classes,
            "num_classes": len(classes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ConfigUpdate(BaseModel):
    images_path: Optional[str] = None
    labels_path: Optional[str] = None
    dataset_yaml: Optional[str] = None

@app.post("/config")
async def update_config(config: ConfigUpdate):
    """Update configuration paths"""
    try:
        global IMAGES_PATH, LABELS_PATH

        if config.images_path:
            CONFIG["images_path"] = config.images_path
            IMAGES_PATH = config.images_path
        if config.labels_path:
            CONFIG["labels_path"] = config.labels_path
            LABELS_PATH = config.labels_path
        if config.dataset_yaml:
            CONFIG["dataset_yaml"] = config.dataset_yaml

        return {"message": "Configuration updated", "config": CONFIG}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ClassManagement(BaseModel):
    action: str  # "add" or "delete"
    class_name: str

@app.post("/classes")
async def manage_classes(request: ClassManagement):
    """Add or delete classes from dataset.yaml"""
    try:
        import yaml
        yaml_path = Path(CONFIG["dataset_yaml"])

        if not yaml_path.exists():
            raise HTTPException(status_code=404, detail="Dataset YAML not found")

        with open(yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        classes = dataset_config.get('names', [])

        if request.action == "add":
            if request.class_name not in classes:
                classes.append(request.class_name)
                dataset_config['names'] = classes
                dataset_config['nc'] = len(classes)

                with open(yaml_path, 'w') as f:
                    yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

                return {"message": f"Added class '{request.class_name}'", "classes": classes}
            else:
                return {"message": f"Class '{request.class_name}' already exists", "classes": classes}

        elif request.action == "delete":
            if request.class_name in classes:
                classes.remove(request.class_name)
                dataset_config['names'] = classes
                dataset_config['nc'] = len(classes)

                with open(yaml_path, 'w') as f:
                    yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

                return {"message": f"Deleted class '{request.class_name}'", "classes": classes}
            else:
                raise HTTPException(status_code=404, detail=f"Class '{request.class_name}' not found")

        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'add' or 'delete'")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
