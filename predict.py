import os
import argparse
import json
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models.detector import ConvNeXtFPNDetector
from utils.nms import decode_predictions, non_maximum_suppression

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and generate object detection predictions.")
    parser.add_argument("--image_dir", required=True, type=str, help="Directory containing images to predict")
    parser.add_argument("--output", required=True, type=str, help="Path to save predictions predictions.json")
    parser.add_argument("--checkpoint", nargs="+", default=["./models/best.pth"], help="One or more checkpoint paths")
    parser.add_argument("--conf_threshold", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.50, help="IoU threshold for NMS")
    parser.add_argument("--tta_flip", action="store_true", help="Run horizontal flip test-time augmentation")
    parser.add_argument("--tta_sizes", default="448", help="Comma-separated inference sizes, e.g. 416,448,480")
    parser.add_argument("--max_detections", type=int, default=100, help="Maximum detections per image")
    return parser.parse_args()


def load_checkpoint_state(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at '{path}'. Make sure you train the model first.")

    print(f"Loading checkpoint from: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("model_config", {})
    return checkpoint, {}


def load_models(checkpoint_paths, device):
    models = []
    for checkpoint_path in checkpoint_paths:
        state_dict, model_config = load_checkpoint_state(checkpoint_path, device)
        model = ConvNeXtFPNDetector(pretrained=False, backbone_name=model_config.get("backbone", "tiny"))
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    return models


def flip_boxes_back(boxes, img_width):
    flipped = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box["bbox"]
        new_box = dict(box)
        new_box["bbox"] = [
            round(max(0.0, img_width - xmax), 1),
            ymin,
            round(min(float(img_width), img_width - xmin), 1),
            ymax,
        ]
        flipped.append(new_box)
    return flipped


def limit_detections(boxes, max_detections):
    return sorted(boxes, key=lambda b: b["confidence"], reverse=True)[:max_detections]


def parse_sizes(value):
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 or size % 32 != 0 for size in sizes):
        raise ValueError("TTA sizes must be positive multiples of 32.")
    return sizes


def prepare_image(img, size, normalize, device):
    resized = img.resize((size, size), Image.BILINEAR)
    return normalize(TF.to_tensor(resized)).unsqueeze(0).to(device)


@torch.no_grad()
def predict_image_boxes(models, img, conf_threshold, tta_flip, tta_sizes, normalize, device):
    w_orig, h_orig = img.size
    raw_boxes = []
    for size in tta_sizes:
        img_tensor = prepare_image(img, size, normalize, device)
        for model in models:
            output = model(img_tensor)
            flipped_output = model(torch.flip(img_tensor, dims=[3])) if tta_flip else None

            image_outputs = [scale_output[0] for scale_output in output]
            raw_boxes.extend(decode_predictions(image_outputs, w_orig, h_orig, conf_threshold=conf_threshold))
            if flipped_output is not None:
                flipped_image_outputs = [scale_output[0] for scale_output in flipped_output]
                flipped_boxes = decode_predictions(flipped_image_outputs, w_orig, h_orig, conf_threshold=conf_threshold)
                raw_boxes.extend(flip_boxes_back(flipped_boxes, w_orig))
    return raw_boxes


def main():
    args = parse_args()
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Instantiate and Load Model(s). Multiple checkpoints are ensembled by box merging + NMS.
    models = load_models(args.checkpoint, device)
    tta_sizes = parse_sizes(args.tta_sizes)
    
    # Image Net normalization transforms
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 3. Locate Images
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(valid_exts)]
    print(f"Found {len(img_files)} images in '{args.image_dir}' for prediction.")
    
    predictions_json = []
    
    # 4. Inference loop
    with torch.no_grad():
        for filename in tqdm(img_files, desc="Inferring"):
            img_path = os.path.join(args.image_dir, filename)
            
            try:
                img = Image.open(img_path).convert("RGB")
                w_orig, h_orig = img.size
                
                raw_boxes = predict_image_boxes(
                    models,
                    img,
                    args.conf_threshold,
                    args.tta_flip,
                    tta_sizes,
                    normalize,
                    device,
                )
                
                # Apply class-wise NMS
                final_boxes = limit_detections(non_maximum_suppression(
                    raw_boxes, 
                    iou_threshold=args.iou_threshold
                ), args.max_detections)
                
                # Append result
                predictions_json.append({
                    "image_id": filename,
                    "boxes": final_boxes
                })
                
            except Exception as e:
                print(f"Error predicting image {filename}: {e}")
                # Ensure the entry is still generated even if failed (empty list)
                predictions_json.append({
                    "image_id": filename,
                    "boxes": []
                })
                
    # 5. Save output predictions JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(predictions_json, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated predictions file: '{args.output}'")

if __name__ == "__main__":
    main()
