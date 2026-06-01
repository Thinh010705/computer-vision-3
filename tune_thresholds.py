import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from predict import flip_boxes_back, load_models
from train import compute_ap
from utils.nms import bbox_iou, decode_predictions, non_maximum_suppression


CLASSES = ["person", "car", "dog", "cat", "chair"]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Tune confidence and NMS thresholds on validation mAP@0.5.")
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint", nargs="+", default=["./models/best.pth"])
    parser.add_argument("--conf_values", default="0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--iou_values", default="0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--tta_flip", action="store_true")
    return parser.parse_args()


def load_validation(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt_by_image = {img["id"]: [] for img in data["images"]}
    for ann in data.get("annotations", []):
        if ann["class"] in CLASSES:
            gt_by_image.setdefault(ann["image_id"], []).append(ann)
    return data["images"], gt_by_image


@torch.no_grad()
def predict_raw_for_val(images, image_dir, models, min_conf, use_tta, device):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    predictions = {}
    for info in tqdm(images, desc="Collecting predictions"):
        filename = info["id"]
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            path = os.path.join(image_dir, os.path.basename(info["file_name"]))
        img = Image.open(path).convert("RGB")
        w_orig, h_orig = img.size
        img_resized = img.resize((448, 448), Image.BILINEAR)
        img_tensor = normalize(TF.to_tensor(img_resized)).unsqueeze(0).to(device)

        boxes = []
        for model in models:
            output = model(img_tensor)
            boxes.extend(decode_predictions(output[0], w_orig, h_orig, conf_threshold=min_conf))
            if use_tta:
                flipped_output = model(torch.flip(img_tensor, dims=[3]))
                flipped_boxes = decode_predictions(flipped_output[0], w_orig, h_orig, conf_threshold=min_conf)
                boxes.extend(flip_boxes_back(flipped_boxes, w_orig))
        predictions[filename] = boxes
    return predictions


def map50_from_dict(images, gt_by_image, raw_predictions, conf_threshold, iou_threshold):
    gt_boxes_by_class = {cls: {} for cls in CLASSES}
    pred_boxes_by_class = {cls: [] for cls in CLASSES}
    gt_counts = {cls: 0 for cls in CLASSES}

    for info in images:
        image_id = info["id"]
        for ann in gt_by_image.get(image_id, []):
            cls = ann["class"]
            gt_boxes_by_class[cls].setdefault(image_id, []).append({"bbox": ann["bbox"], "matched": False})
            gt_counts[cls] += 1

        filtered = [box for box in raw_predictions.get(image_id, []) if box["confidence"] >= conf_threshold]
        final_boxes = non_maximum_suppression(filtered, iou_threshold=iou_threshold)
        for box in final_boxes:
            pred_boxes_by_class[box["class"]].append(
                {"image_id": image_id, "confidence": box["confidence"], "bbox": box["bbox"]}
            )

    aps = []
    for cls in CLASSES:
        num_gt = gt_counts[cls]
        if num_gt == 0:
            continue
        preds = sorted(pred_boxes_by_class[cls], key=lambda x: x["confidence"], reverse=True)
        tp_flags, fp_flags = [], []
        for pred in preds:
            candidates = gt_boxes_by_class[cls].get(pred["image_id"], [])
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= 0.5:
                candidates[best_idx]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        tp_sum, fp_sum, recalls, precisions = 0, 0, [], []
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            recalls.append(tp_sum / num_gt)
            precisions.append(tp_sum / max(tp_sum + fp_sum, 1))
        aps.append(compute_ap(recalls, precisions))
    return float(np.mean(aps)) if aps else 0.0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args.checkpoint, device)
    conf_values = parse_float_list(args.conf_values)
    iou_values = parse_float_list(args.iou_values)
    images, gt_by_image = load_validation(args.val_data)
    raw_predictions = predict_raw_for_val(images, args.val_image_dir, models, min(conf_values), args.tta_flip, device)

    best = (-1.0, None, None)
    for conf in conf_values:
        for iou in iou_values:
            score = map50_from_dict(images, gt_by_image, raw_predictions, conf, iou)
            print(f"conf={conf:.2f} iou={iou:.2f} mAP@0.5={score:.4f}")
            if score > best[0]:
                best = (score, conf, iou)
    print(f"BEST conf={best[1]:.2f} iou={best[2]:.2f} mAP@0.5={best[0]:.4f}")


if __name__ == "__main__":
    main()
