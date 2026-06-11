import json
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASSES = ["person", "car", "dog", "cat", "chair"]
STRIDES = (8, 16, 32)


class DetectionDataset(Dataset):
    def __init__(self, json_path, image_dir, resolution=448, is_train=True):
        self.image_dir = image_dir
        self.is_train = is_train
        self.resolution = resolution
        self.base_mosaic_prob = 0.15
        self.mosaic_prob = self.base_mosaic_prob
        self.training_stage = "strong"
        self.classes = CLASSES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        annotations_by_image = {}
        for ann in data.get("annotations", []):
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        self.examples = []
        for info in data["images"]:
            boxes, labels = [], []
            for ann in annotations_by_image.get(info["id"], []):
                if ann["class"] in self.class_to_idx:
                    boxes.append(ann["bbox"])
                    labels.append(self.class_to_idx[ann["class"]])
            self.examples.append({
                "id": info["id"],
                "file_name": info["file_name"],
                "width": info["width"],
                "height": info["height"],
                "bboxes": boxes,
                "labels": labels,
            })
        self._build_transform()

    def _build_transform(self):
        if not self.is_train:
            transforms = [A.Resize(self.resolution, self.resolution)]
            min_visibility = 0.0
        elif self.training_stage == "fine":
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.97, 1.03), translate_percent=(-0.02, 0.02), rotate=(-3, 3), border_mode=0, p=0.20),
                A.Resize(self.resolution, self.resolution),
                A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.30),
                A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=10, p=0.25),
                A.GaussNoise(p=0.04),
            ]
            min_visibility = 0.3
        else:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(size=(self.resolution, self.resolution), scale=(0.85, 1.0), ratio=(0.90, 1.10), p=0.40),
                A.Resize(self.resolution, self.resolution),
                A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-5, 5), shear=(-3, 3), border_mode=0, p=0.30),
                A.CLAHE(clip_limit=2.0, p=0.15),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.35),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=12, p=0.30),
                A.GaussNoise(p=0.08),
                A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(6, 20), hole_width_range=(6, 20), p=0.12),
            ]
            min_visibility = 0.3

        transforms.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=min_visibility),
        )

    def set_training_stage(self, stage):
        self.training_stage = stage
        self.mosaic_prob = 0.0 if stage == "fine" else self.base_mosaic_prob
        self._build_transform()

    def set_resolution(self, resolution):
        self.resolution = resolution
        self._build_transform()

    def __len__(self):
        return len(self.examples)

    def _load_raw_example(self, idx):
        example = self.examples[idx]
        path = os.path.join(self.image_dir, os.path.basename(example["file_name"]))
        image = np.array(Image.open(path).convert("RGB"))
        return image, list(example["bboxes"]), list(example["labels"])

    def _load_mosaic(self, idx):
        indices = [idx] + random.sample(range(len(self.examples)), 3)
        out_size = self.resolution * 2
        mosaic = np.full((out_size, out_size, 3), 114, dtype=np.uint8)
        boxes_all, labels_all = [], []
        placements = [
            (0, 0), (self.resolution, 0),
            (0, self.resolution), (self.resolution, self.resolution),
        ]
        for sample_idx, (left, top) in zip(indices, placements):
            image, boxes, labels = self._load_raw_example(sample_idx)
            height, width = image.shape[:2]
            resized = np.array(Image.fromarray(image).resize((self.resolution, self.resolution), Image.BILINEAR))
            mosaic[top:top + self.resolution, left:left + self.resolution] = resized
            scale_x, scale_y = self.resolution / max(width, 1), self.resolution / max(height, 1)
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                boxes_all.append([x1 * scale_x + left, y1 * scale_y + top, x2 * scale_x + left, y2 * scale_y + top])
                labels_all.append(label)

        crop_x, crop_y = random.randint(0, self.resolution), random.randint(0, self.resolution)
        cropped = mosaic[crop_y:crop_y + self.resolution, crop_x:crop_x + self.resolution]
        boxes_out, labels_out = [], []
        for box, label in zip(boxes_all, labels_all):
            x1 = max(0.0, min(self.resolution, box[0] - crop_x))
            y1 = max(0.0, min(self.resolution, box[1] - crop_y))
            x2 = max(0.0, min(self.resolution, box[2] - crop_x))
            y2 = max(0.0, min(self.resolution, box[3] - crop_y))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                boxes_out.append([x1, y1, x2, y2])
                labels_out.append(label)
        return cropped, boxes_out, labels_out

    @staticmethod
    def _assigned_scales(width, height):
        # Primary scale from the longest normalized side, plus one adjacent scale.
        longest = max(width, height)
        if longest < 0.18:
            return (0, 1)
        if longest < 0.45:
            return (1, 0) if longest < 0.30 else (1, 2)
        return (2, 1)

    def _build_targets(self, bboxes, labels):
        targets, occupied_areas = [], []
        for stride in STRIDES:
            size = self.resolution // stride
            targets.append(torch.zeros((10, size, size), dtype=torch.float32))
            occupied_areas.append(torch.full((size, size), float("inf")))

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = [float(v) / self.resolution for v in bbox]
            xc, yc = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            xc, yc = min(0.9999, max(0.0, xc)), min(0.9999, max(0.0, yc))
            width, height = min(1.0, width), min(1.0, height)
            area = width * height

            for scale_idx in self._assigned_scales(width, height):
                target = targets[scale_idx]
                size = target.shape[-1]
                col, row = int(xc * size), int(yc * size)
                # Multi-scale assignment greatly reduces collisions. If a collision
                # remains, retain the smaller object because it is harder to recover.
                if area >= occupied_areas[scale_idx][row, col]:
                    continue
                occupied_areas[scale_idx][row, col] = area
                target[:, row, col] = 0.0
                target[0, row, col] = 1.0
                target[1 + int(label), row, col] = 1.0
                target[6, row, col] = xc * size - col
                target[7, row, col] = yc * size - row
                target[8, row, col] = width
                target[9, row, col] = height
        return targets

    def __getitem__(self, idx):
        example = self.examples[idx]
        if self.is_train and random.random() < self.mosaic_prob:
            image, boxes, labels = self._load_mosaic(idx)
        else:
            image, boxes, labels = self._load_raw_example(idx)

        try:
            transformed = self.transform(image=image, bboxes=boxes, category_ids=labels)
            image_tensor = transformed["image"]
            boxes, labels = transformed["bboxes"], transformed["category_ids"]
        except Exception:
            # Preserve annotations in the fallback instead of turning the image into
            # an incorrect background-only sample.
            fallback = A.Compose(
                [A.Resize(self.resolution, self.resolution), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=0.0),
            )
            transformed = fallback(image=image, bboxes=boxes, category_ids=labels)
            image_tensor = transformed["image"]
            boxes, labels = transformed["bboxes"], transformed["category_ids"]

        meta = {
            "image_id": example["id"],
            "width_orig": example["width"],
            "height_orig": example["height"],
            "bboxes_orig": torch.tensor(example["bboxes"], dtype=torch.float32).reshape(-1, 4),
            "labels_orig": torch.tensor(example["labels"], dtype=torch.long),
        }
        return image_tensor, self._build_targets(boxes, labels), meta
