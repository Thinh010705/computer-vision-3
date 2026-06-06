import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DetectionDataset(Dataset):
    """
    Custom Dataset for Object Detection.
    Fully integrated with Albumentations for advanced augmentation (Cutout, ShiftScaleRotate, crops)
    and Multi-Scale Stride 16 grid target generation.
    """
    def __init__(self, json_path, image_dir, resolution=448, is_train=True):
        self.image_dir = image_dir
        self.is_train = is_train
        self.resolution = resolution
        self.grid_size = resolution // 16  # Stable Stride 16 Resolution Grid
        
        # Load classes
        self.classes = ["person", "car", "dog", "cat", "chair"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Load annotations JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create mapping from image ID to image info and annotations
        images_info = {img['id']: img for img in data['images']}
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
            
        # Parse into examples list
        self.examples = []
        for img_id, img_info in images_info.items():
            anns = annotations_by_image.get(img_id, [])
            
            bboxes = []
            labels = []
            for ann in anns:
                bbox = ann['bbox']
                class_name = ann['class']
                if class_name in self.class_to_idx:
                    bboxes.append(bbox)
                    labels.append(self.class_to_idx[class_name])
                    
            self.examples.append({
                'id': img_id,
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'bboxes': bboxes,
                'labels': labels
            })
            
        # Define Albumentations pipelines
        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(
                    size=(self.resolution, self.resolution),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.11),
                    p=0.5
                ),
                A.Resize(self.resolution, self.resolution),  # ALWAYS ensure final resolution matches self.resolution
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), shear=(-10, 10), border_mode=0, p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.GaussNoise(p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.3),  # Cutout
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3))
        else:
            self.transform = A.Compose([
                A.Resize(self.resolution, self.resolution),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.0))

    def set_resolution(self, resolution):
        """
        Dynamically update the target resolution and grid size for Multi-Scale training.
        """
        self.resolution = resolution
        self.grid_size = resolution // 16  # Stable Stride 16 Resolution Grid
        
        # Re-initialize transform pipeline with new resolution
        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(
                    size=(self.resolution, self.resolution),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.11),
                    p=0.5
                ),
                A.Resize(self.resolution, self.resolution),  # ALWAYS ensure final resolution matches self.resolution
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), shear=(-10, 10), border_mode=0, p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.GaussNoise(p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.3),  # Cutout
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3))
        else:
            self.transform = A.Compose([
                A.Resize(self.resolution, self.resolution),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.0))

    def __len__(self):
        return len(self.examples)

    def _load_raw_example(self, idx):
        example = self.examples[idx]
        img_path = os.path.join(self.image_dir, os.path.basename(example['file_name']))
        img = Image.open(img_path).convert("RGB")
        return np.array(img), list(example['bboxes']), list(example['labels'])

    def _load_mosaic(self, idx):
        indices = [idx] + random.sample(range(len(self.examples)), 3)
        out_size = self.resolution * 2
        mosaic = np.full((out_size, out_size, 3), 114, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []
        placements = [
            (0, 0, self.resolution, self.resolution),
            (self.resolution, 0, out_size, self.resolution),
            (0, self.resolution, self.resolution, out_size),
            (self.resolution, self.resolution, out_size, out_size),
        ]

        for sample_idx, (x1_dst, y1_dst, x2_dst, y2_dst) in zip(indices, placements):
            img, boxes, labels = self._load_raw_example(sample_idx)
            h, w = img.shape[:2]
            resized = np.array(Image.fromarray(img).resize((self.resolution, self.resolution), Image.BILINEAR))
            mosaic[y1_dst:y2_dst, x1_dst:x2_dst] = resized

            scale_x = self.resolution / max(w, 1)
            scale_y = self.resolution / max(h, 1)
            for box, label in zip(boxes, labels):
                bx1, by1, bx2, by2 = box
                nx1 = bx1 * scale_x + x1_dst
                ny1 = by1 * scale_y + y1_dst
                nx2 = bx2 * scale_x + x1_dst
                ny2 = by2 * scale_y + y1_dst
                if nx2 > nx1 and ny2 > ny1:
                    mosaic_boxes.append([nx1, ny1, nx2, ny2])
                    mosaic_labels.append(label)

        crop_x = random.randint(0, self.resolution)
        crop_y = random.randint(0, self.resolution)
        cropped = mosaic[crop_y:crop_y + self.resolution, crop_x:crop_x + self.resolution]
        cropped_boxes = []
        cropped_labels = []
        for box, label in zip(mosaic_boxes, mosaic_labels):
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(float(self.resolution), x1 - crop_x))
            y1 = max(0.0, min(float(self.resolution), y1 - crop_y))
            x2 = max(0.0, min(float(self.resolution), x2 - crop_x))
            y2 = max(0.0, min(float(self.resolution), y2 - crop_y))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                cropped_boxes.append([x1, y1, x2, y2])
                cropped_labels.append(label)
        return cropped, cropped_boxes, cropped_labels

    def __getitem__(self, idx):
        example = self.examples[idx]
        if self.is_train and random.random() < 0.35:
            img_np, bboxes, labels = self._load_mosaic(idx)
        else:
            img_np, bboxes, labels = self._load_raw_example(idx)
        
        # Apply Albumentations Compose Pipeline
        try:
            transformed = self.transform(image=img_np, bboxes=bboxes, category_ids=labels)
            img_tensor = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['category_ids']
        except Exception as e:
            # Fallback transform if crop/visibility filters out all bboxes
            fallback_transform = A.Compose([
                A.Resize(self.resolution, self.resolution),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            img_tensor = fallback_transform(image=img_np)['image']
            bboxes = []
            labels = []
            
        # Grid target generation (Stride 16)
        S = self.grid_size
        target = torch.zeros((10, S, S), dtype=torch.float32)
        
        # To handle overlapping centers in a cell, keep the one with smaller area
        cell_areas = torch.full((S, S), float('inf'))
        
        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            
            # Normalize bounding box coordinates to [0, 1] relative to the post-transformation resolution
            x1 = xmin / self.resolution
            y1 = ymin / self.resolution
            x2 = xmax / self.resolution
            y2 = ymax / self.resolution
            
            # Bounding box center, width, and height in [0, 1]
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            
            # Safe boundary check
            xc = min(0.9999, max(0.0, xc))
            yc = min(0.9999, max(0.0, yc))
            w = min(1.0, max(0.0, w))
            h = min(1.0, max(0.0, h))
            
            # Determine grid cell row and column
            col = int(xc * S)
            row = int(yc * S)
            
            area = w * h
            # If this cell is empty or has a larger object, assign this smaller object to it
            if area < cell_areas[row, col]:
                cell_areas[row, col] = area
                
                # Clear former classes if any
                target[0:6, row, col] = 0.0
                
                # 0. Objectness = 1.0
                target[0, row, col] = 1.0
                # 1-5. One-hot class label
                target[1 + int(label), row, col] = 1.0
                # 6-7. Bounding box center relative to cell top-left (range [0, 1])
                target[6, row, col] = xc * S - col
                target[7, row, col] = yc * S - row
                # 8-9. Bounding box width and height relative to image size (range [0, 1])
                target[8, row, col] = w
                target[9, row, col] = h
                
        # Return image, target grid, and metadata for inference/eval
        meta = {
            'image_id': example['id'],
            'width_orig': example['width'],
            'height_orig': example['height'],
            'bboxes_orig': torch.tensor(example['bboxes'], dtype=torch.float32),
            'labels_orig': torch.tensor(example['labels'], dtype=torch.long)
        }
        
        return img_tensor, target, meta
