import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.dataset import DetectionDataset
from models.detector import ConvNeXtFPNDetector
from utils.loss import DetectionLoss
from utils.nms import decode_predictions, non_maximum_suppression, bbox_iou

def parse_args():
    parser = argparse.ArgumentParser(description="Train custom ConvNeXt FPN anchor-free detector.")
    parser.add_argument("--train_data", required=True, type=str, help="Path to train.json")
    parser.add_argument("--val_data", required=True, type=str, help="Path to val.json")
    parser.add_argument("--image_dir", required=True, type=str, help="Path to train images")
    parser.add_argument("--val_image_dir", required=True, type=str, help="Path to val images")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="Directory to save checkpoints")
    
    # Tham sô huấn luyện và đánh giá
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--multi_scale", action="store_true", default=True, help="Enable multi-scale training")
    parser.add_argument("--no_multi_scale", action="store_true", help="Disable multi-scale training")
    parser.add_argument("--conf_threshold", type=float, default=0.05, help="Confidence threshold used during validation mAP")
    parser.add_argument("--iou_threshold", type=float, default=0.50, help="NMS IoU threshold used during validation mAP")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint for fine-tuning/resume")
    parser.add_argument("--no_pretrained", action="store_true", help="Initialize ConvNeXt backbone without ImageNet weights")
    parser.add_argument("--backbone", choices=["tiny", "small"], default="tiny", help="ConvNeXt backbone size")
    parser.add_argument("--mosaic_prob", type=float, default=0.15, help="Mosaic probability during the strong augmentation phase")
    parser.add_argument("--close_mosaic_epochs", type=int, default=15, help="Disable mosaic and strong augmentation for final epochs")
    parser.add_argument("--fine_tune_lr_scale", type=float, default=0.25, help="Multiply LR when entering final fine-tune phase")
    parser.add_argument("--save_top_k", type=int, default=5, help="Keep top-k checkpoints by validation mAP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_ap(recalls, precisions):
    # Tính Average Precision (AP) bằng cách tính diện tích dưới đường cong Precision-Recall.
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Prevents PyTorch default_collate from trying to stack variable-size tensors (bboxes/labels) in metadata.
    """
    images = [item[0] for item in batch]
    target_lists = [item[1] for item in batch]
    metas = [item[2] for item in batch]
    targets = [
        torch.stack([sample_targets[scale_idx] for sample_targets in target_lists], 0)
        for scale_idx in range(len(target_lists[0]))
    ]
    return torch.stack(images, 0), targets, metas


def evaluate_map(model, val_loader, device, conf_threshold=0.05, iou_threshold=0.5):
    """
        Tính mAP@0.5 trên tập validation bằng cách so sánh dự đoán của mô hình với Ground Truth.
    """
    model.eval()
    
    # Lưu Ground Truth và dự đoán theo lớp để tính toán mAP.
    classes = ["person", "car", "dog", "cat", "chair"]
    
    gt_boxes_by_class = {cls: {} for cls in classes}
    pred_boxes_by_class = {cls: [] for cls in classes}
    
    # Đếm số lượng Ground Truth cho mỗi lớp để tính recall.
    gt_counts = {cls: 0 for cls in classes}
    
    with torch.no_grad():
        for images, _, metas in val_loader:
            images = images.to(device)
            outputs = model(images)
            
            for b in range(images.shape[0]):
                img_id = metas[b]['image_id']
                w_orig = metas[b]['width_orig']
                h_orig = metas[b]['height_orig']
                
                # Lấy Ground Truth gốc (trước khi resize) để tính mAP.
                bboxes_gt = metas[b]['bboxes_orig']  # tensor
                labels_gt = metas[b]['labels_orig']
                
                # Lưu Ground Truth theo lớp.
                for bbox, label in zip(bboxes_gt, labels_gt):
                    # Bỏ qua các bounding box không hợp lệ
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    cls_name = classes[label.item()]
                    if img_id not in gt_boxes_by_class[cls_name]:
                        gt_boxes_by_class[cls_name][img_id] = []
                    gt_boxes_by_class[cls_name][img_id].append({
                        "bbox": bbox.tolist(),
                        "matched": False
                    })
                    gt_counts[cls_name] += 1
                
                # Decode all P3/P4/P5 predictions for this image.
                image_outputs = [scale_output[b] for scale_output in outputs]
                raw_predictions = decode_predictions(image_outputs, w_orig, h_orig, conf_threshold=conf_threshold)
                # Apply class-wise NMS
                final_predictions = non_maximum_suppression(raw_predictions, iou_threshold=iou_threshold)
                
                # Group predictions by class
                for pred in final_predictions:
                    cls_name = pred["class"]
                    pred_boxes_by_class[cls_name].append({
                        "image_id": img_id,
                        "confidence": pred["confidence"],
                        "bbox": pred["bbox"]
                    })
                    
    # Calculate AP for each class
    aps = []
    for cls_name in classes:
        num_gt = gt_counts[cls_name]
        class_preds = sorted(pred_boxes_by_class[cls_name], key=lambda x: x["confidence"], reverse=True)
        class_gts = gt_boxes_by_class[cls_name]
        
        if num_gt == 0:
            continue
            
        tp_flags = []
        fp_flags = []
        
        for pred in class_preds:
            img_id = pred["image_id"]
            candidates = class_gts.get(img_id, [])
            
            best_iou = 0.0
            best_idx = -1
            
            for idx, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    
            if best_idx >= 0 and best_iou >= 0.5:
                candidates[best_idx]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
                
        # Calculate precision-recall curves
        cumulative_tp = []
        cumulative_fp = []
        tp_sum = 0
        fp_sum = 0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            cumulative_tp.append(tp_sum)
            cumulative_fp.append(fp_sum)
            
        recalls = [tp / num_gt if num_gt else 0.0 for tp in cumulative_tp]
        precisions = [tp / max(tp + fp, 1) for tp, fp in zip(cumulative_tp, cumulative_fp)]
        
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        
    mAP = np.mean(aps) if aps else 0.0
    return mAP

def train(args):
    set_seed(args.seed)
    args.multi_scale = args.multi_scale and not args.no_multi_scale
    
    # 1. Setup Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Create Datasets
    # Default resolution is 448, yielding P3/P4/P5 grids 56x56, 28x28, 14x14.
    train_dataset = DetectionDataset(args.train_data, args.image_dir, resolution=448, is_train=True)
    val_dataset = DetectionDataset(args.val_data, args.val_image_dir, resolution=448, is_train=False)
    train_dataset.base_mosaic_prob = args.mosaic_prob
    train_dataset.set_training_stage("strong")
    
    print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    
    # Validation uses num_workers=2
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    
    resume_checkpoint = None
    if args.resume:
        print(f"Resuming/fine-tuning from: {args.resume}")
        try:
            resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            resume_checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(resume_checkpoint, dict):
            resume_config = resume_checkpoint.get("model_config", {})
            args.backbone = resume_config.get("backbone", args.backbone)

    # 4. Instantiate Model, Loss, Optimizer, and Cosine Scheduler
    model = ConvNeXtFPNDetector(pretrained=not args.no_pretrained, backbone_name=args.backbone).to(device)
    
    # Inverse-frequency class weights used by the proven baseline.
    # Frequency counts: person: 5829, car: 1339, dog: 1028, cat: 833, chair: 1613
    # Absolute counts sum to 10642 annotations. Inverse frequency weights are:
    class_weights = torch.tensor([1.83, 7.95, 10.35, 12.78, 6.60], dtype=torch.float32).to(device)
    # Normalize weights so that their mean is 1.0 (sums to num_classes = 5)
    class_weights = class_weights / class_weights.sum() * 5.0
    
    criterion = DetectionLoss(class_weights=class_weights).to(device)
    
    # Differential Learning Rates: fine-tune backbone 10x slower than the head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone_features" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1}, # 10x smaller learning rate for backbone parameters
        {"params": head_params, "lr": args.lr}            # normal learning rate for head parameters
    ], weight_decay=args.weight_decay)

    start_epoch = 0
    resume_best_map = 0.0
    if resume_checkpoint is not None:
        if isinstance(resume_checkpoint, dict) and "model_state_dict" in resume_checkpoint:
            model.load_state_dict(resume_checkpoint["model_state_dict"])
            if "optimizer_state_dict" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
            resume_best_map = float(resume_checkpoint.get("mAP", 0.0))
        else:
            model.load_state_dict(resume_checkpoint)

    # Cosine learning rate decay for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Version-safe modern GradScaler for Mixed Precision (AMP)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    except (TypeError, ValueError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
    best_map = resume_best_map
    scales = [416, 448, 480]
    fine_start_epoch = max(0, args.epochs - args.close_mosaic_epochs)
    fine_lr_applied = False
    top_checkpoints = []
    model_config = {
        "detector": "ConvNeXtFPNDetector",
        "backbone": args.backbone,
        "model_version": getattr(model, "model_version", "unknown"),
        "strides": list(getattr(model, "strides", (8, 16, 32))),
    }
    
    for epoch in range(start_epoch, args.epochs):
        model.train()

        fine_phase = epoch >= fine_start_epoch
        if fine_phase:
            train_dataset.set_training_stage("fine")
            train_dataset.set_resolution(448)
            if not fine_lr_applied:
                for group in optimizer.param_groups:
                    group["lr"] *= args.fine_tune_lr_scale
                fine_lr_applied = True
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | Fine-tune phase: mosaic OFF, light augmentation, 448x448 ---")
        elif args.multi_scale and torch.cuda.is_available():
            train_dataset.set_training_stage("strong")
            new_res = random.choice(scales)
            train_dataset.set_resolution(new_res)
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | Multi-scale target resolution set to: {new_res}x{new_res} ---")
        else:
            train_dataset.set_training_stage("strong")
            train_dataset.set_resolution(448)
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | Target resolution: 448x448 ---")
            
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (images, targets, _) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            targets = [target.to(device, non_blocking=True) for target in targets]
            
            # Linear Learning Rate Warm-up during first 3 epochs to protect pre-trained weights
            warmup_epochs = 3
            total_warmup_steps = warmup_epochs * len(train_loader)
            global_step = epoch * len(train_loader) + batch_idx
            if global_step < total_warmup_steps:
                factor = (global_step + 1) / total_warmup_steps
                for g_idx, g in enumerate(optimizer.param_groups):
                    # g_idx 0 is backbone, g_idx 1 is head
                    base_lr = args.lr * 0.1 if g_idx == 0 else args.lr
                    g['lr'] = base_lr * factor
            
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast context helper for version safety in PyTorch 2.6+
            try:
                autocast_context = torch.amp.autocast('cuda', enabled=torch.cuda.is_available())
            except (TypeError, ValueError, AttributeError):
                autocast_context = torch.cuda.amp.autocast(enabled=torch.cuda.is_available())
                
            # Forward pass under Mixed Precision autocast
            with autocast_context:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            # Backward and Optimizer step using GradScaler with Gradient Norm Clipping
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        
        # 6. Evaluation Phase
        print(f"Calculating Validation mAP@0.5...")
        val_map = evaluate_map(model, val_loader, device, conf_threshold=args.conf_threshold, iou_threshold=args.iou_threshold)
        
        print(f"Epoch {epoch+1} Summary: Avg Train Loss = {avg_train_loss:.4f} | Val mAP@0.5 = {val_map:.4f}")
        
        # 7. Checkpoint Saving (Save best checkpoint)
        if val_map > best_map:
            best_map = val_map
            best_path = os.path.join(args.checkpoint_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': val_map,
                'conf_threshold': args.conf_threshold,
                'iou_threshold': args.iou_threshold,
                'model_config': model_config,
            }, best_path)
            print(f"⭐ New Best Model saved with mAP@0.5 = {val_map:.4f} at {best_path}")

        top_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1:03d}_map_{val_map:.4f}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mAP': val_map,
            'model_config': model_config,
        }, top_path)
        top_checkpoints.append((val_map, top_path))
        top_checkpoints.sort(key=lambda item: item[0], reverse=True)
        while len(top_checkpoints) > args.save_top_k:
            _, remove_path = top_checkpoints.pop()
            if os.path.exists(remove_path):
                os.remove(remove_path)
            
        # Also save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mAP': val_map,
            'conf_threshold': args.conf_threshold,
            'iou_threshold': args.iou_threshold,
            'model_config': model_config,
        }, latest_path)

    print(f"\nTraining completed! Best Validation mAP@0.5 = {best_map:.4f}")

if __name__ == "__main__":
    args = parse_args()
    train(args)
