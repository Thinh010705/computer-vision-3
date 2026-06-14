import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.dataset import CLASSES, DetectionDataset
from models.detector import ConvNeXtFPNDetector
from utils.loss import DetectionLoss
from utils.nms import decode_predictions, non_maximum_suppression, bbox_iou

def parse_args():
    """Đọc đường dẫn dữ liệu và các siêu tham số huấn luyện từ dòng lệnh."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--val_data", required=True, type=str)
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--val_image_dir", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    
    # Các siêu tham số huấn luyện và đánh giá.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--multi_scale", default="384,416,448,480")
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.05)
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--conf_threshold", type=float, default=0.02)
    parser.add_argument("--iou_threshold", type=float, default=0.55)
    parser.add_argument("--mosaic_prob", type=float, default=0.15)
    parser.add_argument("--close_mosaic_epochs", type=int, default=5)
    parser.add_argument("--fine_tune_lr_scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou_obj_ratio", type=float, default=0.75)
    return parser.parse_args()

def set_seed(seed):
    """Cố định seed cho Python, NumPy và PyTorch để kết quả dễ tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_image_sizes(value):
    """Chuyển chuỗi multi-scale thành danh sách kích thước ảnh."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def compute_ap(recalls, precisions):
    """Tính AP bằng diện tích dưới đường cong Precision-Recall đã nội suy."""
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
    Gom các mẫu thành batch mà không stack metadata có số hộp thay đổi.

    Ảnh và target lưới có kích thước cố định nên được stack; metadata chứa
    hộp/nhãn gốc được giữ dưới dạng danh sách để phục vụ validation.
    """
    images = [item[0] for item in batch]
    target_lists = [item[1] for item in batch]
    metas = [item[2] for item in batch]
    targets = [
        torch.stack([sample_targets[scale_idx] for sample_targets in target_lists], 0)
        for scale_idx in range(len(target_lists[0]))
    ]
    return torch.stack(images, 0), targets, metas


def evaluate_map(model, val_loader, device, conf_threshold=0.02, iou_threshold=0.55, max_detections=150):
    """Tính mAP@0.5 bằng cách ghép dự đoán với ground truth theo từng lớp."""
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
                
                # Giải mã và gộp toàn bộ dự đoán P3/P4/P5 của ảnh hiện tại.
                image_outputs = [scale_output[b] for scale_output in outputs]
                raw_predictions = decode_predictions(image_outputs, w_orig, h_orig, conf_threshold=conf_threshold)
                # NMS theo lớp loại các hộp trùng trước khi tính AP.
                final_predictions = sorted(
                    non_maximum_suppression(raw_predictions, iou_threshold=iou_threshold),
                    key=lambda prediction: prediction["confidence"],
                    reverse=True,
                )[:max_detections]
                
                # Gom dự đoán theo lớp để xây dựng đường Precision-Recall.
                for pred in final_predictions:
                    cls_name = pred["class"]
                    pred_boxes_by_class[cls_name].append({
                        "image_id": img_id,
                        "confidence": pred["confidence"],
                        "bbox": pred["bbox"]
                    })
                    
    # Tính AP độc lập cho từng lớp rồi lấy trung bình thành mAP.
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
                
        # Tích lũy TP/FP theo thứ tự confidence để tạo đường Precision-Recall.
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
    """Thực hiện toàn bộ quy trình train, validation và lưu best.pth."""
    set_seed(args.seed)
    scales = parse_image_sizes(args.multi_scale)
    
    # Tạo thư mục lưu checkpoint nếu chưa tồn tại.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Tự động chọn GPU CUDA nếu khả dụng.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Train dùng augmentation; validation chỉ resize và normalize.
    train_dataset = DetectionDataset(args.train_data, args.image_dir, resolution=args.image_size, is_train=True)
    val_dataset = DetectionDataset(args.val_data, args.val_image_dir, resolution=args.image_size, is_train=False)
    train_dataset.base_mosaic_prob = args.mosaic_prob
    train_dataset.set_training_stage("strong")
    
    print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")
    
    # collate_fn riêng xử lý metadata có số lượng hộp thay đổi theo ảnh.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    
    # Khởi tạo ConvNeXt-Small pretrained cùng detector và hàm mất mát.
    model = ConvNeXtFPNDetector(pretrained=True).to(device)
    
    # Trọng số lớp theo inverse-frequency có làm mềm bằng số mũ, giúp cân bằng
    # lớp mà không quá ưu tiên các chi tiết hiếm chỉ xuất hiện ở train/validation.
    class_counts = [0] * len(CLASSES)
    for example in train_dataset.examples:
        for label in example["labels"]:
            class_counts[label] += 1
    class_counts = torch.tensor(class_counts, dtype=torch.float32, device=device).clamp_min_(1.0)
    class_weights = class_counts.pow(-args.class_weight_power)
    class_weights = class_weights / class_weights.mean()
    
    criterion = DetectionLoss(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        iou_obj_ratio=args.iou_obj_ratio,
    ).to(device)
    
    # Backbone dùng learning rate nhỏ hơn để bảo vệ đặc trưng ImageNet; FPN và
    # detection head học nhanh hơn để thích nghi với bài toán mới.
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone_features" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
        {"params": head_params, "lr": args.lr}
    ], weight_decay=args.weight_decay)

    # CosineAnnealingLR giảm learning rate mượt dần trong toàn bộ quá trình.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # GradScaler hỗ trợ AMP và tương thích nhiều phiên bản PyTorch.
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    except (TypeError, ValueError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
    best_map = -1.0
    fine_start_epoch = max(0, args.epochs - args.close_mosaic_epochs)
    fine_lr_applied = False
    model_config = {
        "detector": "ConvNeXtFPNDetector",
        "backbone": "small",
        "model_version": getattr(model, "model_version", "unknown"),
        "strides": list(getattr(model, "strides", (8, 16, 32))),
        "image_size": args.image_size,
        "iou_aware_obj": True,
        "iou_obj_ratio": args.iou_obj_ratio,
    }
    
    for epoch in range(args.epochs):
        model.train()

        fine_phase = epoch >= fine_start_epoch
        if fine_phase:
            train_dataset.set_training_stage("fine")
            train_dataset.set_resolution(args.image_size)
            if not fine_lr_applied:
                for group in optimizer.param_groups:
                    group["lr"] *= args.fine_tune_lr_scale
                fine_lr_applied = True
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | Fine-tune phase: mosaic OFF, light augmentation, {args.image_size}x{args.image_size} ---")
        else:
            train_dataset.set_training_stage("strong")
            new_res = random.choice(scales)
            train_dataset.set_resolution(new_res)
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | Multi-scale target resolution set to: {new_res}x{new_res} ---")
            
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (images, targets, _) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            targets = [target.to(device, non_blocking=True) for target in targets]
            
            # Warm-up tuyến tính trong 3 epoch đầu để tránh cập nhật quá mạnh
            # lên backbone pretrained khi head còn được khởi tạo ngẫu nhiên.
            warmup_epochs = 3
            total_warmup_steps = warmup_epochs * len(train_loader)
            global_step = epoch * len(train_loader) + batch_idx
            if global_step < total_warmup_steps:
                factor = (global_step + 1) / total_warmup_steps
                for g_idx, g in enumerate(optimizer.param_groups):
                    # Nhóm 0 là backbone, nhóm 1 là FPN và detection head.
                    base_lr = args.lr * args.backbone_lr_scale if g_idx == 0 else args.lr
                    g['lr'] = base_lr * factor
            
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast chọn kiểu dữ liệu phù hợp để tăng tốc và giảm VRAM.
            try:
                autocast_context = torch.amp.autocast('cuda', enabled=torch.cuda.is_available())
            except (TypeError, ValueError, AttributeError):
                autocast_context = torch.cuda.amp.autocast(enabled=torch.cuda.is_available())
                
            # Forward và tính loss dưới chế độ mixed precision.
            with autocast_context:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            # GradScaler tránh underflow khi backward với số thực FP16.
            scaler.scale(loss).backward()
            
            # Phải unscale trước khi clip norm để giới hạn gradient chính xác.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        
        epoch_number = epoch + 1
        # Validation sau mọi epoch và chỉ giữ checkpoint có mAP tốt nhất.
        print("Calculating Validation mAP@0.5...")
        val_map = evaluate_map(
            model,
            val_loader,
            device,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
        )
        print(f"Epoch {epoch_number} Summary: Avg Train Loss = {avg_train_loss:.4f} | Val mAP@0.5 = {val_map:.4f}")

        if val_map > best_map:
            best_map = val_map
            best_path = os.path.join(args.checkpoint_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'mAP': val_map,
                'conf_threshold': args.conf_threshold,
                'iou_threshold': args.iou_threshold,
                'model_config': model_config,
            }, best_path)
            print(f"New best Model saved with mAP@0.5 = {val_map:.4f} at {best_path}")

    print(f"\nTraining completed! Best Validation mAP@0.5 = {best_map:.4f}")

if __name__ == "__main__":
    args = parse_args()
    train(args)
