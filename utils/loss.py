import torch
import torch.nn as nn
import numpy as np

def focal_loss_with_logits(inputs, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss cho bài toán nhị phân.

    alpha cân bằng positive/negative; gamma giảm trọng số của các mẫu nền dễ
    để mô hình tập trung nhiều hơn vào các mẫu khó.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    inputs_sig = torch.sigmoid(inputs)
    p_t = inputs_sig * targets + (1 - inputs_sig) * (1 - targets)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss

class DetectionLoss(nn.Module):
    """
    Hàm mất mát cho bộ phát hiện anchor-free.

    Gồm IoU-aware objectness cho ô positive, Focal Loss cho nền, Weighted
    Cross Entropy cho phân lớp và CIoU kết hợp Smooth L1 cho hộp bao.
    """
    def __init__(
        self,
        lambda_obj=5.0,
        lambda_noobj=0.5,
        lambda_class=1.0,
        lambda_box=3.0,
        class_weights=None,
        label_smoothing=0.05,
        iou_obj_ratio=0.75,
    ):
        """Khởi tạo trọng số các thành phần loss và hàm loss cơ sở."""
        super(DetectionLoss, self).__init__()
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.lambda_box = lambda_box
        self.iou_obj_ratio = iou_obj_ratio
        
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction='sum',
            label_smoothing=label_smoothing,
        )
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

    def forward(self, predictions, targets):
        """Tính loss cho một tầng hoặc trung bình loss của P3/P4/P5.

        predictions và targets có dạng ``(batch_size, 10, S, S)``, trong đó
        S là kích thước lưới của tầng FPN tương ứng.
        """
        if isinstance(predictions, (list, tuple)):
            total = predictions[0].new_tensor(0.0)
            for pred, target in zip(predictions, targets):
                total = total + self.forward(pred, target)
            return total / max(len(predictions), 1)

        batch_size, _, S, _ = predictions.shape
        
        # Tách objectness, logits lớp và tọa độ hộp từ tensor 10 kênh.
        pred_obj = predictions[:, 0, :, :]           # (batch, S, S)
        pred_class = predictions[:, 1:6, :, :]        # (batch, 5, S, S)
        pred_coords = predictions[:, 6:10, :, :]      # (batch, 4, S, S)
        
        target_obj = targets[:, 0, :, :]             # (batch, S, S)
        target_class = targets[:, 1:6, :, :]          # (batch, 5, S, S)
        target_coords = targets[:, 6:10, :, :]        # (batch, 4, S, S)
        
        # Tạo mask cho ô có vật thể và ô nền.
        obj_mask = (target_obj == 1.0)
        noobj_mask = (target_obj == 0.0)
        
        # Batch không có positive chỉ cần tối ưu khả năng nhận biết nền.
        num_pos = obj_mask.sum().item()
        if num_pos == 0:
            loss_noobj = focal_loss_with_logits(
                pred_obj[noobj_mask],
                target_obj[noobj_mask],
                alpha=0.25,
                gamma=2.0,
            ).sum()
            grid_normalization = (S * S) / 784.0
            return (self.lambda_noobj * loss_noobj / grid_normalization) / batch_size
            
        # Cross Entropy chỉ được tính tại các ô chứa vật thể.
        pred_class_flat = pred_class.permute(0, 2, 3, 1)[obj_mask] # (num_pos, 5)
        target_class_flat = target_class.permute(0, 2, 3, 1)[obj_mask].argmax(dim=-1) # (num_pos)
        
        total_class_loss = self.ce_loss(pred_class_flat, target_class_flat)
        
        # Hồi quy hộp chỉ được tính tại các ô positive.
        p_coords = pred_coords.permute(0, 2, 3, 1)[obj_mask] # (num_pos, 4)
        t_coords = target_coords.permute(0, 2, 3, 1)[obj_mask] # (num_pos, 4)
        
        # Lấy chỉ số batch, hàng và cột để giải mã offset tâm theo ô lưới.
        _, rows, cols = torch.nonzero(obj_mask, as_tuple=True)
        
        # Giải mã hộp dự đoán sang tọa độ chuẩn hóa [0, 1].
        px_c = (cols.float() + torch.sigmoid(p_coords[:, 0])) / S
        py_c = (rows.float() + torch.sigmoid(p_coords[:, 1])) / S
        pw = torch.sigmoid(p_coords[:, 2])
        ph = torch.sigmoid(p_coords[:, 3])
        
        # Giải mã hộp ground truth sang cùng hệ tọa độ chuẩn hóa.
        tx_c = (cols.float() + t_coords[:, 0]) / S
        ty_c = (rows.float() + t_coords[:, 1]) / S
        tw = t_coords[:, 2]
        th = t_coords[:, 3]
        
        # Đổi từ tâm/kích thước sang [x1, y1, x2, y2].
        pred_x1 = px_c - pw / 2.0
        pred_y1 = py_c - ph / 2.0
        pred_x2 = px_c + pw / 2.0
        pred_y2 = py_c + ph / 2.0
        
        target_x1 = tx_c - tw / 2.0
        target_y1 = ty_c - th / 2.0
        target_x2 = tx_c + tw / 2.0
        target_y2 = ty_c + th / 2.0
        
        # Tính diện tích giao nhau giữa hộp dự đoán và ground truth.
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Tính diện tích hợp và IoU.
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + 1e-7
        
        iou = inter_area / union_area

        # IoU-aware objectness giúp hộp định vị tốt có confidence cao hơn.
        # Pha IoU với mục tiêu 1.0 để positive vẫn nhận gradient đủ mạnh ở đầu train.
        quality_target = (1.0 - self.iou_obj_ratio) + self.iou_obj_ratio * iou.detach()
        # Positive chất lượng cao được ưu tiên để confidence phù hợp với thứ tự AP.
        loss_obj = (self.bce_logits(pred_obj[obj_mask], quality_target) * quality_target).sum()
        loss_noobj = focal_loss_with_logits(
            pred_obj[noobj_mask],
            target_obj[noobj_mask],
            alpha=0.25,
            gamma=2.0,
        ).sum()

        # Chuẩn hóa objectness theo lưới stride-16 của ảnh cơ sở 448.
        grid_normalization = (S * S) / 784.0
        total_obj_loss = (self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj) / grid_normalization
        
        # CIoU bổ sung khoảng cách tâm và độ tương đồng tỉ lệ vào IoU.
        # Khoảng cách bình phương giữa hai tâm hộp.
        center_dist_sq = (px_c - tx_c)**2 + (py_c - ty_c)**2
        
        # Bình phương đường chéo của hộp nhỏ nhất bao quanh cả hai hộp.
        convex_x1 = torch.min(pred_x1, target_x1)
        convex_y1 = torch.min(pred_y1, target_y1)
        convex_x2 = torch.max(pred_x2, target_x2)
        convex_y2 = torch.max(pred_y2, target_y2)
        convex_diag_sq = (convex_x2 - convex_x1)**2 + (convex_y2 - convex_y1)**2 + 1e-7
        
        # Thành phần phạt khoảng cách tâm.
        d_term = center_dist_sq / convex_diag_sq
        
        # Thành phần v đo sai khác tỉ lệ khung hình.
        v = (4.0 / (np.pi**2)) * (torch.atan(tw / (th + 1e-7)) - torch.atan(pw / (ph + 1e-7)))**2
        
        # alpha cân bằng mức đóng góp của thành phần tỉ lệ khung hình.
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + 1e-7)
            
        ciou = iou - d_term - alpha * v
        loss_ciou = (1.0 - ciou).sum()
        
        # Smooth L1 trên tọa độ sau sigmoid giúp quá trình hồi quy ổn định hơn.
        p_coords_activated = torch.sigmoid(p_coords)
        loss_l1 = self.smooth_l1(p_coords_activated, t_coords)
        
        total_box_loss = loss_ciou + 0.3 * loss_l1
        
        # Chuẩn hóa class/box loss theo số positive, với mốc 5 vật thể mỗi ảnh.
        avg_objs_per_image = 5.0
        total_class_loss = (total_class_loss / num_pos) * (batch_size * avg_objs_per_image)
        total_box_loss = (total_box_loss / num_pos) * (batch_size * avg_objs_per_image)
        
        # Kết hợp các thành phần loss theo trọng số đã cấu hình.
        loss = (total_obj_loss + self.lambda_class * total_class_loss + self.lambda_box * total_box_loss) / batch_size
        return loss
