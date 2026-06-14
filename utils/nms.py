import torch

def bbox_iou(box_a, box_b):
    """
    Tính IoU giữa hai hộp ``[xmin, ymin, xmax, ymax]``.

    Hàm được dùng khi NMS và khi đánh giá dự đoán với ground truth.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    
    if union <= 0:
        return 0.0
    return intersection / union

def decode_predictions(prediction, img_width, img_height, conf_threshold=0.15):
    """
    Giải mã tensor lưới thô thành hộp bao theo kích thước ảnh gốc.

    prediction có dạng ``(10, S, S)``. Confidence cuối được tính bằng
    objectness nhân xác suất lớp lớn nhất, sau đó lọc theo conf_threshold.
    """
    if isinstance(prediction, (list, tuple)):
        decoded = []
        for scale_prediction in prediction:
            decoded.extend(decode_predictions(scale_prediction, img_width, img_height, conf_threshold))
        return decoded

    S = prediction.shape[1]
    classes = ["person", "car", "dog", "cat", "chair"]
    
    # Kênh 0 là objectness, được đưa qua sigmoid.
    pred_obj = torch.sigmoid(prediction[0, :, :]) # (S, S)
    # Kênh 1-5 là logits lớp, được chuẩn hóa bằng softmax.
    pred_class_probs = torch.softmax(prediction[1:6, :, :], dim=0) # (5, S, S)
    # Kênh 6-9 là offset tâm và kích thước hộp, được giới hạn bằng sigmoid.
    pred_coords = torch.sigmoid(prediction[6:10, :, :]) # (4, S, S)
    
    # Mỗi ô chỉ giữ lớp có xác suất lớn nhất.
    max_class_probs, max_class_indices = torch.max(pred_class_probs, dim=0) # (S, S)
    
    # Confidence kết hợp chất lượng objectness và xác suất phân lớp.
    scores = pred_obj * max_class_probs # (S, S)
    
    # Chỉ giải mã các ô có confidence vượt ngưỡng.
    rows, cols = torch.where(scores >= conf_threshold)
    
    decoded_boxes = []
    for row, col in zip(rows, cols):
        row = row.item()
        col = col.item()
        c_idx = max_class_indices[row, col].item()
        
        score = scores[row, col].item()
        class_name = classes[c_idx]
        
        # Giải mã tâm theo ô lưới; width/height tương đối theo toàn ảnh.
        tx = pred_coords[0, row, col].item()
        ty = pred_coords[1, row, col].item()
        tw = pred_coords[2, row, col].item()
        th = pred_coords[3, row, col].item()
        
        xc = (col + tx) / S
        yc = (row + ty) / S
        w = tw
        h = th
        
        # Đổi sang định dạng góc [xmin, ymin, xmax, ymax].
        xmin = (xc - w / 2.0) * img_width
        ymin = (yc - h / 2.0) * img_height
        xmax = (xc + w / 2.0) * img_width
        ymax = (yc + h / 2.0) * img_height
        
        # Giới hạn hộp trong biên ảnh gốc.
        xmin = max(0.0, min(float(img_width), xmin))
        ymin = max(0.0, min(float(img_height), ymin))
        xmax = max(0.0, min(float(img_width), xmax))
        ymax = max(0.0, min(float(img_height), ymax))
        
        # Chỉ giữ hộp có diện tích hợp lệ.
        if xmax > xmin and ymax > ymin:
            decoded_boxes.append({
                "class": class_name,
                "confidence": score,
                "bbox": [xmin, ymin, xmax, ymax]
            })
            
    return decoded_boxes

def non_maximum_suppression(boxes, iou_threshold=0.5):
    """
    Thực hiện NMS riêng theo từng lớp để loại các hộp trùng lặp.

    Hộp có confidence cao nhất được giữ trước; các hộp cùng lớp có IoU vượt
    ngưỡng với hộp đã giữ sẽ bị loại.
    """
    if not boxes:
        return []
        
    # Gom hộp theo lớp để các lớp khác nhau không triệt tiêu lẫn nhau.
    boxes_by_class = {}
    for box in boxes:
        cls = box["class"]
        if cls not in boxes_by_class:
            boxes_by_class[cls] = []
        boxes_by_class[cls].append(box)
        
    keep_boxes = []
    
    # Áp dụng NMS độc lập cho từng lớp.
    for cls, class_boxes in boxes_by_class.items():
        # Sắp xếp giảm dần theo confidence để ưu tiên hộp tốt nhất.
        sorted_boxes = sorted(class_boxes, key=lambda b: b["confidence"], reverse=True)
        
        while sorted_boxes:
            best_box = sorted_boxes.pop(0)
            keep_boxes.append(best_box)
            
            # Loại các hộp chồng lắp quá nhiều với hộp vừa được giữ.
            sorted_boxes = [
                box for box in sorted_boxes
                if bbox_iou(best_box["bbox"], box["bbox"]) < iou_threshold
            ]
            
    return keep_boxes
