import os
import argparse
import json
import urllib.request
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models.detector import ConvNeXtFPNDetector
from utils.nms import decode_predictions, non_maximum_suppression

DEFAULT_CHECKPOINT = "./models/best.pth"
DEFAULT_WEIGHT_URL = (
    "https://huggingface.co/NangThinh/Object_Detection_ConvNeXt_Small/"
    "resolve/main/best.pth"
)


def parse_args():
    """Đọc các đường dẫn và cấu hình hậu xử lý từ dòng lệnh."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--checkpoint", nargs="+", default=[DEFAULT_CHECKPOINT])
    parser.add_argument("--conf_threshold", type=float, default=0.02)
    parser.add_argument("--iou_threshold", type=float, default=0.55)
    parser.add_argument("--tta_sizes", default="448")
    parser.add_argument("--max_detections", type=int, default=150)
    return parser.parse_args()


def ensure_default_checkpoint(path):
    """Tự tải checkpoint mặc định từ Hugging Face nếu file chưa tồn tại."""
    if os.path.exists(path):
        return

    if os.path.normpath(path) != os.path.normpath(DEFAULT_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found at '{path}'.")

    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Tải vào file tạm để không sử dụng nhầm checkpoint chưa hoàn tất.
    temporary_path = f"{path}.download"
    print(f"Checkpoint not found at '{path}'.")
    print(f"Downloading default checkpoint from Hugging Face to: {path}")
    try:
        urllib.request.urlretrieve(DEFAULT_WEIGHT_URL, temporary_path)
        os.replace(temporary_path, path)
    except Exception as error:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise RuntimeError(
            f"Failed to download checkpoint from '{DEFAULT_WEIGHT_URL}': {error}"
        ) from error
    print("Checkpoint download completed.")


def load_checkpoint_state(path, device):
    """Đọc trọng số và metadata mô hình từ checkpoint."""
    ensure_default_checkpoint(path)

    print(f"Loading checkpoint from: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("model_config", {})
    return checkpoint, {}


def load_models(checkpoint_paths, device):
    """Khởi tạo các mô hình Small phục vụ suy luận hoặc ensemble."""
    models = []
    for checkpoint_path in checkpoint_paths:
        state_dict, _ = load_checkpoint_state(checkpoint_path, device)
        model = ConvNeXtFPNDetector(pretrained=False)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    return models


def flip_boxes_back(boxes, img_width):
    """Đưa hộp dự đoán từ ảnh lật ngang về hệ tọa độ ảnh gốc."""
    flipped = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box["bbox"]
        new_box = dict(box)
        new_box["bbox"] = [
            max(0.0, img_width - xmax),
            ymin,
            min(float(img_width), img_width - xmin),
            ymax,
        ]
        flipped.append(new_box)
    return flipped


def limit_detections(boxes, max_detections):
    """Giữ tối đa số hộp yêu cầu, ưu tiên confidence cao nhất."""
    return sorted(boxes, key=lambda b: b["confidence"], reverse=True)[:max_detections]


def parse_sizes(value):
    """Chuyển chuỗi kích thước TTA thành danh sách số nguyên."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def prepare_image(img, size, normalize, device):
    """Resize, chuyển ảnh thành tensor, chuẩn hóa ImageNet và đưa lên device."""
    resized = img.resize((size, size), Image.BILINEAR)
    return normalize(TF.to_tensor(resized)).unsqueeze(0).to(device)


@torch.no_grad()
def predict_image_boxes(models, img, conf_threshold, tta_sizes, normalize, device):
    """Thu thập hộp thô từ nhiều model, kích thước TTA và ảnh lật ngang."""
    w_orig, h_orig = img.size
    raw_boxes = []
    for size in tta_sizes:
        img_tensor = prepare_image(img, size, normalize, device)
        for model in models:
            output = model(img_tensor)
            # TTA lật ngang luôn bật để bổ sung các dự đoán ổn định hơn.
            flipped_output = model(torch.flip(img_tensor, dims=[3]))

            image_outputs = [scale_output[0] for scale_output in output]
            raw_boxes.extend(decode_predictions(image_outputs, w_orig, h_orig, conf_threshold=conf_threshold))
            flipped_image_outputs = [scale_output[0] for scale_output in flipped_output]
            flipped_boxes = decode_predictions(flipped_image_outputs, w_orig, h_orig, conf_threshold=conf_threshold)
            raw_boxes.extend(flip_boxes_back(flipped_boxes, w_orig))
    return raw_boxes


def main():
    """Chạy suy luận toàn bộ thư mục ảnh và ghi file predictions.json."""
    args = parse_args()
    
    # Chọn GPU nếu khả dụng, nếu không sẽ suy luận bằng CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Nhiều checkpoint được ensemble bằng cách gộp hộp thô rồi áp dụng NMS.
    models = load_models(args.checkpoint, device)
    tta_sizes = parse_sizes(args.tta_sizes)
    
    # Chuẩn hóa theo thống kê ImageNet tương ứng với backbone pretrained.
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Thu thập tất cả file ảnh hợp lệ trong thư mục đầu vào.
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(valid_exts))
    print(f"Found {len(img_files)} images in '{args.image_dir}' for prediction.")
    
    predictions_json = []
    
    # Suy luận lần lượt từng ảnh để giữ mức sử dụng bộ nhớ ổn định.
    with torch.no_grad():
        for filename in tqdm(img_files, desc="Inferring"):
            img_path = os.path.join(args.image_dir, filename)
            
            try:
                img = Image.open(img_path).convert("RGB")
                raw_boxes = predict_image_boxes(
                    models,
                    img,
                    args.conf_threshold,
                    tta_sizes,
                    normalize,
                    device,
                )
                
                # Áp dụng NMS theo lớp rồi giới hạn số lượng hộp mỗi ảnh.
                final_boxes = limit_detections(non_maximum_suppression(
                    raw_boxes, 
                    iou_threshold=args.iou_threshold,
                ), args.max_detections)
                
                # Luôn xuất đúng cấu trúc JSON yêu cầu của đề bài.
                predictions_json.append({
                    "image_id": filename,
                    "boxes": final_boxes
                })
                
            except OSError as error:
                print(f"Không thể đọc ảnh {filename}: {error}")
                # Nếu một ảnh lỗi, vẫn xuất image_id với danh sách hộp rỗng.
                predictions_json.append({
                    "image_id": filename,
                    "boxes": []
                })
                
    # Ghi toàn bộ kết quả suy luận thành predictions.json.
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(predictions_json, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated predictions file: '{args.output}'")

if __name__ == "__main__":
    main()
