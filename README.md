# Mô hình Phát hiện Đối tượng Anchor-Free Đa Tỉ lệ với ConvNeXt và FPN P3/P4/P5

Kiến trúc sử dụng backbone **ConvNeXt-Tiny** hoặc **ConvNeXt-Small** pretrained ImageNet kết hợp FPN và các detection head tự cài đặt. Mô hình tạo ba lưới dự đoán tại stride `8`, `16` và `32`. Với ảnh đầu vào `448 x 448`, các lưới tương ứng có kích thước:

```text
P3: 56 x 56 - ưu tiên vật thể nhỏ
P4: 28 x 28 - ưu tiên vật thể trung bình
P5: 14 x 14 - ưu tiên vật thể lớn
```

Toàn bộ pipeline phát hiện đối tượng phía sau backbone được tự cài đặt bằng PyTorch, bao gồm FPN, decoupled detection heads, target assignment đa tỉ lệ, hàm mất mát, giải mã hộp bao, confidence filtering và class-wise NMS.

---

## 📂 Cấu trúc Thư mục Nộp bài

```text
<my_submission>/
├── models/                       # Định nghĩa mô hình và lưu checkpoint sau khi train
│   └── detector.py               # ConvNeXtFPNDetector: backbone, FPN P3/P4/P5 và heads
├── utils/                        # Các thành phần dùng chung cho dữ liệu, loss và hậu xử lý
│   ├── dataset.py                # Đọc JSON/ảnh, augmentation, Mosaic và tạo target đa tỉ lệ
│   ├── loss.py                   # Focal Loss, Weighted CE, CIoU và Smooth L1 tự cài đặt
│   └── nms.py                    # Decode, IoU và Class-wise NMS tự cài đặt
├── train.py                      # Huấn luyện, validation mAP, AMP và lưu top-k checkpoint
├── predict.py                    # Suy luận ảnh, TTA/ensemble và xuất predictions.json
├── tune_thresholds.py            # Quét confidence/NMS threshold tốt nhất trên validation
├── README.md                     # Hướng dẫn cài đặt, huấn luyện, suy luận và đánh giá
└── requirements.txt              # Danh sách thư viện Python cần cài đặt
```

---

## Hướng dẫn Thiết lập và chạy chương trình

### Bước 1: Cài đặt Môi trường

```bash
pip install -r requirements.txt
```

### Bước 2: Huấn luyện Mô hình

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

### Bước 3: Tune Confidence và NMS Threshold

Sau khi train xong, chạy `tune_thresholds.py` trên validation để tìm confidence threshold và NMS IoU threshold tốt nhất:

```bash
python tune_thresholds.py \
  --val_data ./public/annotations/val.json \
  --val_image_dir ./public/val/images \
  --checkpoint ./models/best.pth \
  --tta_flip
```

Script sẽ in dòng kết quả cuối dạng:

```text
BEST conf=0.03 iou=0.45 mAP@0.5=...
```

Sử dụng đúng hai giá trị `conf` và `iou` này khi chạy `predict.py`. Nếu tune có `--tta_flip`, suy luận cuối cũng phải bật `--tta_flip`.

Có thể thử multi-scale TTA:

```bash
python tune_thresholds.py \
  --val_data ./public/annotations/val.json \
  --val_image_dir ./public/val/images \
  --checkpoint ./models_p3p5_tiny/best.pth \
  --tta_sizes 416,448,480 \
  --tta_flip
```

Multi-scale TTA chậm hơn đáng kể và chỉ nên sử dụng nếu evaluator chính thức cho kết quả cao hơn.

### Bước 4: Chạy Suy luận
Không có threshold:
```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json \
  --checkpoint ./models/best.pth

```

Nếu không truyền `--checkpoint`, chương trình mặc định sử dụng
`./models/best.pth`. Khi file này chưa tồn tại, `predict.py` tự động tải
checkpoint từ Hugging Face:

```text
https://huggingface.co/NangThinh/Object_Detection_ConvNeXt_Small/resolve/main/best.pth
```

Repository Hugging Face cần được giữ ở chế độ public và môi trường chấm cần
có kết nối mạng. Checkpoint được tải vào file tạm `./models/best.pth.download`
rồi mới đổi tên thành `./models/best.pth`, tránh sử dụng file tải chưa hoàn tất.

Với threshold
```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json \
  --checkpoint ./models/best.pth \
  --conf_threshold <BEST_CONF_SMALL> \
  --iou_threshold <BEST_IOU_SMALL> \
  --tta_flip
```

### Bước 5: Tự chấm điểm và Đánh giá mAP@0.5

Sau khi tạo `predictions.json`, chạy evaluator chính thức:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions predictions.json \
  --output val_score.json
```
Lệnh chấm mẫu:
```bash
python tools/evaluate_predictions.py \
  --ground_truth ./private/hidden_test_annotations.json \
  --predictions predictions.json \
  --output score.json
```
