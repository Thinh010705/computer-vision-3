# Phát hiện đối tượng Anchor-Free với ConvNeXt-Tiny và FPN tự cài đặt

Dự án xây dựng một detector anchor-free cho 5 lớp:

```text
person, car, dog, cat, chair
```

Phần phát hiện đối tượng được cài đặt trực tiếp bằng PyTorch: tạo target lưới, detection heads, hàm mất mát, giải mã hộp bao, confidence filtering và class-wise NMS. Dự án không sử dụng detector hoàn chỉnh như YOLOv5/v8, Detectron2, MMDetection, Faster R-CNN hoặc SSD có sẵn.

Backbone `ConvNeXt-Tiny` pretrained ImageNet được sử dụng làm mạng trích xuất đặc trưng. Điều này phù hợp với quy định cho phép dùng backbone đã huấn luyện trước; toàn bộ detection pipeline phía sau backbone vẫn được tự cài đặt.

> `ResNetYOLO` là tên lớp giữ lại từ phiên bản đầu. Kiến trúc thực tế trong `models/detector.py` sử dụng ConvNeXt-Tiny, không phải ResNet-50.

## Kết quả đã đo

Kết quả trên tập validation bằng công cụ chấm chính thức:

```text
mAP@0.5:         0.778076
Performance:     20/20
Micro recall:    0.914399
Micro precision: 0.110269
```

Kết quả trên được tạo bằng checkpoint tốt nhất đã lưu, threshold đã tune và TTA lật ngang. Các thử nghiệm làm thay đổi mạnh loss đã được loại bỏ vì làm giảm mAP; phiên bản hiện tại giữ loss baseline đã chứng minh hiệu quả và bổ sung các kỹ thuật inference/checkpoint ít rủi ro.

## Cấu trúc dự án

```text
<submission>/
├── public/
├── models/
│   ├── detector.py          # ConvNeXt-Tiny, FPN fusion và decoupled heads
│   └── best.pth             # Checkpoint tốt nhất theo validation mAP@0.5
├── utils/
│   ├── dataset.py           # Đọc JSON, augment, mosaic và sinh target lưới
│   ├── loss.py              # Focal objectness, weighted CE, CIoU và Smooth L1
│   └── nms.py               # Decode bbox, IoU và class-wise NMS tự cài đặt
├── train.py                 # Train, AMP, staged augmentation, top-k checkpoint và validation mAP
├── predict.py               # Suy luận, flip/multi-scale TTA, ensemble và predictions.json
├── tune_thresholds.py       # Tìm confidence/NMS threshold tốt nhất trên validation
├── average_checkpoints.py   # Tạo model soup bằng trung bình trọng số
├── EXPERIMENT_VARIANT.md
├── README.md
└── requirements.txt
```

## Kiến trúc mô hình

### Backbone và FPN

Ảnh được chuẩn hóa theo ImageNet và đưa qua `ConvNeXt-Tiny`:

```text
Ảnh đầu vào
    ↓
ConvNeXt-Tiny pretrained ImageNet
    ├── feature stride 16, 384 channels
    └── feature stride 32, 768 channels
              ↓
       projection 1x1 + upsample
              ↓
       FPN feature fusion
              ↓
       feature stride 16
```

Với ảnh `448×448`, feature cuối có kích thước `28×28`.

### Detection heads

Mô hình sử dụng hai nhánh dự đoán tách biệt:

- Classification head: `objectness + 5 class logits`.
- Regression head: `x, y, width, height`.

Mỗi cell dự đoán tối đa một đối tượng. Đầu ra có dạng:

```text
[objectness, class_1 ... class_5, x, y, w, h]
```

### Hàm mất mát

Loss được tự cài đặt và gồm:

- Focal Loss cho objectness.
- Weighted Cross Entropy với class weights cho phân lớp.
- CIoU Loss và Smooth L1 cho hồi quy hộp bao.

Class weights sử dụng inverse-frequency như cấu hình baseline đạt mAP tốt nhất.

## Quy trình dữ liệu

Dataset đọc trực tiếp `train.json` và `val.json`, hỗ trợ nhiều đối tượng trong một ảnh và tạo target lưới stride 16.

Huấn luyện sử dụng staged augmentation:

### Giai đoạn đầu

- Horizontal flip.
- Mosaic 4 ảnh với xác suất mặc định `0.15`.
- Random resized crop nhẹ.
- Affine nhẹ.
- Brightness, contrast, hue và saturation.
- Noise và cutout nhẹ.
- Multi-scale training ở `416`, `448`, `480`.

### Giai đoạn cuối

Trong 15 epoch cuối:

- Tắt mosaic.
- Cố định kích thước `448×448`.
- Chỉ giữ augmentation nhẹ.
- Giảm learning rate để fine-tune trên phân phối gần ảnh thật.

Resolution `384×384` đã được loại bỏ vì các thí nghiệm cho thấy nó làm giảm validation mAP đáng kể.

## Cài đặt môi trường

```bash
pip install -r requirements.txt
pip install albumentations
```

Khuyến nghị sử dụng GPU NVIDIA hỗ trợ CUDA. Code vẫn có thể chạy trên CPU nhưng quá trình huấn luyện sẽ chậm.

## Huấn luyện

### Lệnh bắt buộc

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

### Cấu hình khuyến nghị

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models_generalized/ \
  --epochs 60 \
  --batch_size 32 \
  --lr 1e-3 \
  --mosaic_prob 0.15 \
  --close_mosaic_epochs 15 \
  --fine_tune_lr_scale 0.25 \
  --save_top_k 5
```

Checkpoint được lưu:

```text
models_generalized/best.pth    # validation mAP tốt nhất
models_generalized/latest.pth  # Checkpoint mới nhất
models_generalized/epoch_*.pth # top-k checkpoint để ensemble/model soup
```

Mặc định backbone dùng trọng số ImageNet. Để khởi tạo toàn bộ mô hình ngẫu nhiên:

```bash
python train.py ... --no_pretrained
```

Để tiếp tục train từ checkpoint:

```bash
python train.py ... --resume ./models/best.pth
```

## Tune confidence và NMS

`tune_thresholds.py` thử nhiều cặp confidence/NMS threshold trên validation. Script không thay đổi trọng số mô hình.

Tune không dùng TTA:

```bash
python tune_thresholds.py \
  --val_data ./public/annotations/val.json \
  --val_image_dir ./public/val/images \
  --checkpoint ./models/best.pth
```

Tune với TTA lật ngang:

```bash
python tune_thresholds.py \
  --val_data ./public/annotations/val.json \
  --val_image_dir ./public/val/images \
  --checkpoint ./models/best.pth \
  --tta_flip
```

Tune với multi-scale TTA:

```bash
python tune_thresholds.py \
  --val_data ./public/annotations/val.json \
  --val_image_dir ./public/val/images \
  --checkpoint ./models/best.pth \
  --tta_sizes 416,448,480 \
  --tta_flip
```

Trong thí nghiệm hiện tại:

```text
Không TTA flip: mAP@0.5 = 0.7713
Có TTA flip:    mAP@0.5 = 0.7781
```

Vì vậy TTA flip được khuyến nghị cho checkpoint hiện tại. Khi predict hidden test, phải sử dụng cùng TTA và threshold đã tune.

## Suy luận

### Lệnh bắt buộc

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

### Cấu hình khuyến nghị

Thay `<best_conf>` và `<best_iou>` bằng kết quả từ `tune_thresholds.py`:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint ./models/best.pth \
  --conf_threshold <best_conf> \
  --iou_threshold <best_iou> \
  --tta_flip
```

`predict.py` hỗ trợ ensemble nhiều checkpoint:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint ./models/model_a.pth ./models/model_b.pth \
  --tta_flip
```

Prediction của ảnh gốc, ảnh lật và các checkpoint được gộp trước khi chạy class-wise NMS.

Multi-scale TTA chạy cùng model ở nhiều kích thước:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions_multiscale.json \
  --checkpoint ./models/best.pth \
  --tta_sizes 416,448,480 \
  --tta_flip
```

Multi-scale TTA chậm hơn đáng kể, vì vậy chỉ sử dụng nếu evaluator chính thức cho kết quả cao hơn.

## Model soup

`train.py` giữ lại top-k checkpoint tốt nhất. Có thể trung bình trọng số của các checkpoint có mAP gần nhau:

```bash
python average_checkpoints.py \
  --checkpoints \
    ./models/epoch_039_map_0.75xx.pth \
    ./models/epoch_043_map_0.76xx.pth \
    ./models/epoch_047_map_0.75xx.pth \
  --output ./models/model_soup.pth
```

Sau đó tune và đánh giá `model_soup.pth` như checkpoint bình thường. Không nên trộn checkpoint có mAP quá thấp hoặc đến từ kiến trúc khác.

## Đánh giá bằng công cụ chính thức

Tạo prediction trên validation:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json \
  --checkpoint ./models/best.pth \
  --conf_threshold <best_conf> \
  --iou_threshold <best_iou> \
  --tta_flip
```

Chấm bằng evaluator:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions predictions.json \
  --output score.json
```

Điểm trong `score.json` là kết quả gần nhất với cách hệ thống chấm hidden test. mAP hiển thị trong `train.py` chủ yếu dùng để chọn checkpoint tốt nhất.

## Định dạng predictions.json

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

- `image_id`: tên file ảnh.
- `class`: một trong 5 lớp quy định.
- `confidence`: thuộc `[0, 1]`.
- `bbox`: `[xmin, ymin, xmax, ymax]` theo tọa độ ảnh gốc.
- Ảnh không có detection vẫn được xuất với `"boxes": []`.

## Lưu ý khi so sánh mô hình

- Giữ checkpoint đạt `0.778076` làm baseline an toàn.
- Train các cải tiến mới vào thư mục checkpoint khác.
- So sánh bằng cùng evaluator, cùng TTA và cùng quy trình tune threshold.
- Không chọn model chỉ vì một lớp tăng; ưu tiên tổng mAP và độ ổn định qua nhiều epoch.
- Hidden test có thể khác validation, vì vậy không sử dụng threshold riêng theo lớp hoặc các điều chỉnh quá sát validation.
