# Mô hình Phát hiện Đối tượng Anchor-Free Đa Tỉ lệ với ConvNeXt và FPN P3/P4/P5

Kiến trúc sử dụng duy nhất backbone **ConvNeXt-Small pretrained ImageNet** kết hợp FPN và các detection head tự cài đặt. Mô hình tạo ba lưới dự đoán tại stride `8`, `16` và `32`. Với ảnh đầu vào `448 x 448`, các lưới tương ứng có kích thước:

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
├── train.py                      # Huấn luyện, validation mAP, AMP và lưu best.pth
├── predict.py                    # Suy luận ảnh, TTA/ensemble và xuất predictions.json
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

### Bước 3: Chạy Suy luận

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

Khi không truyền cấu hình hậu xử lý, chương trình tự dùng các giá trị mặc định
đã chọn trước: confidence threshold `0.02`, NMS IoU threshold `0.55`, kích
thước TTA `448`, TTA lật ngang luôn bật và tối đa `150` hộp trên mỗi ảnh.

### Bước 4: Tự chấm điểm và Đánh giá mAP@0.5

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

---

## Chạy và Chấm bằng Docker

Các lệnh dưới đây cần được chạy lần lượt từ thư mục gốc của dự án, nơi chứa
`Dockerfile`, `predict.py`, `models/` và `public/`.

### Bước 1: Di chuyển vào thư mục dự án

```bash
cd /Users/nguyennangthinh/Downloads/computer-vision
ls Dockerfile predict.py public/val/images
```

### Bước 2: Build Docker image

Trên máy Linux x86_64 hoặc máy chấm của giảng viên:

```bash
docker build -t object-detection-exam:2026 .
```

Trên Mac Apple Silicon, có thể build image tương thích Linux AMD64:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t object-detection-exam:2026 .
```

Kiểm tra image đã được tạo:

```bash
docker images object-detection-exam:2026
```

### Bước 3: Tạo thư mục chứa kết quả chấm

```bash
mkdir -p grading_outputs
```

### Bước 4: Chạy inference bằng Docker

```bash
docker run --rm --gpus all \
  -w /workspace \
  -v "$PWD/public/val/images:/exam/val_images:ro" \
  -v "$PWD:/workspace" \
  -v "$PWD/grading_outputs:/exam/outputs" \
  object-detection-exam:2026 \
  python predict.py \
    --image_dir /exam/val_images \
    --output /exam/outputs/val_predictions.json
```

Nếu `./models/best.pth` chưa tồn tại, `predict.py` tự tải weight từ Hugging
Face về thư mục `models/`. Do dự án được mount vào `/workspace`, file weight
đã tải vẫn được giữ trên máy sau khi container kết thúc.

### Bước 5: Kiểm tra file predictions

```bash
ls -lh grading_outputs/val_predictions.json
```

### Bước 6: Chấm mAP trên validation

Chạy evaluator bên ngoài Docker:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions grading_outputs/val_predictions.json \
  --output grading_outputs/val_score.json
```

Xem kết quả:

```bash
cat grading_outputs/val_score.json
```

Khi giảng viên chạy lệnh inference tối giản, `predict.py` tự sử dụng toàn bộ
cấu hình hậu xử lý mặc định được khai báo trong script.
