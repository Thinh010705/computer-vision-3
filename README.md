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
├── average_checkpoints.py        # Tạo model soup bằng trung bình trọng số checkpoint
├── README.md                     # Hướng dẫn cài đặt, huấn luyện, suy luận và đánh giá
└── requirements.txt              # Danh sách thư viện Python cần cài đặt
```

---

## 🚀 Hướng dẫn Thiết lập và chạy chương trình

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

Mặc định chương trình:

- Dùng backbone `ConvNeXt-Tiny` pretrained ImageNet.
- Huấn luyện `50` epoch.
- Dùng batch size `16`.
- Bật multi-scale training.
- Lưu mô hình tốt nhất vào `./models/best.pth`.
- Lưu trạng thái gần nhất vào `./models/latest.pth`.
- Giữ lại các top-k checkpoint theo validation mAP.

#### Cấu hình khuyến nghị với ConvNeXt-Tiny

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models_p3p5_tiny/ \
  --epochs 60 \
  --batch_size 16 \
  --lr 1e-3 \
  --weight_decay 3e-4 \
  --backbone tiny \
  --backbone_lr_scale 0.05 \
  --class_weight_power 0.5 \
  --label_smoothing 0.05 \
  --mosaic_prob 0.15 \
  --close_mosaic_epochs 5 \
  --fine_tune_lr_scale 0.25 \
  --val_interval 5 \
  --dense_val_epochs 10 \
  --save_top_k 5
```

Theo cấu hình này, validation mAP được tính mỗi 5 epoch trong giai đoạn
đầu và được tính sau mỗi epoch trong 10 epoch cuối. Epoch cuối luôn được
đánh giá. `latest.pth` vẫn được lưu sau mọi epoch, còn `best.pth` và
top-k checkpoint chỉ được cập nhật tại các epoch có chạy validation.

- `--val_interval 5`: khoảng cách validation trong giai đoạn đầu.
- `--dense_val_epochs 10`: số epoch cuối được validation liên tục.

#### Cấu hình thử nghiệm với ConvNeXt-Small

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models_p3p5_small/ \
  --epochs 60 \
  --batch_size 8 \
  --lr 7e-4 \
  --weight_decay 3e-4 \
  --backbone small \
  --backbone_lr_scale 0.05 \
  --class_weight_power 0.5 \
  --label_smoothing 0.05 \
  --mosaic_prob 0.15 \
  --close_mosaic_epochs 5 \
  --fine_tune_lr_scale 0.25 \
  --val_interval 5 \
  --dense_val_epochs 10 \
  --save_top_k 5
```

ConvNeXt-Small có khả năng biểu diễn mạnh hơn nhưng tốn nhiều VRAM và có nguy cơ overfit cao hơn. Nên huấn luyện ConvNeXt-Tiny trước để đo lợi ích của kiến trúc đa tỉ lệ, sau đó mới so sánh với Small bằng cùng evaluator.

#### Cấu hình ưu tiên khả năng tổng quát trên hidden test

Cấu hình này giảm việc học quá sát validation bằng regularization mạnh hơn,
fine-tune ngắn hơn và giảm số lần lựa chọn checkpoint theo validation:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models_generalized/ \
  --epochs 50 \
  --batch_size 8 \
  --lr 6e-4 \
  --weight_decay 3e-4 \
  --backbone small \
  --backbone_lr_scale 0.05 \
  --class_weight_power 0.5 \
  --label_smoothing 0.05 \
  --mosaic_prob 0.15 \
  --close_mosaic_epochs 5 \
  --fine_tune_lr_scale 0.25 \
  --val_interval 5 \
  --dense_val_epochs 10 \
  --save_top_k 3
```

- `--class_weight_power 0.5`: dùng căn bậc hai của inverse frequency thay
  vì inverse frequency hoàn toàn, giảm nguy cơ học quá sát các lớp hiếm.
- `--label_smoothing 0.05`: hạn chế mô hình trở nên quá tự tin trên train/val.
- `--backbone_lr_scale 0.05`: bảo vệ đặc trưng ImageNet có tính tổng quát.
- `--close_mosaic_epochs 5`: chỉ dùng augmentation nhẹ trong 5 epoch cuối.
- `--dense_val_epochs 10`: giảm số lần quan sát và lựa chọn checkpoint theo val.

Để đánh giá công bằng, nên giữ nguyên threshold của cấu hình tốt trước đó
trong lần nộp đầu tiên. Chỉ tune lại threshold nếu mức tăng xuất hiện ổn định
trên nhiều checkpoint, tránh chọn một cặp threshold quá sát validation.

#### Huấn luyện không dùng pretrained ImageNet

```bash
python train.py ... --no_pretrained
```

### Bước 3: Tune Confidence và NMS Threshold

Sau khi train xong, chạy `tune_thresholds.py` trên validation để tìm confidence threshold và NMS IoU threshold tốt nhất:

```bash
python tune_thresholds.py \
  --val_data ./public/annotations/val.json \
  --val_image_dir ./public/val/images \
  --checkpoint ./models_p3p5_tiny/best.pth \
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

Lệnh suy luận bắt buộc theo đề bài:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Suy luận trên validation với threshold đã tune:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json \
  --checkpoint ./models_p3p5_tiny/best.pth \
  --conf_threshold <BEST_CONF> \
  --iou_threshold <BEST_IOU> \
  --tta_flip
```

Suy luận với ConvNeXt-Small sử dụng cùng lệnh, chỉ thay checkpoint:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions_small.json \
  --checkpoint ./models_p3p5_small/best.pth \
  --conf_threshold <BEST_CONF_SMALL> \
  --iou_threshold <BEST_IOU_SMALL> \
  --tta_flip
```

Mỗi ảnh luôn xuất hiện trong file kết quả. Nếu không phát hiện đối tượng, chương trình xuất:

```json
{
  "image_id": "image.jpg",
  "boxes": []
}
```

### Bước 5: Tự chấm điểm và Đánh giá mAP@0.5

Sau khi tạo `predictions.json`, chạy evaluator chính thức:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions predictions.json \
  --output score.json
```

Xem kết quả:

```bash
cat score.json
```

Điểm trong `score.json` là kết quả cần dùng để so sánh mô hình. Validation mAP hiển thị trong `train.py` chủ yếu dùng để lựa chọn checkpoint.

---

## 🛠️ Điểm nhấn Công nghệ của Giải pháp

### 1. Backbone ConvNeXt-Tiny hoặc ConvNeXt-Small

- Mặc định sử dụng **ConvNeXt-Tiny pretrained ImageNet** để đạt cân bằng tốt giữa độ chính xác, tốc độ và bộ nhớ GPU.
- Hỗ trợ **ConvNeXt-Small** nhằm thử nghiệm backbone có khả năng biểu diễn mạnh hơn.
- Backbone được fine-tune với learning rate nhỏ hơn detection head để bảo vệ đặc trưng pretrained.
- Toàn bộ FPN, detection heads, target assignment, loss và hậu xử lý được tự cài đặt.

### 2. FPN Đa tỉ lệ P3/P4/P5

Mô hình lấy ba feature map từ ConvNeXt:

```text
C2: stride 8,  192 channels
C3: stride 16, 384 channels
C4: stride 32, 768 channels
```

Các feature được chiếu về `256` kênh và dung hợp theo hướng top-down:

```text
P5 = projection(C4)
P4 = fusion(projection(C3), upsample(P5))
P3 = fusion(projection(C2), upsample(P4))
```

Ba head dự đoán độc lập:

- `P3`, stride 8: tăng khả năng phát hiện vật thể nhỏ.
- `P4`, stride 16: xử lý vật thể trung bình.
- `P5`, stride 32: cung cấp receptive field lớn cho vật thể lớn.

### 3. Anchor-Free Target Assignment Đa tỉ lệ

Mô hình không sử dụng anchor box. Mỗi object được mã hóa thành:

```text
[objectness, 5 class targets, center_x, center_y, width, height]
```

Mỗi ground-truth box được gán vào:

- Một scale chính dựa trên kích thước đối tượng.
- Một scale lân cận để tạo thêm positive supervision.

Cơ chế này giúp:

- Giảm mất nhãn khi nhiều object có tâm rơi vào cùng cell.
- Tăng tín hiệu huấn luyện cho vật thể nhỏ.
- Giữ thiết kế anchor-free và dễ giải thích.

Nếu vẫn xảy ra xung đột cùng một cell trên cùng scale, hệ thống ưu tiên đối tượng nhỏ hơn vì đây thường là đối tượng khó phát hiện hơn.

### 4. Decoupled Detection Heads với Depthwise-Separable Convolution

Mỗi scale sử dụng hai nhánh riêng:

- Classification/Objectness Head: dự đoán objectness và 5 class logits.
- Regression Head: dự đoán `x, y, width, height`.

Việc tách nhánh giúp giảm xung đột giữa nhiệm vụ phân lớp và định vị. Các head sử dụng **depthwise-separable convolution tự cài đặt** để giảm FLOPs và bộ nhớ, đặc biệt quan trọng với lưới P3 `56 x 56`.

### 5. Hàm mất mát Focal Loss, Weighted CE, CIoU và Smooth L1

Hàm mất mát gồm:

- **Focal Loss** cho objectness, giúp giảm ảnh hưởng của số lượng lớn background cell.
- **Weighted Cross Entropy** cho phân lớp, xử lý mất cân bằng giữa 5 lớp.
- **CIoU Loss** cho hộp bao, tối ưu độ chồng lắp, khoảng cách tâm và tỷ lệ khung hình.
- **Smooth L1 Loss** hỗ trợ ổn định quá trình học tọa độ.

Loss được tính riêng trên từng scale và lấy trung bình giữa P3/P4/P5.

### 6. Staged Augmentation và Multi-Scale Training

Trong giai đoạn đầu, mô hình sử dụng:

- Horizontal Flip.
- Mosaic bốn ảnh.
- Random Resized Crop.
- Affine nhẹ.
- Brightness, Contrast, Hue và Saturation.
- Gaussian Noise và CoarseDropout.
- Multi-scale training tại `416`, `448`, `480`.

Trong `15` epoch cuối:

- Tắt Mosaic.
- Cố định kích thước `448 x 448`.
- Chỉ giữ augmentation nhẹ.
- Giảm learning rate để fine-tune trên phân phối gần ảnh thật.

### 7. Differential Learning Rates, Warm-up, Cosine Decay và AMP

- Backbone ConvNeXt dùng learning rate bằng `0.1` learning rate của FPN và heads.
- Ba epoch đầu sử dụng linear warm-up.
- Sau đó learning rate giảm theo Cosine Annealing.
- Mixed Precision AMP giúp giảm bộ nhớ và tăng tốc huấn luyện.
- Gradient clipping hạn chế cập nhật bất thường.

### 8. Class-wise NMS, TTA và Ensemble

- Confidence được tính bằng `objectness × class probability`.
- Dự đoán từ P3/P4/P5 được hợp nhất trước khi chạy NMS.
- Class-wise NMS tự cài đặt loại bỏ hộp trùng lặp theo từng lớp.
- Hỗ trợ horizontal-flip TTA và multi-scale TTA.
- Hỗ trợ ensemble nhiều checkpoint và model soup.

---

## 📊 Kết quả Baseline và Cách So sánh

Kết quả tốt nhất đã ghi nhận của kiến trúc stride-16 trước khi nâng cấp:

```text
mAP@0.5 = 0.77937
```

Đây là baseline đối chứng, không phải kết quả đã xác nhận của kiến trúc P3/P4/P5 mới. Sau khi train Tiny hoặc Small, cần dùng cùng quy trình:

```text
train -> tune threshold -> predict -> official evaluator
```

Chỉ chọn kiến trúc mới nếu `score.json` tốt hơn baseline hoặc cho kết quả ổn định hơn trên nhiều checkpoint. Đặc biệt nên theo dõi AP của `chair`, `car`, số lượng predictions, micro precision và micro recall.

---

## 📄 Định dạng predictions.json

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

Quy định:

- `image_id` là tên file ảnh trong thư mục suy luận.
- `class` thuộc một trong 5 lớp quy định.
- `confidence` nằm trong đoạn `[0, 1]`.
- `bbox` có dạng `[xmin, ymin, xmax, ymax]` trên tọa độ ảnh gốc.
- Ảnh không có phát hiện vẫn phải xuất `"boxes": []`.

---

## ⚠️ Lưu ý Quan trọng

- Giữ checkpoint baseline cũ ở thư mục riêng; không dùng nó để resume kiến trúc P3/P4/P5.
- ConvNeXt-Tiny nên được thử trước vì ít rủi ro overfit và nhẹ hơn.
- ConvNeXt-Small nên dùng batch size nhỏ hơn, ví dụ `8`.
- Nếu tune threshold với `--tta_flip` hoặc `--tta_sizes`, khi predict cũng phải dùng đúng cấu hình đó.
- Không tăng confidence threshold chỉ để cải thiện micro precision; cần lựa chọn theo mAP chính thức.
- File `WORK_DONE_SUMMARY.md` trình bày chi tiết cơ chế và đánh giá kỹ thuật của mô hình.
