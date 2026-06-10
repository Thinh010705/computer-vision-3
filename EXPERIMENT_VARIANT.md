# Experiment Variant: Anchor-Free P3/P4/P5

Phiên bản này nâng cấp baseline ConvNeXt stride-16 thành detector anchor-free đa tỉ lệ:

- `P3`, stride 8: ưu tiên vật thể nhỏ.
- `P4`, stride 16: vật thể trung bình.
- `P5`, stride 32: vật thể lớn.
- Mỗi ground-truth box được gán vào scale chính và một scale lân cận.
- Hỗ trợ backbone `ConvNeXt-Tiny` và `ConvNeXt-Small`.
- Hỗ trợ neck FPN top-down hoặc PAN-FPN hai chiều tự cài đặt.
- Hỗ trợ EMA trọng số trong quá trình train.
- Giữ staged augmentation, Mosaic, multi-scale training, TTA và top-k checkpoints.

## Mục tiêu thí nghiệm

Baseline một head stride 16 có hai giới hạn:

1. Vật thể nhỏ chỉ được biểu diễn trên lưới `28 x 28`.
2. Nhiều đối tượng có tâm rơi vào cùng cell có thể ghi đè target.

Ba head giúp tăng độ phân giải cho vật thể nhỏ và tạo thêm không gian target. Gán mỗi vật thể vào hai scale cung cấp nhiều positive supervision mà vẫn giữ cơ chế offset tâm hợp lệ trong `[0, 1]`.

## Lưu ý checkpoint

Checkpoint baseline một head không tương thích với kiến trúc mới. Không ghi đè thư mục baseline.

## Thí nghiệm khuyến nghị

### Thí nghiệm 1: đo riêng lợi ích của head đa tỉ lệ

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
  --backbone tiny \
  --mosaic_prob 0.15 \
  --close_mosaic_epochs 15 \
  --save_top_k 5
```

### Thí nghiệm 2: thử ConvNeXt-Small

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
  --backbone small \
  --mosaic_prob 0.15 \
  --close_mosaic_epochs 15 \
  --save_top_k 5
```

Tiny nên được chạy trước. Chỉ giữ Small nếu official evaluator tăng ổn định, vì backbone lớn hơn tốn VRAM và có nguy cơ overfit cao hơn.

### Thí nghiệm 3: PAN-FPN + EMA, giữ riêng mô hình FPN tốt nhất

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models_p3p5_small_pan_ema/ \
  --epochs 45 \
  --batch_size 8 \
  --lr 7e-4 \
  --weight_decay 2e-4 \
  --backbone small \
  --neck pan \
  --ema_decay 0.9998 \
  --mosaic_prob 0.15 \
  --close_mosaic_epochs 5 \
  --fine_tune_lr_scale 0.25 \
  --save_top_k 5
```

Mục tiêu của thí nghiệm này là tạo một mô hình có sai số khác FPN thuần. Hãy đánh giá riêng PAN trước, sau đó tune ensemble FPN + PAN. Không ghi đè checkpoint FPN đã đạt `0.782` trên Kaggle.

Sau khi có PAN, tune ensemble hai lần với `--fusion nms` và `--fusion wbf`. WBF tự cài đặt trung bình tọa độ của các hộp đồng thuận, có thể cải thiện localization của ensemble; chỉ chọn nó khi validation mAP tăng ổn định.

## Đánh giá

Sau train, tune và chấm bằng cùng quy trình:

```bash
python tune_thresholds.py \
  --val_data ./public/annotations/val.json \
  --val_image_dir ./public/val/images \
  --checkpoint ./models_p3p5_tiny/best.pth \
  --tta_flip
```

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions_p3p5.json \
  --checkpoint ./models_p3p5_tiny/best.pth \
  --conf_threshold <BEST_CONF> \
  --iou_threshold <BEST_IOU> \
  --tta_flip
```

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions predictions_p3p5.json \
  --output score_p3p5.json
```

So sánh với baseline `mAP@0.5 = 0.77937`, đặc biệt theo dõi AP của `chair`, `car`, số predictions và micro precision.
