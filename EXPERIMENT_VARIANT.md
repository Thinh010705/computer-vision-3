# Variant: Direction 10 - Strong Data Augmentation

This version keeps the original architecture and uses a staged augmentation schedule:

- Early phase: light mosaic (`p=0.15`), moderate augmentations, multi-scale `416/448/480`.
- Final 15 epochs: mosaic disabled, light augmentations only, fixed `448x448`, reduced learning rate.

The `384x384` scale was removed because experiment logs showed a consistent validation mAP drop after training at that resolution.

Generalization-focused training improvements:

- IoU-aware objectness: confidence targets reflect localization quality.
- Online hard-negative mining: training focuses on confusing background cells instead of millions of easy negatives.
- Mild classification label smoothing.
- Square-root inverse-frequency class weights instead of aggressive raw inverse-frequency weights.
- EMA model weights are used for validation and saved as `model_state_dict`.

Recommended command:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --mosaic_prob 0.15 \
  --close_mosaic_epochs 15 \
  --fine_tune_lr_scale 0.25 \
  --ema_decay 0.9998 \
  --hard_negative_ratio 20 \
  --label_smoothing 0.05
```

These changes are class-agnostic and do not use validation-specific per-class thresholds, so they are intended to transfer better to the hidden test set.
