# Variant: Direction 10 - Strong Data Augmentation

This version keeps the original architecture and uses a staged augmentation schedule:

- Early phase: light mosaic (`p=0.15`), moderate augmentations, multi-scale `416/448/480`.
- Final 15 epochs: mosaic disabled, light augmentations only, fixed `448x448`, reduced learning rate.

The `384x384` scale was removed because experiment logs showed a consistent validation mAP drop after training at that resolution.

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
  --fine_tune_lr_scale 0.25
```
