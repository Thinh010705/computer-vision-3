# Variant: Direction 10 - Strong Data Augmentation

This version keeps the original architecture and adds custom 4-image mosaic augmentation before the existing Albumentations pipeline.

The goal is to test whether stronger data diversity improves validation and hidden-test mAP without changing the model architecture.

