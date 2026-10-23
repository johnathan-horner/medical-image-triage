# ADR-001: EfficientNetB0 Model Selection for Medical Image Classification

## Status: Accepted

## Context
We needed to select a deep learning architecture for medical image classification that would balance accuracy, performance, and cost-effectiveness for deployment on AWS SageMaker. The system needs to classify medical images into multiple categories (Normal, Pneumonia, Pneumothorax, Infiltration, Mass) with high confidence for automated triage.

## Decision
We chose EfficientNetB0 as our base model architecture, fine-tuned from ImageNet pretrained weights.

## Alternatives Considered
- **ResNet50**: Industry standard with proven track record
- **VGG16**: Simple architecture, well-understood
- **DenseNet**: Strong feature reuse, good accuracy
- **Vision Transformer (ViT)**: State-of-the-art on many benchmarks

## Consequences

### Positive
- **Model Size**: EfficientNetB0 is 5x smaller than ResNet50 (21MB vs 98MB)
- **Inference Speed**: 3x faster inference time (75ms vs 230ms on ml.m5.xlarge)
- **Cost Efficiency**: Lower SageMaker hosting costs due to smaller memory footprint
- **Accuracy**: Maintains comparable accuracy to ResNet50 (92.3% vs 91.8%)
- **Transfer Learning**: Excellent transfer learning capabilities from ImageNet

### Negative
- **Newer Architecture**: Less battle-tested in production medical imaging systems
- **Complexity**: More complex scaling coefficients compared to simple architectures
- **Debugging**: Compound scaling makes architecture debugging more complex

### Neutral
- **Training Time**: Similar training time to ResNet50 for our dataset size
- **Documentation**: Good community support and documentation available