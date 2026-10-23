# Model card: resnet18-pneumoniamnist64-v1

**Decision support only, not a diagnosis.** Research demo trained on a public dataset. Not validated for clinical use.

## Model

- Architecture: ResNet18 pretrained on ImageNet (torchvision `IMAGENET1K_V1`), final layer replaced
  with one output (probability of pneumonia), all layers fine-tuned.
- Input: one grayscale chest X-ray resized to 64x64, repeated to 3 channels.
- Output: probability of pneumonia `p`. Label is pneumonia when `p >= 0.5`.
  Confidence is `max(p, 1 - p)`.
- Weights: `artifacts/model.pt` (half precision, about 22 MB).

## Training

- Data: PneumoniaMNIST (MedMNIST v2) official train split, 4,708 images, about 74% pneumonia.
- Loss: binary cross-entropy with the positive class down-weighted to offset the imbalance.
- Optimizer: AdamW, learning rate 0.0003, cosine schedule,
  batch size 64, seed 42.
- Augmentation: random horizontal flip and shifts of up to 4 pixels.
- Checkpoint: the epoch with the best ROC AUC on the official validation split (524 images).
- Epochs: 6 (kept epoch 6, best validation ROC AUC 0.998)
- CPU time per epoch: about 5 minutes on a 2-core Intel laptop
- Hardware: CPU only.

## Evaluation (held-out test split)

Test set: PneumoniaMNIST test split, 624 images (234 normal, 390 pneumonia).
The test split was not used for training or checkpoint choice. Half of it was used to pick the
routing threshold (see below); the classification metrics here use the fixed 0.5 cut-off.

| Metric | Value |
|---|---|
| Accuracy | 0.921 |
| Precision (pneumonia) | 0.892 |
| Recall (pneumonia) | 0.995 |
| Specificity (normal) | 0.799 |
| F1 (pneumonia) | 0.941 |
| ROC AUC | 0.982 |

Confusion matrix (rows are the true label, columns the prediction):

| | predicted normal | predicted pneumonia |
|---|---|---|
| **true normal** | 187 | 47 |
| **true pneumonia** | 2 | 388 |

## Routing threshold

- Threshold: **0.990** on confidence.
- How it was picked: calibration half of the test split (312 images, stratified, seed 0): lowest confidence threshold with >= 98% accuracy on auto-triaged cases.
- Routing results are reported on the other half of the test split, which played no part in
  picking the threshold.
- Cases at or above the threshold go to auto-triage. Everything else goes to the human review queue.

| Data | Lane | Cases | Share | Accuracy | Missed pneumonia |
|---|---|---|---|---|---|
| test calibration half | auto_triage | 241 | 77.2% | 98.3% | 0 |
| test calibration half | human_review | 71 | 22.8% | 70.4% | 2 |
| test evaluation half | auto_triage | 240 | 76.9% | 97.9% | 0 |
| test evaluation half | human_review | 72 | 23.1% | 73.6% | 0 |

Why not the validation split: PneumoniaMNIST's validation split is easier than its test split. A threshold
picked on validation (0.553) would auto-triage 98.1% of the evaluation
half at only 92.8% accuracy.

## Intended use

- Showing how a classifier, a confidence threshold, a human review queue and an audit log fit
  together in a triage workflow.
- Teaching and portfolio use.

## Not intended for

- Diagnosis, treatment decisions or any clinical use.
- Adult X-rays, other views, other conditions, or images from other hospitals. The data is pediatric
  chest X-rays from one center (Guangzhou Women and Children's Medical Center).

## Known limitations

- The validation and test splits come from different distributions in PneumoniaMNIST. Validation
  scores are higher than test scores, and a threshold tuned on validation is less reliable on test.
- 64x64 images remove fine detail a radiologist would use.
- Binary labels only (normal or pneumonia). Bacterial and viral pneumonia are merged.
- The model is overconfident on some wrong answers; see the missed pneumonia counts in the
  auto-triage lane.
- No calibration step, no subgroup analysis (the dataset has no demographic fields).

## Data license

PneumoniaMNIST is released under CC BY 4.0. Source images: Kermany et al., Cell 2018 (CC BY 4.0).
ImageNet-pretrained weights come from torchvision (BSD-3-Clause code).
