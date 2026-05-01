# Model Card: Medical Image Triage Classifier

## Model Details
- Model: EfficientNetB0 (transfer learning from ImageNet)
- Framework: TensorFlow
- Type: Image classification (CNN)
- Version: 1.0.0
- Owner: Johnathan Horner, Shoot It Analytics LLC

## Intended Use
- Primary: Classify medical images and route to appropriate physician queue based on confidence score
- Users: Healthcare systems requiring automated triage with human-in-the-loop oversight
- Out of scope: Not intended as a standalone diagnostic tool. Always requires physician review for scores below 0.9

## Training Data
- Base: ImageNet (pretrained weights)
- Fine-tuned on: Medical image dataset
- Classes: [list the classes from the project]

## Evaluation Metrics
- Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion matrix available via dashboard endpoint

## Routing Thresholds
- >0.9: Auto-triage (standard queue)
- 0.7-0.9: Expedited physician review
- <0.7: Senior physician review

## Monitoring
- PSI drift detection via CloudWatch
- Confidence distribution tracked daily
- Alert threshold triggers revalidation

## Ethical Considerations
- Model should not be used without human oversight
- Confidence calibration required before deployment
- Bias evaluation recommended across demographic groups

## Compliance
- HIPAA: No raw images stored post-classification
- Encryption: KMS at rest
- Access: Cognito RBAC
- Audit: DynamoDB immutable log