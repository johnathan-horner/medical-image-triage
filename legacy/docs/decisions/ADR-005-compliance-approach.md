# ADR-005: No Raw Image Retention for HIPAA Compliance

## Status: Accepted

## Context
We needed to establish a data retention and privacy strategy for medical images that balances HIPAA compliance requirements, audit trail needs, and operational efficiency. The system processes sensitive medical images that contain Protected Health Information (PHI).

## Decision
We chose to implement a zero raw image retention policy, storing only SHA256 hashes and metadata for audit purposes, with immediate deletion of raw images after processing.

## Alternatives Considered
- **Encrypted Long-term Storage**: Store encrypted images with strict access controls
- **De-identified Storage**: Remove PHI and store anonymized images for research
- **Hybrid Approach**: Store images for 30 days then automatic deletion
- **External Archive**: Partner with certified medical data archiving service

## Consequences

### Positive
- **HIPAA Simplified**: Eliminates most PHI storage compliance requirements
- **Security**: No raw medical images available for potential data breaches
- **Cost Efficiency**: No long-term storage costs for large image files
- **Audit Sufficiency**: SHA256 hashes provide unique identification for audit trails
- **Processing Focus**: Clear separation between image processing and data retention
- **KMS Integration**: All stored metadata encrypted with customer-managed keys

### Negative
- **No Reprocessing**: Cannot reprocess images if model is updated
- **Limited Research**: Cannot perform retrospective analysis on historical images
- **Debugging Limitations**: Cannot review original images for model debugging

### Neutral
- **Compliance Overhead**: Simpler compliance but still requires proper metadata handling
- **Performance**: No impact on inference performance
- **Integration**: Compatible with existing PACS and medical systems

### Implementation Details
- **Image Hash**: SHA256 hash computed before processing for unique identification
- **Metadata Retention**: Patient ID hash, timestamp, predictions, confidence scores
- **Automatic Deletion**: Lambda function deletes images immediately after successful processing
- **Error Handling**: Failed processing images deleted after 24 hours maximum
- **Audit Trail**: Complete processing trail stored in DynamoDB with 7-year retention

### Risk Mitigation
- **Model Versioning**: Comprehensive model versioning and validation before deployment
- **Test Coverage**: Extensive testing to minimize need for production debugging
- **Monitoring**: Real-time monitoring to catch issues before they affect many images