# ADR-003: S3 Event-Driven Processing for Image Ingestion

## Status: Accepted

## Context
We needed to design the image ingestion and processing pipeline for medical images. The system must ensure no data loss, provide audit trails, and handle varying loads while maintaining HIPAA compliance. Images need to be processed for classification and then securely disposed of.

## Decision
We chose an S3 event-driven architecture where images are uploaded to S3, which triggers Lambda functions for processing, rather than direct API upload to processing services.

## Alternatives Considered
- **Direct API Upload**: Images sent directly to API Gateway → Lambda → SageMaker
- **SQS Queue-based**: Upload to S3 → SQS → Lambda processing
- **Kinesis Streams**: Real-time streaming processing of image data
- **Step Functions**: Orchestrated workflow with direct uploads

## Consequences

### Positive
- **Durability First**: S3 provides 99.999999999% (11 9's) durability
- **Decoupled Processing**: Upload and processing are independent operations
- **No Data Loss**: Images persisted before processing begins
- **Automatic Retry**: S3 events automatically retry on Lambda failures
- **Audit Trail**: Complete S3 access logs and CloudTrail for compliance
- **Scalability**: S3 can handle unlimited concurrent uploads

### Negative
- **Latency**: Additional network hop adds ~100-200ms to total processing time
- **Storage Costs**: Temporary storage costs for images before deletion
- **Complexity**: More components in the architecture to monitor

### Neutral
- **Security**: Same encryption and access control capabilities
- **Monitoring**: Standard CloudWatch metrics for S3 and Lambda
- **Cost**: Storage costs offset by improved reliability and reduced error handling

### Implementation Details
- **Lifecycle Policy**: Automatic deletion of processed images after 24 hours
- **Event Configuration**: S3 triggers Lambda on PUT events with image file extensions
- **Error Handling**: Dead letter queues for failed processing attempts
- **Encryption**: Server-side encryption with AWS KMS for all stored images