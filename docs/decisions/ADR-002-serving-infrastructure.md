# ADR-002: SageMaker Serverless Inference for Model Serving

## Status: Accepted

## Context
We needed to choose a serving infrastructure for our medical image classification model that would balance cost, performance, and scalability. The system needs to handle variable loads throughout the day with periods of low activity (nights/weekends) and higher activity during business hours.

## Decision
We chose SageMaker Serverless Inference over Real-Time Endpoints for model serving.

## Alternatives Considered
- **SageMaker Real-Time Endpoints**: Always-on endpoints with guaranteed capacity
- **Lambda + TensorFlow**: Custom Lambda deployment with TensorFlow
- **ECS Fargate**: Containerized deployment with auto-scaling
- **SageMaker Batch Transform**: Batch processing only

## Consequences

### Positive
- **Cost Efficiency**: $0 cost during idle periods vs $35+/month for always-on endpoints
- **Auto-scaling**: Automatic scaling from 0 to handle traffic spikes
- **No Cold Start Management**: AWS handles cold start optimization
- **Simplified Operations**: No capacity planning or instance management required
- **Pay-per-Use**: Only pay for actual inference requests

### Negative
- **Cold Start Latency**: 10-15 second cold start for first request after idle period
- **Concurrency Limits**: Maximum 200 concurrent requests (can be increased)
- **Less Control**: Limited control over underlying infrastructure and caching

### Neutral
- **Warm Performance**: Similar performance to real-time endpoints when warm (200ms avg)
- **Integration**: Same API interface as real-time endpoints
- **Monitoring**: Same CloudWatch metrics and monitoring capabilities

### Mitigation
- **Warm-up Strategy**: Implement periodic warm-up requests during expected usage periods
- **User Experience**: Display loading indicators for potentially slower first requests
- **Hybrid Approach**: Can switch to real-time endpoints if usage patterns change