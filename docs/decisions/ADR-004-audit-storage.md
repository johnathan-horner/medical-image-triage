# ADR-004: DynamoDB for Audit Trail Storage

## Status: Accepted

## Context
We needed to select a database for storing audit trails, prediction metadata, and compliance records for the medical image triage system. The system requires HIPAA-compliant audit logging with high write throughput, fast queries, and long-term retention capabilities.

## Decision
We chose Amazon DynamoDB with on-demand billing for audit trail and metadata storage.

## Alternatives Considered
- **Amazon RDS (PostgreSQL)**: Traditional relational database with ACID compliance
- **Amazon DocumentDB**: MongoDB-compatible document database
- **Amazon ElastiSearch**: Search-optimized storage with analytics
- **Amazon S3**: Object storage with metadata indexing

## Consequences

### Positive
- **Write Performance**: Single-digit millisecond write latency for high-throughput logging
- **Pay-per-Request**: On-demand billing scales cost with actual usage
- **NoSQL Flexibility**: Schema flexibility for evolving audit requirements
- **Auto-scaling**: Automatic capacity scaling without manual intervention
- **HIPAA Eligible**: Fully compliant with HIPAA requirements when properly configured
- **Backup/Recovery**: Point-in-time recovery and automated backups

### Negative
- **Query Limitations**: Limited query flexibility compared to SQL databases
- **Learning Curve**: DynamoDB-specific query patterns and best practices
- **Eventually Consistent**: Default eventual consistency (strong consistency available)

### Neutral
- **Cost Predictability**: On-demand pricing eliminates capacity planning but less predictable
- **Integration**: Native AWS service integration with Lambda and other services
- **Monitoring**: CloudWatch integration for performance monitoring

### Design Patterns
- **Primary Key**: Composite key with prediction_id (partition) and timestamp (sort)
- **GSI**: Global Secondary Index on patient_id_hash for patient lookups
- **TTL**: Time-to-live for automatic data lifecycle management (7-year retention)
- **Encryption**: Server-side encryption with AWS KMS customer-managed keys

### Performance Characteristics
- **Expected Load**: 1K-10K writes/day, 100-1K reads/day
- **Item Size**: Average 2KB per audit record
- **Query Patterns**: Point lookups, time-range queries, patient history