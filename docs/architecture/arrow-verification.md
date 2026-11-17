# Arrow check

Every arrow that touches an AWS service, with the official AWS page that confirms it and its direction. Arrows between non-AWS parts (Streamlit, Gemini, SQLite, scikit-learn and so on) were checked against the code only. Links come from docs.aws.amazon.com only.

## architecture.png (live)

| From | To | Action | AWS doc |
|---|---|---|---|
| Summary writer | Amazon Bedrock, Claude Haiku 4.5 | InvokeModel through the Anthropic SDK (AnthropicBedrock, Messages format) | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html |
| Summary writer | Amazon Bedrock | Request body uses the Anthropic Claude Messages API | https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html |
| IAM user | Amazon Bedrock | Authorizes InvokeModel on Haiku 4.5 only | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html |

## UNVERIFIED

None.
