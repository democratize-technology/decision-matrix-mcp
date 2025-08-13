# Decision Matrix MCP API Reference

This document provides detailed information about all available MCP tools in the decision-matrix-mcp server.

## Table of Contents
- [Overview](#overview)
- [Tools](#tools)
  - [start_decision_analysis](#start_decision_analysis)
  - [add_criterion](#add_criterion)
  - [evaluate_options](#evaluate_options)
  - [get_decision_matrix](#get_decision_matrix)
  - [add_option](#add_option)
  - [list_sessions](#list_sessions)
  - [clear_all_sessions](#clear_all_sessions)

## Overview

The Decision Matrix MCP server provides tools for structured decision analysis through parallel criterion evaluation. Each tool operates within the context of a session, allowing for complex multi-criteria decision making with weighted scoring.

## Tools

### start_decision_analysis

Initializes a new decision analysis session with options to evaluate.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic` | string | ✓ | - | The decision topic or question to analyze |
| `options` | array[string] | ✓ | - | List of options to evaluate (minimum 2) |
| `initial_criteria` | array[object] | ✗ | null | Optional initial criteria with name, description, and weight |
| `model_backend` | enum | ✗ | "bedrock" | LLM backend: "bedrock", "litellm", or "ollama" |
| `model_name` | string | ✗ | null | Specific model to use (backend-dependent) |

#### Initial Criteria Object Structure
```json
{
  "name": "string",
  "description": "string",
  "weight": "number (0.1-10.0)"
}
```

#### Returns
```json
{
  "session_id": "uuid",
  "topic": "string",
  "options": ["string"],
  "criteria": [{
    "name": "string",
    "description": "string",
    "weight": "number"
  }],
  "model_backend": "string",
  "model_name": "string"
}
```

#### Example
```python
result = await mcp.call_tool(
    "start_decision_analysis",
    request={
        "topic": "Choose a cloud provider for our new application",
        "options": ["AWS", "Google Cloud", "Azure"],
        "initial_criteria": [
            {
                "name": "cost",
                "description": "Total cost of ownership including compute, storage, and network",
                "weight": 2.0
            },
            {
                "name": "performance",
                "description": "Latency, throughput, and reliability metrics",
                "weight": 1.5
            }
        ],
        "model_backend": "bedrock",
        "model_name": "claude-3-sonnet"
    }
)
```

---

### add_criterion

Adds an evaluation criterion to an existing session.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string (uuid) | ✓ | - | Session ID to add criterion to |
| `name` | string | ✓ | - | Name of the criterion (e.g., 'performance', 'cost') |
| `description` | string | ✓ | - | What this criterion evaluates |
| `weight` | number | ✗ | 1.0 | Importance weight (0.1-10.0) |
| `custom_prompt` | string | ✗ | null | Custom evaluation prompt for this criterion |
| `model_backend` | enum | ✗ | "bedrock" | LLM backend for this criterion |
| `model_name` | string | ✗ | null | Specific model to use |

#### Returns
```json
{
  "criterion": {
    "name": "string",
    "description": "string",
    "weight": "number",
    "model_backend": "string",
    "model_name": "string"
  },
  "total_criteria": "number"
}
```

#### Example
```python
result = await mcp.call_tool(
    "add_criterion",
    request={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "security",
        "description": "Security features, compliance certifications, and data protection capabilities",
        "weight": 3.0,
        "custom_prompt": "Focus on SOC2, HIPAA compliance, and encryption at rest/transit"
    }
)
```

---

### evaluate_options

Runs parallel evaluation of all options against all criteria.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string (uuid) | ✓ | - | Session ID for evaluation |

#### Returns
```json
{
  "evaluations": {
    "option_name": {
      "criterion_name": {
        "score": "number (1-10)",
        "explanation": "string",
        "abstained": "boolean"
      }
    }
  },
  "options_evaluated": "number",
  "criteria_evaluated": "number",
  "total_evaluations": "number",
  "abstentions": "number",
  "duration_ms": "number"
}
```

#### Example
```python
result = await mcp.call_tool(
    "evaluate_options",
    request={
        "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
)
```

#### Notes
- Evaluations run in parallel for all option-criterion pairs
- Criteria can abstain with `[NO_RESPONSE]` if not applicable
- Typical evaluation time: 10-30 seconds depending on number of options and criteria

---

### get_decision_matrix

Retrieves the complete decision matrix with weighted scores and rankings.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string (uuid) | ✓ | - | Session ID to get matrix for |

#### Returns
```json
{
  "matrix": {
    "topic": "string",
    "options": ["string"],
    "criteria": [{
      "name": "string",
      "description": "string",
      "weight": "number"
    }],
    "scores": {
      "option_name": {
        "criterion_name": {
          "score": "number",
          "weighted_score": "number",
          "explanation": "string",
          "abstained": "boolean"
        }
      }
    },
    "totals": {
      "option_name": {
        "total": "number",
        "weighted_total": "number",
        "rank": "number"
      }
    },
    "recommendation": {
      "best_option": "string",
      "reasoning": "string"
    }
  }
}
```

#### Example
```python
result = await mcp.call_tool(
    "get_decision_matrix",
    request={
        "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
)

# Access results
best_option = result["matrix"]["recommendation"]["best_option"]
for option, scores in result["matrix"]["totals"].items():
    print(f"{option}: {scores['weighted_total']:.2f} (Rank: {scores['rank']})")
```

---

### add_option

Adds a new option to an existing analysis session.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string (uuid) | ✓ | - | Session ID to add option to |
| `option_name` | string | ✓ | - | Name of the new option |
| `description` | string | ✗ | null | Optional description of the option |

#### Returns
```json
{
  "option": "string",
  "total_options": "number",
  "needs_evaluation": "boolean"
}
```

#### Example
```python
result = await mcp.call_tool(
    "add_option",
    request={
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "option_name": "DigitalOcean",
        "description": "Developer-friendly cloud platform with simpler pricing"
    }
)
```

#### Notes
- New options need to be evaluated before appearing in the decision matrix
- Call `evaluate_options` after adding new options

---

### list_sessions

Lists all active decision analysis sessions.

#### Parameters

None

#### Returns
```json
{
  "sessions": [{
    "session_id": "string",
    "topic": "string",
    "created_at": "ISO 8601 timestamp",
    "options_count": "number",
    "criteria_count": "number",
    "evaluated": "boolean",
    "model_backend": "string"
  }],
  "total": "number"
}
```

#### Example
```python
result = await mcp.call_tool("list_sessions")
for session in result["sessions"]:
    print(f"{session['topic']} ({session['session_id'][:8]}...)")
```

---

### clear_all_sessions

Clears all active decision analysis sessions. Useful for cleanup or testing.

#### Parameters

None

#### Returns
```json
{
  "cleared": "number",
  "message": "string"
}
```

#### Example
```python
result = await mcp.call_tool("clear_all_sessions")
print(result["message"])  # "Cleared 3 sessions"
```

## Error Handling

All tools may return errors in the following format:

```json
{
  "error": {
    "type": "string",
    "message": "string"
  }
}
```

Common error types:
- `ValueError`: Invalid input parameters
- `KeyError`: Session or resource not found
- `RuntimeError`: Evaluation or backend errors

## Best Practices

1. **Session Management**: Sessions are automatically cleaned up after 24 hours of inactivity
2. **Criterion Weights**: Use weights between 1.0-3.0 for most cases; extreme weights (>5.0) can skew results
3. **Parallel Evaluation**: Evaluations run concurrently - expect 10-30 seconds for completion
4. **Abstentions**: Criteria may abstain when not applicable; this doesn't affect weighted totals
5. **Model Selection**: Different models may provide different perspectives; consistency within a session is maintained

## Environment Configuration

The following environment variables affect tool behavior:

- `AWS_PROFILE`: Required for AWS Bedrock backend
- `LITELLM_API_KEY`: Required for LiteLLM backend (OpenAI/Anthropic)
- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)
- `DECISION_MATRIX_LOG_LEVEL`: Logging verbosity (default: INFO)

## Rate Limits and Quotas

- Maximum concurrent sessions: 100
- Maximum options per session: 20
- Maximum criteria per session: 15
- Session timeout: 24 hours
- Evaluation timeout per criterion: 30 seconds
