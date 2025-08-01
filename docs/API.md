# Decision Matrix MCP API Reference

This document provides a comprehensive reference for all MCP tools exposed by the Decision Matrix MCP server.

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
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

Decision Matrix MCP provides structured decision analysis through the Model Context Protocol. All tools operate on decision sessions that maintain state throughout the analysis process.

### Base URL

MCP servers don't have traditional URLs. They communicate via stdio with MCP clients like Claude Desktop.

### Authentication

Authentication is handled by the MCP client. Ensure your client is properly configured.

## Tools

### start_decision_analysis

Initialize a new decision analysis session with options and optional criteria.

**Description**: When facing multiple options and need structured evaluation - create a decision matrix to systematically compare choices across weighted criteria

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| topic | string | Yes | The decision topic or question to analyze |
| options | array[string] | Yes | List of options to evaluate (min: 2) |
| initial_criteria | array[object] | No | Optional initial criteria with name, description, and weight |
| model_backend | string | No | LLM backend to use (default: "bedrock") |
| model_name | string | No | Specific model to use |

**Initial Criteria Object**:
```json
{
  "name": "string",
  "description": "string",
  "weight": "number (0.1-10.0)"
}
```

**Response**:
```json
{
  "session_id": "uuid",
  "topic": "string",
  "options": ["option1", "option2"],
  "criteria_added": ["criterion1", "criterion2"],
  "model_backend": "bedrock|litellm|ollama",
  "model_name": "string|null",
  "message": "string",
  "next_steps": ["string"]
}
```

**Errors**:
- Invalid topic (empty or too long)
- Invalid options (less than 2, empty names, duplicates)
- Invalid criteria specification
- Resource limit exceeded (max sessions)

### add_criterion

Add a new evaluation criterion to an existing decision session.

**Description**: When you identify another factor to consider - add evaluation criteria with weights to structure your decision analysis

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes | Session ID to add criterion to |
| name | string | Yes | Name of the criterion (e.g., 'performance', 'cost') |
| description | string | Yes | What this criterion evaluates |
| weight | number | No | Importance weight (0.1-10.0, default: 1.0) |
| custom_prompt | string | No | Custom evaluation prompt for this criterion |
| model_backend | string | No | LLM backend to use (default: "bedrock") |
| model_name | string | No | Specific model to use |

**Response**:
```json
{
  "session_id": "uuid",
  "criterion_added": "string",
  "description": "string",
  "weight": 1.0,
  "total_criteria": 3,
  "all_criteria": ["criterion1", "criterion2", "criterion3"],
  "message": "string"
}
```

**Errors**:
- Invalid session ID
- Session not found or expired
- Criterion already exists
- Invalid criterion name/description/weight

### evaluate_options

Evaluate all options across all criteria using parallel thread orchestration.

**Description**: When ready to score your options systematically - run parallel evaluation where each criterion scores every option independently

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes | Session ID for evaluation |

**Response**:
```json
{
  "session_id": "uuid",
  "evaluation_complete": true,
  "summary": {
    "options_evaluated": 3,
    "criteria_used": 4,
    "total_evaluations": 12,
    "successful_scores": 10,
    "abstentions": 1,
    "errors": 1
  },
  "errors": ["Error details"] | null,
  "message": "string",
  "next_steps": ["get_decision_matrix - See the complete results"]
}
```

**Errors**:
- Invalid session ID
- No options to evaluate
- No criteria defined
- LLM backend errors

### get_decision_matrix

Get the complete decision matrix with scores, rankings, and recommendations.

**Description**: When you need the complete picture - see the scored matrix with weighted totals, rankings, and clear recommendations

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes | Session ID to get matrix for |

**Response**:
```json
{
  "session_id": "uuid",
  "topic": "string",
  "matrix": {
    "option1": {
      "criterion1": {
        "raw_score": 8.5,
        "weighted_score": 17.0,
        "justification": "string"
      }
    }
  },
  "rankings": [
    {
      "option": "option1",
      "weighted_total": 45.5,
      "breakdown": [
        {
          "criterion": "criterion1",
          "weight": 2.0,
          "raw_score": 8.5,
          "weighted_score": 17.0,
          "justification": "string",
          "abstained": false
        }
      ]
    }
  ],
  "recommendation": "string",
  "criteria_weights": {
    "criterion1": 2.0
  },
  "timestamp": "ISO 8601 string",
  "session_info": {
    "created_at": "ISO 8601 string",
    "evaluations_run": 1,
    "total_options": 3,
    "total_criteria": 4
  }
}
```

**Errors**:
- Invalid session ID
- Session not found
- No evaluations run yet
- Matrix generation error

### add_option

Add a new option to an existing decision analysis.

**Description**: When new alternatives emerge during analysis - add additional options to your existing decision matrix

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes | Session ID to add option to |
| option_name | string | Yes | Name of the new option |
| description | string | No | Optional description |

**Response**:
```json
{
  "session_id": "uuid",
  "option_added": "string",
  "description": "string|null",
  "total_options": 4,
  "all_options": ["option1", "option2", "option3", "option4"],
  "message": "string",
  "next_steps": ["evaluate_options - Re-run evaluation to include new option"]
}
```

**Errors**:
- Invalid session ID
- Session not found
- Option already exists
- Invalid option name

### list_sessions

List all active decision analysis sessions.

**Description**: List all active decision analysis sessions

**Parameters**: None

**Response**:
```json
{
  "sessions": [
    {
      "session_id": "uuid",
      "topic": "string",
      "created_at": "ISO 8601 string",
      "options": ["option1", "option2"],
      "criteria": ["criterion1", "criterion2"],
      "evaluations_run": 1,
      "status": "evaluated|setup"
    }
  ],
  "total_active": 2,
  "stats": {
    "total_sessions": 10,
    "active_sessions": 2,
    "sessions_expired": 5,
    "sessions_cleaned": 3,
    "max_concurrent": 4,
    "last_cleanup": "ISO 8601 string"
  }
}
```

### clear_all_sessions

Clear all active sessions from the session manager.

**Description**: Clear all active decision analysis sessions

**Parameters**: None

**Response**:
```json
{
  "cleared": 3,
  "message": "Cleared 3 active sessions",
  "stats": {
    "total_sessions": 0,
    "active_sessions": 0,
    "sessions_expired": 0,
    "sessions_cleaned": 0,
    "max_concurrent": 4,
    "last_cleanup": "ISO 8601 string"
  }
}
```

## Data Models

### ModelBackend Enum

Available LLM backends:
- `bedrock` - AWS Bedrock (Claude models)
- `litellm` - LiteLLM (OpenAI, Anthropic)
- `ollama` - Ollama (local models)

### Score Range

All evaluation scores are on a scale of 0-10:
- 0-3: Poor/Weak
- 4-6: Average/Moderate
- 7-8: Good/Strong
- 9-10: Excellent/Outstanding

### Weight Range

Criterion weights range from 0.1 to 10.0:
- 0.1-0.5: Low importance
- 0.6-1.5: Normal importance (default: 1.0)
- 1.6-3.0: High importance
- 3.1-10.0: Critical importance

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Error message describing what went wrong"
}
```

### Common Error Types

1. **Validation Errors**
   - Invalid input parameters
   - Out of range values
   - Empty or missing required fields

2. **Session Errors**
   - Session not found
   - Session expired
   - Invalid session ID format

3. **Resource Errors**
   - Maximum sessions reached
   - LLM backend unavailable
   - Rate limits exceeded

4. **Operation Errors**
   - No options/criteria defined
   - Evaluation not yet run
   - Duplicate names

## Examples

### Complete Decision Analysis Workflow

```python
# 1. Start a new decision analysis
response = await start_decision_analysis({
    "topic": "Choose a cloud provider",
    "options": ["AWS", "Google Cloud", "Azure"],
    "initial_criteria": [
        {
            "name": "Cost",
            "description": "Total cost of ownership",
            "weight": 3.0
        },
        {
            "name": "Features",
            "description": "Available services and capabilities",
            "weight": 2.0
        }
    ]
})
session_id = response["session_id"]

# 2. Add another criterion
await add_criterion({
    "session_id": session_id,
    "name": "Support",
    "description": "Quality of documentation and support",
    "weight": 1.5
})

# 3. Run the evaluation
await evaluate_options({
    "session_id": session_id
})

# 4. Get the results
matrix = await get_decision_matrix({
    "session_id": session_id
})

# Rankings will show weighted scores and recommendation
print(matrix["recommendation"])
# Output: "AWS is the clear winner with 24.5 points"
```

### Using Custom Evaluation Prompts

```python
await add_criterion({
    "session_id": session_id,
    "name": "Innovation",
    "description": "Track record of innovation",
    "weight": 2.0,
    "custom_prompt": "You are an expert in technology trends. Evaluate how innovative this option is based on their recent product launches, R&D investment, and market leadership.",
    "model_backend": "litellm",
    "model_name": "gpt-4"
})
```

### Handling Abstentions

When a criterion doesn't apply to an option, the LLM may abstain:

```json
{
  "matrix": {
    "Local Server": {
      "Cloud Features": {
        "raw_score": null,
        "weighted_score": null,
        "justification": "Abstained - criterion not applicable"
      }
    }
  }
}
```

### Session Management

```python
# List all active sessions
sessions = await list_sessions()
for session in sessions["sessions"]:
    print(f"{session['topic']} - {session['status']}")

# Clean up when done
await clear_all_sessions()
```