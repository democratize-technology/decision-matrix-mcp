# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running the MCP Server
```bash
# Direct execution
python3 -m decision_matrix_mcp

# Via run script (for Claude Desktop integration)
./run.sh
```

### Development Tools
```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_decision_matrix.py::TestModels::test_score_creation

# Linting and formatting
ruff check src/
ruff format src/
black src/

# Type checking
mypy src/

# Install development dependencies
pip install -e ".[dev]"
```

## Architecture Overview

This is a Model Context Protocol (MCP) server that implements structured decision analysis through parallel criterion evaluation. The architecture is designed for thread orchestration where each evaluation criterion runs independently.

### Core Components

1. **MCP Server (`__init__.py`)**:
   - FastMCP-based server using stdio transport
   - Exposes decision analysis tools to Claude Desktop
   - CRITICAL: All output must go to stderr, never stdout (reserved for MCP protocol)

2. **Session Manager (`session_manager.py`)**:
   - UUID-based session management with TTL cleanup
   - Enforces limits on active sessions
   - Validates all inputs for security

3. **Orchestrator (`orchestrator.py`)**:
   - Parallel evaluation of options across criteria
   - Multi-backend LLM support (Bedrock, LiteLLM, Ollama)
   - Retry logic with exponential backoff
   - Graceful handling of abstentions with `[NO_RESPONSE]`

4. **Models (`models.py`)**:
   - `DecisionSession`: Core session state management
   - `CriterionThread`: Individual evaluation threads with conversation history
   - `Score`: Evaluation results with optional abstention
   - Weighted scoring calculation across criteria

### Key Design Patterns

- **Thread-per-Criterion**: Each criterion maintains its own conversation thread for consistent evaluation perspective
- **Parallel Evaluation**: All criterion-option pairs evaluated concurrently using asyncio
- **Graceful Abstention**: Criteria can return `[NO_RESPONSE]` when not applicable
- **Session Isolation**: Each decision analysis runs in an isolated session with cleanup

### Environment Configuration

Required for LLM backends:
- `AWS_PROFILE`: For AWS Bedrock access
- `LITELLM_API_KEY`: For OpenAI/Anthropic via LiteLLM
- `OLLAMA_HOST`: For local Ollama models (default: http://localhost:11434)

### MCP Integration Points

The server exposes these tools to Claude:
- `start_decision_analysis`: Initialize session with options and criteria
- `add_criterion`: Add weighted evaluation criteria
- `evaluate_options`: Run parallel evaluation across all criteria
- `get_decision_matrix`: Retrieve scored results with rankings
- `add_option`: Add new alternatives during analysis
- `list_sessions`: View active sessions
