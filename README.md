# Decision Matrix MCP

A Model Context Protocol (MCP) server for structured decision analysis using parallel criterion evaluation. Make better decisions through systematic comparison of options across weighted criteria.

## Features

- **Thread Orchestration**: Each criterion evaluates options in parallel threads
- **Weighted Scoring**: 1-10 scale with importance weights (0.1-10.0)
- **Graceful Abstention**: Criteria can return `[NO_RESPONSE]` when not applicable
- **Multi-Backend Support**: Bedrock, LiteLLM, and Ollama
- **Session Management**: UUID-based sessions with TTL cleanup
- **Professional Presentation**: Clear rankings and recommendations

## Quick Start

1. **Install**:
```bash
git clone https://github.com/yourusername/decision-matrix-mcp.git
cd decision-matrix-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. **Configure Claude Desktop**:
```json
{
  "mcpServers": {
    "decision-matrix": {
      "command": "/path/to/decision-matrix-mcp/run.sh",
      "args": []
    }
  }
}
```

3. **Use in Claude**:
```
"Help me decide between AWS, GCP, and Azure for our startup"
```

## Available Tools

- `start_decision_analysis` - Initialize decision matrix
- `add_criterion` - Add weighted evaluation criteria  
- `evaluate_options` - Run parallel evaluation
- `get_decision_matrix` - View results and rankings
- `add_option` - Add new alternatives
- `list_sessions` - Manage sessions

## Example Usage

```
User: "Help me choose a cloud provider for our startup"
Claude: *creates decision matrix with AWS, GCP, Azure, DigitalOcean*

User: "Cost is most important (weight 3), then ease of use (2), then features"
Claude: *adds three criteria with appropriate weights*

User: "Evaluate the options"
Claude: *runs parallel evaluation across all criteria*

User: "Show me the results"
Claude: *displays ranked matrix with scores and justifications*
```

## Configuration

Set environment variables for LLM backends:
- `AWS_PROFILE` - For Bedrock access
- `LITELLM_API_KEY` - For OpenAI/Anthropic via LiteLLM
- `OLLAMA_HOST` - For local Ollama models

## License

MIT License - See LICENSE file for details