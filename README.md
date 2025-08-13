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
git clone https://github.com/democratize-technology/decision-matrix-mcp.git
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

## Troubleshooting

### Common Issues

#### MCP Server Not Found in Claude Desktop

**Symptom**: Claude Desktop doesn't recognize the decision-matrix server

**Solution**:
1. Verify the path in your Claude Desktop config is absolute, not relative
2. Ensure `run.sh` has execute permissions: `chmod +x run.sh`
3. Check the server starts manually: `python -m decision_matrix_mcp`
4. Review Claude Desktop logs for error messages

#### LLM Backend Connection Errors

**AWS Bedrock Issues**:
```bash
# Check AWS credentials
aws sts get-caller-identity

# Ensure region has Bedrock access
export AWS_REGION=us-east-1  # or another supported region
```

**LiteLLM/OpenAI Issues**:
```bash
# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Ollama Connection Issues**:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If using Docker, use host networking
docker run --network host ...
```

#### Session Expired Errors

**Symptom**: "Session not found or expired" after some time

**Solution**:
- Sessions expire after 24 hours by default
- Use `list_sessions` to see active sessions
- Start a new analysis if needed

#### Evaluation Timeout

**Symptom**: Evaluations take too long or timeout

**Solutions**:
1. Reduce number of options or criteria
2. Use a faster LLM model
3. Check network connectivity
4. Consider using Ollama for local execution

#### Import Errors

**Symptom**: `ModuleNotFoundError` when starting

**Solution**:
```bash
# Ensure you're in the virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -e .
```

### Debug Mode

Enable detailed logging by setting the environment variable:
```bash
export LOG_LEVEL=DEBUG
```

### Getting Help

1. Check the [Getting Started Guide](docs/GETTING_STARTED.md)
2. Review [Architecture Documentation](docs/ARCHITECTURE.md)
3. Search [existing issues](https://github.com/democratize-technology/decision-matrix-mcp/issues)
4. Ask in [Discussions](https://github.com/democratize-technology/decision-matrix-mcp/discussions)
5. Report bugs using our [issue templates](.github/ISSUE_TEMPLATE)

## Documentation

- [Getting Started Guide](docs/GETTING_STARTED.md) - Step-by-step setup and first analysis
- [API Reference](docs/API.md) - Complete tool documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Contributing](CONTRIBUTING.md) - How to contribute to the project

## License

MIT License - See LICENSE file for details
