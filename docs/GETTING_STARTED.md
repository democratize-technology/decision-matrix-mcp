# Getting Started with Decision Matrix MCP

This guide will walk you through setting up and using Decision Matrix MCP for structured decision analysis.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Your First Decision Analysis](#your-first-decision-analysis)
5. [Advanced Usage](#advanced-usage)
6. [Best Practices](#best-practices)

## Prerequisites

Before you begin, ensure you have:

- Python 3.10 or higher
- Claude Desktop (or another MCP-compatible client)
- An LLM backend configured (AWS Bedrock, OpenAI, or Ollama)

## Installation

### Option 1: Install from Source

```bash
# Clone the repository
git clone https://github.com/democratize-technology/decision-matrix-mcp.git
cd decision-matrix-mcp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

### Option 2: Using Docker

```bash
# Clone the repository
git clone https://github.com/democratize-technology/decision-matrix-mcp.git
cd decision-matrix-mcp

# Build and run with Docker Compose
docker-compose up -d
```

### Option 3: Direct Installation

```bash
pip install decision-matrix-mcp
```

## Configuration

### 1. Configure Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "decision-matrix": {
      "command": "python",
      "args": ["-m", "decision_matrix_mcp"],
      "env": {
        "AWS_PROFILE": "default",
        "LITELLM_API_KEY": "your-api-key",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

### 2. Configure LLM Backend

Choose and configure at least one LLM backend:

#### AWS Bedrock (Recommended)

```bash
# Configure AWS credentials
aws configure

# Set environment variable
export AWS_PROFILE=default
export AWS_REGION=us-east-1
```

#### OpenAI via LiteLLM

```bash
export OPENAI_API_KEY=sk-...
# or
export LITELLM_API_KEY=sk-...
```

#### Ollama (Local)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama2

# Ollama runs on http://localhost:11434 by default
```

### 3. Verify Installation

```bash
# Test the server starts correctly
python -m decision_matrix_mcp
```

You should see:
```
Starting Decision Matrix MCP server...
```

## Your First Decision Analysis

Let's walk through a complete decision analysis example using Claude Desktop.

### Step 1: Start a Decision Session

Ask Claude:
```
I need to choose a web framework for my new project.
My options are: React, Vue, Angular, and Svelte.
```

Claude will use the `start_decision_analysis` tool to create a session.

### Step 2: Add Evaluation Criteria

```
Add these criteria:
- Learning Curve (weight: 2.0) - How easy it is to learn
- Performance (weight: 3.0) - Runtime performance and bundle size
- Ecosystem (weight: 1.5) - Available libraries and tools
- Community Support (weight: 2.0) - Documentation and help availability
```

### Step 3: Run the Evaluation

```
Now evaluate all the frameworks against these criteria
```

Claude will run parallel evaluations across all framework-criterion pairs.

### Step 4: View Results

```
Show me the decision matrix with rankings
```

You'll get a comprehensive matrix showing:
- Individual scores for each framework-criterion pair
- Weighted scores based on criterion importance
- Total weighted scores and rankings
- A recommendation based on the analysis

## Advanced Usage

### Using Custom Evaluation Prompts

You can provide custom prompts for specific criteria:

```
Add a criterion for "TypeScript Support" with a custom prompt:
"Evaluate the quality of TypeScript support including type definitions,
IDE integration, and community tooling. Consider both official support
and third-party contributions."
```

### Adding Options Mid-Analysis

```
I just remembered Preact - can you add it as another option?
```

Then re-run the evaluation to include the new option.

### Working with Multiple LLM Backends

You can specify different backends for different criteria:

```
Add "Innovation Potential" criterion using GPT-4 for evaluation
Add "Security Track Record" using Claude on Bedrock
```

### Session Management

```
List all my active decision sessions
```

```
Clear all sessions to start fresh
```

## Best Practices

### 1. Choosing Good Criteria

- **Be Specific**: "Performance" → "Page load time under 3 seconds"
- **Make it Measurable**: Criteria should be evaluable objectively
- **Avoid Overlap**: Each criterion should measure something unique
- **Consider Stakeholders**: Include criteria important to all stakeholders

### 2. Setting Appropriate Weights

- **1.0**: Standard importance (default)
- **0.1-0.5**: Nice to have
- **1.5-2.5**: Important
- **3.0-5.0**: Critical
- **5.0+**: Deal-breaker

### 3. Interpreting Results

- **Clear Winner**: One option significantly ahead (>20% margin)
- **Close Call**: Multiple options within 10% of each other
- **Abstentions**: When criteria don't apply to certain options
- **Review Justifications**: Always read the reasoning, not just scores

### 4. Iterative Refinement

1. Start with broad criteria
2. Review initial results
3. Add more specific criteria if needed
4. Adjust weights based on importance
5. Re-evaluate with refined criteria

### 5. Handling Edge Cases

**Too Many Options**: Group similar options first, then evaluate groups

**Subjective Criteria**: Use specific, scenario-based prompts

**Missing Information**: The LLM will note uncertainty in justifications

## Example Scenarios

### Technology Selection

```python
# Choosing a database
options = ["PostgreSQL", "MongoDB", "DynamoDB", "Redis"]
criteria = [
    ("ACID Compliance", 3.0),
    ("Scalability", 2.5),
    ("Query Flexibility", 2.0),
    ("Operational Complexity", 1.5),
    ("Cost at Scale", 2.0)
]
```

### Business Decisions

```python
# Market expansion strategy
options = ["Enter Europe", "Enter Asia", "Expand in Americas", "Stay Local"]
criteria = [
    ("Market Size", 2.5),
    ("Regulatory Complexity", 2.0),
    ("Cultural Fit", 1.5),
    ("Investment Required", 3.0),
    ("Risk Level", 2.5)
]
```

### Personal Decisions

```python
# Career move evaluation
options = ["Stay Current Job", "Startup Offer", "Big Tech Offer", "Freelance"]
criteria = [
    ("Compensation", 2.0),
    ("Work-Life Balance", 3.0),
    ("Growth Potential", 2.5),
    ("Job Security", 1.5),
    ("Interest/Passion", 3.0)
]
```

## Next Steps

1. **Explore the API**: See [API Documentation](API.md) for all available tools
2. **Understand the Architecture**: Read [Architecture Guide](ARCHITECTURE.md)
3. **Contribute**: Check [Contributing Guidelines](../CONTRIBUTING.md)
4. **Get Help**: Use GitHub Discussions for questions

## Tips for Success

1. **Start Simple**: Begin with 3-4 options and 3-4 criteria
2. **Trust the Process**: Let the systematic evaluation reveal insights
3. **Question Assumptions**: If results surprise you, examine your criteria
4. **Document Decisions**: Save session IDs for future reference
5. **Iterate**: Refine criteria based on initial results

Happy decision making! 🎯
