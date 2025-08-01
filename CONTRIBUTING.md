# Contributing to Decision Matrix MCP

Thank you for your interest in contributing to Decision Matrix MCP! We welcome contributions from the community and are excited to work with you.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Push to your fork and submit a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/democratize-technology/decision-matrix-mcp.git
cd decision-matrix-mcp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Development Workflow

### 1. Before You Start

- Check existing issues and pull requests to avoid duplicating work
- For significant changes, open an issue first to discuss your proposal
- Ensure your development environment is properly set up

### 2. Making Changes

- Follow the existing code style and conventions
- Write clear, self-documenting code with appropriate comments
- Keep commits atomic and write clear commit messages
- Update documentation as needed

### 3. Code Quality Standards

We maintain high code quality standards. Before submitting:

```bash
# Run linting
ruff check src/

# Fix linting issues
ruff check src/ --fix

# Check type annotations
mypy src/

# Format code
black src/

# Run tests
pytest tests/
```

### 4. Testing

- Add tests for new functionality
- Ensure all existing tests pass
- Aim for high test coverage
- Test with different LLM backends if applicable

## Pull Request Process

1. **Update Documentation**: Ensure README.md and other docs reflect your changes
2. **Add Tests**: Include tests that demonstrate your fix or feature works
3. **Update CHANGELOG.md**: Add a note about your changes in the Unreleased section
4. **Clean Commits**: Squash or organize commits for clarity
5. **PR Description**: Use the PR template and provide a clear description

## Commit Message Guidelines

We follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example: `feat: add support for custom evaluation prompts`

## Code Style Guide

### Python Style

- Follow PEP 8 with a line length of 100 characters
- Use type hints for all function signatures
- Prefer f-strings for string formatting
- Use descriptive variable names

### Import Organization

```python
# Standard library imports
import asyncio
import logging

# Third-party imports
from pydantic import BaseModel

# Local imports
from .models import Criterion
```

### Error Handling

- Use custom exceptions from `exceptions.py`
- Provide clear error messages
- Log errors appropriately (stderr only in MCP context)

## Architecture Guidelines

### MCP Protocol Compliance

- Never write to stdout (reserved for MCP protocol)
- All logging must go to stderr
- Handle connection lifecycle properly

### Thread Safety

- Use proper async/await patterns
- Avoid global mutable state
- Implement proper session isolation

### LLM Backend Support

When adding LLM backend features:
- Follow the existing backend pattern
- Handle authentication properly
- Implement retry logic
- Add appropriate error handling

## Testing Guidelines

### Unit Tests

- Test individual components in isolation
- Mock external dependencies
- Cover edge cases and error conditions

### Integration Tests

- Test complete workflows
- Verify MCP protocol compliance
- Test with different configurations

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include parameter descriptions and return types
- Provide usage examples where helpful

### User Documentation

- Update README.md for user-facing changes
- Add examples for new features
- Keep installation instructions current

## Community

### Getting Help

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Check existing issues before creating new ones

### Review Process

- All submissions require review
- Be patient and responsive to feedback
- Address reviewer comments promptly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions help make Decision Matrix MCP better for everyone. We appreciate your time and effort in improving this project!