# Changelog

All notable changes to Decision Matrix MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive OSS documentation and community files
- Issue templates for bugs, features, and questions
- Pull request template
- Contributing guidelines
- Security policy
- MIT license headers to all source files
- NOTICE file with dependency attributions

### Changed
- Updated author information to Democratize Technology organization
- Made run.sh script portable by removing hardcoded paths

### Fixed
- All linting errors (26 whitespace issues)
- All type checking errors (11 mypy issues)
- Test suite compatibility with dependency injection

### Security
- Implemented proper input validation for all user inputs
- Added session validation guards
- Documented security best practices

## [0.1.0] - 2025-01-01

### Added
- Initial release of Decision Matrix MCP
- Multi-criteria decision analysis with weighted scoring
- Parallel evaluation across all option-criterion pairs
- Support for multiple LLM backends:
  - AWS Bedrock (Claude models)
  - LiteLLM (OpenAI, Anthropic)
  - Ollama (local models)
- Thread-based conversation context per criterion
- Graceful handling of abstentions
- Session management with TTL and cleanup
- Comprehensive test suite with ~100% coverage
- FastMCP-based server implementation
- Dependency injection architecture

### Features
- Create decision analysis sessions with topics and options
- Add weighted evaluation criteria
- Run parallel evaluations across all combinations
- Generate decision matrices with rankings
- Support for custom evaluation prompts
- Proper MCP stdio transport handling

### Technical
- Type-safe implementation with Pydantic models
- Comprehensive error handling
- Retry logic with exponential backoff
- Resource limits and validation
- Thread-safe session isolation

[Unreleased]: https://github.com/democratize-technology/decision-matrix-mcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/democratize-technology/decision-matrix-mcp/releases/tag/v0.1.0
