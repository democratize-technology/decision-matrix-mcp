# Decision Matrix MCP - Custom Agents

This directory contains specialized agents tailored for the decision-matrix-mcp project. Each agent addresses specific development, testing, and maintenance needs of this MCP server.

## Available Agents

### 🧪 mcp-integration-tester
Comprehensive testing specialist for MCP protocol implementation. Tests all tool endpoints, validates stdio/stderr separation, mocks LLM responses for deterministic testing, and ensures proper session lifecycle management.

**Use when**: Writing tests, debugging MCP tool behavior, or ensuring protocol compliance.

### 🛡️ llm-resilience-engineer
LLM backend resilience specialist. Implements circuit breakers, connection pooling, fallback strategies, and comprehensive timeout handling for multi-backend LLM orchestration.

**Use when**: Improving reliability, handling LLM failures, or optimizing backend connections.

### 🔒 mcp-security-auditor
Security specialist for MCP servers. Implements rate limiting, prompt injection detection, input validation hardening, and authentication hooks.

**Use when**: Reviewing security, implementing access controls, or hardening against attacks.

### ⚡ decision-performance-optimizer
Performance optimization specialist for parallel decision analysis. Profiles asyncio operations, implements caching strategies, optimizes memory usage, and creates benchmarks.

**Use when**: Improving performance, reducing latency, or handling large-scale evaluations.

### 📚 mcp-docs-generator
Documentation specialist for MCP servers. Generates API references, creates example scenarios, documents Claude Desktop integration, and writes comprehensive guides.

**Use when**: Creating documentation, writing examples, or explaining functionality.

## Usage

These agents can be invoked directly:
```
/chat-with mcp-integration-tester
```

Or will be automatically selected based on context when using the Task tool.

## Agent Selection Guide

- **Testing Issues** → mcp-integration-tester
- **Reliability Problems** → llm-resilience-engineer  
- **Security Concerns** → mcp-security-auditor
- **Performance Issues** → decision-performance-optimizer
- **Documentation Needs** → mcp-docs-generator

Each agent has specific expertise and tools optimized for their domain.