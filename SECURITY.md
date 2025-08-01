# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Decision Matrix MCP seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please do:

- **Email us at**: security@democratize.technology
- **Include**: A description of the vulnerability and steps to reproduce
- **Be patient**: We'll acknowledge receipt within 48 hours
- **Work with us**: We may ask for additional information

### Please don't:

- Don't disclose the vulnerability publicly until we've addressed it
- Don't exploit the vulnerability beyond what's necessary to demonstrate it

## What to Include in Your Report

The more information you provide, the quicker we can validate and fix the issue:

1. **Type of issue** (e.g., injection, authentication bypass, data exposure)
2. **Affected components** (e.g., specific MCP tools, LLM backend handlers)
3. **Step-by-step reproduction instructions**
4. **Proof-of-concept code** (if possible)
5. **Impact assessment** - what can an attacker achieve?
6. **Suggested fix** (if you have one)

## Our Commitment

When you report a vulnerability:

1. **Acknowledgment**: We'll confirm receipt within 48 hours
2. **Assessment**: We'll assess the vulnerability and determine severity
3. **Updates**: We'll keep you informed of our progress
4. **Fix**: We'll work on a fix and test it thoroughly
5. **Credit**: We'll credit you in the security advisory (unless you prefer to remain anonymous)
6. **Disclosure**: We'll coordinate disclosure timing with you

## Security Considerations for MCP Servers

### Input Validation

- All user inputs are validated using Pydantic models
- Session IDs are validated as proper UUIDs
- Option and criterion names have length limits
- Weights are constrained to valid ranges (0.1-10.0)

### LLM Prompt Safety

- System prompts are not user-modifiable by default
- Custom prompts go through validation
- LLM responses are parsed defensively

### Resource Limits

- Maximum number of concurrent sessions
- Maximum options and criteria per session
- Evaluation timeouts to prevent hanging

### Authentication & Authorization

Decision Matrix MCP relies on the MCP client (e.g., Claude Desktop) for authentication. Ensure:

- Your MCP client is properly configured
- You trust the MCP client you're using
- API keys for LLM backends are properly secured

## Security Best Practices for Users

1. **API Key Management**
   - Never commit API keys to version control
   - Use environment variables for sensitive configuration
   - Rotate keys regularly

2. **LLM Backend Security**
   - Use official LLM provider endpoints
   - Verify SSL certificates
   - Monitor API usage for anomalies

3. **Session Management**
   - Don't share session IDs
   - Clear sessions when done
   - Monitor active sessions

## Known Security Limitations

1. **Trust Boundary**: MCP servers trust the MCP client completely
2. **No Built-in Auth**: Authentication must be handled by the MCP client
3. **LLM Responses**: We cannot guarantee LLM responses are safe or accurate

## Security Updates

Security updates will be released as patch versions. Monitor:

- GitHub Security Advisories
- Release notes
- The CHANGELOG.md file

## Contact

For security concerns, contact: security@democratize.technology

For general bugs, please use the GitHub issue tracker.

Thank you for helping keep Decision Matrix MCP secure!