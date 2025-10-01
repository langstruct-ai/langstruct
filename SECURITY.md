# Security Policy

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues via one of these methods:

1. **Email**: Send details to the maintainers (check `pyproject.toml` for contact info)
2. **GitHub Security Advisories**: Use the [private vulnerability reporting feature](https://github.com/langstruct-ai/langstruct/security/advisories/new)

### What to Include

Please provide:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if you have one)

## Security Best Practices

When using LangStruct:

### API Key Management

- **Never commit API keys** to version control
- Use environment variables for all credentials
- Rotate API keys regularly
- Use separate keys for development/production

```python
# ✅ GOOD: Use environment variables
import os
extractor = LangStruct(schema=MySchema)  # Reads from env vars

# ❌ BAD: Hardcoded keys
extractor = LangStruct(schema=MySchema, api_key="sk-...")
```

### Input Validation

- **Sanitize user input** before extraction
- Set reasonable limits on text length
- Validate extraction results before using them
- Be cautious with user-provided schemas or examples

### LLM Provider Security

- Review your LLM provider's security policies
- Understand data retention policies
- Use appropriate models for sensitive data
- Consider local models (Ollama) for highly sensitive content

### Production Deployment

- Use rate limiting to prevent abuse
- Monitor API usage and costs
- Implement proper error handling
- Don't expose raw LLM outputs to end users without validation
- Log extractions for audit trails (without logging sensitive data)

### Dependencies

- Keep LangStruct and dependencies updated
- Review security advisories for DSPy and other dependencies
- Use `pip-audit` or similar tools to scan for vulnerable packages

```bash
# Check for vulnerabilities
pip install pip-audit
pip-audit
```

## Known Security Considerations

### Prompt Injection

Like all LLM-based systems, LangStruct is potentially vulnerable to prompt injection attacks. Mitigations:

- Validate and sanitize all user inputs
- Use structured outputs (Pydantic schemas) to constrain results
- Implement output validation
- Consider using separate models for untrusted content

### Data Privacy

- LangStruct sends text to LLM providers for processing
- Text may be logged by providers (depending on their policies)
- For sensitive data:
  - Use providers with strong privacy guarantees
  - Consider local/on-premise models (Ollama, vLLM)
  - Implement PII redaction before extraction
  - Review provider data retention policies

### Supply Chain

- LangStruct depends on DSPy, LiteLLM, and Pydantic
- We monitor dependencies for security issues
- Pin your dependencies in production
- Verify package signatures when possible

## Contact

For security-related questions that aren't vulnerabilities:

- Open a discussion on GitHub Discussions
- Check existing documentation and issues first

Thank you for helping keep LangStruct secure!
