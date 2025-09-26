---
name: Bug report
about: Create a report to help us improve LangStruct
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## Reproduction Steps
Steps to reproduce the behavior:
1. Go to '...'
2. Run '...'
3. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Error Message
```
Paste any error messages or stack traces here
```

## Environment
- **LangStruct version**: [e.g., 0.1.0]
- **Python version**: [e.g., 3.12.0]
- **Operating System**: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- **LLM Provider**: [e.g., OpenAI GPT-4, Google Gemini]

## Minimal Code Example
```python
from langstruct import LangStruct

# Minimal code that reproduces the issue
extractor = LangStruct(example={"name": "John"})
result = extractor.extract("Some text")  # This fails
```

## Additional Context
Add any other context about the problem here, such as:
- Does this happen with all models or just specific ones?
- Is this related to a specific type of text/document?
- Any workarounds you've found?

## Checklist
- [ ] I have searched existing issues to make sure this isn't a duplicate
- [ ] I have provided a minimal code example that reproduces the issue
- [ ] I have included the complete error message/stack trace
- [ ] I have specified my environment details