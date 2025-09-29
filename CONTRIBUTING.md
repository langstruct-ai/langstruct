# Contributing to LangStruct

Thank you for your interest in contributing to LangStruct! We welcome contributions from everyone.

## Development Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/langstruct-ai/langstruct.git
   cd langstruct
   ```

2. **Install dependencies**
   ```bash
   # With uv (recommended)
   uv sync --extra dev

   # Or with pip
   pip install -e ".[dev]"
   ```

3. **Set up API keys for testing**
   ```bash
   # Get a free API key from aistudio.google.com
   export GOOGLE_API_KEY="your-key-here"

   # Or use OpenAI
   export OPENAI_API_KEY="your-key-here"
   ```

4. **Set up pre-commit hooks (optional but recommended)**
   ```bash
   uv run pre-commit install
   ```

5. **Run tests**
   ```bash
   uv run pytest
   ```

## Development Workflow

### Before Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run the test suite**
   ```bash
   uv run pytest -v
   ```

### Code Quality

We maintain high code quality standards:

- **Formatting**: Use `black` and `isort`
  ```bash
  uv run black .
  uv run isort .
  ```

- **Type Checking**: Use `mypy`
  ```bash
  uv run mypy langstruct/
  ```

- **Testing**: Write tests for new features
  ```bash
  uv run pytest tests/
  ```

### Making Changes

1. **Follow existing patterns**: Look at similar code in the project
2. **Update docstrings**: All public functions should have comprehensive docstrings
3. **Add tests**: Include tests for new functionality
4. **Update examples**: Add examples for new features when appropriate

### Submitting Changes

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** with:
   - Clear title describing the change
   - Description of what you changed and why
   - Links to any relevant issues

## Project Structure

```
langstruct/
├── langstruct/           # Main source code
│   ├── api.py           # Main LangStruct class
│   ├── core/            # Core extraction modules
│   ├── optimizers/      # DSPy optimization modules
│   ├── providers/       # LLM provider interfaces
│   └── visualization/   # HTML visualization tools
├── tests/               # Test suite
├── examples/            # Example scripts
└── docs/               # Documentation source
```

## Types of Contributions

### 🐛 Bug Reports
- Use the bug report template
- Include reproduction steps
- Provide error messages and logs

### ✨ Feature Requests
- Use the feature request template
- Explain the use case
- Consider implementation approach

### 📝 Documentation
- Fix typos, improve clarity
- Add examples and tutorials
- Update API documentation

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Test coverage improvements

## Testing Guidelines

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=langstruct

# Run integration tests (requires API key)
uv run pytest -m integration
```

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names
- Test both success and error cases
- Mock external API calls for unit tests
- Use integration tests sparingly (they cost money)

## Documentation

### Building Documentation

```bash
cd docs
pnpm install
pnpm dev  # Local development server
pnpm build  # Build static site
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Test all code examples
- Follow existing structure and style

## Code Style

### Python Code Style

We follow these conventions:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for consistent import ordering
- **Type hints**: Required for all public functions
- **Docstrings**: Google style docstrings for all public functions

### Example

```python
def extract_entities(text: str, schema: Type[Schema]) -> ExtractionResult:
    """Extract structured entities from text using the provided schema.

    Args:
        text: Input text to extract from
        schema: Pydantic schema defining extraction structure

    Returns:
        ExtractionResult with entities and metadata

    Raises:
        ExtractionError: If extraction fails
    """
    # Implementation here
    pass
```

## Release Process

Releases are automated via GitHub Actions when tags are pushed:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create and push tag: `git tag v0.1.1 && git push origin v0.1.1`

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

---

Thank you for contributing to LangStruct! 🚀
