# LangStruct Examples

This directory contains progressive examples demonstrating LangStruct features from basic to advanced usage.

## 📚 Progressive Learning Path

Start with `01_hello_world.py` and work through the numbered examples:

### Beginner Examples
- **`01_hello_world.py`** - Simplest possible LangStruct usage (start here!)
- **`02_with_schema.py`** - Using custom Pydantic schemas for more control
- **`03_source_tracking.py`** - Understanding where extracted data comes from
- **`04_batch_processing.py`** - Processing multiple documents efficiently

### Advanced Examples
- **`05_error_handling.py`** - Production-ready error handling patterns
- **`06_rag_integration.py`** - RAG system enhancement with structured metadata
- **`07_optimization.py`** - Self-optimization for better accuracy over time
- **`08_query_parsing.py`** - Bidirectional RAG with intelligent query parsing

### Refinement Examples
- **`09_refinement_basics.py`** - Boost accuracy with Best-of-N and iterative refinement
- **`10_refinement_advanced.py`** - Custom strategies, judges, and configuration options
- **`11_refinement_comparison.py`** - Before/after accuracy comparison across document types
- **`12_refinement_budget.py`** - Cost control and budget management for production

### Persistence Examples
- **`13_save_load_extractors.py`** - Save and load extractors with full state preservation

## Running Examples

### Recommended Learning Order

```bash
# Start here - simplest example
uv run examples/01_hello_world.py

# Learn custom schemas
uv run examples/02_with_schema.py

# Understand source tracking
uv run examples/03_source_tracking.py

# Process multiple documents
uv run examples/04_batch_processing.py

# Production error handling
uv run examples/05_error_handling.py

# RAG system enhancement
uv run examples/06_rag_integration.py

# Performance optimization
uv run examples/07_optimization.py

# Bidirectional RAG with query intelligence
uv run examples/08_query_parsing.py

# Boost accuracy with refinement
uv run examples/09_refinement_basics.py

# Advanced refinement configuration
uv run examples/10_refinement_advanced.py

# Accuracy comparison study
uv run examples/11_refinement_comparison.py

# Budget and cost management
uv run examples/12_refinement_budget.py

# Save and load extractors
uv run examples/13_save_load_extractors.py
```

## Generated Files

Examples will create output files in the current directory:
- `*.html` - Interactive visualizations
- `*.json` - JSON export files
- `*.csv` - CSV export files
- `*.xlsx` - Excel files (if pandas installed)
- `saved_extractors/` - Saved extractor directories (from persistence examples)

These output files are automatically ignored by git (see `.gitignore`).

## API Key Setup

Examples require an API key for the LLM provider:

```bash
# Choose one API key:
export GOOGLE_API_KEY="your-key-here"       # Google Gemini (free tier)
export OPENAI_API_KEY="sk-your-key-here"    # OpenAI
export ANTHROPIC_API_KEY="sk-ant-key"       # Claude models
```

For more provider options, see the [installation guide](https://langstruct.dev/installation/).
