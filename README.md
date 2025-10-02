# LangStruct

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DSPy 3.0](https://img.shields.io/badge/DSPy-3.0+-orange.svg)](https://github.com/stanfordnlp/dspy)
[![Docs](https://img.shields.io/badge/docs-langstruct.dev-blue.svg)](https://langstruct.dev)

**Extract structured data from any text – automatically optimized for any LLM**

You know when you have a pile of invoices, medical records, or contracts, and you need to pull out specific fields or metadata? That's what this does.

Point it at messy text, show it an example of what you want, and get back typed JSON. Built on DSPy so it auto-optimizes the prompts for you. You can even parse user queries into filters for your RAG system.

```python
from langstruct import LangStruct

# Just show an example of what you want
extractor = LangStruct(example={
    "invoice_number": "INV-001",
    "amount": 1250.00,
    "due_date": "2024-03-15",
})

# Extract from any similar text
result = extractor.extract("Invoice INV-2024-789 for $3,450 is due April 20th, 2024")
print(result.entities)
# {"invoice_number": "INV-2024-789", "amount": 3450.00, "due_date": "2024-04-20"}
```

## Why use this

- **Auto-optimizes prompts** – DSPy's optimizers do the tuning for you, learns from your examples
- **Type-safe with Pydantic** – Full validation and type hints
- **End-to-end RAG** – Extract structured metadata from docs AND parse user queries into filters
- **Shows sources** – Character-level mapping back to original text
- **Works with any LLM** – OpenAI, Claude, Gemini, or local models. Switch providers without rewriting code.

We built this because we got tired of writing extraction code over and over. The DSPy foundation means your extractor improves with feedback, and you're not locked into one LLM provider.

## Installation

```bash
pip install langstruct

# You'll need an API key for any LLM provider
export GOOGLE_API_KEY="your-key"
# OR
export OPENAI_API_KEY="your-key"
# OR
export ANTHROPIC_API_KEY="your-key"
```

Works with free tier APIs, paid providers, or local models (Ollama). See [docs](https://langstruct.dev/installation/) for all options.

## Basic usage

Two ways to define what you want:

**Option 1: Show an example** (easiest)
```python
extractor = LangStruct(example={
    "name": "Dr. Sarah Johnson",
    "age": 34,
    "specialty": "cardiology"
})
```

**Option 2: Use a Pydantic schema** (more control)
```python
from pydantic import BaseModel, Field

class PersonSchema(BaseModel):
    name: str = Field(description="Full name")
    age: int
    specialty: str = Field(description="Medical specialty")

extractor = LangStruct(schema=PersonSchema)
```

Then extract:
```python
text = "Dr. Sarah Johnson, 34, is a cardiologist at Boston General."
result = extractor.extract(text)

print(result.entities)  # {"name": "Dr. Sarah Johnson", "age": 34, "specialty": "cardiology"}
print(result.confidence)  # 0.94
```

## RAG integration

If you're building a RAG system, LangStruct helps with both sides:

**1. Extract structured metadata from documents:**
```python
metadata_extractor = LangStruct(example={
    "company": "Apple Inc.",
    "revenue": 125.3,
    "quarter": "Q3 2024"
})

# Add to your vector DB with structured filters
metadata = metadata_extractor.extract(document).entities
```

**2. Parse user queries into filters:**
```python
query = "Q3 2024 tech companies with revenue over $100B"
parsed = metadata_extractor.query(query)

print(parsed.semantic_terms)  # ["tech companies"]
print(parsed.structured_filters)  # {"quarter": "Q3 2024", "revenue": {"$gte": 100.0}}

# Now query your vector DB with both
results = vector_db.search(
    semantic=parsed.semantic_terms,
    filters=parsed.structured_filters
)
```

This gives you precise retrieval instead of just semantic search. See [RAG integration guide](https://langstruct.dev/rag-integration/) for details.

## Boost accuracy with refinement

Add `refine=True` to get 15-30% better accuracy. It generates multiple candidates and picks the best one:

```python
result = extractor.extract(text, refine=True)
# Takes longer but significantly more accurate
```

## Source tracking

See exactly where each field came from:

```python
for field, spans in result.sources.items():
    for span in spans:
        print(f"{field}: '{span.text}' at position {span.start}-{span.end}")
```

This is helpful for:
- Debugging extraction issues
- Building citation systems
- Compliance/audit trails

## Batch processing

```python
documents = [doc1, doc2, doc3, ...]  # Hundreds or thousands

results = extractor.extract(
    documents,
    max_workers=8,         # Parallel processing
    rate_limit=60,         # Respect API limits
    show_progress=True     # Show progress bar
)
```

Handles retries with exponential backoff automatically.

## What it's good at

We use this for:
- Processing invoices, receipts, purchase orders
- Extracting data from medical records and clinical notes
- Parsing contracts for key terms and dates
- Structuring customer feedback and reviews
- Pulling metrics from financial reports

## What it's NOT good at

- Real-time applications (LLM latency adds up)
- Perfect accuracy on every extraction (it's an LLM, not a regex)
- Complex tables with merged cells or unusual layouts
- Documents where formatting is critical to meaning

For those cases you might want a traditional parser or OCR solution.

## Comparison with alternatives

### LangStruct vs LangExtract

Both are solid for structured extraction. Here's how they differ:

| Feature               | LangStruct                          | LangExtract                                          |
| --------------------- | ----------------------------------- | ---------------------------------------------------- |
| **Optimization**      | ✅ Automatic (DSPy MIPROv2)          | ❌ Manual prompt tuning                               |
| **Refinement**        | ✅ Best-of-N + iterative improvement | ⚠️ Multi-pass extraction; no Best-of-N/judge pipeline |
| **Schema Definition** | ✅ From examples OR Pydantic         | ⚠️ Prompt + examples (no Pydantic models)             |
| **Source Grounding**  | ✅ Character-level tracking          | ✅ Character-level tracking                           |
| **Confidence Scores** | ✅ Built-in                          | ⚠️ Not surfaced as scores                             |
| **Query Parsing**     | ✅ Bidirectional (docs + queries)    | ❌ Documents only                                     |
| **Model Support**     | ✅ Any LLM (via DSPy/LiteLLM)        | ✅ Gemini, OpenAI, local via Ollama; extensible       |
| **Learning Curve**    | ✅ Simple (example-based)            | ⚠️ Requires prompt + example design                   |
| **Performance**       | ✅ Self-optimizing                   | Depends on manual tuning                             |
| **Project Type**      | Community open-source               | Google open-source                                   |

Use LangStruct if you want automatic optimization and don't want to tune prompts. Use LangExtract if you prefer direct control or want Google's backing.

## Advanced features

Once you've got the basics working, there's more:

**Custom optimization** on your data:
```python
extractor.optimize(
    texts=your_examples,
    expected_results=expected_outputs
)
```

**Save and reuse** extractors:
```python
extractor.save("./my_extractor")
loaded = LangStruct.load("./my_extractor")
```

**Visualize extractions** with highlighted sources:
```python
from langstruct import HTMLVisualizer

viz = HTMLVisualizer()
viz.save_visualization(text, result, "output.html")
```

**Process huge documents** with chunking:
```python
from langstruct import ChunkingConfig

config = ChunkingConfig(
    max_tokens=1500,
    overlap_tokens=150,
    preserve_sentences=True
)

extractor = LangStruct(schema=YourSchema, chunking_config=config)
```

See [examples](https://langstruct.dev/examples/) for medical records, financial docs, legal contracts, and more.

## Common issues

**"No API key found"**
```bash
# Make sure it's exported in your current shell
echo $GOOGLE_API_KEY

# For persistence, add to ~/.bashrc or ~/.zshrc
export GOOGLE_API_KEY="your-key"
```

**Rate limits / costs getting too high**
```python
# Control API usage
results = extractor.extract(
    texts,
    rate_limit=30,           # Calls per minute
    max_workers=2,           # Fewer parallel calls
    refine=False             # Skip refinement to save calls
)
```

**Low extraction quality**
- Try `refine=True` for better accuracy
- Use more specific field descriptions
- Optimize on a few examples with `.optimize()`

## Contributing

We're open to contributions. If you find bugs or have ideas:
- [Open an issue](https://github.com/langstruct-ai/langstruct/issues)
- [Start a discussion](https://github.com/langstruct-ai/langstruct/discussions)
- Submit a PR (see [CONTRIBUTING.md](CONTRIBUTING.md))

Development setup:
```bash
git clone https://github.com/langstruct-ai/langstruct.git
cd langstruct
uv sync --extra dev
uv run pytest
```

## License

MIT – see [LICENSE](LICENSE)

## Credits

Built on [DSPy](https://github.com/stanfordnlp/dspy) for self-optimizing prompts and [Pydantic](https://pydantic.dev) for schemas.
