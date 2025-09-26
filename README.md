# 🧠 LangStruct

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DSPy 3.0](https://img.shields.io/badge/DSPy-3.0+-orange.svg)](https://github.com/stanfordnlp/dspy)
[![Docs](https://img.shields.io/badge/docs-langstruct.dev-blue.svg)](https://langstruct.dev)

**Extract structured data from any text – no prompt engineering required**

> **TL;DR:** Extract structured information from any text - documents, emails, reports, transcripts - into clean JSON data. No prompt engineering required. Built on DSPy 3.0 for automatic optimization.

LangStruct turns messy, unstructured text into clean, typed, validated data. Whether you're processing medical records, financial documents, customer feedback, or legal contracts, LangStruct extracts exactly what you need with source tracking and confidence scores.


## What LangStruct Does

LangStruct extracts **structured information** from **unstructured text**:

```
Input (messy text)                    →  Output (clean data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Dr. Smith diagnosed the 45-year-old  →  {
 patient with Type 2 diabetes and     →    "physician": "Dr. Smith",
 prescribed metformin 500mg twice     →    "patient_age": 45,
 daily. Follow-up in 3 months."       →    "diagnosis": "Type 2 diabetes",
                                      →    "medication": "metformin",
                                      →    "dosage": "500mg",
                                      →    "frequency": "twice daily",
                                      →    "followup": "3 months"
                                      →  }
```

## Key Features

### Core Capabilities
- **Automatic Optimization**: Uses DSPy MIPROv2 for prompt optimization
- **Refinement System**: Best-of-N + iterative improvement for 15-30% accuracy boost
- **Source Tracking**: Character-level mapping of extracted data to source text
- **Schema Generation**: Create Pydantic schemas from examples
- **Type Safety**: Full Pydantic validation and type hints
- **Model Support**: Compatible with OpenAI, Anthropic, Google, Ollama, and other LLMs
- **Persistence**: Save and load extractors with full state preservation
- **Visualization**: HTML output with source highlighting

## Quick Example

> **⚠️ API Key Required**: You need an API key to run LangStruct. **[Get one free here →](https://aistudio.google.com/app/apikey)** or see [setup options](#api-key-setup) below.

```python
from langstruct import LangStruct

# Define what you want to extract with a simple example
extractor = LangStruct(example={
    "invoice_number": "INV-001",
    "amount": 1250.00,
    "due_date": "2024-03-15",
    "line_items": ["Widget A", "Service B"]
})

# Extract from any text
text = """
Dear Customer,

Your invoice INV-2024-789 for $3,450.00 is due on April 20th, 2024.

Items:
- Premium Widget Set
- Installation Service
- Extended Warranty

Thank you for your business!
"""

result = extractor.extract(text)
print(result.entities)
# {
#   "invoice_number": "INV-2024-789",
#   "amount": 3450.00,
#   "due_date": "2024-04-20",
#   "line_items": ["Premium Widget Set", "Installation Service", "Extended Warranty"]
# }

# Boost accuracy with refinement (15-30% improvement)
result = extractor.extract(text, refine=True)
print(f"Confidence: {result.confidence:.1%}")  # Higher confidence score
```

## Quick Start

### 1. Get an API Key (Required)

**Choose one option:**

<div align="center">

| Provider       | Get Key                                                  | Best For                  |
| -------------- | -------------------------------------------------------- | ------------------------- |
| Google Gemini  | [Get Free Key →](https://aistudio.google.com/app/apikey) | Fast & generous free tier |
| OpenAI         | [Get Key →](https://platform.openai.com/api-keys)        | GPT models                |
| Anthropic      | [Get Key →](https://console.anthropic.com/)              | Claude models             |
| Local (Ollama) | [Install Ollama →](https://ollama.ai/)                   | Privacy, no API needed    |

</div>

**Set your API key:**
```bash
# Google Gemini (free)
export GOOGLE_API_KEY="your-key-here"

# Or use others:
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### 2. Installation

Install from PyPI:

```bash
# uv (recommended)
uv add langstruct

# or pip
pip install langstruct

# Optional extras
pip install langstruct[viz]        # Visualization tools (HTML helpers)
pip install langstruct[examples]   # Example integrations (ChromaDB, LangChain)
pip install langstruct[parallel]   # tqdm for nicer progress bars
pip install langstruct[dev]        # Test and lint toolchain
pip install langstruct[all]        # Everything above
```

### 3. Basic Usage

```python
from langstruct import LangStruct

# Create an extractor from an example (simplest approach)
extractor = LangStruct(example={
    "name": "Dr. Sarah Johnson",
    "age": 34,
    "location": "Cambridge, Massachusetts",
    "occupation": "cardiologist"
})

# Extract structured data from text
text = """
Dr. Sarah Johnson is a 34-year-old cardiologist working at Boston General Hospital.
She currently lives in Cambridge, Massachusetts, with her family.
"""

result = extractor.extract(text)

print(result.entities)
# Output: {
#   "name": "Dr. Sarah Johnson",
#   "age": 34,
#   "location": "Cambridge, Massachusetts",
#   "occupation": "cardiologist"
# }

print(f"Confidence: {result.confidence:.2%}")
# Output: Confidence: 94%
```

That's it! LangStruct automatically handles schema generation, optimization, and source tracking.

## 📚 Common Applications

### 1. Data Pipeline Automation
Extract structured data from documents for databases, analytics, or APIs:
```python
# Process invoices, receipts, reports, emails
invoice_data = extractor.extract(invoice_pdf_text)
# → {"invoice_no": "INV-2024-001", "amount": 5420.00, "due_date": "2024-03-15"}
```

### 2. Content Analysis & Research
Analyze transcripts, reviews, surveys, or social media:
```python
# Extract insights from customer feedback
feedback = extractor.extract(review_text)
# → {"sentiment": "positive", "product_issues": [], "feature_requests": ["dark mode"]}
```

### 3. Compliance & Validation
Extract and validate required information from legal or regulatory documents:
```python
# Check contracts for specific clauses
contract_data = extractor.extract(contract_text)
# → {"term_length": "2 years", "termination_clause": true, "liability_cap": 1000000}
```

## 🚀 RAG System Enhancement

**Transform your RAG system** from simple search to intelligent retrieval:

> **Note**: LangStruct enhances ANY vector database or search system (Pinecone, Weaviate, Elasticsearch, etc.).

### 1. Document → Structured Metadata
```python
# Extract structured metadata from documents
extractor = LangStruct(example={
    "company": "Apple Inc.",
    "revenue": 125.3,
    "quarter": "Q3 2024"
})

metadata = extractor.extract(document).entities
# Now your documents have precise, filterable metadata
```

### 2. Query → Structured Filters
```python
from langstruct import LangStruct

# Parse natural language queries into filters
ls = LangStruct(example=same_schema)  # Same schema as extraction!

query = "Show me Q3 2024 tech companies with revenue over $100B discussing AI investments"
parsed = ls.query(query)

print(parsed.semantic_terms)
# ["tech companies", "AI investments", "artificial intelligence"]

print(parsed.structured_filters)
# {"quarter": "Q3 2024", "revenue": {"$gte": 100.0}}
```

### 3. Precise Retrieval
```python
# Combine semantic search with exact filters
rag_results = vector_store.similarity_search(
    query=' '.join(parsed.semantic_terms),  # Semantic search
    where=parsed.structured_filters         # Exact filters
)
# Returns only docs matching BOTH semantic AND structural requirements
```

### Why RAG + LangStruct?

Traditional RAG systems struggle with **structured requirements**. LangStruct solves this:

| Query                            | Traditional RAG                             | With LangStruct                       |
| -------------------------------- | ------------------------------------------- | ------------------------------------- |
| "invoices over $10k from Q3"     | Returns any document with "invoice" OR "Q3" | Returns ONLY invoices >$10k from Q3   |
| "patients over 65 with diabetes" | Returns any medical document                | Returns ONLY matching patient records |
| "contracts expiring in 2024"     | Returns any contract                        | Returns ONLY 2024 expirations         |

See our [complete RAG integration guide](https://langstruct.dev/rag-integration/) for implementation.

## 🌟 Where LangStruct Excels

### Perfect for:
- **📄 Document Processing**: Invoices, reports, forms, emails
- **🏥 Healthcare**: Medical records, clinical notes, lab results
- **💼 Financial**: Statements, filings, contracts, reports
- **⚖️ Legal**: Contracts, agreements, regulations, cases
- **🔬 Research**: Papers, patents, technical documentation
- **🎯 Customer Data**: Reviews, feedback, support tickets

### Key Advantages:
- **No prompt engineering**: DSPy handles optimization automatically
- **Type safety**: Pydantic schemas with full validation
- **Source grounding**: Know exactly where each field came from
- **Confidence scores**: Understand extraction reliability
- **Model agnostic**: Works with any LLM provider

## 📊 Comparison with Alternatives

### LangStruct vs LangExtract

Both are excellent tools for structured extraction with different strengths:

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

Note: Comparison verified on 2025-09-10 against the latest LangExtract README and examples. See LangExtract: https://github.com/google/langextract and example walkthroughs (e.g., longer text extraction): https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md

**Choose LangStruct if you want:**
- Automatic optimization without prompt engineering
- Best-of-N refinement for higher accuracy
- Flexibility to define schemas from examples
- Query parsing for RAG systems
- Confidence scores for extraction quality
- Support for any LLM provider

**Choose LangExtract if you prefer:**
- Direct control over prompts
- Google's backing and support
- Simpler architecture without DSPy

## 🎯 Getting Started

Once you're comfortable with the basics, you can:

**Define Custom Schemas** for more control:
```python
from pydantic import BaseModel, Field
from langstruct import LangStruct

class PersonSchema(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    location: str = Field(description="Current location")

extractor = LangStruct(schema=PersonSchema)
```

**Process Multiple Documents** at once:
```python
documents = [doc1, doc2, doc3]
results = extractor.extract(documents)  # Handles batch processing automatically
```

**Save and Load Extractors** for reuse:
```python
# Save an optimized extractor (preserves all state)
extractor.save("./my_extractor")

# Load anywhere (API keys must be available in environment)
loaded_extractor = LangStruct.load("./my_extractor")
result = loaded_extractor.extract("New text")
```

**View Source Locations** to see where data came from:
```python
for field, spans in result.sources.items():
    for span in spans:
        print(f"{field}: '{span.text}' at chars {span.start}-{span.end}")
```

## 📋 Supported Models

LangStruct works with any LLM provider:

- **Google Gemini**: gemini/gemini-2.5-flash, gemini/gemini-2.5-pro
- **OpenAI**: gpt-5-pro, gpt-5-mini, gpt-4o, gpt-4o-mini
- **Anthropic**: claude-opus-4-1, claude-sonnet-4-0, claude-3-7-sonnet-latest, claude-3-5-haiku-latest
- **Local**: Any model via Ollama (llama3, mistral, etc.)

## 🎨 Visualization & Export

**Create Interactive Visualizations:**
```python
from langstruct import HTMLVisualizer

viz = HTMLVisualizer()
viz.save_visualization(text, result, "results.html")  # Shows highlighted sources
```

**Export Results:**
```python
# Save to various formats
result.save_json("data.json")
extractor.export_batch(results, "data.csv")  # CSV, Excel, Parquet supported
```

**JSONL Round‑Trip + Visualization:**
```python
# Save annotated documents to JSONL
results = extractor.extract(texts, validate=False)
extractor.save_annotated_documents(results, "extractions.jsonl")

# Load later
loaded = extractor.load_annotated_documents("extractions.jsonl")

# Generate interactive HTML
extractor.visualize(loaded, "results.html")
```

## 🧵 Batch, Rate Limits, Retries

LangStruct batches efficiently and helps respect provider quotas.

```python
# Control concurrency and quotas
results = extractor.extract(
    texts,
    max_workers=8,        # Thread workers
    show_progress=True,   # Requires langstruct[parallel]
    rate_limit=60,        # Calls per minute
    retry_failed=True     # Raise on failures or surface warnings
)
```

- Retries: exponential backoff (3 attempts by default) for transient errors.
- Rate limiting: simple token‑bucket; set `rate_limit=None` for unlimited.
- Failures: when `retry_failed=False`, failures are warned and skipped; otherwise an exception summarizes first errors.

---

## 🚀 Advanced Features

### Optimization (For Power Users)

LangStruct optimizes automatically, but you can fine-tune for your specific data:

```python
# Train on your examples
training_texts = ["Your domain-specific texts..."]
expected_results = [{"name": "Expected outputs..."}]

extractor.optimize(
    texts=training_texts,
    expected_results=expected_results,
    num_trials=50  # More trials = better results
)

# Evaluate performance
scores = extractor.evaluate(test_texts, test_expected)
print(f"Accuracy: {scores['accuracy']:.2%}")
```

### Refinement for Higher Accuracy

Boost extraction accuracy by 15-30% with Best-of-N candidate selection and iterative improvement:

```python
# Simple refinement
result = extractor.extract(text, refine=True)

# Advanced refinement with custom configuration
result = extractor.extract(text, refine={
    "strategy": "bon_then_refine",  # Best-of-N + iterative improvement
    "n_candidates": 5,              # Generate 5 candidates
    "judge": "Prefer candidates that exactly match cited text spans",
    "max_refine_steps": 2,
    "budget": {"max_calls": 10}     # Cost control
})

print(f"Accuracy improvement: {result.confidence:.1%}")
```

### Custom Configuration

```python
from langstruct import ChunkingConfig

# For large documents
config = ChunkingConfig(
    max_tokens=1500,
    overlap_tokens=150,
    preserve_sentences=True
)

extractor = LangStruct(
    schema=YourSchema,
    model="gemini/gemini-2.5-flash",
    chunking_config=config,
    optimize=True  # Enabled for training data
)
```

## 🔧 Troubleshooting

### API Key Issues

**Error: "No API keys found" or "Authentication failed"**

1. **Check your API key is set:**
   ```bash
   echo $GOOGLE_API_KEY  # Should show your key
   ```

2. **Common fixes:**
   ```bash
   # Make sure you're using the right format
   export GOOGLE_API_KEY="your-actual-key-here"  # No quotes in the key itself

   # For persistent setup, add to your shell profile:
   echo 'export GOOGLE_API_KEY="your-key"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Test your key works:**
   ```python
   import os
   print("API key set:", bool(os.getenv("GOOGLE_API_KEY")))

   # Quick test
   from langstruct import LangStruct
   ls = LangStruct(example={"name": "test"})
   result = ls.extract("Hello John")  # Should work without errors
   ```

**Error: "Model not found" or "Rate limit exceeded"**

- **Model not found**: Your API key might be for a different provider
- **Rate limits**: Try a different model or wait a few minutes
- **Billing**: Check your account has credits (OpenAI/Anthropic)

### Installation Issues

**Error: Package not found on PyPI**

If you encounter package installation issues, try:
```bash
# Update pip and try again
pip install --upgrade pip
pip install langstruct

# Or install from source for development
git clone https://github.com/langstruct-ai/langstruct.git
cd langstruct
uv sync --extra dev
uv pip install -e .
```

**Import errors or missing dependencies**

```bash
# Reinstall with all dependencies
pip install -e ".[dev,examples,viz,parallel]"
```

### Getting Help

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/langstruct-ai/langstruct/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/langstruct-ai/langstruct/discussions)
- 📖 **Documentation**: [langstruct.dev](https://langstruct.dev)

## 🤝 Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/langstruct.git
cd langstruct

# Install dependencies with uv
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run black . && uv run isort .
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on [DSPy](https://github.com/stanfordnlp/dspy) for self-optimizing LM pipelines
- Uses [Pydantic](https://pydantic.dev) for type-safe schemas
