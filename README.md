# JobFit RAG Assistant

Resume vs job description analyzer powered by a local LLM (SmolLM2-1.7B). Compares skills, scores fit, and answers queries through an LLM-driven agent loop.

## Features

- **Match Scoring** — Sentence-BERT embeddings + cosine similarity to score resume-JD alignment
- **Skill Extraction** — Extracts technical and professional skills from any text
- **JD Summarization** — Condenses job descriptions into key points
- **Tool Agent** — LLM decides which tool to call based on your natural language query
- **Hallucination Control** — Validates LLM outputs against expected schemas
- **ChromaDB Vector Store** — In-memory semantic retrieval of resume and JD chunks per session

## Installation

```bash
git clone https://github.com/muhammedriswanp/jobfit-rag-assistant.git
cd jobfit-rag-assistant
pip install -r requirements.txt
```

> **Note on hardware**: The models used (SmolLM2-1.7B-Instruct + all-MiniLM-L6-v2) require significant GPU memory and RAM. All testing and development were done inside the `notebooks/` directory on Google Colab due to local GPU/RAM constraints.

## Usage

### CLI

```bash
python -m src.jobfit
```

### Python API

```python
from src.jobfit import (
    run_tool_workflow,
    calculate_match_score,
    extract_skills,
    summarize_jd_llm,
)

# Tool agent — LLM routes your query to the right tool
result = run_tool_workflow(
    "How well does my resume match this job?",
    resume_text="...",
    jd_text="...",
    verbose=True,
)
print(result["answer"])

# Or call tools directly
skills = extract_skills("Experienced software engineer skilled in Python, Java, and C++.")
print(skills)
```

### Notebook (recommended for testing)

Due to GPU/RAM requirements, the recommended way to test is via the Colab-compatible notebook:

```
notebooks/jobfit_function_caller.ipynb
```

This notebook lazy-loads models so the embedding model and LLM are never in RAM simultaneously.

## Project Structure

```
src/jobfit/
├── __init__.py            # Package exports
├── __main__.py            # CLI entry point
├── function_caller.py     # LLM-driven tool agent loop with keyword fallback
├── llm.py                 # SmolLM2 text generation (lazy-loaded)
├── similarity.py          # Sentence embedding + cosine matching
├── tools.py               # Tool implementations (match, extract, summarize)
├── validator.py           # Schema validation for hallucination control
└── vector_store.py        # ChromaDB vector store for semantic retrieval
notebooks/
└── jobfit_function_caller.ipynb   # Colab-compatible test notebook
```

## Tech Stack

| Component | Model / Library |
|---|---|
| LLM Agent | HuggingFaceTB/SmolLM2-1.7B-Instruct |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Similarity | scikit-learn cosine similarity |
| Vector Store | ChromaDB (EphemeralClient — in-memory per session) |
