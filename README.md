# JobFit RAG Assistant

Resume vs job description analyzer powered by a local LLM (SmolLM2-1.7B). Compares skills, scores fit, and answers queries through an LLM-driven agent loop.

## Features

- **Match Scoring** — Sentence-BERT embeddings + cosine similarity to score resume-JD alignment
- **Skill Extraction** — Extracts technical and professional skills from any text
- **JD Summarization** — Condenses job descriptions into key points
- **Tool Agent** — LLM decides which tool to call based on your natural language query

## Installation

```bash
git clone https://github.com/muhammedriswanp/jobfit-rag-assistant.git
cd jobfit-rag-assistant
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Default — match score
python -m src.jobfit

# Summarize a job description
python -m src.jobfit "Summarize this job description for me"

# Extract skills from resume
python -m src.jobfit "What skills are in this resume?"
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
    context={"resume_path": "data/resume_sentences.txt", "jd_path": "data/jd_sentences.txt"},
    verbose=True,
)
print(result["answer"])

# Or call tools directly
skills = extract_skills("Experienced software engineer skilled in Python, Java, and C++.")
print(skills)
```

## Project Structure

```
src/jobfit/
├── __init__.py            # Package exports
├── __main__.py            # CLI entry point
├── function_caller.py     # LLM-driven tool agent loop
├── llm.py                 # SmolLM2 text generation
├── similarity.py          # Sentence embedding + cosine matching
└── tools.py               # Tool implementations
data/
├── resume_sentences.txt
└── jd_sentences.txt
```

## Tech Stack

| Component | Model / Library |
|---|---|
| LLM Agent | HuggingFaceTB/SmolLM2-1.7B-Instruct |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Similarity | scikit-learn cosine similarity |


