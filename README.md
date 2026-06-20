# JobFit RAG Assistant

AI-powered resume ↔ job description analyzer. Scores skill fit, summarizes JDs, and answers questions about requirements.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.jobfit
```

## How It Works

1. **Similarity Engine** — Sentence-BERT (`all-MiniLM-L6-v2`) embeds resume + JD sentences, then cosine similarity finds top matches
2. **JD Summarizer** — BART (`facebook/bart-large-cnn`) condenses the job description
3. **QA Pipeline** — DistilBERT (`distilbert-base-cased-distilled-squad`) answers questions about the JD

## Notes

- Uses raw model classes (`AutoTokenizer` + `AutoModelForSeq2SeqLM`) instead of `pipeline()` because **transformers v5 removed the `"summarization"` and `"question-answering"` pipeline tasks**. The raw approach is equivalent and the standard recommendation for v5.

## Project Structure

```
src/jobfit/
├── __init__.py
├── __main__.py       # CLI entry point
├── similarity.py     # Sentence embedding + cosine matching
├── summarizer.py     # BART summarization
├── qa.py             # DistilBERT extractive QA
└── analyzer.py       # Orchestrator: runs all modules + prints report
```
