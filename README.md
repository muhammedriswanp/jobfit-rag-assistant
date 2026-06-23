# JobFit RAG Assistant

AI-powered resume ↔ job description analyzer. Scores skill fit, summarizes JDs, answers questions, and generates gap-analysis suggestions using a local LLM.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.jobfit
```

## How It Works

1. **Similarity Engine** — Sentence-BERT (`all-MiniLM-L6-v2`) embeds resume + JD sentences, then cosine similarity finds top matches
2. **JD Summarizer** — BART (`facebook/bart-large-cnn`) condenses the job description
3. **QA Pipeline** — DistilBERT (`distilbert-base-cased-distilled-squad`) answers questions about the JD
4. **LLM Generation** — SmolLM2-1.7B-Instruct (`AutoModelForCausalLM`) generates resume improvement suggestions for skill gaps
5. **Integration** — `analyzer.py` orchestrates all modules and prints a full report

## Notes

- Uses raw model classes (`AutoTokenizer` + `AutoModelForSeq2SeqLM`/`AutoModelForCausalLM`) instead of `pipeline()` because **transformers v5 removed several pipeline tasks**. The raw approach is equivalent and the standard recommendation for v5.
- LLM uses lazy loading — model is loaded only on first `generate_text()` call (not at import time).

## Project Structure

```
src/jobfit/
├── __init__.py
├── __main__.py       # CLI entry point
├── similarity.py     # Sentence embedding + cosine matching
├── summarizer.py     # BART summarization
├── qa.py             # DistilBERT extractive QA
├── llm.py            # SmolLM2 causal LM generation (Phase 2)
└── analyzer.py       # Orchestrator: runs all modules + prints report
notebook/
└── test_colab.ipynb  # Colab notebook for GPU-accelerated testing
experiments/
└── temperature_comparison.md  # Day 1 temperature experiment results

## Limitations
- **Context window:** SmolLM2-1.7B struggles with JDs >30 lines. Chunking and RAG retrieval will be added (Day 10) to handle long documents.
```
