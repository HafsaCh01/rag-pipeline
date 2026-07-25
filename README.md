# RAG Pipeline

A retrieval-augmented generation pipeline built on the AG News dataset — from raw data cleaning through chunking, embedding, and vector search.

---

## Week 1: Data Acquisition & Cleaning

**Results:**

| Metric | Value |
|---|---|
| Raw documents ingested | 6,060 |
| Dropped (empty after cleaning) | 120 |
| Duplicates removed | 60 |
| **Final clean dataset size** | **5,880** |
| Unit tests passing | 41 / 41 |

Full before/after numbers are in `reports/data_quality_report.md`.

---

## Week 2: Chunking, Embeddings & Vector DB

### Overview
This week builds on the cleaned dataset (`data/clean_dataset.csv`) from Week 1 to implement chunking, generate embeddings, store them in ChromaDB, and benchmark retrieval performance.

### Approach

**Chunking strategy: Recursive character splitting**
Text is split using a prioritized list of separators — paragraph breaks, then line breaks, then sentence boundaries, then spaces, then hard character cuts as a last resort. The splitter tries the largest boundary first and only falls back to smaller ones if a piece is still too big. This keeps chunks aligned to natural text boundaries instead of cutting mid-sentence.

- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Implementation: `scripts/chunk_text.py`
- Tests: `tests/test_chunking.py` (14 unit tests covering empty input, short/long text, overlap behavior, invalid params, and edge cases like a single very long word)

**Embeddings**
Generated using `sentence-transformers` with the `all-MiniLM-L6-v2` model. Embedding time and per-chunk latency are logged to `reports/embedding_performance.json`.

- Script: `scripts/generate_embeddings.py`
- Output: `data/embeddings.npy`, `data/chunks_meta.json`

**Vector database: ChromaDB**
Chunks and embeddings are ingested into a persistent ChromaDB collection (`ag_news_chunks`) stored at `data/chroma_db`. Ingestion is batched (500 items/batch) to handle large datasets, and the script skips re-ingesting if the collection is already populated.

- Script: `scripts/setup_chromadb.py`

**Retrieval benchmarking**
Semantic search is tested against 5 sample queries spanning different news categories. Latency per query and summary stats (avg/min/max/median) are logged.

- Script: `scripts/test_retrieval.py`
- Output: `reports/retrieval_latency.json`

### Edge Cases Handled
- Empty or whitespace-only text produces no chunks
- `chunk_overlap >= chunk_size` raises a `ValueError`
- A single word longer than `chunk_size` is hard-split by character
- Mismatched embeddings/metadata counts raise an error before ingestion
- Re-running ingestion on an already-populated collection is a no-op

---

## How to Run This Project

```bash
git clone <your-repo-url>
cd rag-pipeline
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Week 1: data acquisition and cleaning
python scripts/verify_setup.py
python scripts/build_raw_dataset.py
python scripts/validate_dataset.py
pytest tests/test_clean_text.py -v
python scripts/run_pipeline.py

# Week 2: chunking, embeddings, vector DB, retrieval
python scripts/generate_embeddings.py
python scripts/setup_chromadb.py
python scripts/test_retrieval.py
pytest tests/test_chunking.py -v
```

---

## Dependencies

See `requirements.txt`: pandas, numpy, datasets, ftfy, langdetect, beautifulsoup4, lxml, pytest, tqdm, sentence-transformers, chromadb.

---

## Weekly Progress Log

### Week 1 (Complete)
- Set up environment, verified all libraries import and function correctly
- Acquired 6,060 real-world documents (AG News)
- Validated dataset: found 90 nulls, 30 empty/whitespace rows, 123 encoding issues, 90 HTML rows, 144 duplicates
- Built defensive text-cleaning module handling nulls, HTML, encoding, whitespace, unicode, mixed language
- Wrote 41 unit tests, all passing
- Produced final clean dataset: 5,880 documents, ready for chunking/embedding
- Generated data quality report

### Week 2 (Complete)
- Implemented recursive character-based chunking (`scripts/chunk_text.py`), 500 char chunks with 50 char overlap
- Wrote 14 unit tests for chunking, all passing (`tests/test_chunking.py`)
- Generated embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`), logged embedding time (`reports/embedding_performance.json`)
- Set up ChromaDB, ingested all chunks/embeddings into a persistent collection, handled batching and re-ingestion edge cases
- Built semantic search + latency benchmarking script across 5 sample queries, logged results (`reports/retrieval_latency.json`)