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

## How to Run This Project

```bash
git clone <your-repo-url>
cd rag-pipeline
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/verify_setup.py
python scripts/build_raw_dataset.py
python scripts/validate_dataset.py
pytest tests/test_clean_text.py -v
python scripts/run_pipeline.py
```

---

## Dependencies

See `requirements.txt`: pandas, numpy, datasets, ftfy, langdetect, beautifulsoup4, lxml, pytest, tqdm.

---

## Weekly Progress Log

### Week 1 ( Complete)
- Set up environment, verified all libraries import and function correctly
- Acquired 6,060 real-world documents (AG News)
- Validated dataset: found 90 nulls, 30 empty/whitespace rows, 123 encoding issues, 90 HTML rows, 144 duplicates
- Built defensive text-cleaning module handling nulls, HTML, encoding, whitespace, unicode, mixed language
- Wrote 41 unit tests, all passing
- Produced final clean dataset: 5,880 documents, ready for chunking/embedding
- Generated data quality report

