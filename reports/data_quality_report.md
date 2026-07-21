# Data Quality Report
 
## Source
AG News (real-world news articles), sampled and augmented with realistic edge cases, downloaded from a public GitHub mirror.
 
## Pipeline Summary
- Raw documents ingested: 6060
- Documents dropped (empty/null after cleaning): 120
- Duplicate documents removed: 60
- Documents dropped (too short, <10 chars): 0
- **Final clean dataset size: 5880**
 
## Pre-Cleaning Validation Findings
- Total raw rows: 6060
- Null text rows: 90
- Empty/whitespace-only rows: 30
- Mojibake/encoding-issue rows: 123
- Rows containing HTML tags: 90
- Duplicate text rows: 144 (58 groups)
- Non-ASCII heavy (mixed-language signal) rows: 66
- Text length: min=7, max=1010, mean=240.7, median=236.0
 
## Post-Cleaning Stats
- Language distribution (detected): {"en": 5876, "fr": 2, "it": 1, "de": 1}
- Mixed-language docs (foreign sentences stripped): 1238
- Docs that had HTML removed: 90
- Docs that had encoding fixes applied: 375
- Clean text length: min=27, max=959, mean=229.5
 
## Unit Tests
41/41 passing (tests/test_clean_text.py) covering null/NaN handling, empty/whitespace text,
mojibake repair, HTML stripping (incl. malformed markup), whitespace normalization,
unicode normalization, URL removal, language detection, and mixed-language splitting.
 
## Output
Clean dataset ready for chunking/embedding: `data/clean_dataset.csv` (5880 rows, columns: ['doc_id', 'label_name', 'clean_text', 'char_len', 'detected_language', 'had_html', 'had_encoding_issue', 'was_mixed_language'])
