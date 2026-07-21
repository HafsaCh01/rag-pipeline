import pandas as pd
import numpy as np
import re
import ftfy

RAW_PATH = "data/raw_dataset.csv"

MOJIBAKE_PATTERNS = [
    "Ã¢â‚¬", "Ã¢â‚¬â„¢", "â€œ", "â€", "âœ“", "Â ", "Ã©", "Ã¨", "Ã¯", "Ã¼"
]

HTML_TAG_RE = re.compile(r"<[^>]+>")


def load_raw(path=RAW_PATH):
    return pd.read_csv(path, keep_default_na=True)


def check_nulls(df):
    null_counts = df.isnull().sum().to_dict()
    null_text_rows = df[df["text"].isnull()].shape[0]
    return {"null_counts_per_column": null_counts, "null_text_rows": null_text_rows}


def check_empty_or_whitespace(df):
    non_null = df["text"].dropna()
    empty = non_null[non_null.str.strip() == ""]
    return {"empty_or_whitespace_rows": len(empty), "empty_row_ids": empty.index.tolist()[:10]}


def check_encoding_issues(df):
    non_null = df["text"].dropna()
    flagged = [idx for idx, text in non_null.items() if any(pat in text for pat in MOJIBAKE_PATTERNS)]
    return {"mojibake_suspected_rows": len(flagged), "sample_ids": flagged[:10]}


def check_html_content(df):
    non_null = df["text"].dropna()
    flagged = non_null[non_null.apply(lambda t: bool(HTML_TAG_RE.search(t)))]
    return {"html_tag_rows": len(flagged), "sample_ids": flagged.index.tolist()[:10]}


def check_duplicates(df):
    dup_mask = df.duplicated(subset=["text"], keep=False)
    dup_mask_notnull = dup_mask & df["text"].notnull()
    return {
        "duplicate_text_rows": int(dup_mask_notnull.sum()),
        "unique_duplicate_groups": int(df.loc[dup_mask_notnull, "text"].nunique()),
    }


def check_length_stats(df):
    non_null = df["text"].dropna()
    lengths = non_null.str.len()
    return {
        "min_len": int(lengths.min()) if len(lengths) else None,
        "max_len": int(lengths.max()) if len(lengths) else None,
        "mean_len": float(lengths.mean()) if len(lengths) else None,
        "median_len": float(lengths.median()) if len(lengths) else None,
    }


def check_mixed_language(df, sample_size=500):
    non_null = df["text"].dropna()
    non_ascii_heavy = non_null[non_null.apply(lambda t: sum(1 for c in t if ord(c) > 127) > 15)]
    return {"non_ascii_heavy_rows": len(non_ascii_heavy), "sample_ids": non_ascii_heavy.index.tolist()[:10]}


def run_all_checks(path=RAW_PATH):
    df = load_raw(path)
    results = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "nulls": check_nulls(df),
        "empty_or_whitespace": check_empty_or_whitespace(df),
        "encoding_issues": check_encoding_issues(df),
        "html_content": check_html_content(df),
        "duplicates": check_duplicates(df),
        "length_stats": check_length_stats(df),
        "mixed_language_signal": check_mixed_language(df),
    }
    return results, df


def print_summary(results):
    print("=== Dataset Validation Summary ===")
    print(f"Total rows: {results['total_rows']}")
    print(f"Columns: {results['columns']}")
    print(f"\nNulls: {results['nulls']}")
    print(f"\nEmpty/whitespace-only: {results['empty_or_whitespace']}")
    print(f"\nEncoding issues (mojibake suspected): {results['encoding_issues']}")
    print(f"\nHTML content detected: {results['html_content']}")
    print(f"\nDuplicates: {results['duplicates']}")
    print(f"\nLength stats: {results['length_stats']}")
    print(f"\nMixed-language signal (non-ASCII heavy): {results['mixed_language_signal']}")


if __name__ == "__main__":
    results, df = run_all_checks()
    print_summary(results)