from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    chunk_id: int
    source_id: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


def _split_on_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _merge_splits(splits: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    current = []
    current_len = 0
    sep_len = len(separator)

    for piece in splits:
        piece_len = len(piece)

        if current_len + piece_len + (sep_len if current else 0) > chunk_size and current:
            chunk_str = separator.join(current)
            chunks.append(chunk_str)

            overlap_pieces = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) > chunk_overlap:
                    break
                overlap_pieces.insert(0, p)
                overlap_len += len(p) + sep_len
            current = overlap_pieces
            current_len = sum(len(p) for p in current) + sep_len * max(0, len(current) - 1)

        current.append(piece)
        current_len += piece_len + (sep_len if len(current) > 1 else 0)

    if current:
        chunks.append(separator.join(current))

    return [c.strip() for c in chunks if c.strip()]


def _recursive_split(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    sep, *rest_separators = separators
    splits = _split_on_separator(text, sep)

    if len(splits) == 1:
        return _recursive_split(text, rest_separators, chunk_size, chunk_overlap)

    merged = _merge_splits(splits, sep, chunk_size, chunk_overlap)

    final = []
    for piece in merged:
        if len(piece) > chunk_size:
            final.extend(_recursive_split(piece, rest_separators, chunk_size, chunk_overlap))
        else:
            final.append(piece)
    return final


def chunk_text(text: Optional[str], chunk_size: int = 500, chunk_overlap: int = 50,
               separators: Optional[List[str]] = None) -> List[str]:
    if not text or not text.strip():
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    separators = separators or DEFAULT_SEPARATORS
    return _recursive_split(text.strip(), separators, chunk_size, chunk_overlap)


def chunk_dataframe(df: pd.DataFrame, text_col: str = "cleaned_text", id_col: Optional[str] = None,
                     chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    global_id = 0

    for idx, row in df.iterrows():
        source_id = str(row[id_col]) if id_col else str(idx)
        pieces = chunk_text(row[text_col], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for piece in pieces:
            all_chunks.append(Chunk(text=piece, chunk_id=global_id, source_id=source_id))
            global_id += 1

    return all_chunks


if __name__ == "__main__":
    sample = (
        "Reuters reported today that markets rallied. Investors were optimistic.\n\n"
        "In other news, tech stocks led gains this quarter as earnings beat expectations. "
        "Analysts remain cautiously bullish heading into next year."
    )
    result = chunk_text(sample, chunk_size=80, chunk_overlap=15)
    for i, c in enumerate(result):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c)