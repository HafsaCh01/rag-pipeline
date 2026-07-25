import time
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from chunk_text import chunk_dataframe


MODEL_NAME = "all-MiniLM-L6-v2"
TEXT_COL = "clean_text"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 32


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(model: SentenceTransformer, texts: list, batch_size: int = BATCH_SIZE):
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    elapsed = time.time() - start
    return embeddings, elapsed


def run(input_csv: str, output_npy: str, output_meta_json: str, log_json: str):
    df = pd.read_csv(input_csv)

    chunk_start = time.time()
    chunks = chunk_dataframe(df, text_col=TEXT_COL, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunk_elapsed = time.time() - chunk_start

    texts = [c.text for c in chunks]

    model = load_model()
    embeddings, embed_elapsed = embed_chunks(model, texts)

    np.save(output_npy, embeddings)

    meta = [
        {"chunk_id": c.chunk_id, "source_id": c.source_id, "text": c.text, "char_count": c.char_count}
        for c in chunks
    ]
    with open(output_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log = {
        "model_name": MODEL_NAME,
        "num_articles": len(df),
        "num_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_time_seconds": round(chunk_elapsed, 4),
        "embedding_time_seconds": round(embed_elapsed, 4),
        "avg_time_per_chunk_ms": round((embed_elapsed / max(1, len(chunks))) * 1000, 4),
        "embedding_dim": embeddings.shape[1]
    }
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    run(
        input_csv="data/clean_dataset.csv",
        output_npy="data/embeddings.npy",
        output_meta_json="data/chunks_meta.json",
        log_json="reports/embedding_performance.json"
    )