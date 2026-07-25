import json
import time
import numpy as np
import chromadb


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "ag_news_chunks"
EMBEDDINGS_PATH = "data/embeddings.npy"
META_PATH = "data/chunks_meta.json"
BATCH_SIZE = 500


def get_client(path: str = CHROMA_PATH):
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(client, name: str = COLLECTION_NAME):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def load_data(embeddings_path: str = EMBEDDINGS_PATH, meta_path: str = META_PATH):
    embeddings = np.load(embeddings_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if len(embeddings) != len(meta):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs {len(meta)} metadata entries"
        )

    return embeddings, meta


def ingest(collection, embeddings, meta, batch_size: int = BATCH_SIZE):
    start = time.time()
    total = len(meta)

    for i in range(0, total, batch_size):
        batch_embeddings = embeddings[i:i + batch_size].tolist()
        batch_meta = meta[i:i + batch_size]

        ids = [f"chunk_{m['chunk_id']}" for m in batch_meta]
        documents = [m["text"] for m in batch_meta]
        metadatas = [
            {"source_id": m["source_id"], "char_count": m["char_count"]}
            for m in batch_meta
        ]

        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas
        )

    elapsed = time.time() - start
    return elapsed


def run():
    client = get_client()
    collection = get_or_create_collection(client)

    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection already has {existing_count} items. Skipping ingest.")
        return collection

    embeddings, meta = load_data()
    elapsed = ingest(collection, embeddings, meta)

    print(f"Ingested {len(meta)} chunks in {elapsed:.4f}s")
    print(f"Collection count: {collection.count()}")

    return collection


if __name__ == "__main__":
    run()