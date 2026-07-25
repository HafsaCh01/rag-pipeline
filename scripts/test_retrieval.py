import time
import json
import statistics
from sentence_transformers import SentenceTransformer

from setup_chromadb import get_client, get_or_create_collection, COLLECTION_NAME


MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

TEST_QUERIES = [
    "stock market rally and investor sentiment",
    "sports championship results",
    "new government policy announcement",
    "technology company product launch",
    "international conflict and diplomacy"
]


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def semantic_search(collection, model, query: str, top_k: int = TOP_K):
    start = time.time()
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    elapsed_ms = (time.time() - start) * 1000

    return results, elapsed_ms


def run_benchmark(queries: list = TEST_QUERIES, top_k: int = TOP_K):
    client = get_client()
    collection = get_or_create_collection(client, COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError("Collection is empty. Run setup_chromadb.py first.")

    model = load_model()
    latencies = []
    log_entries = []

    for query in queries:
        results, elapsed_ms = semantic_search(collection, model, query, top_k)
        latencies.append(elapsed_ms)

        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        log_entries.append({
            "query": query,
            "latency_ms": round(elapsed_ms, 4),
            "top_result_preview": documents[0][:150] if documents else None,
            "top_result_distance": round(distances[0], 4) if distances else None,
            "num_results": len(documents)
        })

        print(f"Query: {query}")
        print(f"  Latency: {elapsed_ms:.2f} ms")
        print(f"  Top result: {documents[0][:150]}..." if documents else "  No results")
        print()

    summary = {
        "collection_size": collection.count(),
        "num_queries": len(queries),
        "top_k": top_k,
        "avg_latency_ms": round(statistics.mean(latencies), 4),
        "min_latency_ms": round(min(latencies), 4),
        "max_latency_ms": round(max(latencies), 4),
        "median_latency_ms": round(statistics.median(latencies), 4),
        "queries": log_entries
    }

    with open("reports/retrieval_latency.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "queries"}, indent=2))

    return summary


if __name__ == "__main__":
    run_benchmark()