import json
import time
from datetime import datetime
from llm_integration import call_llm, check_hallucination_risk, is_query_in_domain
from test_retrieval import search


def answer_query(query, k=5, log_path="reports/e2e_latency.json"):
    overall_start = time.time()

    retrieval_start = time.time()
    results = search(query, k=k)
    retrieval_latency = time.time() - retrieval_start

    distances = [r.get("distance") for r in results if r.get("distance") is not None]
    in_domain, domain_reason = is_query_in_domain(
        results, min_chunks=1, max_distance=1.5, distances=distances or None
    )

    if not in_domain:
        log_entry = {
            "query": query,
            "answer": "This question appears to be outside the scope of this dataset (AG News articles).",
            "in_domain": False,
            "domain_reason": domain_reason,
            "retrieval_latency_seconds": round(retrieval_latency, 3),
            "generation_latency_seconds": None,
            "total_latency_seconds": round(time.time() - overall_start, 3),
            "timestamp": datetime.now().isoformat()
        }
        _append_log(log_entry, log_path)
        return log_entry

    generation_start = time.time()
    gen_result = call_llm(query, results)
    generation_latency = time.time() - generation_start

    hallucination_check = check_hallucination_risk(gen_result["answer"], results)

    log_entry = {
        "query": query,
        "answer": gen_result["answer"],
        "in_domain": True,
        "num_chunks_retrieved": len(results),
        "retrieval_latency_seconds": round(retrieval_latency, 3),
        "generation_latency_seconds": round(generation_latency, 3),
        "total_latency_seconds": round(time.time() - overall_start, 3),
        "hallucination_check": hallucination_check,
        "llm_error": gen_result.get("error"),
        "timestamp": datetime.now().isoformat()
    }

    _append_log(log_entry, log_path)
    return log_entry


def _append_log(entry, log_path):
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logs = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    test_queries = [
        "What happened in the stock market recently?",
        "Who won the latest Olympic gold medal in curling?",
        "What are the latest developments in technology?"
    ]

    for q in test_queries:
        result = answer_query(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Total latency: {result['total_latency_seconds']}s")