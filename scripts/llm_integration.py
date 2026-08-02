import os
import json
import time
import requests
from datetime import datetime

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/free"

SYSTEM_PROMPT = """You are a factual question-answering assistant for a news retrieval system.

Rules you must follow:
1. Answer ONLY using information found in the provided context chunks.
2. If the context does not contain enough information to answer, say exactly: "I don't have enough information in the retrieved context to answer this question."
3. Do not use outside knowledge, even if you are confident about the answer.
4. Do not speculate or fill in gaps with assumptions.
5. Keep answers concise and directly grounded in the context.
6. If you use information from a chunk, you may reference it as (Source: chunk_id)."""


class RAGGenerationError(Exception):
    pass


def build_prompt(query, chunks):
    if not chunks:
        context_block = "No relevant context was retrieved."
    else:
        context_block = "\n\n".join(
            f"[chunk_id: {c.get('chunk_id', i)}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

    prompt = f"""Context:
{context_block}

Question: {query}

Answer using only the context above."""
    return prompt


def call_llm(query, chunks, max_retries=3, timeout=30, max_tokens=500):
    if not OPENROUTER_API_KEY:
        raise RAGGenerationError(
            "OPENROUTER_API_KEY not set. Run: setx OPENROUTER_API_KEY \"your-key\" "
            "and restart PowerShell."
        )

    user_prompt = build_prompt(query, chunks)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            latency = time.time() - start

            if response.status_code == 200:
                data = response.json()
                message = data["choices"][0]["message"]
                answer = message.get("content") or message.get("reasoning") or None
                usage = data.get("usage", {})

                return {
                    "query": query,
                    "answer": answer,
                    "num_chunks_used": len(chunks),
                    "latency_seconds": round(latency, 3),
                    "attempt": attempt,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "timestamp": datetime.now().isoformat(),
                    "error": None
                }

            elif response.status_code == 429:
                last_error = f"Rate limited (429) on attempt {attempt}"
                wait_time = 2 ** attempt
                print(f"{last_error}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            elif response.status_code in (401, 403):
                raise RAGGenerationError(
                    f"Auth error ({response.status_code}): check your API key"
                )

            elif response.status_code >= 500:
                last_error = f"Server error ({response.status_code}) on attempt {attempt}"
                time.sleep(2 ** attempt)
                continue

            else:
                last_error = f"Unexpected status {response.status_code}: {response.text[:200]}"
                break

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {timeout}s on attempt {attempt}"
            time.sleep(2 ** attempt)
            continue

        except requests.exceptions.ConnectionError:
            last_error = f"Connection error on attempt {attempt}"
            time.sleep(2 ** attempt)
            continue

        except requests.exceptions.RequestException as e:
            last_error = f"Request failed: {str(e)}"
            break

    return {
        "query": query,
        "answer": None,
        "num_chunks_used": len(chunks),
        "latency_seconds": None,
        "attempt": max_retries,
        "timestamp": datetime.now().isoformat(),
        "error": last_error
    }


def check_hallucination_risk(answer, chunks, min_overlap_ratio=0.1):
    if answer is None:
        return {"flagged": False, "reason": "no answer generated"}

    refusal_phrase = "i don't have enough information"
    if refusal_phrase in answer.lower():
        return {"flagged": False, "reason": "model correctly refused"}

    if not chunks:
        return {"flagged": True, "reason": "answer generated with zero retrieved chunks"}

    context_words = set()
    for c in chunks:
        context_words.update(c["text"].lower().split())

    answer_words = set(answer.lower().split())
    if not answer_words:
        return {"flagged": True, "reason": "empty answer"}

    overlap = answer_words & context_words
    overlap_ratio = len(overlap) / len(answer_words)

    if overlap_ratio < min_overlap_ratio:
        return {
            "flagged": True,
            "reason": f"low context overlap ({overlap_ratio:.2%}), possible hallucination"
        }

    return {"flagged": False, "reason": f"context overlap {overlap_ratio:.2%}"}


def is_query_in_domain(chunks, min_chunks=1, max_distance=None, distances=None):
    if not chunks or len(chunks) < min_chunks:
        return False, "no chunks retrieved — likely off-topic for this dataset"

    if distances is not None and max_distance is not None:
        best_distance = min(distances)
        if best_distance > max_distance:
            return False, f"best match distance {best_distance:.3f} exceeds threshold {max_distance}"

    return True, "in-domain"


if __name__ == "__main__":
    test_chunks = [
        {"chunk_id": 12, "text": "The stock market rallied today after the Federal Reserve announced no change in interest rates."},
        {"chunk_id": 45, "text": "Tech stocks led gains as investors reacted positively to the Fed decision."}
    ]

    result = call_llm("What did the Federal Reserve announce?", test_chunks)
    print(json.dumps(result, indent=2))

    if result["answer"]:
        risk = check_hallucination_risk(result["answer"], test_chunks)
        print(json.dumps(risk, indent=2))