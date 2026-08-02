import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest
from unittest.mock import patch, MagicMock
import requests

from llm_integration import (
    build_prompt,
    call_llm,
    check_hallucination_risk,
    is_query_in_domain,
    RAGGenerationError
)


SAMPLE_CHUNKS = [
    {"chunk_id": 1, "text": "The Federal Reserve announced no change in interest rates today."},
    {"chunk_id": 2, "text": "Stock markets reacted positively to the Fed's decision."}
]


def make_mock_response(status_code=200, json_data=None, text=""):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {}
    mock_resp.text = text
    return mock_resp


class TestBuildPrompt:
    def test_includes_all_chunks(self):
        prompt = build_prompt("What did the Fed announce?", SAMPLE_CHUNKS)
        assert "Federal Reserve" in prompt
        assert "Stock markets" in prompt
        assert "chunk_id: 1" in prompt
        assert "chunk_id: 2" in prompt

    def test_includes_query(self):
        prompt = build_prompt("What did the Fed announce?", SAMPLE_CHUNKS)
        assert "What did the Fed announce?" in prompt

    def test_empty_chunks(self):
        prompt = build_prompt("Any question?", [])
        assert "No relevant context was retrieved" in prompt


class TestCallLLM:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setattr("llm_integration.OPENROUTER_API_KEY", None)
        with pytest.raises(RAGGenerationError):
            call_llm("test query", SAMPLE_CHUNKS)

    @patch("llm_integration.requests.post")
    def test_successful_response(self, mock_post, monkeypatch):
        monkeypatch.setattr("llm_integration.OPENROUTER_API_KEY", "fake-key")
        mock_post.return_value = make_mock_response(
            status_code=200,
            json_data={
                "choices": [{"message": {"content": "The Fed announced no rate change."}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}
            }
        )
        result = call_llm("What did the Fed announce?", SAMPLE_CHUNKS)
        assert result["answer"] == "The Fed announced no rate change."
        assert result["error"] is None
        assert result["prompt_tokens"] == 100

    @patch("llm_integration.requests.post")
    def test_reasoning_field_fallback(self, mock_post, monkeypatch):
        monkeypatch.setattr("llm_integration.OPENROUTER_API_KEY", "fake-key")
        mock_post.return_value = make_mock_response(
            status_code=200,
            json_data={
                "choices": [{"message": {"content": None, "reasoning": "Answer via reasoning field."}}],
                "usage": {}
            }
        )
        result = call_llm("test query", SAMPLE_CHUNKS)
        assert result["answer"] == "Answer via reasoning field."

    @patch("llm_integration.requests.post")
    def test_auth_error_raises_immediately(self, mock_post, monkeypatch):
        monkeypatch.setattr("llm_integration.OPENROUTER_API_KEY", "bad-key")
        mock_post.return_value = make_mock_response(status_code=401, text="Unauthorized")
        with pytest.raises(RAGGenerationError):
            call_llm("test query", SAMPLE_CHUNKS)

    @patch("llm_integration.requests.post")
    @patch("llm_integration.time.sleep", return_value=None)
    def test_rate_limit_retries_then_fails(self, mock_sleep, mock_post, monkeypatch):
        monkeypatch.setattr("llm_integration.OPENROUTER_API_KEY", "fake-key")
        mock_post.return_value = make_mock_response(status_code=429, text="Rate limited")
        result = call_llm("test query", SAMPLE_CHUNKS, max_retries=2)
        assert result["answer"] is None
        assert "Rate limited" in result["error"]
        assert mock_post.call_count == 2

    @patch("llm_integration.requests.post")
    @patch("llm_integration.time.sleep", return_value=None)
    def test_timeout_retries_then_fails(self, mock_sleep, mock_post, monkeypatch):
        monkeypatch.setattr("llm_integration.OPENROUTER_API_KEY", "fake-key")
        mock_post.side_effect = requests.exceptions.Timeout()
        result = call_llm("test query", SAMPLE_CHUNKS, max_retries=2, timeout=5)
        assert result["answer"] is None
        assert "Timeout" in result["error"]

    @patch("llm_integration.requests.post")
    @patch("llm_integration.time.sleep", return_value=None)
    def test_server_error_retries_then_succeeds(self, mock_sleep, mock_post, monkeypatch):
        monkeypatch.setattr("llm_integration.OPENROUTER_API_KEY", "fake-key")
        mock_post.side_effect = [
            make_mock_response(status_code=500, text="Server error"),
            make_mock_response(
                status_code=200,
                json_data={
                    "choices": [{"message": {"content": "Recovered answer"}}],
                    "usage": {}
                }
            )
        ]
        result = call_llm("test query", SAMPLE_CHUNKS, max_retries=3)
        assert result["answer"] == "Recovered answer"
        assert result["attempt"] == 2


class TestHallucinationCheck:
    def test_flags_low_overlap(self):
        result = check_hallucination_risk(
            "Purple giraffes juggle spaghetti under moonlit pyramids.",
            SAMPLE_CHUNKS
        )
        assert result["flagged"] is True

    def test_passes_high_overlap(self):
        result = check_hallucination_risk(
            "The Federal Reserve announced no change in interest rates.",
            SAMPLE_CHUNKS
        )
        assert result["flagged"] is False

    def test_correct_refusal_not_flagged(self):
        result = check_hallucination_risk(
            "I don't have enough information in the retrieved context to answer this question.",
            SAMPLE_CHUNKS
        )
        assert result["flagged"] is False
        assert "refused" in result["reason"]

    def test_none_answer_not_flagged(self):
        result = check_hallucination_risk(None, SAMPLE_CHUNKS)
        assert result["flagged"] is False

    def test_zero_chunks_flagged(self):
        result = check_hallucination_risk("Some answer here.", [])
        assert result["flagged"] is True


class TestDomainDetection:
    def test_no_chunks_out_of_domain(self):
        in_domain, reason = is_query_in_domain([])
        assert in_domain is False

    def test_chunks_present_in_domain(self):
        in_domain, reason = is_query_in_domain(SAMPLE_CHUNKS)
        assert in_domain is True

    def test_distance_threshold_exceeded(self):
        in_domain, reason = is_query_in_domain(
            SAMPLE_CHUNKS, max_distance=1.0, distances=[1.8, 2.1]
        )
        assert in_domain is False

    def test_distance_within_threshold(self):
        in_domain, reason = is_query_in_domain(
            SAMPLE_CHUNKS, max_distance=2.0, distances=[0.8, 1.1]
        )
        assert in_domain is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])