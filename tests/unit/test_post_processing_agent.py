"""Unit tests for PostProcessingAgent with LLM-as-judge."""

import json
from typing import Dict, Any, List

import pytest
from unittest.mock import Mock, patch

from agents.post_processing_agent import PostProcessingAgent


class TestPostProcessingAgent:
    """Test suite for PostProcessingAgent."""

    @pytest.fixture
    def agent(self) -> PostProcessingAgent:
        """Create PostProcessingAgent with mocked dependencies."""
        with patch("agents.post_processing_agent.OPENAI_API_KEY", "test-key"):
            with patch("agents.post_processing_agent.USE_LLM_JUDGE", False):
                agent = PostProcessingAgent()
                agent.client = Mock()
                return agent

    def _dummy_sources(self) -> List[Dict[str, Any]]:
        return [
            {
                "content": "This is a source document about AI and machine learning.",
                "file_name": "doc1.txt",
                "modality": "text",
            }
        ]

    def test_process_without_judge_uses_heuristics(self, agent: PostProcessingAgent):
        """When judge is disabled, process should work with heuristic-only logic."""
        answer = "AI is a field of computer science. [Source 1]"
        sources = self._dummy_sources()

        # Ensure we are not using the judge in this test
        agent.use_llm_judge = False

        result = agent.process(answer=answer, sources=sources, query="What is AI?")

        assert "final_answer" in result
        assert "confidence" in result
        assert "hallucination_score" in result
        # No judge details should be attached
        assert "llm_judge" not in result

    def test_process_with_judge_result_attached(self):
        """When judge is enabled and returns a result, it is attached and used."""
        with patch("agents.post_processing_agent.OPENAI_API_KEY", "test-key"):
            with patch("agents.post_processing_agent.USE_LLM_JUDGE", True):
                agent = PostProcessingAgent()
                agent.client = Mock()

        answer = "AI is a field of computer science. [Source 1]"
        sources = self._dummy_sources()

        # Provide a stable fake judge result
        fake_judge = {
            "faithfulness_score": 0.9,
            "hallucination_score": 0.1,
            "confidence_score": 0.8,
            "rationale": "The answer closely matches the provided sources.",
        }

        with patch.object(agent, "llm_judge", return_value=fake_judge):
            result = agent.process(answer=answer, sources=sources, query="What is AI?")

        # Judge details should be surfaced
        assert result.get("llm_judge") == fake_judge
        # Hallucination score should primarily come from judge signal
        assert abs(result.get("hallucination_score", 0.0) - 0.1) < 1e-6

    def test_process_falls_back_when_judge_raises(self, agent: PostProcessingAgent):
        """If the judge fails, the agent should fall back cleanly to heuristics."""
        answer = "AI is a field of computer science. [Source 1]"
        sources = self._dummy_sources()

        agent.use_llm_judge = True

        with patch.object(agent, "llm_judge", side_effect=RuntimeError("boom")):
            result = agent.process(answer=answer, sources=sources, query="What is AI?")

        # Still returns a complete result
        assert "final_answer" in result
        assert "confidence" in result
        assert "hallucination_score" in result
        # No judge key when judge fails
        assert "llm_judge" not in result

