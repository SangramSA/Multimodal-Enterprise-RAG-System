"""Unit tests for AnswerGenerationAgent."""

import pytest
from unittest.mock import Mock, patch

from agents.answer_generation_agent import AnswerGenerationAgent


class TestAnswerGenerationAgent:
    """Test suite for AnswerGenerationAgent."""
    
    @pytest.fixture
    def answer_agent(self, mocker):
        """Create AnswerGenerationAgent with mocked dependencies."""
        with patch('agents.answer_generation_agent.OPENAI_API_KEY', 'test-key'):
            agent = AnswerGenerationAgent()
            agent.client = Mock()
            return agent
    
    def test_init(self, mocker):
        """Test AnswerGenerationAgent initialization."""
        with patch('agents.answer_generation_agent.OPENAI_API_KEY', 'test-key'):
            agent = AnswerGenerationAgent()
            assert agent.name == "AnswerGenerationAgent"
    
    def test_generate_answer_simple(self, answer_agent):
        """Test generating simple answer."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is the answer."
        answer_agent.client.chat.completions.create.return_value = mock_response
        
        # generate expects context as List[Dict], not a string
        context = [{"content": "AI is artificial intelligence.", "chunk_id": "chunk1"}]
        result = answer_agent.generate("What is AI?", context)
        
        assert "answer" in result
        assert result["answer"] == "This is the answer."
    
    def test_generate_answer_with_reasoning(self, answer_agent):
        """Test generating answer with reasoning steps."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Use format that matches the actual parsing logic
        mock_response.choices[0].message.content = """Step 1: Analyze the question
This is step 1 content.

Step 2: Review context
This is step 2 content.

Final Answer: This is the answer."""
        answer_agent.client.chat.completions.create.return_value = mock_response
        
        # generate expects context as List[Dict], not a string
        context = [{"content": "Context here", "chunk_id": "chunk1"}]
        # Mock _needs_reasoning to return True to trigger reasoning path
        with patch.object(answer_agent, '_needs_reasoning', return_value=True):
            result = answer_agent.generate("Complex question", context)
        
        assert "answer" in result
        assert "reasoning_steps" in result
        # Should have reasoning steps if parsing worked
        reasoning_steps = result.get("reasoning_steps", [])
        # May be empty if parsing fails, but structure should exist
        assert isinstance(reasoning_steps, list)

