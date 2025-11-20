"""Common unit tests for agent classes."""

import pytest
from unittest.mock import Mock, patch

from agents.query_validation_agent import QueryValidationAgent


class TestAgentCommon:
    """Common test patterns for all agents."""
    
    @pytest.fixture
    def agent_instance(self, mocker):
        """Create a sample agent instance for testing."""
        with patch('agents.query_validation_agent.OPENAI_API_KEY', 'test-key'):
            agent = QueryValidationAgent()
            agent.validator = Mock()
            agent.validator.validate_query.return_value = {
                "is_valid": True,
                "sanitized_query": "test"
            }
            return agent
    
    def test_agent_has_name(self, agent_instance):
        """Test that agent has a name attribute."""
        assert hasattr(agent_instance, 'name')
        assert agent_instance.name is not None
    
    def test_agent_has_process_method(self, agent_instance):
        """Test that agent implements process method."""
        assert hasattr(agent_instance, 'process')
        assert callable(agent_instance.process)
    
    def test_agent_logging_methods(self, agent_instance):
        """Test that agent has logging methods."""
        assert hasattr(agent_instance, 'log_info')
        assert hasattr(agent_instance, 'log_error')
        assert hasattr(agent_instance, 'log_warning')
        assert hasattr(agent_instance, 'log_debug')

