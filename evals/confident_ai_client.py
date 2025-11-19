"""Utility client for pushing evaluation results to Confident AI."""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import requests
from loguru import logger

from utils.config import (
    CONFIDENT_AI_API_KEY,
    CONFIDENT_AI_PROJECT,
    CONFIDENT_AI_BASE_URL,
    CONFIDENT_AI_ENABLED,
)


class ConfidentAIClient:
    """Thin wrapper around Confident AI's Evaluation API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.api_key = api_key or CONFIDENT_AI_API_KEY
        self.project = project or CONFIDENT_AI_PROJECT
        self.base_url = (base_url or CONFIDENT_AI_BASE_URL or "").rstrip("/")
        self.enabled = enabled if enabled is not None else CONFIDENT_AI_ENABLED

        if not self.api_key or not self.project:
            self.enabled = False
            logger.debug(
                "Confident AI client disabled (missing CONFIDENT_AI_API_KEY or CONFIDENT_AI_PROJECT)."
            )

    def is_enabled(self) -> bool:
        return self.enabled

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _format_test_cases(self, per_test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for case in per_test_results:
            formatted.append(
                {
                    "query": case.get("query"),
                    "expected_answer": case.get("expected_answer"),
                    "answer": case.get("answer"),
                    "dataset": case.get("dataset"),
                    "modality": case.get("modality"),
                    "latency_seconds": case.get("latency"),
                    "metrics": case.get("deepeval", {}),
                }
            )
        return formatted

    def upload_results(self, evaluation_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Note: This method is deprecated. DeepEval automatically handles Confident AI uploads
        when using the evaluate() function and CONFIDENT_API_KEY is set.
        
        For manual metric evaluation (using measure() directly), Confident AI uploads
        should be handled through DeepEval's native integration.
        
        This method now only logs a warning and returns None.
        """
        if not self.is_enabled():
            return None

        logger.warning(
            "Custom Confident AI upload is deprecated. "
            "DeepEval automatically uploads results when using evaluate() function. "
            "Set CONFIDENT_API_KEY environment variable for automatic uploads."
        )
        logger.info(
            "To enable automatic Confident AI uploads: "
            "1. Set CONFIDENT_API_KEY environment variable (not CONFIDENT_AI_API_KEY) "
            "2. Use DeepEval's evaluate() function instead of measure() directly"
        )
        
        return None


def get_confident_ai_client() -> ConfidentAIClient:
    return ConfidentAIClient()

