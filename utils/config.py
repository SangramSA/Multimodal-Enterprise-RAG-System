"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# LLM-as-judge configuration
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "false").lower() in {"1", "true", "yes"}
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", OPENAI_MODEL)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = "multimodal_rag"

# Application Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
UPLOAD_DIR = PROJECT_ROOT / "uploads"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Evaluation Configuration
EVAL_MODE = os.getenv("EVAL_MODE", "development")
EVAL_LOG_PATH = os.getenv("EVAL_LOG_PATH", str(LOGS_DIR / "eval_results.json"))
DEEPEVAL_CACHE_ENABLED = os.getenv("DEEPEVAL_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}

# Hallucination / confidence thresholds (used for future rule layers / telemetry)
HALLUCINATION_ALERT_THRESHOLD = float(os.getenv("HALLUCINATION_ALERT_THRESHOLD", "0.5"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.4"))

# Confident AI Configuration
# Note: DeepEval uses CONFIDENT_API_KEY (not CONFIDENT_AI_API_KEY) for automatic uploads
CONFIDENT_API_KEY = os.getenv("CONFIDENT_API_KEY") or os.getenv("CONFIDENT_AI_API_KEY")
CONFIDENT_AI_API_KEY = os.getenv("CONFIDENT_AI_API_KEY")  # Legacy support
CONFIDENT_AI_PROJECT = os.getenv("CONFIDENT_AI_PROJECT")
CONFIDENT_AI_BASE_URL = os.getenv("CONFIDENT_AI_BASE_URL", "https://api.confident-ai.com")
CONFIDENT_AI_ENABLED = os.getenv("CONFIDENT_AI_ENABLED", "false").lower() in {"1", "true", "yes"}

# Set CONFIDENT_API_KEY for DeepEval's automatic integration
if CONFIDENT_API_KEY and not os.getenv("CONFIDENT_API_KEY"):
    import os as os_module
    os_module.environ["CONFIDENT_API_KEY"] = CONFIDENT_API_KEY

# Domain Tags
DOMAIN_TAGS = [
    "finance",
    "legal",
    "technical",
    "medical",
    "HR",
    "marketing",
    "operations",
    "sales",
    "customer_support"
]

# Create necessary directories
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def validate_config() -> tuple[bool, Optional[str]]:
    """Validate that required configuration is present."""
    if not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY is required"
    return True, None

