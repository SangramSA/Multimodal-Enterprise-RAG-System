"""Utility script to manage DeepEval cache."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.deepeval_cache import get_cache
from loguru import logger


def main():
    """Main entry point for cache management."""
    parser = argparse.ArgumentParser(description="Manage DeepEval evaluation cache")
    parser.add_argument(
        "action",
        choices=["clear", "stats", "show"],
        help="Action to perform: clear (delete cache), stats (show statistics), show (display cache contents)"
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable caching (for this run only)"
    )
    
    args = parser.parse_args()
    
    cache = get_cache(enabled=not args.disable)
    
    if args.action == "clear":
        cache.clear()
        logger.success("Cache cleared successfully")
        return 0
    
    elif args.action == "stats":
        stats = cache.get_stats()
        logger.info("DeepEval Cache Statistics:")
        logger.info(f"  Enabled: {stats['enabled']}")
        logger.info(f"  Cache file: {stats['cache_file']}")
        logger.info(f"  Cached entries: {stats['cached_entries']}")
        logger.info(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
        return 0
    
    elif args.action == "show":
        if not cache.cache:
            logger.info("Cache is empty")
            return 0
        
        logger.info(f"Cache contains {len(cache.cache)} entries:")
        for i, (key, value) in enumerate(list(cache.cache.items())[:10], 1):
            logger.info(f"\nEntry {i} (key: {key[:16]}...):")
            logger.info(f"  Hallucination: {value.get('hallucination_score')}")
            logger.info(f"  Relevancy: {value.get('answer_relevancy_score')}")
            logger.info(f"  Faithfulness: {value.get('faithfulness_score')}")
        
        if len(cache.cache) > 10:
            logger.info(f"\n... and {len(cache.cache) - 10} more entries")
        
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())

