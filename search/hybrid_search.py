"""Hybrid search combining keyword, vector, and graph search."""

from typing import List, Dict, Any, Optional
from loguru import logger

from search.keyword_search import KeywordSearch
from search.vector_search import VectorSearch
from search.graph_search import GraphSearch
from middleware.reranker import CrossEncoderReranker, RerankerConfig


class HybridSearch:
    """Combine multiple search methods with RRF reranking."""
    
    def __init__(
        self,
        keyword_search: KeywordSearch,
        vector_search: VectorSearch,
        graph_search: GraphSearch,
        *,
        use_final_rerank: bool = False,
        final_rerank_top_n: int = 20,
        final_rerank_config: Optional[RerankerConfig] = None,
    ):
        self.keyword_search = keyword_search
        self.vector_search = vector_search
        self.graph_search = graph_search
        self.rrf_k = 60  # RRF constant

        self.use_final_rerank = use_final_rerank
        self.final_rerank_top_n = max(0, int(final_rerank_top_n))
        self._final_reranker: Optional[CrossEncoderReranker]
        if self.use_final_rerank and self.final_rerank_top_n > 0:
            # Use a dedicated config so we can tune top-N separately from vector leg.
            config = final_rerank_config or RerankerConfig(
                rerank_k=self.final_rerank_top_n
            )
            self._final_reranker = CrossEncoderReranker(config)
        else:
            self._final_reranker = None
    
    def reciprocal_rank_fusion(self, result_lists: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Args:
            result_lists: List of ranked result lists
        
        Returns:
            Dictionary mapping result IDs to RRF scores
        """
        scores = {}
        
        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                result_id = result.get("chunk_id") or result.get("file_id") or str(id(result))
                if result_id not in scores:
                    scores[result_id] = 0.0
                scores[result_id] += 1.0 / (self.rrf_k + rank)
        
        return scores
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_keyword: bool = True,
        use_vector: bool = True,
        use_graph: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining all methods.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional metadata filters
            use_keyword: Whether to use keyword search
            use_vector: Whether to use vector search
            use_graph: Whether to use graph search
        
        Returns:
            Combined and reranked results
        """
        logger.info(
            "HybridSearch.search | query_prefix='{}' | limit={} | keyword={} | vector={} | graph={} | final_rerank={}",
            query[:80],
            limit,
            use_keyword,
            use_vector,
            use_graph,
            self.use_final_rerank,
        )

        all_results = []
        result_lists = []
        
        # Keyword search
        if use_keyword:
            try:
                keyword_results = self.keyword_search.search(query, limit=limit * 2, filters=filters)
                result_lists.append(keyword_results)
                all_results.extend(keyword_results)
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")
        
        # Vector search
        if use_vector:
            try:
                vector_results = self.vector_search.search(query, limit=limit * 2, filters=filters)
                result_lists.append(vector_results)
                all_results.extend(vector_results)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # Graph search (extract entities from query first)
        if use_graph:
            try:
                # Simple entity extraction from query (in production, use proper NER)
                query_words = query.split()
                # Try to find entities in graph
                graph_results = []
                for word in query_words:
                    if len(word) > 3:  # Skip short words
                        entity_results = self.graph_search.search_by_entity(word, limit=5)
                        graph_results.extend(entity_results)
                
                if graph_results:
                    result_lists.append(graph_results)
                    all_results.extend(graph_results)
            except Exception as e:
                logger.warning(f"Graph search failed: {e}")
        
        # Combine using RRF
        if not result_lists:
            logger.info("HybridSearch.search | no results from any modality")
            return []
        
        rrf_scores = self.reciprocal_rank_fusion(result_lists)
        
        # Create result map
        result_map = {}
        for result in all_results:
            result_id = result.get("chunk_id") or result.get("file_id") or str(id(result))
            if result_id not in result_map:
                result_map[result_id] = result
                result_map[result_id]["rrf_score"] = rrf_scores.get(result_id, 0.0)
            else:
                # Merge results from different sources
                existing = result_map[result_id]
                existing["rrf_score"] = max(existing.get("rrf_score", 0.0), rrf_scores.get(result_id, 0.0))
                # Combine scores
                if "keyword_score" in result:
                    existing["keyword_score"] = max(existing.get("keyword_score", 0.0), result.get("keyword_score", 0.0))
                if "score" in result:
                    existing["vector_score"] = max(existing.get("vector_score", 0.0), result.get("score", 0.0))
        
        # Sort by RRF score and deduplicate
        final_results = list(result_map.values())
        final_results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        
        # Remove duplicates based on content similarity
        deduplicated = []
        seen_contents = set()
        for result in final_results:
            content = result.get("content", "")[:100]  # Use first 100 chars for deduplication
            content_hash = hash(content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                deduplicated.append(result)
                if len(deduplicated) >= limit:
                    break

        # Optional final-stage cross-encoder rerank over top-N fused results.
        if (
            self.use_final_rerank
            and self._final_reranker is not None
            and deduplicated
            and self.final_rerank_top_n > 0
        ):
            top_n = min(self.final_rerank_top_n, len(deduplicated))
            prefix = deduplicated[:top_n]
            suffix = deduplicated[top_n:]

            logger.info(
                "HybridSearch.final_rerank | query_prefix='{}' | top_n={} | total_results={}",
                query[:80],
                top_n,
                len(deduplicated),
            )

            try:
                reranked_prefix, _timings = self._final_reranker.rerank(
                    query=query,
                    docs=prefix,
                    rerank_k=top_n,
                    score_key="rrf_score",
                )
                deduplicated = reranked_prefix + suffix
            except Exception as e:
                logger.warning("HybridSearch.final_rerank failed, using RRF ordering: {}", e)

        logger.info(
            "HybridSearch.search | final_results={}",
            len(deduplicated),
        )

        return deduplicated

