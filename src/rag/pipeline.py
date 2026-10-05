import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.embeddings.chunk_embedder import ChunkRetriever
from src.llm.client import Generation, MistralClient
from src.llm.config import MANIFEST_PATH, PipelineConfig
from src.llm.prompts import build_legal_prompt
from src.mitigation import MitigationArm
from src.rag.response_parsing import parse_structured_response

# Configure logging at module level (called once)
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class RAGPipeline:

    # Mitigation arms are applied in the order given; with none, the pipeline is the frozen system
    def __init__(self, predict_articles: bool = False, arms: Sequence[MitigationArm] = ()):
        self.config = PipelineConfig.load_from_manifest()
        self.predict_articles = predict_articles
        self.arms = list(arms)

        self.retriever = ChunkRetriever(
            index_path=self.config.index_path,
            metadata_path=self.config.metadata_path,
        )

        self.llm_client = MistralClient()
        if predict_articles:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                variant = json.load(f)["prompt_variants"]["predict_articles"]
            self.llm_client.config.max_tokens = int(variant["max_tokens"])

        # Country/jurisdiction isn't carried in the chunk-level FAISS metadata,
        # so it's joined at logging time from the per-case dataset file instead.
        self._case_country = {}
        train_metadata_path = Path("dataset/train_with_metadata.jsonl")
        if train_metadata_path.exists():
            with open(train_metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    defendants = rec.get("defendants") or []
                    self._case_country[rec["case_id"]] = defendants[0] if defendants else None

    def _case_log_entry(self, rank: int, case: Dict) -> Dict:
        return {
            'rank': rank,
            'case_id': case['case_id'],
            'case_no': case['case_no'],
            'title': case['title'],
            'judgment_date': case['judgment_date'],
            'avg_similarity': float(case['avg_similarity']),
            'max_similarity': float(case['max_similarity']),
            'num_chunks': case['num_chunks'],
            'country': self._case_country.get(case['case_id']),
            'gender': case['chunks'][0].get('gender'),
            'violated_articles': case['violated_articles'],
            'outcome': 'violation' if case['violated_articles'] else 'no_violation',
            'chunks': [
                {
                    'chunk_id': chunk.get('chunk_id'),
                    'similarity_score': float(chunk['similarity_score']),
                    'word_count': chunk.get('word_count'),
                }
                for chunk in case['chunks']
            ],
        }

    def _log_interaction(self, timestamp: str, query: str, query_id: str, effective_query: str,
                         cases: List[Dict], generation: Generation) -> None:
        try:
            log_path = Path(self.config.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if generation.text:
                parsed = parse_structured_response(cases, generation.text)
            else:
                parsed = {"cited_case_ids": [], "case_articles": {}, "json_parse_ok": False}

            log_entry = {
                'timestamp': timestamp,
                'query_id': query_id,
                'query': query,
                # Only present when an arm rewrote the query, so frozen-system rows keep their exact format
                **({'rewritten_query': effective_query} if effective_query != query else {}),
                'retrieved_cases': [self._case_log_entry(rank, case) for rank, case in enumerate(cases, start=1)],
                'cited_case_ids': parsed['cited_case_ids'],
                'case_articles': parsed['case_articles'],
                'json_parse_ok': parsed['json_parse_ok'],
                **({'predicted_articles': parsed.get('predicted_articles')} if self.predict_articles else {}),
                'prompt_variant': 'predict_articles' if self.predict_articles else 'baseline',
                'arms': [arm.name for arm in self.arms],
                'response': generation.text,
                'generation_error': generation.error,
                'token_usage': {
                    'total_tokens': generation.total_tokens,
                    'prompt_tokens': generation.prompt_tokens,
                    'completion_tokens': generation.completion_tokens,
                },
            }

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    def _transform_query_vector(self, vector: np.ndarray) -> np.ndarray:
        for arm in self.arms:
            vector = arm.transform_query_vector(vector)
        return vector

    # Returns the query after any rewriting arm (used for both search and generation) and the ranked cases
    def retrieve(self, legal_query: str) -> Tuple[str, List[Dict]]:
        effective_query = legal_query
        for arm in self.arms:
            effective_query = arm.rewrite_query(effective_query)

        chunks = self.retriever.retrieve_chunks(effective_query, self.config.top_k_chunks, transform=self._transform_query_vector)
        for arm in self.arms:
            chunks = arm.rescore_chunks(effective_query, chunks, self.retriever, self.config.top_k_chunks)

        return effective_query, self.retriever.aggregate_chunks_to_cases(chunks, self.config.top_k_cases)

    def query(self, legal_query: str, query_id: str = None) -> Dict:
        timestamp = datetime.now().isoformat()
        effective_query, cases = self.retrieve(legal_query)

        # The LLM sees the same text that was searched, so a query-rewriting arm applies to the whole system
        messages = build_legal_prompt(effective_query, cases, predict_articles=self.predict_articles)
        generation = self.llm_client.generate(messages)
        if generation.error:
            logger.error(f"Generation failed for query_id={query_id}: {generation.error}")

        # Retrieval is logged even if generation failed, so a flaky LLM call never drops a query from the audit
        self._log_interaction(timestamp, legal_query, query_id, effective_query, cases, generation)

        return {
            'query': legal_query,
            'cases': cases,
            'response': generation.text,
            'generation_error': generation.error,
        }
