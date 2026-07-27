
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.embeddings.chunk_embedder import ChunkRetriever
from src.llm.client import MistralClient
from src.llm.config import PipelineConfig
from src.llm.prompts import build_legal_prompt
from src.rag.response_parsing import parse_structured_response

# Configure logging at module level (called once)
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class RAGPipeline():

    def __init__(self):
        self.config = PipelineConfig.load_from_manifest()

        self.retriever = ChunkRetriever(
            index_path = self.config.index_path,
            metadata_path = self.config.metadata_path
        )

        self.llm_client = MistralClient()

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

    def _log_interaction(self, timestamp: str, query: str, query_id: str, cases: List[Dict], response_text: str, token_usage: Dict, generation_error: str = None) -> None:
        try:
            log_path = Path(self.config.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if response_text:
                parsed = parse_structured_response(cases, response_text)
            else:
                parsed = {"cited_case_ids": [], "case_articles": {}, "json_parse_ok": False}

            retrieved_cases = []
            for rank,case in enumerate(cases, start=1):

                first_chunk = case['chunks'][0]
                outcome = 'violation' if case['violated_articles'] else 'no_violation'

                chunk_entries = [
                    {
                        'chunk_id': chunk.get('chunk_id'),
                        'similarity_score': float(chunk['similarity_score']),
                        'word_count': chunk.get('word_count'),
                    }
                    for chunk in case['chunks']
                ]

                case_entry = {
                    'rank': rank,
                    'case_id': case['case_id'],
                    'case_no': case['case_no'],
                    'title': case['title'],
                    'judgment_date': case['judgment_date'],
                    'avg_similarity': float(case['avg_similarity']),
                    'max_similarity': float(case['max_similarity']),
                    'num_chunks': case['num_chunks'],
                    'country': self._case_country.get(case['case_id']),
                    'gender': first_chunk.get('gender'),
                    'age': first_chunk.get('age_at_judgment'),
                    'violated_articles': case['violated_articles'],
                    'outcome': outcome,
                    'chunks': chunk_entries,
                }

                retrieved_cases.append(case_entry)

            log_entry = {
                'timestamp': timestamp,
                'query_id': query_id,
                'query': query,
                'retrieved_cases': retrieved_cases,
                'cited_case_ids': parsed['cited_case_ids'],
                'case_articles': parsed['case_articles'],
                'json_parse_ok': parsed['json_parse_ok'],
                'response': response_text,
                'generation_error': generation_error,
                'token_usage': {
                    'total_tokens': token_usage['token_count'],
                    'prompt_tokens': token_usage['prompt_tokens'],
                    'completion_tokens': token_usage['completion_tokens']
                }
            }

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")


    def query(self, legal_query: str, query_id: str = None) -> Dict:

        timestamp = datetime.now().isoformat()
        chunks = self.retriever.retrieve_chunks(legal_query, self.config.top_k_chunks)
        cases = self.retriever.aggregate_chunks_to_cases(chunks, self.config.top_k_cases)

        messages = build_legal_prompt(legal_query,cases)

        # Retrieval must be logged even if generation fails, so a flaky LLM
        # call never silently drops a query's retrieval results from the audit.
        try:
            llm_response = self.llm_client.generate(messages)
            response_text = llm_response['response']
            token_usage = {
                'token_count': llm_response['token_count'],
                'prompt_tokens': llm_response['prompt_tokens'],
                'completion_tokens': llm_response['completion_tokens']
            }
            generation_error = None
        except Exception as e:
            logger.error(f"Generation failed for query_id={query_id}: {e}")
            response_text = None
            token_usage = {'token_count': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
            generation_error = str(e)

        self._log_interaction(
            timestamp = timestamp,
            query = legal_query,
            query_id = query_id,
            cases = cases,
            response_text = response_text,
            token_usage = token_usage,
            generation_error = generation_error,
        )

        return {
            'query': legal_query,
            'cases': cases,
            'response': response_text,
            'generation_error': generation_error,
        }
