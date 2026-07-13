
import json
import logging
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from src.embeddings.chunk_embedder import ChunkRetriever
from src.llm.client import MistralClient
from src.llm.prompts import build_legal_prompt
from src.llm.config import PipelineConfig

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
    
    def _log_interaction(self, timestamp: str, query: str, cases: List[Dict], response_text: str, token_usage: Dict) -> None:
        try:
            log_path = Path(self.config.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            retrieved_cases = []
            for rank,case in enumerate(cases, start=1):
                
                first_chunk = case['chunks'][0]
                outcome = 'violation' if case['violated_articles'] else 'no_violation'
                
                case_entry = {
                    'rank': rank,
                    'case_id': case['case_id'],
                    'case_no': case['case_no'],
                    'title': case['title'],
                    'judgment_date': case['judgment_date'],
                    'avg_similarity': float(case['avg_similarity']),
                    'max_similarity': float(case['max_similarity']),
                    'num_chunks': case['num_chunks'],
                    'country': first_chunk.get('defendants', [None])[0],
                    'gender': first_chunk.get('classification', {}).get('gender'),
                    'age': first_chunk.get('age_info', {}).get('age_at_judgment'),
                    'violated_articles': case['violated_articles'],
                    'outcome': outcome
                }
                
                retrieved_cases.append(case_entry)
            
            log_entry = {
                'timestamp': timestamp,
                'query': query,
                'retrieved_cases': retrieved_cases,
                'response': response_text,
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
            
        
    def query(self, legal_query: str) -> Dict:
        
        timestamp = datetime.now().isoformat()
        chunks = self.retriever.retrieve_chunks(legal_query, self.config.top_k_chunks)
        cases = self.retriever.aggregate_chunks_to_cases(chunks, self.config.top_k_cases)
        
        messages = build_legal_prompt(legal_query,cases)
        
        llm_response = self.llm_client.generate(messages)
        
        response_text = llm_response['response']
        
        self._log_interaction(
            timestamp = timestamp,
            query = legal_query,
            cases = cases,
            response_text = response_text,
            token_usage = {
                'token_count': llm_response['token_count'],
                'prompt_tokens': llm_response['prompt_tokens'], 
                'completion_tokens': llm_response['completion_tokens'] 
            }
        )
        
        return {
            'query': legal_query,
            'cases': cases,
            'response': response_text 
        }

