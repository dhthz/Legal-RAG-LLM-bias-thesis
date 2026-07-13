from typing import List, Dict, Any
import httpx
from openai import OpenAI
from .config import PipelineConfig

class MistralClient:
    
    def __init__(self):
        self.config = PipelineConfig.load_from_manifest()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            http_client=httpx.Client(verify=False),
        )
    
    def generate(self, messages: List[Dict]) -> Dict[str, Any]:
        # Generated the response from Mistral, returns a dictionary with response:str, token_count:int (prompt+completion), prompt_tokens:int & completion_tokens:int
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return {
            'response': response.choices[0].message.content,
            'token_count': response.usage.total_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens
        }
    
    def test_connection(self) -> bool:
        try:
            test_messages = [
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Say OK if you can read this."}
            ]
            result = self.generate(test_messages)
            # Check if we got a valid response back
            return 'response' in result and len(result['response']) > 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False