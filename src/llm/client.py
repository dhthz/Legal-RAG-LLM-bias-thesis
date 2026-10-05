from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

from .config import PipelineConfig


@dataclass
class Generation:
    text: Optional[str]
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


class MistralClient:

    def __init__(self):
        self.config = PipelineConfig.load_from_manifest()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            http_client=httpx.Client(verify=False),
        )

    # A failed call is returned as a Generation with an error rather than raised, so callers can still log retrieval
    def generate(self, messages: List[Dict]) -> Generation:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as e:
            return Generation(text=None, error=str(e))

        return Generation(
            text=response.choices[0].message.content,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    def test_connection(self) -> bool:
        result = self.generate([
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Say OK if you can read this."},
        ])
        if result.error:
            print(f"Connection test failed: {result.error}")
        return bool(result.text)
