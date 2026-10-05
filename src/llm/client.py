from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from .config import PipelineConfig

# Ollama's native chat API: unlike its OpenAI-compatible endpoint it honours num_ctx, so long prompts are not
# silently cut to Ollama's default 4096-token context
CHAT_ENDPOINT = "/api/chat"
REQUEST_TIMEOUT_S = 600


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
        self.client = httpx.Client(base_url=self.config.base_url, timeout=REQUEST_TIMEOUT_S)

    # A failed call is returned as a Generation with an error rather than raised, so callers can still log retrieval
    def generate(self, messages: List[Dict]) -> Generation:
        try:
            response = self.client.post(CHAT_ENDPOINT, json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "num_ctx": self.config.num_ctx,
                },
            })
            response.raise_for_status()
            body = response.json()
        except Exception as e:
            return Generation(text=None, error=str(e))

        prompt_tokens, completion_tokens = body.get("prompt_eval_count", 0), body.get("eval_count", 0)
        # Ollama truncates an over-long prompt without failing; treat a full context window as an error, never a silent cut
        if prompt_tokens + completion_tokens >= self.config.num_ctx:
            return Generation(text=None, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                              total_tokens=prompt_tokens + completion_tokens,
                              error=f"context window full ({prompt_tokens}+{completion_tokens} >= num_ctx {self.config.num_ctx})")

        return Generation(
            text=body["message"]["content"],
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def test_connection(self) -> bool:
        result = self.generate([
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Say OK if you can read this."},
        ])
        if result.error:
            print(f"Connection test failed: {result.error}")
        return bool(result.text)
