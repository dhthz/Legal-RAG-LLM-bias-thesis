import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "FROZEN_BASELINE_MANIFEST.json"


@dataclass
class PipelineConfig:
    # LLM Settings
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "mistral"
    temperature: float = 0.1
    max_tokens: int = 500

    # RAG/Retrieval Settings
    top_k_chunks: int = 25
    top_k_cases: int = 3

    # Index Paths
    index_path: str = "faiss_indices/paragraph_chunks_l2.index"
    metadata_path: str = "faiss_indices/paragraph_chunks_metadata_enriched.json"

    # Log path
    log_path: str = "logs/pipeline_interactions.jsonl"

    @classmethod
    def load_from_manifest(cls, manifest_path: Path = MANIFEST_PATH):
        if not manifest_path.exists():
            print(f"Warning: manifest not found at {manifest_path}, using defaults")
            return cls()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        llm = manifest.get("llm", {})
        gen = manifest.get("generation_params", {})
        paths = manifest.get("paths", {})

        return cls(
            base_url=llm.get("base_url", cls.base_url),
            api_key=llm.get("api_key", cls.api_key),
            model=llm.get("model", cls.model),
            temperature=float(gen.get("temperature", cls.temperature)),
            max_tokens=int(gen.get("max_tokens", cls.max_tokens)),
            top_k_chunks=int(gen.get("top_k_chunks", cls.top_k_chunks)),
            top_k_cases=int(gen.get("top_k_cases", cls.top_k_cases)),
            index_path=paths.get("index_path", cls.index_path),
            metadata_path=paths.get("metadata_path", cls.metadata_path),
            log_path=paths.get("log_path", cls.log_path),
        )
