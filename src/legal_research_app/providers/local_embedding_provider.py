"""Local embedding provider: BGE-M3 via sentence-transformers, runs in-process
on this machine/server -- no network call, works under LOCAL_ONLY privacy mode.

Model instances are cached per model name at module scope: loading BGE-M3 takes
real time and memory, and every task/session sharing one loaded model instead
of reloading it per-call is essential on a 16GB machine (master prompt section 51).
"""

from __future__ import annotations

from legal_research_app.providers.base import EmbeddingProvider, ProviderError, ProviderHealth

_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer  # deferred: heavy import

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class LocalEmbeddingProvider(EmbeddingProvider):
    name = "local"

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True, detail="Runs in-process; no external dependency.")

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        try:
            embedder = _get_model(model)
            vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:
            raise ProviderError(
                f"Local embedding failed for model '{model}': {exc}. "
                "Check the model name and that enough memory/disk is available."
            ) from exc
        return [v.tolist() for v in vectors]
