from __future__ import annotations

import httpx
import ollama

from legal_research_app.providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ProviderError,
    ProviderHealth,
)


class OllamaProvider(LLMProvider):
    """Local Ollama. Fully private -- no network call leaves the machine/server
    this process runs on. Suitable for LOCAL_ONLY privacy mode and for dev/testing,
    not for concurrent multi-user production serving on modest hardware.
    """

    name = "ollama"

    def __init__(self, host: str) -> None:
        self._host = host
        self._client = ollama.Client(host=host)

    def health(self) -> ProviderHealth:
        try:
            response = self._client.list()
        except Exception as exc:  # connection refused, daemon not running, etc.
            return ProviderHealth(
                available=False,
                detail=f"Could not reach Ollama at {self._host}: {exc}",
            )
        tags = {m.model for m in response.models}
        return ProviderHealth(available=True, detail=f"{len(tags)} model(s) pulled.")

    def _model_is_pulled(self, model: str) -> bool:
        try:
            response = self._client.list()
        except Exception:
            return False
        return any(m.model == model or m.model.startswith(f"{model}:") for m in response.models)

    def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self._model_is_pulled(model):
            raise ProviderError(
                f"Ollama model '{model}' is not pulled on this host ({self._host}). "
                f"Run `ollama pull {model}` first, or switch the active provider profile. "
                "Local mode does not fall back to a cloud provider."
            )
        try:
            response = self._client.chat(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                options={"temperature": temperature, **({"num_predict": max_tokens} if max_tokens else {})},
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderError(
                f"Ollama request failed: could not connect to {self._host}. "
                "Is the Ollama daemon running? Local mode does not fall back to a cloud provider."
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Ollama request failed: {exc}. Local mode does not fall back to a cloud provider."
            ) from exc

        return LLMResponse(
            content=response.message.content or "",
            provider=self.name,
            model=model,
        )
