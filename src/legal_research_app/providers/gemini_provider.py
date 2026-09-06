from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors

from legal_research_app.providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ProviderError,
    ProviderHealth,
    TokenUsage,
)


class GeminiProvider(LLMProvider):
    """Google Gemini. Note: intended as an optional/secondary provider only --
    see config.py profile comments on free-tier reliability. Never used as a
    silent fallback for another provider's failure.
    """

    name = "gemini"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client = genai.Client(api_key=api_key) if api_key else None

    def health(self) -> ProviderHealth:
        if not self._api_key:
            return ProviderHealth(available=False, detail="GEMINI_API_KEY is not set.")
        return ProviderHealth(
            available=True,
            detail="API key configured (free-tier: reliability not guaranteed).",
        )

    def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self._client is None:
            raise ProviderError(
                "Gemini request failed: GEMINI_API_KEY is not set. "
                "Set it in .env or switch the active provider profile."
            )

        system_parts = [m.content for m in messages if m.role == "system"]
        contents = [m.content for m in messages if m.role != "system"]

        config: dict = {"temperature": temperature}
        if system_parts:
            config["system_instruction"] = "\n".join(system_parts)
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as exc:
            raise ProviderError(
                f"Gemini request failed (client error): {exc}. "
                "This may be a free-tier rate limit -- retry, or switch provider profile. "
                "No fallback provider will be used automatically."
            ) from exc
        except genai_errors.ServerError as exc:
            raise ProviderError(
                f"Gemini request failed (server error): {exc}. "
                "Retry, or switch to a different provider profile."
            ) from exc

        usage = None
        if getattr(response, "usage_metadata", None) is not None:
            um = response.usage_metadata
            usage = TokenUsage(
                prompt_tokens=um.prompt_token_count or 0,
                completion_tokens=um.candidates_token_count or 0,
                total_tokens=um.total_token_count or 0,
            )
        return LLMResponse(
            content=response.text or "",
            provider=self.name,
            model=model,
            usage=usage,
        )
