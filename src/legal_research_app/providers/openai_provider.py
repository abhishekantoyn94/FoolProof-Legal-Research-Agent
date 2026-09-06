from __future__ import annotations

from openai import APIError, APIConnectionError, AuthenticationError, OpenAI

from legal_research_app.providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ProviderError,
    ProviderHealth,
    TokenUsage,
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key) if api_key else None

    def health(self) -> ProviderHealth:
        if not self._api_key:
            return ProviderHealth(available=False, detail="OPENAI_API_KEY is not set.")
        return ProviderHealth(available=True, detail="API key configured.")

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
                "OpenAI request failed: OPENAI_API_KEY is not set. "
                "Set it in .env or switch the active provider profile."
            )
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except AuthenticationError as exc:
            raise ProviderError(
                "OpenAI request failed: the API key was rejected. "
                "Check OPENAI_API_KEY in .env. No document data was sent successfully."
            ) from exc
        except APIConnectionError as exc:
            raise ProviderError(
                "OpenAI request failed: could not reach the OpenAI API (network issue). "
                "Retry, or switch to a different provider profile."
            ) from exc
        except APIError as exc:
            raise ProviderError(
                f"OpenAI request failed: {exc.message if hasattr(exc, 'message') else exc}. "
                "Retry, or switch to a different provider profile."
            ) from exc

        choice = response.choices[0]
        usage = None
        if response.usage is not None:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return LLMResponse(
            content=choice.message.content or "",
            provider=self.name,
            model=model,
            usage=usage,
        )
