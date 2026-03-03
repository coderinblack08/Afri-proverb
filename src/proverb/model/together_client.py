import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from together import Together


@dataclass
class TogetherGenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    max_retries: int = 3
    max_tokens_ceiling: int = 4096


class TogetherGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        api_key: Optional[str] = None,
        config: Optional[TogetherGenerationConfig] = None,
    ):
        resolved_api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Together API key is missing. Set TOGETHER_API_KEY or pass --together_api_key."
            )

        self.client = Together(api_key=resolved_api_key)
        self.model_name_or_path = model_name_or_path
        self.config = config or TogetherGenerationConfig()

    def generate(self, prompt: str) -> str:
        return self.generate_with_info(prompt)["text"]

    def generate_with_info(self, prompt: str) -> dict[str, Any]:
        attempts = max(1, self.config.max_retries)
        max_tokens = self.config.max_tokens

        last_error: Optional[str] = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name_or_path,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning={"enabled": False},
                    chat_template_kwargs={"enable_thinking": False},
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )

                text, details = self._extract_with_details(response)
                should_retry_for_length = (
                    not text
                    and details["finish_reason"] == "length"
                    and max_tokens < self.config.max_tokens_ceiling
                )
                if should_retry_for_length:
                    max_tokens = min(max_tokens * 2, self.config.max_tokens_ceiling)
                    continue

                return {
                    "text": text,
                    "attempts": attempt,
                    "max_tokens_used": max_tokens,
                    **details,
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == attempts:
                    raise
                time.sleep(min(2**attempt, 8))

        return {
            "text": "",
            "attempts": attempts,
            "max_tokens_used": max_tokens,
            "finish_reason": None,
            "reasoning": "",
            "reasoning_len": 0,
            "usage": None,
            "error": last_error,
        }

    @staticmethod
    def _extract_with_details(response) -> tuple[str, dict[str, Any]]:
        details: dict[str, Any] = {
            "finish_reason": None,
            "reasoning": "",
            "reasoning_len": 0,
            "usage": None,
        }
        if not getattr(response, "choices", None):
            return "", details

        choice = response.choices[0]
        details["finish_reason"] = getattr(choice, "finish_reason", None)

        if hasattr(choice, "message") and choice.message is not None:
            content = choice.message.content
            reasoning = getattr(choice.message, "reasoning", "") or ""
            details["reasoning"] = reasoning
            details["reasoning_len"] = len(reasoning)
        else:
            content = getattr(choice, "text", "")

        usage = getattr(response, "usage", None)
        if usage is not None:
            details["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        if isinstance(content, str):
            return content.strip(), details

        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(str(item.get("text", "")))
                else:
                    chunks.append(str(getattr(item, "text", "")))
            return "".join(chunks).strip(), details

        return str(content).strip(), details
