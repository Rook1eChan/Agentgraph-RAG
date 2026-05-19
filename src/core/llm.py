"""LLM clien"""

import os
import logging
from typing import Any, Dict, List, Optional

import requests
import tiktoken

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client for OpenAI-compatible APIs."""

    def __init__(
            self,
            model: str = None,
            api_key: str = None,
            base_url: str = None,
            temperature: float = 0.0,
            max_tokens: int = 16384,
            reasoning_effort: str = None,
    ):
        self.model = model or os.getenv("ARAG_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("ARAG_API_KEY")
        self.base_url = (base_url or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1")).rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        if not self.api_key:
            raise ValueError("API key required. Set ARAG_API_KEY environment variable or pass api_key parameter.")

        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""))
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.count_tokens(str(tc.get("function", {})))
        return total

    def chat(
            self,
            messages: List[Dict[str, Any]],
            tools: List[Dict[str, Any]] = None,
            temperature: float = None,
            max_tokens: int = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        logger.debug(f"LLM Input: {messages}\n\n")

        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()

        logger.debug(f"LLM Output: {result}\n\n")

        usage = result.get("usage", {})

        return {
            "message": result["choices"][0]["message"],
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "raw_response": result,
        }

    def generate(
            self,
            messages: List[Dict[str, Any]],
            system: str = None,
            tools: List[Dict[str, Any]] = None,
            temperature: float = None,
            **kwargs
    ) -> tuple:
        """Generate response (compatibility method for eval script)."""
        if system:
            messages = [{"role": "system", "content": system}] + messages

        result = self.chat(messages=messages, tools=tools, temperature=temperature)
        content = result["message"].get("content", "")
        return content, 0.0
