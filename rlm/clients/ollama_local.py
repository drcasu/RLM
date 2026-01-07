from collections import defaultdict
from typing import Any, TYPE_CHECKING

from ollama import chat

if TYPE_CHECKING:
    from ollama import ChatResponse

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class OllamaLocalClient(BaseLM):
    """
    LM Client for running models with the Ollama API.
    Uses the official Ollama SDK.
    """

    def __init__(
        self,
        model_name: str | None = "gemma3",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs) # type: ignore

        self.model_name = model_name
        
        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OllamaLocal client.")
        
        response: 'ChatResponse' = chat(
            model=model,
            messages=messages,
            stream=False,
        )
        
        if not isinstance(response.message.content, str):
            raise ValueError(f"Invalid `response.message.content` type: {type(response.message.content)}")
        
        return response.message.content
    
    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OllamaLocal client.")
        
        response: 'ChatResponse' = chat(
            model=model,
            messages=messages,
            stream=False,
        )
        
        if not isinstance(response.message.content, str):
            raise ValueError(f"Invalid `response.message.content` type: {type(response.message.content)}")
        
        self._track_cost(response, model)
        return response.message.content
    
    def _track_cost(self, response: 'ChatResponse', model: str):
        self.model_call_counts[model] += 1
        
        input_token_count  = response.prompt_eval_count or 0
        output_token_count = response.eval_count or 0
        total_token_count  = input_token_count + output_token_count

        self.model_input_tokens[model]  += input_token_count
        self.model_output_tokens[model] += output_token_count
        self.model_total_tokens[model]  += total_token_count

        # Track last call for handler to read
        self.last_prompt_tokens     = input_token_count
        self.last_completion_tokens = output_token_count

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
