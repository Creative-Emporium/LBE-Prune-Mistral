import torch
from deepeval.models.base_model import DeepEvalBaseLLM

from main import generate
from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer


class PrunedMistral(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        temperature: float,
        max_tokens: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        final_prompt = f"[INST]{prompt}[/INST]"

        result, _ = generate(
            prompts=[final_prompt],
            model=model,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        answer = result[0].rsplit(sep="[/INST]", maxsplit=1)[1]
        answer = answer.strip()
        if len(answer) == 0:
            answer = " "  # return non-zero string to avoid crashes during eval
        return answer

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt=prompt)

    def get_model_name(self):
        return "Pruned Mistral 7B"
