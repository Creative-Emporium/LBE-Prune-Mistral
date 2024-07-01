import torch
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks.tasks import MMLUTask
from main import generate
from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer


def extract_task_from_prompt_header(header: str) -> MMLUTask:
    import re

    regex_str = (
        "The following are multiple choice questions "
        + re.escape("(with answers)")
        + " about ([a-zA-Z ]+)"
    )
    extract_task_regex = re.compile(regex_str)
    regex_result = extract_task_regex.search(header)
    task = regex_result.group(1)
    assert len(task) > 0
    task_formatted = task.replace(" ", "_")
    return MMLUTask(task_formatted)


def split_prompt(prompt: str) -> list:
    """splits prompt into subparts;
    first element should be header (to extract task),
    last element should be actual question for model
    """
    all_splits = prompt.split(sep="\n\n", maxsplit=7)
    assert len(all_splits) == 7
    assert all(split is not None for split in all_splits)
    return all_splits


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
        self.response_per_subtask = {}

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
        prompt_splits = split_prompt(prompt)  # needed to extract header from prompt
        header = prompt_splits[0]
        subtask = extract_task_from_prompt_header(header)
        self.save_response(task=subtask, response=answer)
        return answer

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt=prompt)

    def save_response(self, task: MMLUTask, response: str):
        if task not in self.response_per_subtask:
            self.response_per_subtask[task] = [response]
        else:
            self.response_per_subtask[task].append(response)

    def get_model_name(self):
        return "Pruned Mistral 7B"
