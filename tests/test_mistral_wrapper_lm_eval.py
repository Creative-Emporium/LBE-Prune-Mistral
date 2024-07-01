import pytest
from mistral_wrapper_lm_eval import extract_task_from_prompt_header, split_prompt
from deepeval.benchmarks.mmlu.task import MMLUTask
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from mistral_wrapper_lm_eval import PrunedMistral
from pathlib import Path


@pytest.fixture()
def model_path():
    return "model_weights/mistral-7B-v0.2-instruct"


@pytest.fixture()
def test_prompt():
    with open("benchmark_prompts/high_school_european_history_prompt.txt", "r") as f:
        prompt = f.read()
    return prompt


@pytest.mark.parametrize(
    "input,expected",
    [
        ("high school geography", MMLUTask.HIGH_SCHOOL_GEOGRAPHY),
        ("human aging", MMLUTask.HUMAN_AGING),
        ("management", MMLUTask.MANAGEMENT),
    ],
)
def test_extract_task_from_prompt_header(input: str, expected: str):
    prompt_header = (
        "The following are multiple choice questions (with answers) about " + input
    )
    result = extract_task_from_prompt_header(header=prompt_header)
    assert result == expected


def test_split_prompts(test_prompt):
    splits = split_prompt(test_prompt)
    assert (
        splits[0]
        == "The following are multiple choice questions (with answers) about high school european history."
    )
