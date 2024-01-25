import re
import fire
from pathlib import Path

import torch
import datasets
import evaluate

from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer



from main import generate

def __construct_mmlu_prompt(data):
    """construct correct prompt from a dataset entry"""
    
    question = data["question"]
    question +="\n"
    answers =""
    for i, answer in enumerate(data["choices"]):
        answers +=f"{i}: {answer}\n"
    answers +="\n"
    prompt = f"{question} {answers} Answer: "
    return prompt

def mmlu(model_path: str, max_tokens: int = 35, temperature: float = 0.7):
    """run MMLU benchmark on mistral 7b
        param: model_path: path to downloaded model weights
    """
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
    mmlu_dataset = datasets.load_dataset("Stevross/mmlu", "high_school_geography", split="test")
    five_shot_prompt = ""
    with open("mmlu_5_shot_prompt.txt","r") as f:
        five_shot_prompt=f.read()
    
    ground_truths = mmlu_dataset["answer"]
    predictions = []
    for data in mmlu_dataset:

        assert(type(data["answer"])==int)
        prompt = five_shot_prompt + __construct_mmlu_prompt(data=data)
        prompt = f"[INST] {prompt} [/INST]"
        res, _logprobs = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        regex = "[0-3]"
        ints_in_string = re.findall(regex, res[0])
        ints_in_string = [int(x) for x in ints_in_string]
        if len(ints_in_string) == 0:
            print("Error no int found in llm response with prompt \n {prompt}")
            predictions.append(99999) # class which doesnt exist -> counts as wrong prediction
        else:
            assert(type(ints_in_string[0]) == int)
            predictions.append(ints_in_string[0]) # re.findall returns a list and parses from left to right; pred id should be in first list element

    
    accuracy_metric = evaluate.load("accuracy")
    result = accuracy_metric.compute(references=ground_truths, predictions=predictions)
    res_acc = result["accuracy"]
    print(f"accuracy for high school geography test set: {res_acc}")
    return res_acc






if __name__ == "__main__":
    fire.Fire({
        "mmlu": mmlu
        })
