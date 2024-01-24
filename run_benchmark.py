import torch

import fire
from pathlib import Path
import datasets

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

def mmlu(model_path: str):
    """run MMLU benchmark on mistral 7b
        param: model_path: path to downloaded model weights
    """
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
    mmlu_dataset = datasets.load_dataset("Stevross/mmlu", "high_school_geography", split="test")
    five_shot_prompt = ""
    with open("mmlu_5_shot_prompt.txt","r") as f:
        five_shot_prompt=f.read()
    
    for data in mmlu_dataset:

        #generate()
        pass

    print("first entry:")
    print(__construct_mmlu_prompt(data=mmlu_dataset[0]))






if __name__ == "__main__":
    fire.Fire({
        "mmlu": mmlu
        })
