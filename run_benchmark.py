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

def mmlu(model_path: str, trans:Transformer = None, tok: Tokenizer = None,  max_tokens: int = 40, temperature: float = 0.0):
    """run MMLU benchmark on mistral 7b
        param: model_path: path to downloaded model weights
        
    """
    print(f"running mmlu benchmark with max_tokens {max_tokens} and temperature {temperature}")
    if trans is None:
        transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
    else:
        transformer = trans
    if tok is None:
        tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    else:
        tokenizer = tok
    ground_truths = [] 
    predictions = []

    subset_list = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'] # mmlu topics; loop over them to run entire dataset
    for subset in subset_list:
         mmlu_dataset = datasets.load_dataset("Stevross/mmlu",subset, split="test")
         five_shot_prompt = ""
         with open("mmlu_5_shot_prompt.txt","r") as f:
             five_shot_prompt=f.read()
         ground_truths.extend(mmlu_dataset["answer"])
         for data in mmlu_dataset:

             prompt = five_shot_prompt + __construct_mmlu_prompt(data=data)
             prompt = f"[INST] {prompt} [/INST]"
             res, _logprobs = generate(
                 [prompt],
                 transformer,
                 tokenizer,
                 max_tokens=max_tokens,
                 temperature=temperature,
             )
             print("---------------result--------------------")
             print(f"{res[0]}")
             print("---------------result end--------------------")
             regex = "(\[/INST\]) ([0-3])"
             ints_in_string = re.findall(regex, res[0])
             print("regex before cast:")
             print(ints_in_string)
             ints_in_string = [int(x[1]) for x in ints_in_string]
             print("regex after cast:")
             print(ints_in_string)
             if len(ints_in_string) == 0:
                 print("Error no int found in llm response with prompt \n {prompt}")
                 predictions.append(99999) # class which doesnt exist -> counts as wrong prediction
             else:
                 print(f"prediction: {ints_in_string[0]}")
                 assert(type(ints_in_string[0]) == int)
                 predictions.append(ints_in_string[0]) # re.findall returns a list and parses from left to right; pred id should be in first list element

    
    accuracy_metric = evaluate.load("accuracy")
    result = accuracy_metric.compute(references=ground_truths, predictions=predictions)
    res_acc = result["accuracy"]
    print("---------------------------ACCURACY------------------------------------")
    print(f"Accuracy for MMLU test set: {res_acc}")
    return res_acc

def main(model_path: str, max_tokens: int = 40, temperature: float = 0.0):
    mmlu(model_path=model_path, trans=None, tok=None, max_tokens=max_tokens, temperature=temperature)






if __name__ == "__main__":
    fire.Fire({
        "mmlu": main
        })
