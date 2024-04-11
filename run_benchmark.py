import re
from pathlib import Path

import datasets
import evaluate
import fire
import torch
from torch.utils.data import DataLoader

from main import generate
from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer


def _construct_mmlu_batch(batch: dict, context: str):
    """
    construct
    """
    questions = batch["question"]

    prompts = [
        _construct_mmlu_prompt(question, batch["choices"], i, context)
        for i, question in enumerate(questions)
    ]
    return prompts


def _construct_mmlu_prompt(
    question: str, choices: list, answer_index: int, context: str
):
    """construct correct prompt from dataset batch entries; the entries are aranged
    as lists of tuples, where each tuple contains all answers with code 0 for the entire batch:
     [(ans0_to_quest0, ans0_to_quest1... ans0_to_quest_n), (ans1_to_quest0,..ans1_to_quest_n), (ans2_to_quest0,..ans2_to_quest_n), (ans3_to_quest0,..ans3_to_quest_n)]
     hence we pass answer_index to this method from _construct_mmlu_batch"""

    question += "\n"
    answers = ""
    for i, ans_tuple in enumerate(choices):
        answers += f"{i}: {ans_tuple[answer_index]}\n"
    answers += "\n"
    prompt = f"{question} {answers} Answer: "
    prompt = context + prompt
    prompt = f"[INST] {prompt} [/INST]"
    return prompt


def mmlu(
    transformer: Transformer,
    tokenizer: Tokenizer,
    subset_list: list,
    max_tokens: int = 80,
    temperature: float = 0.0,
):
    """run MMLU benchmark on mistral 7b
    param: model_path: path to downloaded model weights

    """
    print(
        f"running mmlu benchmark with max_tokens {max_tokens} and temperature {temperature}"
    )
    ground_truths = []
    predictions = []
    five_shot_prompt = ""
    with open("benchmark_prompts/mmlu_5_shot_prompt.txt", "r") as f:
        five_shot_prompt = f.read()

    for subset in subset_list:
        gt, pred = _generate_from_batched_mmlu(
            transformer=transformer,
            tokenizer=tokenizer,
            subset=subset,
            five_shot_prompt=five_shot_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        ground_truths.extend(gt)
        predictions.extend(pred)
    assert len(ground_truths) == len(predictions)
    accuracy_metric = evaluate.load("accuracy")
    result = accuracy_metric.compute(references=ground_truths, predictions=predictions)
    res_acc = result["accuracy"]
    print("---------------------------ACCURACY------------------------------------")
    print(f"Accuracy for MMLU test set: {res_acc}")
    return res_acc


def _generate_from_batched_mmlu(
    transformer: Transformer,
    tokenizer: Tokenizer,
    subset: str,
    five_shot_prompt: str,
    max_tokens: int = 80,
    temperature: float = 0.0,
):
    hf_dataset = datasets.load_dataset("Stevross/mmlu", subset, split="test")
    mmlu_dataset = hf_dataset.with_format(type="torch")
    print(f"type of mmlu dataset is {type(mmlu_dataset)}")
    batch_size = transformer.args.max_batch_size
    mmlu_dataloader = DataLoader(mmlu_dataset, batch_size=batch_size, shuffle=False)
    print(f"length of subset {subset}: {len(mmlu_dataset)}")
    print(f"len of dataloader: {len(mmlu_dataloader)}")

    predictions = []
    ground_truths = []
    for batch in mmlu_dataloader:
        batch_of_prompts = _construct_mmlu_batch(batch=batch, context=five_shot_prompt)
        results, _logprobs = generate(
            batch_of_prompts,
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(f"len of generated results: {len(results)}")
        for res in results:
            print("---------------result--------------------")
            print(f"{res}")
            print("---------------result end--------------------")
            regex = "(\[/INST\]) ([0-3])"
            ints_in_string = re.findall(regex, res)
            print("regex before cast:")
            print(ints_in_string)
            ints_in_string = [int(x[1]) for x in ints_in_string]
            print("regex after cast:")
            print(ints_in_string)
            if len(ints_in_string) == 0:
                print("Error no int found in llm response with prompt \n {prompt}")
                predictions.append(
                    99999
                )  # class which doesnt exist -> counts as wrong prediction
            else:
                print(f"prediction: {ints_in_string[0]}")
                assert type(ints_in_string[0]) == int
                predictions.append(
                    ints_in_string[0]
                )  # re.findall returns a list and parses from left to right; pred id should be in first list element

        ground_truths.extend(batch["answer"].tolist())

    return ground_truths, predictions


def __construct_hellaswag_prompt(data):
    """construct correct prompt from a dataset entry"""

    context = data["ctx"]
    context += "\n"
    answers = ""
    for i, answer in enumerate(data["endings"]):
        answers += f"{i}: {answer}\n"
    answers += "\n"
    prompt = f"{context} {answers} Answer: "
    return prompt


def hellaswag(
    model_path: str,
    trans: Transformer = None,
    tok: Tokenizer = None,
    max_tokens: int = 80,
    temperature: float = 0.0,
):
    """run HELLASWAG benchmark on mistral 7b
    param: model_path: path to downloaded model weights

    """
    print(
        f"running hellaswag benchmark with max_tokens {max_tokens} and temperature {temperature}"
    )
    if trans is None:
        transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
    else:
        transformer = trans
    if tok is None:
        tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    else:
        tokenizer = tok
    predictions = []
    hellaswag_dataset = datasets.load_dataset("Rowan/hellaswag", split="validation")
    ground_truths = [int(x) for x in hellaswag_dataset["label"]]
    prompt_instructions = ""
    with open("benchmark_prompts/hellaswag_0_shot_prompt.txt", "r") as f:
        prompt_instructions = f.read()

    for data in hellaswag_dataset:
        prompt = prompt_instructions + __construct_hellaswag_prompt(data=data)
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
            predictions.append(
                99999
            )  # class which doesnt exist -> counts as wrong prediction
        else:
            print(f"prediction: {ints_in_string[0]}")
            assert type(ints_in_string[0]) == int
            predictions.append(
                ints_in_string[0]
            )  # re.findall returns a list and parses from left to right; pred id should be in first list element

    accuracy_metric = evaluate.load("accuracy")
    result = accuracy_metric.compute(references=ground_truths, predictions=predictions)
    res_acc = result["accuracy"]
    print("---------------------------ACCURACY------------------------------------")
    print(f"Accuracy for HELLASWAG test set: {res_acc}")
    return res_acc


def main():
    subset_list = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]  # mmlu topics; loop over them to run entire dataset
    max_tokens: int = 40
    temperature: float = 0.0
    batch_size = 8
    model_path = "model_weights/mistral-7B-v0.2-instruct"
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=batch_size)
    mmlu(
        transformer=transformer,
        tokenizer=tokenizer,
        subset_list=subset_list,
        max_tokens=max_tokens,
        temperature=temperature,
    )


if __name__ == "__main__":
    main()
