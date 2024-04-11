import re
from pathlib import Path

import datasets
import fire
import numpy as np
import torch
from torch import nn

import wandb
from main import generate
from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from run_benchmark import mmlu


def get_mistral_attention_activations(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int = 40,
    temperature: float = 0.0,
):
    """
    extracts activations of each Transformer Block after self attention layer (before residual connection)
    """
    activations = {k: [] for k in range(transformer.n_local_layers)}
    activations_query = {k: [] for k in range(transformer.n_local_layers)}

    query_tensor_size = 0
    for p in prompt:
        query_tensor_size += len(tokenizer.encode(p))

    # helper method to attach forward hook to layer; returns hook method
    def get_activation(index: int):

        def hook(module, input, output):
            if output.detach().size() == torch.Size(
                [query_tensor_size, 4096]
            ):  # check if current activations belong to input query (check for size of query length)
                activations_query[index].append(output.detach())
            else:
                activations[index].append(output.detach())
            # print(f"activation at module {module}: {output.detach().size()}")

        return hook

    hook_handles = []  # list of hooks handles for cleanup
    for index, layer in transformer.layers.items():
        handle = layer.attention.wo.register_forward_hook(get_activation(int(index)))
        hook_handles.append(handle)

    result, _logprobs = generate(
        prompt, transformer, tokenizer, max_tokens=max_tokens, temperature=temperature
    )

    print(f"Answer of Mistral: {result}")
    for handle in hook_handles:  # cleanup handles
        handle.remove()

    for index, activation in activations.items():
        assert activation is not None
        # print(f"---------------------- layer {index} ---------------------------------")
        # for index_t, tensor in enumerate(activation):
        # print(f"activation size of token {index_t}: {tensor.size()}")

    for layer_index, activation_list in activations.items():
        activations[layer_index] = torch.stack(activation_list, dim=-1)

    for layer_index, activation_list in activations_query.items():
        activations_query[layer_index] = torch.stack(activation_list, dim=-1)

    return activations, activations_query


def get_mistral_linear_activations(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int = 40,
    temperature: float = 0.0,
):
    """
    extracts activations of each TransformerBlock after fully connected (linear) layer (before residual connection)
    """
    activations_dict_size = transformer.n_local_layers
    activations = {k: [] for k in range(activations_dict_size)}
    activations_query = {k: [] for k in range(activations_dict_size)}

    query_tensor_size = 0
    for p in prompt:
        query_tensor_size += len(tokenizer.encode(p))

    # helper method to attach forward hook to layer; returns hook method
    def get_activation(index: int):

        def hook(module, input, output):
            if output.detach().size() == torch.Size(
                [query_tensor_size, 4096]
            ):  # check if current activations belong to input query (check for size of query length)
                activations_query[index].append(output.detach())
            else:
                activations[index].append(output.detach())
            # print(f"activation at module {module}: {output.detach().size()}")

        return hook

    # input tensors need to be extracted and later substracted from measured activations
    inputs = {k: [] for k in range(activations_dict_size)}
    inputs_query = {k: [] for k in range(activations_dict_size)}

    def get_input(index: int):

        def hook(module, input, output):
            if input[0].detach().size() == torch.Size(
                [query_tensor_size, 4096]
            ):  # check if current input is input query (check for size of query length)
                inputs_query[index].append(input[0].detach())
            else:
                inputs[index].append(input[0].detach())

        return hook

    activation_hook_handles = []  # list of hooks handles for cleanup
    input_hook_handles = []
    for index, layer in transformer.layers.items():
        act_handle = layer.feed_forward.w2.register_forward_hook(
            get_activation(int(index))
        )
        activation_hook_handles.append(act_handle)
        input_handle = layer.attention_norm.register_forward_hook(get_input(int(index)))
        input_hook_handles.append(input_handle)

    result, _logprobs = generate(
        prompt, transformer, tokenizer, max_tokens=max_tokens, temperature=temperature
    )
    assert len(activations) == len(inputs) and len(activations_query) == len(
        inputs_query
    )

    print(f"Answer of Mistral: {result}")
    for handle in activation_hook_handles:  # cleanup handles
        handle.remove()

    for handle in input_hook_handles:  # cleanup handles
        handle.remove()

    # substract inputs from activations to avoid signal pollution by recurrent connection after attention layer
    for layer_index, activation_list in activations.items():
        for index, activation in enumerate(activation_list):
            activation_list[index] = torch.sub(
                activation_list[index], inputs[layer_index][index], alpha=1
            )

    for layer_index, activation_list in activations_query.items():
        for index, activation in enumerate(activation_list):
            activation_list[index] = torch.sub(
                activation_list[index], inputs_query[layer_index][index], alpha=1
            )

    for layer_index, activation_list in activations.items():
        if len(activation_list) == 0:
            print(f"list is empty at layer index: {layer_index}")
        activations[layer_index] = torch.stack(activation_list, dim=-1)
        # print(f"size of activations at layer {layer_index}: {activations[layer_index].size()}")

    for layer_index, activation_list in activations_query.items():
        if len(activation_list) == 0:
            print(f"list is empty at layer index: {layer_index}")
        activations_query[layer_index] = torch.stack(activation_list, dim=-1)
        # print(f"size of query activations at layer {layer_index}: {activations_query[layer_index].size()}")

    return activations, activations_query


def layerwise_batch_entropy(x):
    """Estimate the differential entropy by assuming a gaussian distribution of
    values for different samples of a set of token activations.
    """
    if x.shape[0] <= 1:
        raise Exception("The batch entropy can only be calculated for |batch| > 1.")
    # print(f"shape before flatten {x.size()}")
    x = torch.flatten(x, start_dim=1)
    # print(f"shape after flatten {x.size()}")
    x_std = torch.std(x, dim=0)
    # print(f"shape of std: {x_std.size()}")
    entropies = 0.5 * torch.log(np.pi * np.e * x_std**2 + 1)
    # sigmoid = torch.nn.Sigmoid()

    # entropies = sigmoid(entropies) # map non normalized entropies to a range between 0 and 1
    return torch.mean(entropies)


def compute_lbe(activations: dict):
    """computes Layerwise Batch Entropy from model activations.
    takes dict with layer index as index and layer activations as value as parameter
    returns: dict with layer index as index and layerwise batch entropy at that layer index as value
    """
    lbe = {}
    for index, activation in activations.items():
        lbe[index] = layerwise_batch_entropy(activation)
        lbe[index] = lbe[index].item()  # convert scalar tensor to python float
        assert type(lbe[index]) is float
        print(f"lbe at index {index}: {lbe[index]}")

    return lbe


def _reindex_pruned_transformer(
    start_index: int, amount_removed: int, transformer: Transformer
) -> Transformer:
    print(f"length of transformer.layers {len(transformer.layers)}")
    for index in range(start_index, len(transformer.layers) + amount_removed, 1):
        if index >= start_index:
            try:
                block_to_reindex = transformer.layers.pop(str(index))
                new_index = index - amount_removed
                transformer.layers[str(new_index)] = block_to_reindex
            except KeyError:
                continue

    transformer.n_local_layers = len(transformer.layers)
    return transformer


def _prune_single_layer(lbe: dict, transformer: Transformer) -> Transformer:
    max_entropy: tuple = (0, 0.0)  # (index, lbe)
    last_layer_index = transformer.n_local_layers - 1
    for index, token_entropy in lbe.items():
        if index == last_layer_index:
            continue
        if token_entropy >= max_entropy[1]:
            max_entropy = (index, token_entropy)

    removed_block = transformer.layers.pop(str(max_entropy[0]))
    print(f"layer with index {max_entropy[0]} removed")
    assert removed_block is not None
    transformer = _reindex_pruned_transformer(max_entropy[0], 1, transformer)
    # for index, layer in transformer.layers.items():
    #    if int(index) >= max_entropy[0]:
    #        block_to_reindex = transformer.layers.pop(index)
    #        new_index = int(index) - 1
    #        transformer.layers[str(new_index)] = block_to_reindex

    # transformer.n_local_layers = len(transformer.layers)
    return transformer


def _lbe_similarity_pruning(
    lbe: dict, transformer: Transformer, amount: int, start_at_layer: int
) -> Transformer:
    if amount < 1:
        print("amount must be more than 1! Aborting without similarity pruning!!")
        return transformer
    highest_layer_index = transformer.n_local_layers - 1
    highest_differential: tuple = (
        31,
        99999.9,
    )  # (index, differential between index lbe and index + amount lbe)
    for index, entropy in lbe.items():
        comparison_index = index + amount + 1
        # we want at least one layer in between
        if comparison_index >= highest_layer_index:
            continue
        if index < start_at_layer:
            continue
        differential = _log_ratio(x=lbe[index], y=lbe[comparison_index])
        if differential < highest_differential[1]:
            highest_differential = (index, differential)

    target_index = highest_differential[0] + amount + 1
    for index in range(highest_differential[0] + 1, target_index, 1):
        print(
            f"removing layer with index {index}; should be between {highest_differential[0]} and {target_index}"
        )
        removed_layer = transformer.layers.pop(str(index))
        assert removed_layer is not None

    transformer = _reindex_pruned_transformer(
        start_index=(highest_differential[0] + 1),
        amount_removed=amount,
        transformer=transformer,
    )
    return transformer


def _log_ratio(x: float, y: float):
    return abs(np.log2(x / y))


def prune_lbe(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int,
    amount: int = 2,
) -> Transformer:
    pruned_transformer = transformer
    for i in range(amount):
        activations, _ = get_mistral_linear_activations(
            tokenizer=tokenizer,
            transformer=pruned_transformer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        lbe = compute_lbe(activations)
        pruned_transformer = _prune_single_layer(lbe, pruned_transformer)
    return pruned_transformer


def prune_lbe_similarity(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int,
    amount: int = 2,
    start_at_layer=16,
) -> Transformer:

    activations, _ = get_mistral_linear_activations(
        tokenizer=tokenizer,
        transformer=transformer,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    lbe = compute_lbe(activations)
    transformer = _lbe_similarity_pruning(
        lbe=lbe, transformer=transformer, amount=amount, start_at_layer=start_at_layer
    )
    return transformer


def prune_cosine_similarity(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int,
    amount: int = 2,
    start_at_layer=16,
) -> Transformer:
    activations, _ = get_mistral_linear_activations(
        tokenizer=tokenizer,
        transformer=transformer,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    activations


def fetch_mmlu_batch(batch_size: int, subset_list: list):
    """
    fetch questions from mmlu dataset, fetches batch_size amount of questions
    """
    # sample on example from each topic
    prompts: list = []
    for index, topic in enumerate(subset_list):
        if index >= batch_size:
            break
        mmlu_slice = datasets.load_dataset("Stevross/mmlu", topic, split="test[0:1]")
        sample = mmlu_slice["question"][0]
        prompts.append(f"[INST]{sample}[/INST]")

    return prompts


def main(model_path: str):
    # wandb_run = wandb.init(entity="maxdanelli", project="mistral_prune")

    max_tokens = 40
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
    batch_size: int = 8
    prompt = fetch_mmlu_batch(batch_size=batch_size, subset_list=subset_list)
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=len(prompt))
    # new_transformer = prune_lbe(tokenizer = tokenizer, transformer = transformer, prompt = prompt, max_tokens = max_tokens, amount = 2)
    new_transformer = prune_lbe_similarity(
        tokenizer=tokenizer,
        transformer=transformer,
        prompt=prompt,
        max_tokens=max_tokens,
        amount=7,
        start_at_layer=14,
    )
    result, logits = generate(
        prompts=prompt,
        model=new_transformer,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    for r in result:
        print(f"result after pruning: \n {result}")
    new_transformer_acc = mmlu(
        transformer=new_transformer,
        tokenizer=tokenizer,
        subset_list=subset_list,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    # wandb_run.log(lbe)


if __name__ == "__main__":
    # fire.Fire({
    #    "prune": main
    # }
    # )
    main("model_weights/mistral-7B-v0.2-instruct")
