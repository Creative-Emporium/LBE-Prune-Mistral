import argparse
from pathlib import Path

import datasets
import numpy as np
import torch

import wandb
from main import generate
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer


def get_mistral_last_token_output_activations(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int = 40,
    temperature: float = 0.0,
):
    """
    extracts activations of each Transformer Block at the output; extracts last token only
    used for The Unreasonable Ineffectiveness of the Deeper Layers by Gromov et al."""
    last_output_token_per_layer = {k: [] for k in range(transformer.n_local_layers)}

    # helper method to attach forward hook to layer; returns hook method
    def get_activation(index: int):

        def hook(module, input, output):
            last_output_token_per_layer[index] = output.detach()[-1]

        return hook

    hook_handles = []  # list of hooks handles for cleanup
    for index, layer in transformer.layers.items():
        handle = layer.register_forward_hook(get_activation(int(index)))
        hook_handles.append(handle)

    result, _logprobs = generate(
        prompt, transformer, tokenizer, max_tokens=max_tokens, temperature=temperature
    )

    for handle in hook_handles:  # cleanup handles
        handle.remove()

    for index, activation in last_output_token_per_layer.items():
        assert activation is not None

    return last_output_token_per_layer


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
    activations_generated = {k: [] for k in range(activations_dict_size)}

    query_tensor_size = 0
    for p in prompt:
        query_tensor_size += len(tokenizer.encode(p))

    # helper method to attach forward hook to layer; returns hook method
    def get_activation(index: int):

        def hook(module, input, output):
            if output.detach().size() == torch.Size(
                [query_tensor_size, 4096]
            ):  # check if current activations belong to input query tokens (check for size of query length), skip those
                return
            else:
                activations_generated[index].append(output.detach())

        return hook

    # input tensors need to be extracted and later substracted from measured activations
    inputs_generated = {k: [] for k in range(activations_dict_size)}

    def get_input(index: int):

        def hook(module, input, output):
            if input[0].detach().size() == torch.Size(
                [query_tensor_size, 4096]
            ):  # check if current input belong input query tokens (check for size of query length), skip those
                return
            else:
                inputs_generated[index].append(input[0].detach())

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
    assert len(activations_generated) == len(inputs_generated)

    for handle in activation_hook_handles:  # cleanup handles
        handle.remove()

    for handle in input_hook_handles:  # cleanup handles
        handle.remove()

    # substract inputs from activations to avoid signal pollution by recurrent connection after attention layer
    for layer_index, activation_list in activations_generated.items():
        for index, activation in enumerate(activation_list):
            activation_list[index] = torch.sub(
                activation_list[index], inputs_generated[layer_index][index], alpha=1
            )

    for layer_index, activation_list in activations_generated.items():
        activations_generated[layer_index] = torch.stack(activation_list, dim=-1)

    return activations_generated


def layerwise_batch_entropy(x):
    """Estimate the differential entropy by assuming a gaussian distribution of
    values for different samples of a set of token activations.
    taken from https://github.com/peerdavid/layerwise-batch-entropy
    """
    if x.shape[0] <= 1:
        raise Exception("The batch entropy can only be calculated for |batch| > 1.")
    x = torch.flatten(x, start_dim=1)
    x_std = torch.std(x, dim=0)
    entropies = 0.5 * torch.log(np.pi * np.e * x_std**2 + 1)
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

    return lbe


def _reindex_pruned_transformer(
    start_index: int, amount_removed: int, transformer: Transformer
) -> Transformer:
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


def _prune_single_layer(lbe: dict, transformer: Transformer):
    max_entropy: tuple = (0, 0.0)  # (index, lbe)
    last_layer_index = transformer.n_local_layers - 1
    for index, token_entropy in lbe.items():
        if index == last_layer_index:
            continue
        if token_entropy >= max_entropy[1]:
            max_entropy = (index, token_entropy)

    removed_block = transformer.layers.pop(str(max_entropy[0]))
    assert removed_block is not None
    transformer = _reindex_pruned_transformer(max_entropy[0], 1, transformer)
    return transformer, max_entropy[0]


def _lbe_similarity_pruning(
    lbe: dict, transformer: Transformer, amount: int, start_at_layer: int
):
    if amount < 1:
        print("amount must be more than 1! Aborting without similarity pruning!!")
        return transformer
    highest_layer_index = transformer.n_local_layers - 1
    lowest_differential: tuple = (
        start_at_layer,
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
        if differential < lowest_differential[1]:
            lowest_differential = (index, differential)

    target_index = lowest_differential[0] + amount + 1
    if target_index > highest_layer_index:
        target_index = highest_layer_index

    layers_removed = []
    for index in range(lowest_differential[0] + 1, target_index, 1):
        removed_layer = transformer.layers.pop(str(index))
        assert removed_layer is not None
        layers_removed.append(index)

    transformer = _reindex_pruned_transformer(
        start_index=(lowest_differential[0] + 1),
        amount_removed=amount,
        transformer=transformer,
    )
    return transformer, layers_removed


def _log_ratio(x: float, y: float):
    return abs(np.log2(x / y))


def prune_lbe_naive(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int,
    amount: int = 2,
):
    """
    prunes network using naive algorithm: select maximal lbe at each iteration, prune layer with maximal lbe;
    iterate until amount layers have been removed
    """
    pruned_transformer = transformer
    layers_removed = []
    for i in range(amount):
        activations = get_mistral_linear_activations(
            tokenizer=tokenizer,
            transformer=pruned_transformer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        lbe = compute_lbe(activations)
        pruned_transformer, removed_index = _prune_single_layer(lbe, pruned_transformer)
        layers_removed.append(removed_index)
    return pruned_transformer, layers_removed


def prune_lbe_similarity(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int,
    amount: int = 2,
    start_at_layer=14,
):
    activations = get_mistral_linear_activations(
        tokenizer=tokenizer,
        transformer=transformer,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    lbe = compute_lbe(activations)
    transformer, layers_removed = _lbe_similarity_pruning(
        lbe=lbe, transformer=transformer, amount=amount, start_at_layer=start_at_layer
    )
    return transformer, layers_removed


def _last_token_similarity(
    last_token_start_layer: torch.Tensor, last_token_end_layer: torch.Tensor
) -> float:
    """implements token distance metric from paper The Unreasonable Ineffectiveness of the Deeper Layers by Gromov et al;
    computes distance metric between the last tokens of the start and end layer"""

    token_dot_product = torch.dot(last_token_start_layer, last_token_end_layer)
    token_norms_mult = torch.linalg.norm(last_token_start_layer) * torch.linalg.norm(
        last_token_end_layer
    )
    arc_cos_tensor = torch.arccos(token_dot_product / token_norms_mult)
    angular_distance = (1 / np.pi) * arc_cos_tensor.data
    assert type(angular_distance.item()) is float
    return angular_distance.item()


def _last_token_arccos_similiarity_pruning(
    last_token_per_layer: dict,
    transformer: Transformer,
    amount: int,
    start_at_layer: int,
):
    if amount < 1:
        print("amount must be more than 1! Aborting without similarity pruning!!")
        return transformer
    highest_layer_index = transformer.n_local_layers - 1
    lowest_differential: tuple = (
        start_at_layer,
        99999.9,
    )  # (index, differential between index last_token_per_layer and index + amount last_token_per_layer)
    for index, _ in last_token_per_layer.items():
        comparison_index = index + amount + 1
        # we want at least one layer in between
        if comparison_index >= highest_layer_index:
            continue
        if index < start_at_layer:
            continue
        differential = _last_token_similarity(
            last_token_start_layer=last_token_per_layer[
                index - 1
            ],  # index -1 because we need input to layer of index, which we get by taking the previous layer output
            last_token_end_layer=last_token_per_layer[comparison_index],
        )
        if differential < lowest_differential[1]:
            lowest_differential = (index, differential)

    target_index = lowest_differential[0] + amount
    if target_index > highest_layer_index:
        target_index = highest_layer_index

    layers_removed = []
    for index in range(lowest_differential[0], target_index, 1):
        removed_layer = transformer.layers.pop(str(index))
        assert removed_layer is not None
        layers_removed.append(index)

    transformer = _reindex_pruned_transformer(
        start_index=(lowest_differential[0] + 1),
        amount_removed=amount,
        transformer=transformer,
    )
    return transformer, layers_removed


def prune_last_token_cosine_similarity(
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int,
    amount: int = 2,
    start_at_layer=14,
):
    """implements approach by paper The Unreasonable Ineffectiveness of the Deeper Layers by Gromov et al"""
    last_token_per_layer = get_mistral_last_token_output_activations(
        tokenizer=tokenizer,
        transformer=transformer,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    transformer, layers_removed = _last_token_arccos_similiarity_pruning(
        last_token_per_layer=last_token_per_layer,
        transformer=transformer,
        amount=amount,
        start_at_layer=start_at_layer,
    )
    return transformer, layers_removed


def prune_index_list_block(transformer: Transformer, index_list: list):
    """prune layers with specific consecutive indices from network"""
    assert sorted(index_list) == list(
        range(min(index_list), max(index_list) + 1)
    )  # check if indices are consecutive
    layers_removed = []
    for index in index_list:
        assert isinstance(index, int)
        assert 32 > index >= 0
        removed_layer = transformer.layers.pop(str(index))
        assert removed_layer is not None
        layers_removed.append(index)

    transformer_reindex = _reindex_pruned_transformer(
        start_index=min(index_list),
        amount_removed=len(index_list),
        transformer=transformer,
    )
    return transformer_reindex, layers_removed


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pruning Hyperparameters")
    parser.add_argument(
        "--num_layers_prune",
        type=int,
        default=7,
        help="number of layers to be pruned by the algorithms",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="lbe_sim",
        help="name of the algorithm to be run: options: lbe_sim for lbe similarity; last_token_sim for last token similarity; naive for naive pruning algorithm, baseline for no pruning (vanilla Mistral 7B with all layers)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size for computing LBE forward pass",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20,
        help="how many tokens Mistral should generate at most on each request",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_weights/mistral-7B-v0.2-instruct",
        help="path to subdirectory containing model weights and tokenizer; example: model_weights/mistral-7B-v0.2-instruct if you created model_weights/ directory in root of project",
    )
    parser.add_argument(
        "--log_wandb",
        type=bool,
        default=False,
        help="should the results be logged to weights and biases; include this in your sweep.yaml and set it to True",
    )
    args = parser.parse_args()
    return args


def choose_algorithm(
    algorithm: str,
    tokenizer: Tokenizer,
    transformer: Transformer,
    prompt: list,
    max_tokens: int,
    amount: int,
    start_at_layer: int = 14,
):
    num_layers_before_prune = transformer.n_local_layers
    new_transformer: Transformer
    if algorithm == "lbe_sim":
        print("choosing lbe similarity algorithm")
        new_transformer, layers_removed = prune_lbe_similarity(
            tokenizer=tokenizer,
            transformer=transformer,
            prompt=prompt,
            max_tokens=max_tokens,
            amount=amount,
            start_at_layer=start_at_layer,
        )
        num_layers_after_prune = new_transformer.n_local_layers
        assert num_layers_after_prune < num_layers_before_prune
    elif algorithm == "last_token_sim":
        print("choosing last token similarity algorithm")
        new_transformer, layers_removed = prune_last_token_cosine_similarity(
            tokenizer=tokenizer,
            transformer=transformer,
            prompt=prompt,
            max_tokens=max_tokens,
            amount=amount,
            start_at_layer=start_at_layer,
        )
        num_layers_after_prune = new_transformer.n_local_layers
        assert num_layers_after_prune < num_layers_before_prune
    elif algorithm == "naive":
        print("choosing naive algorithm")
        new_transformer, layers_removed = prune_lbe_naive(
            tokenizer=tokenizer,
            transformer=transformer,
            prompt=prompt,
            max_tokens=max_tokens,
            amount=amount,
        )
        num_layers_after_prune = new_transformer.n_local_layers
        assert num_layers_after_prune < num_layers_before_prune
    elif algorithm == "baseline":
        print(
            "choosing baseline; no pruning algorithm will be used; evaluating on vanilla Mistral 7B"
        )
        new_transformer = transformer
        layers_removed = []
    else:
        print(
            "ERROR! Incorrect algorithm identifier passed to program! correct options:  lbe_sim for lbe similarity; last_token_sim for last token similarity; naive for naive pruning algorithm"
        )
        exit(-1)
    return new_transformer, layers_removed


def eval_mmlu(max_tokens, new_transformer, num_layers_pruned, prune_config, tokenizer):
    from deepeval.benchmarks import MMLU
    from mistral_wrapper_lm_eval import PrunedMistral

    pruned_model_eval = PrunedMistral(
        model=new_transformer,
        tokenizer=tokenizer,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    benchmark = MMLU()
    benchmark.evaluate(model=pruned_model_eval)
    all_tasks_acc = benchmark.overall_score
    tasks_acc_df = benchmark.task_scores
    tasks_acc_table = wandb.Table(dataframe=tasks_acc_df)
    if prune_config.log_wandb:
        wandb.log(
            {
                "all tasks accuracy": all_tasks_acc,
                "subtask accuracy": tasks_acc_table,
                "layers removed": num_layers_pruned,
            }
        )


def wandb_log_layers_removed(layers_removed: list, prune_config):
    if len(layers_removed) == 0:
        print("no layers removed")
        return
    for layer_index in layers_removed:
        print(f"removed layer {layer_index}")
    if prune_config.log_wandb:
        columns = ["algorithm used", "num layers pruned", "indices of pruned layers"]
        data = [[prune_config.algorithm, prune_config.num_layers_prune, layers_removed]]
        wandb_table = wandb.Table(columns=columns, data=data)
        wandb.log({"indices_removed": wandb_table})


def print_mistral_results(
    max_tokens: int, new_transformer: Transformer, prompts: list, tokenizer: Tokenizer
):
    import re

    results, _ = generate(
        prompts=prompts,
        model=new_transformer,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    extract_query_answer_regex = re.compile("\[INST\](.*)\[\/INST\](.*)", flags=re.S)
    for result in results:
        regex_result = extract_query_answer_regex.search(result)
        query = regex_result.group(1)
        answer = regex_result.group(2)
        assert len(query) > 0 and len(answer) > 0
        print(f"Query: {query} \n\nAnswer: {answer} \n")
        print("-------------------------------")


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
    ]  # mmlu topics;
    prune_config = parse_args()
    if prune_config.log_wandb:
        wandb.init(config=vars(prune_config))
    model_path = prune_config.model_path
    max_tokens = prune_config.max_tokens
    batch_size: int = prune_config.batch_size
    num_layers_pruned: int = prune_config.num_layers_prune
    prompt = fetch_mmlu_batch(batch_size=batch_size, subset_list=subset_list)
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=len(prompt))
    new_transformer, layers_removed = choose_algorithm(
        algorithm=prune_config.algorithm,
        tokenizer=tokenizer,
        transformer=transformer,
        prompt=prompt,
        max_tokens=max_tokens,
        amount=num_layers_pruned,
        start_at_layer=14,
    )
    example_mmlu_prompts = [
        "[INST]A horse is attached to a cart that is at rest behind it. Which force, or combination of forces, explains how the horse-cart system can accelerate from rest? A. The forward static friction force of the ground on the horse is greater than any friction forces acting backward on the cart, providing a forward acceleration. B. The forward force of the horse on the cart is greater than the backward force of the cart on the horse, providing a forward acceleration. C. The force of the horse’s muscles on the rest of the horse-cart system provides the necessary acceleration.  D. The upward normal force of the ground on the horse is greater than the horse’s weight, providing an upward acceleration.[/INST]"
    ]
    example_hellaswag_prompts = [
        "[INST]She talks about how different the marathons used to be back in the day where she shows clips of participants from a marathon at least 50 years ago. There are pictures of male participants shown dominating the scene where she was the only female participant. she A. kayaks through the scenic body of water to a river where the ahead rock views are surrounded by tall trees. B. continues to talk about her marathon skills that are so well known around the news and films shes shown during her own marathons. C. talks about her experience as she shows more pictures of her participation against all odds. D. also talks again about the extraordinary waxes shes shown running marathons.[/INST]"
    ]
    example_summary_prompts = [
        "[INST]Please summarize the following text in 3 sentences: We showed some alternatives to Layer-wise pruning in Section 2.2. We showed approaches such as Wanda Sun et al. (2023) and SparseGPT Frantar and Alistarh (2023), which introduce un-structured sparsity into the model. A possible problem with unstructured sparsity is that it can interfere with optimizations introduced by a large number of libraries implementing sparse matrix multiplications Wilkinson et al. (2023). In order to avoid this problem, we decided to develop a Layer-wise pruning approach instead. Furthermore, SparseGPT ran their experiments on Bloom Le Scao et al. (2023) and OPT Zhang et al. (2022). These models are around 175B parameters large, therefore more than 20 times the size of Mistral 7B Jiang et al. (2023). Since our goal is to make LLMs more accessible to a broader audience, we decided to build on approaches (such as Gromov et al. (2024)) which demonstrably work on smaller transformer models also.[/INST]"
    ]
    wandb_log_layers_removed(layers_removed, prune_config)
    print_mistral_results(max_tokens, new_transformer, example_mmlu_prompts, tokenizer)
    eval_mmlu(max_tokens, new_transformer, num_layers_pruned, prune_config, tokenizer)


if __name__ == "__main__":
    main()
