import re
import fire
from pathlib import Path

import torch
from torch import nn
import datasets
import wandb
import numpy as np

import run_benchmark

from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from main import generate
from run_benchmark import mmlu



def get_mistral_attention_activations(tokenizer: Tokenizer, transformer: Transformer, prompt: list, max_tokens: int = 40, temperature: float = 0.0):
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
            if output.detach().size() == torch.Size([query_tensor_size, 4096]): # check if current activations belong to input query (check for size of query length)
                activations_query[index].append(output.detach())
            else:
                activations[index].append(output.detach())
            #print(f"activation at module {module}: {output.detach().size()}")
        return hook

    hook_handles = [] # list of hooks handles for cleanup
    for index, layer in transformer.layers.items():
        handle = layer.attention.wo.register_forward_hook(get_activation(int(index)))
        hook_handles.append(handle)
    

    
    result, _logprobs = generate(prompt, transformer, tokenizer, max_tokens = max_tokens, temperature = temperature)
    
    
    print(f"Answer of Mistral: {result}")
    for handle in hook_handles:# cleanup handles
        handle.remove() 
    
    
    for index, activation in activations.items():
        assert(activation is not None)
        #print(f"---------------------- layer {index} ---------------------------------")
        #for index_t, tensor in enumerate(activation):
            #print(f"activation size of token {index_t}: {tensor.size()}")
    
    for layer_index, activation_list in activations.items():
        activations[layer_index] = torch.stack(activation_list, dim=-1)

    for layer_index, activation_list in activations_query.items():
        activations_query[layer_index] = torch.stack(activation_list, dim=-1)

    return activations, activations_query

    
def get_mistral_linear_activations(tokenizer: Tokenizer, transformer: Transformer, prompt: list, max_tokens: int = 40, temperature: float = 0.0):

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
            if output.detach().size() == torch.Size([query_tensor_size, 4096]): # check if current activations belong to input query (check for size of query length)
                activations_query[index].append(output.detach())
            else:
                activations[index].append(output.detach())
            #print(f"activation at module {module}: {output.detach().size()}")
        return hook

    # input tensors need to be extracted and later substracted from measured activations
    inputs = {k: [] for k in range(activations_dict_size)}
    inputs_query = {k: [] for k in range(activations_dict_size)}
    
    def get_input(index: int):

        def hook(module, input, output):
            if input[0].detach().size() == torch.Size([query_tensor_size, 4096]): # check if current input is input query (check for size of query length)
                inputs_query[index].append(input[0].detach())
            else:
                inputs[index].append(input[0].detach())
        return hook

    activation_hook_handles = [] # list of hooks handles for cleanup
    input_hook_handles = []
    for index, layer in transformer.layers.items():
        act_handle = layer.feed_forward.w2.register_forward_hook(get_activation(int(index)))
        activation_hook_handles.append(act_handle)
        input_handle = layer.attention_norm.register_forward_hook(get_input(int(index)))
        input_hook_handles.append(input_handle)
    

    
    result, _logprobs = generate(prompt, transformer, tokenizer, max_tokens = max_tokens, temperature = temperature)
    assert(len(activations) == len(inputs) and len(activations_query) == len(inputs_query))
    
    print(f"Answer of Mistral: {result}")
    for handle in activation_hook_handles:# cleanup handles
        handle.remove() 
    
    for handle in input_hook_handles:# cleanup handles
        handle.remove() 
    
    #substract inputs from activations to avoid signal pollution by recurrent connection after attention layer
    for layer_index, activation_list in activations.items():
        for index, activation in enumerate(activation_list):
            activation_list[index] = torch.sub(activation_list[index], inputs[layer_index][index], alpha = 1)

    for layer_index, activation_list in activations_query.items():
        for index, activation in enumerate(activation_list):
            activation_list[index] = torch.sub(activation_list[index], inputs_query[layer_index][index], alpha = 1)
    
    for layer_index, activation_list in activations.items():
        if len(activation_list) == 0:
            print(f"list is empty at layer index: {layer_index}")
        activations[layer_index] = torch.stack(activation_list, dim=-1)
        #print(f"size of activations at layer {layer_index}: {activations[layer_index].size()}")

    for layer_index, activation_list in activations_query.items():
        if len(activation_list) == 0:
            print(f"list is empty at layer index: {layer_index}")
        activations_query[layer_index] = torch.stack(activation_list, dim=-1)
        #print(f"size of query activations at layer {layer_index}: {activations_query[layer_index].size()}")

    return activations, activations_query
    
def layerwise_batch_entropy(x):
    """ Estimate the differential entropy by assuming a gaussian distribution of
        values for different samples of a set of token activations.
    """
    if(x.shape[0] <= 1):
        raise Exception("The batch entropy can only be calculated for |batch| > 1.")
    #print(f"shape before flatten {x.size()}")
    x = torch.flatten(x, start_dim=1)
    #print(f"shape after flatten {x.size()}")
    x_std = torch.std(x, dim=0)
    #print(f"shape of std: {x_std.size()}")
    entropies = 0.5 * torch.log(np.pi * np.e * x_std**2 + 1)
    sigmoid = torch.nn.Sigmoid()

    entropies = sigmoid(entropies) # map non normalized entropies to a range between 0 and 1
    return torch.mean(entropies)
    
def compute_lbe(activations: dict):
    """computes Layerwise Batch Entropy from model activations.
    takes dict with layer index as index and layer activations as value as parameter
    returns: dict with layer index as index and layerwise batch entropy at that layer index as value
    """
    lbe = {}
    for index, activation in activations.items():
        lbe[index] = layerwise_batch_entropy(activation)
        print(f"lbe at index {index}: {lbe[index]}")
    
    return lbe

def _prune_single_layer(lbe: dict, transformer : Transformer):
    max_entropy : tuple = (0, 0.0) #(index, lbe)
    last_layer_index = transformer.n_local_layers - 1
    for index, token_entropy in lbe.items():
        if index == last_layer_index:
            continue
        if token_entropy >= max_entropy[1]:
            max_entropy = (index, token_entropy)

    removed_block = transformer.layers.pop(str(max_entropy[0]))
    print(f"layer with index {max_entropy[0]} removed")
    assert(removed_block is not None)
    for index, layer in transformer.layers.items():
        if int(index) >= max_entropy[0]:
            block_to_reindex = transformer.layers.pop(index)
            new_index = int(index) - 1
            transformer.layers[str(new_index)] = block_to_reindex

    transformer.n_local_layers = len(transformer.layers)
    return transformer




def prune_lbe(tokenizer:Tokenizer, transformer: Transformer, prompt: str, max_tokens: int, amount: int = 8):
    pruned_transformer = transformer
    for i in range(amount):
        activations, _ = get_mistral_linear_activations(tokenizer=tokenizer, transformer=pruned_transformer, prompt=prompt, max_tokens=max_tokens)
        lbe = compute_lbe(activations)
        pruned_transformer = _prune_single_layer(lbe, pruned_transformer)
    return pruned_transformer





def fetch_mmlu_batch(batch_size: int):
    """
    fetch questions from mmlu dataset, fetches batch_size amount of questions
    """
    subset_list = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology',
                    'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine',
                      'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 
                      'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 
                      'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 
                      'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 
                      'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics',
                        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 
                        'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 
                        'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 
                        'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology',
                          'us_foreign_policy', 'virology', 'world_religions'] # mmlu topics; loop over them to run entire dataset
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
    #wandb_run = wandb.init(entity="maxdanelli", project="mistral_prune")
    
    max_tokens = 80
    prompt = fetch_mmlu_batch(batch_size=16)
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size = len(prompt))
    new_transformer = prune_lbe(tokenizer = tokenizer, transformer = transformer, prompt = prompt, max_tokens = max_tokens, amount = 2)
    result, logits = generate(prompts = prompt, model = new_transformer,tokenizer = tokenizer, max_tokens = max_tokens, temperature = 0.0)
    print(f"result after pruning: \n {result}")
    #new_transformer_acc = mmlu(model_path=model_path, trans=new_transformer, tok=tokenizer, max_tokens=40, temperature=0.0)

    #wandb_run.log(lbe)







if __name__ == "__main__":
    #fire.Fire({
    #    "prune": main
    #}
    #)
    main("model_weights/mistral-7B-v0.2-instruct")