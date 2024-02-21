import re
import fire
from pathlib import Path

import torch
from torch import nn
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
        activations[layer_index] = torch.stack(activation_list)

    for layer_index, activation_list in activations_query.items():
        activations_query[layer_index] = torch.stack(activation_list)

    return activations, activations_query

    
def get_mistral_linear_activations(tokenizer: Tokenizer, transformer: Transformer, prompt: list, max_tokens: int = 40, temperature: float = 0.0):

    """
    extracts activations of each TransformerBlock after fully connected (linear) layer (before residual connection)
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
        handle = layer.feed_forward.w2.register_forward_hook(get_activation(int(index)))
        hook_handles.append(handle)
    

    
    result, _logprobs = generate(prompt, transformer, tokenizer, max_tokens = max_tokens, temperature = temperature)
    
    
    print(f"Answer of Mistral: {result}")
    for handle in hook_handles:# cleanup handles
        handle.remove() 
    
    
    for index, activation in activations.items():
        assert(activation is not None)
        #print(f"---------------------- layer {index} ---------------------------------")
        #for index_t, tensor in enumerate(activation):
        #    print(f"activation size of token {index_t}: {tensor.size()}")
    
    for layer_index, activation_list in activations.items():
        activations[layer_index] = torch.stack(activation_list)

    for layer_index, activation_list in activations_query.items():
        activations_query[layer_index] = torch.stack(activation_list)

    return activations, activations_query
    
def token_wise_entropy(x):
    """ Estimate the differential entropy by assuming a gaussian distribution of
        values for different samples of a set of token activations.
    """
    #if(x.shape[0] <= 1):
    #    raise Exception("The batch entropy can only be calculated for |batch| > 1.")
    #print(f"shape before flatten {x.size()}")
    #x = torch.flatten(x, start_dim=0, end_dim=1)
    x = torch.squeeze(x)
    #print(f"shape after flatten {x.size()}")
    x_std = torch.std(x, dim=0)
    #print(f"shape of std: {x_std.size()}")
    entropies = 0.5 * torch.log(np.pi * np.e * x_std**2 + 1)
    sigmoid = torch.nn.Sigmoid()

    entropies = sigmoid(entropies) # map non normalized entropies to a range between 0 and 1
    return torch.max(entropies)
    
def compute_lte(activations: dict):
    """computes Layerwise Token Entropy from model activations.
    takes dict with layer index as index and layer activations as value as parameter
    returns: dict with layer index as index and layerwise batch entropy at that layer index as value
    """
    lte = {}
    for index, activation in activations.items():
        lte[index] = token_wise_entropy(activation)
        print(f"lte at index {index}: {lte[index]}")
    
    return lte

def prune_lte(lte: dict, transformer : Transformer, threshold: float = 0.0125):
    layers_to_be_pruned = []
    for index, token_entropy in lte.items():
        if token_entropy < threshold:
            layers_to_be_pruned.append(index)
    
    print(f"layers to be pruned {layers_to_be_pruned}")
    for index in layers_to_be_pruned:
        if index < 14 or index == 31:
            continue
        print(f"pruned layer {index}")
        transformer.layers.pop(str(index))
    
    return transformer



def main(model_path: str):
    #wandb_run = wandb.init(entity="maxdanelli", project="mistral_prune")
    
    prompt = ["[INST] What year was albert Einstein born? [/INST]","[INST]What is the molecular structure of water?[/INST]",
               "[INST] Who was the president of the USA in the year 1992? [/INST]"]

    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size = len(prompt))
    activations, activations_query = get_mistral_linear_activations(tokenizer=tokenizer, transformer=transformer, prompt=prompt)
    
    lte = compute_lte(activations)
    new_transformer = prune_lte(lte=lte, transformer=transformer, threshold=0.55)
    result, logits = generate(prompts=prompt, model=new_transformer,tokenizer= tokenizer, max_tokens = 40, temperature = 0.0)
    print(f"result after pruning: \n {result}")
    #new_transformer_acc = mmlu(model_path=model_path, trans=new_transformer, tok=tokenizer, max_tokens=40, temperature=0.0)

    #wandb_run.log(lbe)







if __name__ == "__main__":
    #fire.Fire({
    #    "prune": main
    #}
    #)
    main("model_weights/mistral-7B-v0.2-instruct")