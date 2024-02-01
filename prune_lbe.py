import re
import fire
from pathlib import Path

import torch
import wandb

from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from main import generate



def get_mistral_activations(model_path: str, max_tokens: int = 35, temperature: float = 0.7):

    # helper method to attach forward hook to layer; returns hook method
    activations = {}
    def get_activation(index: str):

        def hook(module, input, output):
            activations[index] = output.detach()
        return hook

        
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size = 3)

    hook_handles = [] # list of hooks handles for cleanup
    for index, layer in transformer.layers.items():
        handle = layer.attention.wo.register_forward_hook(get_activation(index))
        hook_handles.append(handle)
    

    prompt = "[INST] what is the capital of France? [/INST]"
    
    result, _logprobs = generate([prompt], transformer, tokenizer, max_tokens = max_tokens, temperature = temperature)
    
    print(f"Answer of Mistral: {result}")
    for handle in hook_handles:# cleanup handles
        handle.remove() 
    
    print(f"length of activation dict: {len(activations)}")
    for index, activation in activations.items():
        assert(activation is not None)
        print(f"activation tensor size of index {index}: {activation.size()}")
    
    return activations, transformer
    
    
def batch_entropy(x):
    """ Estimate the differential entropy by assuming a gaussian distribution of
        values for different samples of a mini-batch.
    """
    if(x.shape[0] <= 1):
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
        print(f"size of activation: {activation.shape()}")
        lbe[index] = batch_entropy(activation)
    
    return lbe

def main(model_path: str):
    #wandb_run = wandb.init(entity="maxdanelli", project="mistral_prune")
    
    activations, transformer = get_mistral_activations(model_path=model_path)
    #lbe = compute_lbe(activations)
    #wandb_run.log(lbe)







if __name__ == "__main__":
    fire.Fire({
        "prune": main
    }
    )