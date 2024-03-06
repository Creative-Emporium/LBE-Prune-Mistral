from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from pathlib import Path

from main import generate
from run_benchmark import mmlu







def top_layer_pruning(transformer: Transformer, amount: int):
    if amount == 0:
        return transformer
    num_layers = transformer.n_layers
    if amount >= num_layers -1:
        print(f"desired amount {amount} exceeds supported model size of {num_layers}: skipping pruning of model")
        return transformer
    amount += 1 # increase amount by 1 since we always skip the last layer
    stop_layer :int = num_layers - amount - 1
    layers_to_prune = list(range(num_layers - 1, stop_layer, - 1))
    prune_count = 0
    for index in layers_to_prune:
        if index == num_layers - 1:
            continue
        print(f"pruned layer with index {index}")
        transformer.layers.pop(str(index))
        prune_count += 1
    
    print(f"prune count: {prune_count}")
    return transformer, prune_count


def eval_top_layer_pruning(model_path:str, max_tokens = 40, temperature = 0.0, max_amount: int = 14):
    """ evaluates top layer pruning by pruning layers from top of mistral in steps of 2, starting at at least 2 layers
    """
    mmlu_subset_list = ["high_school_geography"]
    accuracy_amount_pruned = {} # number of pruned layers as key, accuracy as value
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    for i in range(0, max_amount+1, 2):
        transformer, prune_count = Transformer.from_folder(Path(model_path), max_batch_size = 1)
        pruned_transformer = top_layer_pruning(transformer, amount=i)
        accuracy_amount_pruned[prune_count] = mmlu(model_path=model_path, trans=pruned_transformer, tok=tokenizer, subset_list=mmlu_subset_list, max_tokens=max_tokens, temperature=temperature)
        del pruned_transformer
        del transformer
    
    return accuracy_amount_pruned



def main():
    prompt = "say hello!"
    max_tokens = 40
    temperature = 0.0
    model_path = "model_weights/mistral-7B-v0.2-instruct"
    accuracy_amount_pruned = eval_top_layer_pruning(model_path, max_amount=14)
    
    for index, accuracy in accuracy_amount_pruned.items():
        print(f"accuracy when pruning {index} layers is: {accuracy}")
    
    #print(f"accuracy before: {acc_before}\n accuracy after: {acc_after}")














if __name__ == "__main__":
    main()