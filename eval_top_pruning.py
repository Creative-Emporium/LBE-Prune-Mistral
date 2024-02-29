from mistral.cache import RotatingBufferCache
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from pathlib import Path

from main import generate
from run_benchmark import mmlu







def top_layer_pruning(transformer: Transformer, amount: int):
    num_layers = transformer.n_layers
    if amount >= num_layers -1:
        print(f"desired amount {amount} exceeds supported model size of {num_layers}: skipping pruning of model")
        return transformer
    amount += 1 # increase amount by 1 since we always skip the last layer
    stop_layer :int = num_layers - amount - 1
    layers_to_prune = list(range(num_layers - 1, stop_layer, - 1))
    for index in layers_to_prune:
        if index == num_layers - 1:
            continue
        print(f"pruned layer with index {index}")
        transformer.layers.pop(str(index))
    
    return transformer



def main():
    prompt = "say hello!"
    max_tokens = 80
    model_path = "model_weights/mistral-7B-v0.2-instruct"
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size = len(prompt))
    

    top_layer_pruning(transformer, 3)





















if __name__ == "__main__":
    main()