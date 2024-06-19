# Pruning Mistral using Layerwise Batch Entropy


This project is a fork of [mistral-ai](https://github.com/mistralai/mistral-inference). We implement a novel layer-wise pruning Algorithm we call **LBE Similarity Algorithm** which uses [Layer-wise Batch Entropy](https://github.com/peerdavid/layerwise-batch-entropy) to decide which Layers to remove.
The resulting model is evaluated on the **MMLU** dataset using 5-shot evaluation.

## Usage 

Example:
```
python3 prune_lbe.py --algorithm lbe_sim --max-tokens 20 --num_layers_prune 4
```

This command uses the **LBE Similarity Algorithm** to prune 4 layers from the network and subsequently evaluates the resulting architecture on MMLU.

you can run `python3 prune_lbe.py -h` or `python3 prune_lbe.py --help` for more available commands.

## System Requirements

This project uses Mistral, and therefore requires an NVIDIA GPU with at least 16 GB of VRAM.