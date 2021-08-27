# ASGAT

## Installation
You can install the dependencies by one of the following two methods.
### Conda
The conda env file assumes you would use CUDA11.
```sh
conda env create -f environment.yml
```

### pip
1. Install [PyTorch 1.9+](https://pytorch.org/get-started/locally/)
2. Install [PyTorch Geometrics](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
3. `pip install -r requirements.txt`

## Run
```sh
./run.sh
```

### Note
The accuracy performance might differ a little depending on the GPU vs. CPU, CUDA driver, Pytorch version, etc. We try to make the results as deterministic as possible using `torch.use_deterministic` but this feature is experimental and not stable.
For details, see
https://github.com/rusty1s/pytorch_geometric/issues/2788
https://pytorch.org/docs/stable/notes/randomness.html

### Attribution
Some of the benchmark implimentations are forked from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark
