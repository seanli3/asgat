#!/bin/sh

echo "Cornell"
python -m webkb.decimation --dataset=Cornell --alpha=0.2 --chebyshev_order=16 --dropout=0.2 --heads=1 --hidden=128 --lr=0.01 --patience=10 --seed=729 --epochs=2000 --runs=10

echo "Texas"
python -m webkb.decimation --dataset=Texas --alpha=0.2 --chebyshev_order=16 --dropout=0.2 --heads=1 --hidden=32 --lr=0.005 --patience=10 --seed=729 --epochs=2000 --runs=10

echo "Wisconsin"
python -m webkb.decimation --dataset=Wisconsin --alpha=0.2 --chebyshev_order=16 --dropout=0.2 --heads=1 --hidden=128 --lr=0.01 --patience=10 --seed=729 --epochs=2000 --runs=10

