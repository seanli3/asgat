#!/bin/sh

#echo "Cornell"
#python -m webkb.decimation --dataset=Cornell --alpha=0.2 --chebyshev_order=16 --dropout=0.5 --heads=4 --hidden=128 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1
#
#echo "Texas heads"
#python -m webkb.decimation --dataset=Texas --alpha=0.2 --chebyshev_order=16 --dropout=0.4 --heads=4 --hidden=128 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1
#
#echo "Wisconsin"
#python -m webkb.decimation --dataset=Wisconsin --alpha=0.2 --chebyshev_order=16 --dropout=0.8 --heads=14 --hidden=128 --lr=0.01 --patience=10 --seed=729 --epochs=2000 --runs=1

echo "Cornell"
python -m webkb.decimation --dataset=Cornell --alpha=0.2 --chebyshev_order=16 --dropout=0.5 --heads=1 --hidden=128 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1
python -m webkb.decimation --dataset=Cornell --alpha=0.2 --chebyshev_order=16 --dropout=0.5 --heads=2 --hidden=128 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1
python -m webkb.decimation --dataset=Cornell --alpha=0.2 --chebyshev_order=16 --dropout=0.5 --heads=8 --hidden=128 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1
python -m webkb.decimation --dataset=Cornell --alpha=0.2 --chebyshev_order=16 --dropout=0.5 --heads=10 --hidden=128 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1
python -m webkb.decimation --dataset=Cornell --alpha=0.2 --chebyshev_order=16 --dropout=0.5 --heads=12 --hidden=128 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1
