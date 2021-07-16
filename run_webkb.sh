#!/bin/sh

#echo "Cornell"
#echo "===="

#echo "Decimation"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m webkb.decimation --dataset=Texas --order=16 --dropout=0.0 --heads=8 --hidden=128 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0002 --cuda --k=5
#
#echo "MLP"
#python -m webkb.mlp --dataset=Cornell --dropout=0.3 --hidden1=128 --hidden2=64

#echo "GCN"
#python -m webkb.gcn --dataset=Cornell

#echo "GAT"
#python -m webkb.gat --dataset=Cornell
#
#echo "Cheby"
#python -m webkb.cheb --dataset=Cornell --num_hops=1 --hidden=128
#
#echo "SGC"
#python -m webkb.sgc --dataset=Cornell --K=3
#
#echo "ARMA"
#python -m webkb.arma --dataset=Cornell --num_stacks=2 --num_layers=1 --shared_weights=True --hidden=512 --dropout=0.2 --skip_dropout=0.45
#
#echo "APPNP"
#python -m webkb.appnp --dataset=Cornell --alpha=0.8 --K=8


#echo "Texas"
#echo "===="

#echo "Decimation"
#python -m webkb.decimation --dataset=Texas --alpha=0.2 --order=16 --dropout=0.3 --heads=12 --hidden=512 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0005

#echo "MLP"
#python -m webkb.mlp --dataset=Texas --dropout=0.2 --hidden1=1024 --hidden2=64

#echo "GCN"
#python -m webkb.gcn --dataset=Texas --hidden=256

#echo "GAT"
#python -m webkb.gat --dataset=Texas --hidden=128

#echo "Cheby"
#python -m webkb.cheb --dataset=Texas --num_hops=2 --hidden=128
#
#echo "SGC"
#python -m webkb.sgc --dataset=Texas --K=2
#
#echo "ARMA"
#python -m webkb.arma --dataset=Texas --num_stacks=2 --num_layers=1 --shared_weights=True --hidden=512 --dropout=0.2 --skip_dropout=0.45

#
#echo "APPNP"
#python -m webkb.appnp --dataset=Texas --alpha=0.8 --K=8 --hidden=256


#echo "Wisconsin"
#echo "===="

#echo "Wisconsin"
#python -m webkb.decimation --dataset=Wisconsin --alpha=0.2 --order=16 --dropout=0.2 --heads=4 --hidden=256 --lr=0.01 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0008

#echo "MLP"
#python -m webkb.mlp --dataset=Wisconsin --dropout=0.1 --hidden1=512 --hidden2=32

#echo "GCN"
#python -m webkb.gcn --dataset=Wisconsin --hidden=256

#echo "GAT"
#python -m webkb.gat --dataset=Wisconsin --hidden=256

#echo "Cheby"
#python -m webkb.cheb --dataset=Wisconsin --num_hops=1 --hidden=64
#
#echo "SGC"
#python -m webkb.sgc --dataset=Wisconsin --K=2
#
#echo "ARMA"
#python -m webkb.arma --dataset=Wisconsin --num_stacks=2 --num_layers=1 --shared_weights=True --hidden=512 --dropout=0.2 --skip_dropout=0.45
#
#echo "APPNP"
#python -m webkb.appnp --dataset=Wisconsin --alpha=0.8 --K=8 --hidden=512

# echo "Chameleon"
# echo "===="

#echo "MLP"
#python -m webkb.mlp --dataset=Chameleon --dropout=0.5 --hidden1=256 --hidden2=128

#echo "GCN"
#python -m webkb.gcn --dataset=Chameleon
