#!/bin/sh

runs=10

echo "Cora"
echo "===="
#
echo "GCN"
python3 -m citation.gcn --dataset=Cora --runs=$runs --split=public --cuda
##python gcn --dataset=Cora --random_splits=True

echo "GAT"
python3 -m citation.gat --dataset=Cora --runs=$runs --split=public --cuda
##python gat --dataset=Cora --random_splits=True
#
echo "Cheby"
python3 -m citation.cheb --dataset=Cora --num_hops=3 --runs=$runs --split=public --cuda
##python cheb --dataset=Cora --num_hops=3 --random_splits=True
#
echo "SGC"
python3 -m citation.sgc --dataset=Cora --K=3 --weight_decay=0.0005 --runs=$runs --split=public --cuda
##python sgc --dataset=Cora --K=3 --weight_decay=0.0005 --random_splits=True
#
echo "ARMA"
python3 -m citation.arma --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights=True --runs=$runs --split=public --cuda
##python arma --dataset=Cora --num_stacks=3 --num_layers=1 --shared_weights=True --random_splits=True
#
echo "APPNP"
python3 -m citation.appnp --dataset=Cora --alpha=0.1 --runs=$runs --split=public --cuda
##python appnp --dataset=Cora --alpha=0.1 --random_splits=True

echo "CiteSeer"
echo "========"

echo "GCN"
python3 -m citation.gcn --dataset=CiteSeer --runs=$runs --split=public --cuda
#python gcn --dataset=CiteSeer --random_splits=True

echo "GAT"
python3 -m citation.gat --dataset=CiteSeer --runs=$runs --split=public --cuda
#python gat --dataset=CiteSeer --random_splits=True

echo "Cheby"
python3 -m citation.cheb --dataset=CiteSeer --num_hops=2 --runs=$runs --split=public --cuda
#python cheb --dataset=CiteSeer --num_hops=3 --random_splits=True

echo "SGC"
python3 -m citation.sgc --dataset=CiteSeer --K=2 --weight_decay=0.005 --runs=$runs --split=public --cuda
#python sgc --dataset=CiteSeer --K=2 --weight_decay=0.005 --random_splits=True

echo "ARMA"
python3 -m citation.arma --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --runs=$runs --cuda
#python arma --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --random_splits=True

echo "APPNP"
python3 -m citation.appnp --dataset=CiteSeer --alpha=0.1 --runs=$runs --split=public --cuda
#python appnp --dataset=CiteSeer --alpha=0.1 --random_splits=True

echo "PubMed"
echo "======"

echo "GCN"
python3 -m citation.gcn --dataset=PubMed --runs=$runs --split=public --cuda
#python gcn --dataset=PubMed --random_splits=True

echo "GAT"
python3 -m citation.gat --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --runs=$runs --split=public --cuda
#python gat --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits=True

echo "Cheby"
python3 -m citation.cheb --dataset=PubMed --num_hops=2 --runs=$runs --split=public --cuda
#python cheb --dataset=PubMed --num_hops=2 --random_splits=True

echo "SGC"
python3 -m citation.sgc --dataset=PubMed --K=2 --weight_decay=0.0005 --runs=$runs --split=public --cuda
#python sgc --dataset=PubMed --K=2 --weight_decay=0.0005 --random_splits=True

echo "ARMA"
python3 -m citation.arma --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0 --runs=$runs --split=public --cuda
##python arma --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0.5 --random_splits=True

echo "APPNP"
python3 -m citation.appnp --dataset=PubMed --alpha=0.1 --runs=$runs --split=public --cuda
#python appnp --dataset=PubMed --alpha=0.1 --random_splits=True
