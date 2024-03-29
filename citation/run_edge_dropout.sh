#!/bin/sh

echo "Cora"
echo "===="

echo "GCN"
python gcn.py --dataset=Cora --edge_dropout=0

echo "GAT"
python gat.py --dataset=Cora --edge_dropout=0

echo "Cheby"
python cheb.py --dataset=Cora --num_hops=3 --edge_dropout=0

echo "SGC"
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --edge_dropout=0

echo "ARMA"
python arma.py --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights=True --edge_dropout=0

echo "APPNP"
python appnp.py --dataset=Cora --alpha=0.1 --edge_dropout=0

echo "CiteSeer"
echo "========"

echo "GCN"
python gcn.py --dataset=CiteSeer --edge_dropout=0

echo "GAT"
python gat.py --dataset=CiteSeer --edge_dropout=0

echo "Cheby"
python cheb.py --dataset=CiteSeer --num_hops=2 --edge_dropout=0

echo "SGC"
python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --edge_dropout=0

echo "ARMA"
python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --edge_dropout=0

echo "APPNP"
python appnp.py --dataset=CiteSeer --alpha=0.1 --edge_dropout=0

echo "PubMed"
echo "======"

echo "GCN"
python gcn.py --dataset=PubMed --edge_dropout=0

echo "GAT"
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --edge_dropout=0

echo "Cheby"
python cheb.py --dataset=PubMed --num_hops=2 --edge_dropout=0

echo "SGC"
python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --edge_dropout=0

echo "ARMA"
python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0 --edge_dropout=0

echo "APPNP"
python appnp.py --dataset=PubMed --alpha=0.1 --edge_dropout=0

read junk
