export LD_LIBRARY_PATH=/apps/cuda/11.1.1/lib64:$LD_LIBRARY_PATH
export CPATH=/apps/cuda/11.1.1/include:$CPATH

module load cuda/11.1.1 

source ~/.bashrc
conda activate /datastore/li243/.conda/env/asgat-test
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m webkb.decimation --dataset=Texas --dropout=0.1 --heads=8 --hidden=128 --lr=0.06 --patience=100 --epochs=2000 --weight_decay=0.0005 --cuda --k=13 --method=exact
python -m webkb.decimation --dataset=Cornell --dropout=0.1 --heads=3 --hidden=32 --lr=0.1 --patience=100 --epochs=2000 --weight_decay=0.0005 --cuda --k=17 --method=exact
python -m webkb.decimation --dataset=Wisconsin --dropout=0. --heads=5 --hidden=128 --lr=0.1 --patience=100 --epochs=2000 --weight_decay=0.0008 --cuda --k=20 --method=exact
python -m webkb.decimation --dataset=Wisconsin --order=10 --dropout=0. --heads=4 --hidden=128 --lr=0.1 --patience=100 --epochs=2000 --weight_decay=0.0008 --cuda --k=20 --method=chebyshev
python -m webkb.decimation --dataset=Texas --order=16 --dropout=0.1 --heads=2 --hidden=128 --lr=0.06 --patience=100 --epochs=2000 --weight_decay=0.0005 --cuda --k=13 --method=chebyshev
python -m webkb.decimation --dataset=Cornell --order=16 --dropout=0. --heads=6 --hidden=32 --lr=0.1 --patience=100 --epochs=2000 --weight_decay=0.0005 --cuda --k=17 --method=chebyshev
python -m webkb.decimation --dataset=Chameleon --order=8 --dropout=0.4 --heads=12 --hidden=88 --lr=0.01 --patience=100 --epochs=2000 --weight_decay=0.001 --cuda --k=6 --method=chebyshev
python -m webkb.decimation --dataset=Squirrel --order=16 --dropout=0.5 --heads=8 --hidden=256 --lr=0.008 --patience=100 --epochs=2000 --weight_decay=5e-05 --method=Chebyshev --k=6 --cuda

python -m citation.decimation --dataset=Cora --dropout=0.1 --heads=1 --hidden=32 --lr=0.005 --patience=100 --epochs=2000 --runs=10 --weight_decay=0.0001 --k=15 --cuda --order=14 --method=chebyshev
python -m citation.decimation --dataset=CiteSeer --order=14 --dropout=0.2 --heads=9 --hidden=512 --lr=0.005 --patience=100 --epochs=2000 --runs=10 --weight_decay=0.0008 --cuda --k=11 --method=chebyshev --self_loop

