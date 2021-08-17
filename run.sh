export LD_LIBRARY_PATH=/apps/cuda/11.1.1/lib64:$LD_LIBRARY_PATH
export CPATH=/apps/cuda/11.1.1/include:$CPATH

module load cuda/11.1.1 
conda activate /datastore/li243/.conda/env/asgat-test


CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m webkb.decimation --dataset=Wisconsin --order=10 --dropout=0. --heads=4 --hidden=128 --lr=0.1 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0008 --cuda --k=20 --method=chebyshev
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m webkb.decimation --dataset=Texas --order=16 --dropout=0. --heads=5 --hidden=932 --lr=0.007 --patience=100 --seed=720 --epochs=2000 --runs=1 --weight_decay=0.0002 --cuda --k=5 --method=chebyshev
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m webkb.decimation --dataset=Chameleon --order=8 --dropout=0.4 --heads=12 --hidden=88 --lr=0.01 --patience=100 --seed=720 --epochs=2000 --runs=1 --weight_decay=0.001 --cuda --k=6 --method=chebyshev
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m webkb.decimation --dataset=Cornell --order=10 --dropout=0. --heads=4 --hidden=128 --lr=0.1 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0005 --cuda --k=20 --method=chebyshev
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m citation.decimation --dataset=CiteSeer --order=14 --dropout=0.2 --heads=9 --hidden=512 --lr=0.005 --patience=100 --seed=729 --epochs=2000 --runs=10 --weight_decay=0.0008 --cuda --k=11 --method=chebyshev --self_loop
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m citation.citation --dataset=Cora --order=14 --dropout=0.1 --heads=1 --hidden=128 --lr=0.005 --patience=100 --seed=729 --epochs=2000 --runs=10 --weight_decay=0.0001 --cuda --k=15 --method=chebyshev
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m citation.citation --dataset=Cora --order=14 --dropout=0.1 --heads=1 --hidden=128 --lr=0.005 --patience=100 --seed=729 --epochs=2000 --runs=10 --weight_decay=0.0001 --cuda --k=15 --method=chebyshev
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m citation.citation --dataset=Squirrel --order=14 --dropout=0.3 --heads=8 --hidden=256 --lr=0.008 --patience=100 --seed=729 --epochs=2000 --runs=10 --weight_decay=5e-05 --cuda --k=6 --method=chebyshev
