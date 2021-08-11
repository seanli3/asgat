module load miniconda3/4.9.2 
source ~/.bashrc

conda activate /datastore/li243/.conda/env/asgat


CUBLAS_WORKSPACE_CONFIG=:4096:8  python3 -m webkb.decimation $LINE
