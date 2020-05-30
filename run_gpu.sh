#!/bin/bash
#PBS -P go95
#PBS -q gpuvolta
#PBS -l walltime=20:00:00
#PBS -l mem=16GB
#PBS -l jobfs=1GB
#PBS -l ngpus=1 
#PBS -l ncpus=12
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=train_old_splot
## The job will be executed from current working directory instead of home.
#PBS -l wd

module load cuda/10.1
module load python/3.6.1

export LD_LIBRARY_PATH=/apps/cuda/10.1/lib64:$LD_LIBRARY_PATH
export CPATH=/apps/cuda/10.1/include:$CPATH

# pip uninstall -y torch
pip3 install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user scikit_learn
pip3 install --user torch-scatter==2.0.4+cu101 torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-sparse==0.6.4+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-cluster==1.5.4+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-spline-conv==1.2.0+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html 
pip3 install --user torch-geometric


python3 decimation.py $LINE
