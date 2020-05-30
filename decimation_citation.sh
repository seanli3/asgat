module load cuda/10.1
module load python3

export LD_LIBRARY_PATH=/apps/cuda/10.1/lib64:$LD_LIBRARY_PATH
export CPATH=/apps/cuda/10.1/include:$CPATH

pip3 install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user pygsp
pip3 install --user scikit_learn
pip3 install --user torch-scatter==2.0.4+cu101 torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-sparse==0.6.4+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-cluster==1.5.4+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-spline-conv==1.2.0+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html 
pip3 install --user torch-geometric

python3 setup.py install --user

python3 ./citation/decimation.py $LINE
