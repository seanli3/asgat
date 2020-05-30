export LD_LIBRARY_PATH=/apps/cuda/10.1.168/lib64:$LD_LIBRARY_PATH
export CPATH=/apps/cuda/10.1.168/include:$CPATH

python3 ./citation/decimation.py $LINE
