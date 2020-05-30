echo "Reading parameters from file: " $1
while read -r line; do sbatch --export LINE="$line" run_gpu.sh; done < "$1"
