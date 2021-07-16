echo "Reading parameters from file: " $1
while read -r line; do sbatch --export LINE="$line" run_bracewell_gpu_webkb.sh; done < "$1"
