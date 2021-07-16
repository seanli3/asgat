echo "Reading parameters from file: " $1
while read -r line; do sbatch --export LINE="$line" run_pearcey_cpu_citation.sh; done < "$1"
