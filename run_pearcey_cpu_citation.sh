#!/bin/bash
#SBATCH --job-name=squirrel_topk
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --mem=2000GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

LINE=$LINE ./bracewell_decimation_citation.sh
