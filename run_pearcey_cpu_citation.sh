#!/bin/bash
#SBATCH --job-name=pubmed
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

LINE=$LINE ./bracewell_decimation_citation.sh
