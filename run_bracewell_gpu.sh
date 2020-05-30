#!/bin/bash
#SBATCH --job-name=citeseer
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

LINE=$LINE ./decimation_citation.sh
