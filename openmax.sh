#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q long
#$ -l h=!qa-rtx6k-044
#$ -e errors/
#$ -N openmax_s4_tail_1000

# Required modules
module load conda
conda init bash
source activate open_max

python compute_openmax_updated.py