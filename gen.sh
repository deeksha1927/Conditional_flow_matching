#!/bin/bash
#$ -M darun@nd.edu      # Email address for job notification
#$ -m ae               # Send mail when job begins, ends, and aborts
#$ -q gpu
#$ -l h=qa-a10-*|qa-rtx6k-*|qa-l40s-*
#$ -l gpu_card=1
#$ -pe smp 24
#$ -N fm_generate  # Specify job name
#$ -o /users/darun/afs/conditional-flow-matching/logs/   # stdout



conda activate ft


python /users/darun/afs/conditional-flow-matching/generate_ears.py



