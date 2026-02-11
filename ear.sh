#!/bin/bash
#$ -M darun@nd.edu      # Email address for job notification
#$ -m ae               # Send mail when job begins, ends, and aborts
#$ -q gpu
#$ -l h=qa-a10-*|qa-rtx6k-*|qa-l40s-*
#$ -l gpu_card=1
#$ -pe smp 24
#$ -N fm  # Specify job name
#$ -o /users/darun/afs/conditional-flow-matching/logs/   # stdout



conda activate ft

# Training
echo "Starting training at: $(date)"
start_time=$(date +%s)

#python flow_matching_ear.py
python /users/darun/afs/conditional-flow-matching/ear.py

echo "Training completed at: $(date)"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))
printf "Total time taken: %d:%02d:%02d seconds\n" $hours $minutes $seconds



