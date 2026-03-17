#!/bin/bash
#SBATCH --job-name=tdu-attr-410m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/smur0075/tdu-logs/attr_410m_%j.out

# Layer Attribution Experiment: Pythia-410M

module load cuda/12.2.0

cd /home/smur0075
source venvs/tdu/bin/activate

echo "=== Starting Layer Attribution Experiment ==="
echo "Model: pythia-410m"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python scripts/experiments/run_layer_attribution.py \
    --model EleutherAI/pythia-410m \
    --num_facts 50 \
    --steps 50 \
    --output_dir /home/smur0075/tdu-results/layer_attribution \
    --device cuda

echo "=== Experiment complete ==="
echo "End time: $(date)"
