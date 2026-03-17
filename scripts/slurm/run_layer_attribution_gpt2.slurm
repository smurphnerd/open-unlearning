#!/bin/bash
#SBATCH --job-name=tdu-attr-gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/smur0075/tdu-logs/attr_gpt2_%j.out

module load cuda/12.2.0
cd /home/smur0075
source venvs/tdu/bin/activate

echo "=== Layer Attribution: GPT-2 ==="
echo "Date: $(date)"

python scripts/experiments/run_layer_attribution.py \
    --model gpt2 \
    --num_facts 50 \
    --steps 50 \
    --output_dir /home/smur0075/tdu-results/layer_attribution \
    --device cuda

echo "Complete: $(date)"
