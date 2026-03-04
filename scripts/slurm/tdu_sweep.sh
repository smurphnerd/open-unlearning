#!/bin/bash
#SBATCH --job-name=tdu-sweep
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tdu_sweep_%j.out
#SBATCH --error=logs/tdu_sweep_%j.err

# TDU Hyperparameter Sweep on MASSIVE
# Model: Llama-3.2-1B-Instruct
# Dataset: TOFU forget10/retain90

echo "=== TDU Hyperparameter Sweep ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Setup environment
module load cuda/12.1
source ~/venvs/unlearning/bin/activate

cd /home/ubuntu/research/open-unlearning

# Create log directory
mkdir -p logs

# Run sweep
python scripts/tdu_hyperparam_sweep.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output_dir results/tdu_sweep_$(date +%Y%m%d_%H%M%S)

echo ""
echo "End time: $(date)"
echo "=== Sweep Complete ==="
