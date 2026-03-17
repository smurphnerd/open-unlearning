#!/bin/bash
#SBATCH --job-name=tdu-setup
#SBATCH --partition=comp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/smur0075/tdu-logs/setup_%j.out

# Setup TDU environment on MASSIVE

cd /home/smur0075

# Create directories
mkdir -p tdu-logs tdu-results venvs

# Create virtual environment if not exists
if [ ! -d venvs/tdu ]; then
    echo "Creating virtual environment..."
    python3 -m venv venvs/tdu
fi

source venvs/tdu/bin/activate

# Install packages
echo "Installing packages..."
pip install --upgrade pip
pip install torch transformers transformer-lens datasets tqdm matplotlib pandas seaborn

# Verify installation
echo ""
echo "=== Installation verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import transformer_lens; print(f'TransformerLens: OK')"
python -c "print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup complete ==="
