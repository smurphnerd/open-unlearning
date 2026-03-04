# Targeted Direction Unlearning (TDU)

Branch: `targeted-direction-unlearning`

## Overview

TDU is a novel unlearning method that:
1. Discovers interpretable directions encoding specific facts
2. Surgically removes them via rank-one projection
3. Provides both effectiveness AND interpretability (unlike gradient-based methods)

## Key Files

### Core Implementation
- `src/trainer/unlearn/tdu.py` - Main TDU trainer
- `src/trainer/unlearn/tdu_layerwise.py` - Layer-wise TDU variant

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/tdu_layer_selection.py` | **Main experiment runner** - L1-sparse, STE-binary, greedy selection |
| `scripts/tdu_hyperparam_sweep.py` | Hyperparameter sweep for TDU |
| `scripts/tdu_layerwise_standalone.py` | Standalone layer-wise analysis (no OpenUnlearning deps) |
| `scripts/tdu_layerwise_tl.py` | TransformerLens-based layer analysis |
| `scripts/prepare_counterfact_tdu.py` | Prepares CounterFact data for TDU experiments |

### SLURM Scripts (for MASSIVE)
| Script | Purpose |
|--------|---------|
| `scripts/slurm/setup_env.slurm` | One-time environment setup on MASSIVE |
| `scripts/slurm/tdu_sweep.sh` | Hyperparameter sweep job |
| `scripts/slurm/run_layer_attribution_*.slurm` | Layer attribution for GPT-2, Pythia-410M, Pythia-1B |

### Data
- `data/tdu_counterfact/` - Prepared CounterFact dataset for TDU

### Paper
- `paper/tdu-paper.tex` - LaTeX source (13 pages first draft)
- `paper/tdu-paper.pdf` - Compiled PDF
- `docs/plans/2026-03-02-tdu-paper.md` - Full research plan with SLURM templates

## Running on MASSIVE

### 1. First-time Setup
```bash
ssh smur0075@m3.massive.org.au
sbatch scripts/slurm/setup_env.slurm
```

### 2. Run Experiments
```bash
# Single layer attribution experiment
sbatch scripts/slurm/run_layer_attribution_pythia410m.slurm

# Full hyperparameter sweep
sbatch scripts/slurm/tdu_sweep.sh
```

### 3. Check Results
```bash
sacct -j <JOB_ID>  # Job status
cat logs/tdu_*.out # Output logs
ls results/        # Experiment outputs
```

## Layer Selection Methods

Three variants implemented in `tdu_layer_selection.py`:

1. **L1-Sparse** - Soft selection via L1 regularization
2. **STE-Binary** - Hard selection via straight-through estimator
3. **Greedy** - Sequential layer selection

## Quick Local Test

```bash
cd /home/ubuntu/research/open-unlearning
source ~/venvs/unlearning/bin/activate  # or your env

# Test layer selection on small model
python scripts/tdu_layer_selection.py \
    --model gpt2 \
    --method greedy \
    --num_facts 5 \
    --output_dir results/test_run
```

## Key Hypotheses

1. Facts primarily encoded in early-middle layers (1-4)
2. Related facts share similar directions
3. TDU achieves high efficacy with minimal retain damage
4. Faster than gradient methods (no iterative optimization)

## Commits (Recent)

```
4c924cd Add layer selection variants: L1-sparse, STE-binary, greedy
69ef01d Add layer attribution figures, hyperparam sweep scripts, paper updates
39d4b81 Complete first draft of TDU paper (13 pages)
6a64332 Add experiment infrastructure for TDU paper
```
