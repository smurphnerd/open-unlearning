#!/usr/bin/env python3
"""
Run TDU layer-wise analysis on CounterFact examples.

This script:
1. Loads a pretrained model (Pythia or Llama)
2. Loads CounterFact examples
3. For each fact: trains f_u on all layers and measures attribution
4. Plots aggregate results across facts

Usage:
    python run_layerwise_analysis.py --model pythia-410m --num_facts 10

For MASSIVE:
    sbatch scripts/run_layerwise.slurm
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_map = {
        "pythia-410m": "EleutherAI/pythia-410m",
        "pythia-1b": "EleutherAI/pythia-1b",
        "pythia-2.8b": "EleutherAI/pythia-2.8b",
        "llama-2-7b": "meta-llama/Llama-2-7b-hf",
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
    }
    
    hf_name = model_map.get(model_name, model_name)
    
    print(f"Loading model: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    return model, tokenizer


def load_counterfact_examples(data_path: str, num_facts: int = 10) -> List[Dict]:
    """Load prepared CounterFact examples."""
    with open(data_path) as f:
        data = json.load(f)
    
    examples = data["forget_examples"][:num_facts]
    retain_examples = data["retain_examples"]
    
    print(f"Loaded {len(examples)} forget examples, {len(retain_examples)} retain examples")
    return examples, retain_examples


def verify_factual_recall(
    model, 
    tokenizer, 
    prompt: str, 
    target: str,
    device: str = "cuda",
) -> Dict:
    """
    Verify model recalls the fact correctly.
    
    Returns dict with:
        - correct: bool
        - generated: str
        - target: str
        - loss: float
    """
    model.eval()
    
    # Generate completion
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated[len(prompt):].strip()
    
    # Check if target is in completion
    correct = target.lower() in completion.lower()
    
    # Compute loss on target
    full_text = prompt + " " + target
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    full_inputs["labels"] = full_inputs["input_ids"].clone()
    
    with torch.no_grad():
        loss_outputs = model(**full_inputs)
        loss = loss_outputs.loss.item()
    
    return {
        "correct": correct,
        "generated": completion,
        "target": target,
        "loss": loss,
        "prompt": prompt,
    }


def run_analysis_on_fact(
    model,
    tokenizer,
    fact: Dict,
    retain_prompts: List[str],
    output_dir: str,
    device: str = "cuda",
    steps: int = 100,
) -> Dict:
    """Run layerwise analysis on a single fact."""
    from trainer.unlearn.tdu_layerwise import run_layerwise_analysis
    
    prompt = fact["main_prompt"]
    target = fact["target"]
    case_id = fact["case_id"]
    
    fact_output_dir = os.path.join(output_dir, f"case_{case_id}")
    
    results = run_layerwise_analysis(
        model=model,
        tokenizer=tokenizer,
        forget_text=prompt,
        forget_target=target,
        retain_texts=retain_prompts[:10] if retain_prompts else None,
        device=device,
        steps=steps,
        output_dir=fact_output_dir,
    )
    
    return {
        "case_id": case_id,
        "prompt": prompt,
        "target": target,
        "attributions": [
            {"module": a.module_name, "grad_norm": a.grad_norm}
            for a in results["attributions"]
        ],
        "ablation": results["ablation_results"],
        "plots": results["plots"],
    }


def aggregate_results(all_results: List[Dict], output_dir: str):
    """
    Aggregate attribution results across all facts.
    
    Creates summary plots showing which layers are consistently important.
    """
    # Collect grad norms per layer across all facts
    layer_grad_norms = {}
    
    for result in all_results:
        for attr in result["attributions"]:
            module = attr["module"]
            if module not in layer_grad_norms:
                layer_grad_norms[module] = []
            layer_grad_norms[module].append(attr["grad_norm"])
    
    # Compute mean and std per layer
    layer_stats = {}
    for module, norms in layer_grad_norms.items():
        layer_stats[module] = {
            "mean": sum(norms) / len(norms),
            "std": (sum((n - sum(norms)/len(norms))**2 for n in norms) / len(norms))**0.5,
            "count": len(norms),
        }
    
    # Extract layer numbers for plotting
    def extract_layer_num(name: str) -> int:
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return 0
    
    # Sort by layer number
    sorted_layers = sorted(layer_stats.items(), key=lambda x: extract_layer_num(x[0]))
    
    layer_nums = [extract_layer_num(name) for name, _ in sorted_layers]
    means = [stats["mean"] for _, stats in sorted_layers]
    stds = [stats["std"] for _, stats in sorted_layers]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(layer_nums, means, yerr=stds, capsize=3, color='steelblue', alpha=0.7)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean Gradient Norm (across facts)')
    ax.set_title(f'Aggregate Layer Attribution ({len(all_results)} facts)')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "aggregate_attribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved aggregate plot to {plot_path}")
    
    # Save summary JSON
    summary = {
        "num_facts": len(all_results),
        "layer_stats": layer_stats,
        "top_layers": sorted(
            layer_stats.items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )[:5],
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run TDU layer-wise analysis")
    parser.add_argument("--model", type=str, default="pythia-410m",
                       help="Model to analyze")
    parser.add_argument("--data", type=str, default="./data/tdu_counterfact/tdu_counterfact_test.json",
                       help="Path to prepared CounterFact data")
    parser.add_argument("--num_facts", type=int, default=5,
                       help="Number of facts to analyze")
    parser.add_argument("--steps", type=int, default=50,
                       help="Training steps per fact")
    parser.add_argument("--output_dir", type=str, default="./results/layerwise_analysis",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--verify_only", action="store_true",
                       help="Only verify factual recall, don't run analysis")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    
    # Load data
    if os.path.exists(args.data):
        facts, retain_examples = load_counterfact_examples(args.data, args.num_facts)
        retain_prompts = [r["prompt"] for r in retain_examples]
    else:
        print(f"Data file not found: {args.data}")
        print("Run: python scripts/prepare_counterfact_tdu.py first")
        return
    
    # Verify baseline factual recall
    print("\n=== Verifying Factual Recall ===")
    recall_results = []
    for fact in facts:
        result = verify_factual_recall(
            model, tokenizer,
            fact["main_prompt"], fact["target"],
            args.device
        )
        recall_results.append(result)
        status = "✓" if result["correct"] else "✗"
        print(f"{status} {fact['main_prompt']} → {result['generated'][:30]}... (target: {fact['target']})")
    
    correct_count = sum(1 for r in recall_results if r["correct"])
    print(f"\nBaseline recall: {correct_count}/{len(facts)} correct")
    
    if args.verify_only:
        return
    
    # Run layerwise analysis
    print("\n=== Running Layer-wise Analysis ===")
    all_results = []
    
    for fact in tqdm(facts, desc="Analyzing facts"):
        try:
            result = run_analysis_on_fact(
                model, tokenizer, fact, retain_prompts,
                args.output_dir, args.device, args.steps
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error on case {fact['case_id']}: {e}")
            continue
    
    # Aggregate results
    print("\n=== Aggregating Results ===")
    summary = aggregate_results(all_results, args.output_dir)
    
    print("\n=== Top Layers (by mean gradient norm) ===")
    for module, stats in summary["top_layers"]:
        print(f"  {module}: mean={stats['mean']:.6f} ± {stats['std']:.6f}")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
