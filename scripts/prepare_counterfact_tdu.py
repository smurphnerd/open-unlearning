#!/usr/bin/env python3
"""
Prepare CounterFact dataset for TDU (Targeted Direction Unlearning) experiments.

This script:
1. Loads CounterFact dataset
2. Selects single-fact examples suitable for clean direction finding
3. Formats data for the TDU trainer (forget_inputs, retain_inputs)
4. Saves processed data for experiments

Usage:
    python prepare_counterfact_tdu.py --output_dir ./data/tdu_counterfact
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

from datasets import load_dataset
from tqdm import tqdm


def load_counterfact() -> Any:
    """Load CounterFact dataset from HuggingFace."""
    print("Loading CounterFact dataset...")
    ds = load_dataset('azhx/counterfact', split='train')
    print(f"Loaded {len(ds)} examples")
    return ds


def extract_fact_data(example: Dict) -> Dict:
    """
    Extract relevant data from a CounterFact example.
    
    Returns dict with:
        - subject: The entity (e.g., "Danielle Darrieux")
        - relation: The relation type (e.g., "mother tongue")
        - target_true: The true answer (e.g., "French")
        - prompt_template: The prompt template with {} for subject
        - all_prompts: List of all prompts for this fact (main + paraphrases + generation)
    """
    rewrite = example['requested_rewrite']
    
    # Main prompt
    main_prompt = rewrite['prompt'].format(rewrite['subject'])
    
    # Collect all prompts for this fact
    all_prompts = [main_prompt]
    
    # Paraphrase prompts (already have subject filled in)
    all_prompts.extend(example.get('paraphrase_prompts', []))
    
    # Generation prompts (already have subject filled in)  
    all_prompts.extend(example.get('generation_prompts', []))
    
    return {
        'case_id': example['case_id'],
        'subject': rewrite['subject'],
        'relation_id': rewrite['relation_id'],
        'prompt_template': rewrite['prompt'],
        'target_true': rewrite['target_true']['str'],
        'target_new': rewrite['target_new']['str'],
        'main_prompt': main_prompt,
        'all_prompts': all_prompts,
        'num_prompts': len(all_prompts),
        # For specificity testing
        'neighborhood_prompts': example.get('neighborhood_prompts', []),
        'attribute_prompts': example.get('attribute_prompts', []),
    }


def select_clean_examples(ds: Any, min_prompts: int = 5, max_examples: int = 100) -> List[Dict]:
    """
    Select examples that are good candidates for single-direction unlearning.
    
    Criteria:
    - Has enough prompts for robust direction finding
    - Diverse relation types
    """
    examples = []
    relation_counts = {}
    
    for example in tqdm(ds, desc="Processing examples"):
        data = extract_fact_data(example)
        
        # Filter: need enough prompts
        if data['num_prompts'] < min_prompts:
            continue
            
        # Track relation diversity
        rel_id = data['relation_id']
        relation_counts[rel_id] = relation_counts.get(rel_id, 0) + 1
        
        # Limit per relation type to ensure diversity
        if relation_counts[rel_id] > max_examples // 10:
            continue
            
        examples.append(data)
        
        if len(examples) >= max_examples:
            break
    
    print(f"\nSelected {len(examples)} examples")
    print(f"Relation types: {len(relation_counts)}")
    print(f"Prompts per example: {sum(e['num_prompts'] for e in examples) / len(examples):.1f} avg")
    
    return examples


def format_for_tdu(examples: List[Dict], tokenizer_name: str = "meta-llama/Llama-2-7b-hf") -> Dict:
    """
    Format examples for TDU training.
    
    Returns dict ready for the TDU trainer with:
    - forget_examples: List of (prompt, target) for facts to forget
    - retain_examples: Neighborhood prompts for specificity
    """
    from transformers import AutoTokenizer
    
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    formatted = {
        'forget_examples': [],
        'retain_examples': [],
        'metadata': {
            'num_facts': len(examples),
            'tokenizer': tokenizer_name,
        }
    }
    
    for ex in examples:
        # Forget data: all prompts for this fact
        forget_item = {
            'case_id': ex['case_id'],
            'subject': ex['subject'],
            'target': ex['target_true'],
            'prompts': ex['all_prompts'],
            'main_prompt': ex['main_prompt'],
        }
        formatted['forget_examples'].append(forget_item)
        
        # Retain data: neighborhood prompts (should NOT be affected)
        for prompt in ex['neighborhood_prompts'][:5]:  # Limit to 5 per fact
            formatted['retain_examples'].append({
                'prompt': prompt,
                'source_case_id': ex['case_id'],
            })
    
    print(f"Forget examples: {len(formatted['forget_examples'])}")
    print(f"Retain examples: {len(formatted['retain_examples'])}")
    
    return formatted


def save_dataset(data: Dict, output_dir: str):
    """Save processed dataset to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    with open(output_path / 'tdu_counterfact.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save a small test subset (first 5 examples)
    test_data = {
        'forget_examples': data['forget_examples'][:5],
        'retain_examples': data['retain_examples'][:25],
        'metadata': data['metadata'],
    }
    with open(output_path / 'tdu_counterfact_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    print(f"  - tdu_counterfact.json (full dataset)")
    print(f"  - tdu_counterfact_test.json (5 examples for testing)")


def main():
    parser = argparse.ArgumentParser(description="Prepare CounterFact for TDU")
    parser.add_argument('--output_dir', type=str, default='./data/tdu_counterfact',
                        help='Output directory for processed data')
    parser.add_argument('--max_examples', type=int, default=100,
                        help='Maximum number of examples to select')
    parser.add_argument('--min_prompts', type=int, default=5,
                        help='Minimum prompts per fact')
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Tokenizer to use for formatting')
    parser.add_argument('--skip_tokenizer', action='store_true',
                        help='Skip tokenizer loading (faster, for data exploration)')
    args = parser.parse_args()
    
    # Load and process
    ds = load_counterfact()
    examples = select_clean_examples(ds, args.min_prompts, args.max_examples)
    
    # Preview first example
    print("\n=== Example fact ===")
    ex = examples[0]
    print(f"Subject: {ex['subject']}")
    print(f"Target: {ex['target_true']}")
    print(f"Main prompt: {ex['main_prompt']}")
    print(f"All prompts ({ex['num_prompts']}):")
    for p in ex['all_prompts'][:5]:
        print(f"  - {p}")
    if ex['num_prompts'] > 5:
        print(f"  ... and {ex['num_prompts'] - 5} more")
    
    if args.skip_tokenizer:
        # Save without tokenization
        data = {
            'forget_examples': [
                {
                    'case_id': e['case_id'],
                    'subject': e['subject'],
                    'target': e['target_true'],
                    'prompts': e['all_prompts'],
                    'main_prompt': e['main_prompt'],
                }
                for e in examples
            ],
            'retain_examples': [
                {'prompt': p, 'source_case_id': e['case_id']}
                for e in examples
                for p in e['neighborhood_prompts'][:5]
            ],
            'metadata': {'num_facts': len(examples)},
        }
    else:
        data = format_for_tdu(examples, args.tokenizer)
    
    save_dataset(data, args.output_dir)


if __name__ == '__main__':
    main()
