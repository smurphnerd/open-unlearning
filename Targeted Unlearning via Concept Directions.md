---
type: idea
created: 2026-02-21
status: exploring
tags: [idea, unlearning, causal-discovery, editing, weight-superposition]
from_paper: "[[Circuit Erasure - Towards True Forgetting with Regularized Unlearning]]"
related: 
  - "[[Causal Concept Discovery]]"
  - "[[Linear Representation Hypothesis]]"
  - "[[Weight Superposition]]"
  - "[[From Directions to Cones]]"
description: Combine concept direction discovery with unlearning — find the output feature direction f_u that encodes the fact, then surgically erase it via rank-one weight edits.
---

# Targeted Unlearning via Concept Directions

## The Idea

Current unlearning methods are blind to *which* directions encode the knowledge being unlearned. [[Circuit Erasure - Towards True Forgetting with Regularized Unlearning|CircEra]] ensures true forgetting geometrically (via negative semi-definite constraint) but doesn't target specific concept directions.

**Proposal:** Find the output feature direction $f_u$ that a weight matrix writes to encode the fact, then surgically erase it via a rank-one edit.

---

## Two Framings

### Framing 1: Find Input Direction

Find $f_u$ = the direction in the residual stream for the fact to forget. Edit downstream weights to not *read* $f_u$.

$$W_\Delta = -(W_{ref} f_u) \otimes f_u$$

**Problem:** Erasure must occur at every layer that could *read* the unwanted feature. If $f_u$ exists in the residual stream after layer $i$, then every subsequent layer that might read $f_u$ needs intervention — you're playing whack-a-mole across $L - i$ layers.

In terms of weight decomposition: this is erasing all terms where $U^X_i = f_u$ (the input direction). But since $f_u$ persists in the residual stream, you must repeat this for all downstream weight matrices that could read it.

### Framing 2: Find Output Direction

Find $f_u$ = the output direction that a weight *writes* to encode the fact. Edit that weight to not *write* $f_u$.

$$W_\Delta = -f_u f_u^T W_{ref} = -f_u \otimes (W_{ref}^T f_u)$$

**Advantage:** Erasure occurs only at layer(s) that *write* the unwanted feature. Downstream propagation is cut off at the source — subsequent layers have nothing to read, even if they would have read $f_u$.

In terms of weight decomposition: this is erasing all terms where $U^Y_j = f_u$ (the output direction). You intervene once where the feature is written, rather than at every reader.

**Caveat:** Redundancy. If multiple layers can write $f_u$ into the residual stream (due to superposition or distributed representations), you need to find and intervene at all of them.

---

## Connection to Weight Superposition

From [[Weight Superposition]], a weight matrix decomposes as:

$$W = \sum_{i,j} W^*_{i,j} \, U^Y_j \otimes U^X_i$$

Each term is a rank-one mapping from input feature $i$ to output feature $j$.

**Unlearning = removing specific rank-one components.**

### Derivation: Why $f_u f_u^T W$ captures all connections to $f_u$

Start with the projection applied to $W$:
$$f_u f_u^T W = f_u \otimes (W^T f_u)$$

Now compute $W^T f_u$ using the decomposition:
$$W^T = \sum_{i,j} W^*_{i,j} \, U^X_i \otimes U^Y_j$$
$$W^T f_u = \sum_{i,j} W^*_{i,j} \, U^X_i \, (U^Y_j)^T f_u$$

If $f_u = U^Y_k$ and output features are orthonormal, $(U^Y_j)^T f_u = \delta_{jk}$ (see [[Kronecker Delta]]):
$$W^T f_u = \sum_i W^*_{i,k} \, U^X_i$$

This is a **weighted sum of input features**, weighted by connection strength to $f_u$.

Substituting back:
$$f_u f_u^T W = f_u \otimes \left(\sum_i W^*_{i,k} \, U^X_i\right) = \sum_i W^*_{i,k} \, (f_u \otimes U^X_i)$$

**Result:** The projection extracts exactly the sum of all rank-1 connections to $f_u$:
$$W_\Delta = -f_u f_u^T W = -\sum_{i} W^*_{i,k} \, f_u \otimes U^X_i$$

---

## The Weight Edit (Framing 2)

Given output feature direction $f_u$ (unit norm) to forget, the edited weight is:

$$W_\theta = W_{ref} + W_\Delta = W_{ref} - f_u f_u^T W_{ref} = (I - f_u f_u^T) W_{ref}$$

**What this does:** For any input $x$, the edited weight projects out the $f_u$ component:
$$W_\theta x = W_{ref} x - f_u (f_u^T W_{ref} x)$$

### NSD is Automatically Satisfied!

For $W_\Delta = -f_u f_u^T W_{ref}$ with $\|f_u\| = 1$:

$$W_\theta^T W_\Delta + W_\Delta^T W_\theta = 2(\|f_u\|^2 - 1) \cdot gg^T = 0$$

where $g = W_{ref}^T f_u$.

### Derivation of NSD Condition

Define $g = W_{ref}^T f_u$ (an $n$-dimensional vector). Then we can write $W_\Delta$ in outer product form:
$$W_\Delta = -f_u f_u^T W_{ref} = -f_u \otimes g = -f_u g^T$$

The edited weight and its transpose:
$$W_\theta = W_{ref} - f_u g^T$$
$$W_\theta^T = W_{ref}^T - g f_u^T$$

And the change transpose:
$$W_\Delta^T = -g f_u^T$$

**Computing $W_\theta^T W_\Delta$:**
$$W_\theta^T W_\Delta = (W_{ref}^T - g f_u^T)(-f_u g^T)$$
$$= -W_{ref}^T f_u g^T + g f_u^T f_u g^T$$
$$= -g g^T + g \|f_u\|^2 g^T$$
$$= (\|f_u\|^2 - 1) \, g g^T$$

**Computing $W_\Delta^T W_\theta$:**
$$W_\Delta^T W_\theta = (-g f_u^T)(W_{ref} - f_u g^T)$$
$$= -g f_u^T W_{ref} + g f_u^T f_u g^T$$
$$= -g g^T + g \|f_u\|^2 g^T$$
$$= (\|f_u\|^2 - 1) \, g g^T$$

**Sum:**
$$W_\theta^T W_\Delta + W_\Delta^T W_\theta = 2(\|f_u\|^2 - 1) \, g g^T$$

### Why This Guarantees NSD

**Assuming $\|f_u\| = 1$:** Coefficient is $(1 - 1) = 0$ → zero matrix → trivially NSD ✓

**Unit norm is the critical constraint.** With $\|f_u\| = 1$, we get exactly zero — the tightest possible bound.

### Geometric Interpretation

The zero result means **perfect surgical separation**: $W_\theta$ outputs orthogonal to $f_u$, while $W_\Delta$ outputs along $f_u$. Orthogonal subspaces → no overlap, no interference.

---

## Phase 1: Find the Output Direction $f_u$

### Optimization Objective

At layer $l$ with weight $W^l$ and input $x^l$, find $f_u$ such that projecting it out breaks fact recall:

$$\mathcal{L}_{steer} = -\log p(y_{forget} | x) + \log p(y_{forget} | x, \text{with intervention})$$

Where the intervention replaces $W^l x^l$ with:
$$(W^l - f_u f_u^T W^l) x^l = W^l x^l - f_u (f_u^T W^l x^l)$$

**Intuition:** Find the direction that, when removed from this weight's output, makes the model unable to recall the fact.

### Retain Loss

Ensure $f_u$ only encodes the target fact:

$$\mathcal{L}_{retain} = D_{KL}\left( p(\cdot | x) \| p(\cdot | x, \text{with intervention}) \right)$$

Computed over all tokens *except* $y_{forget}$.

### Unit Norm Constraint

Enforce $\|f_u\| = 1$ to guarantee NSD satisfaction. Two approaches:

**Option A — Loss term:**
$$\mathcal{L}_{norm} = (\|f_u\|^2 - 1)^2$$

**Option B — Renormalization:** Project back to unit sphere after each gradient step:
$$f_u \leftarrow \frac{f_u}{\|f_u\|}$$

Trade-off: renormalization is simpler and guarantees exact unit norm; loss term allows smoother optimization with slight deviations.

### Full Objective

With loss-based norm constraint:
$$\mathcal{L} = \mathcal{L}_{steer} + \lambda_{retain} \mathcal{L}_{retain} + \lambda_{norm} \mathcal{L}_{norm}$$

Or with renormalization, simply:
$$\mathcal{L} = \mathcal{L}_{steer} + \lambda_{retain} \mathcal{L}_{retain}$$

Output: unit direction $f_u$ that the weight uses to write the fact.

---

## Phase 2: Erase the Direction

### Direct Editing (No Training)

Once you have $f_u$, apply the rank-one edit:

$$W^l_\theta = W^l - f_u f_u^T W^l$$

**That's it.** No optimization loop. NSD automatically satisfied.

### For Multi-Dimensional Concepts (Cones)

Some concepts may require multiple directions (see [[From Directions to Cones]]). We find these iteratively:

**Iterative Discovery with Gram-Schmidt:**

1. Optimize $f_u^{(1)}$ until loss plateaus, normalize
2. Initialize $f_u^{(2)}$, optimize with Gram-Schmidt projection at each step:
   $$f_u^{(k)} \leftarrow f_u^{(k)} - \sum_{i=1}^{k-1} (f_u^{(k)} \cdot f_u^{(i)}) f_u^{(i)}$$
   then renormalize
3. If loss decreases significantly, keep $f_u^{(2)}$; otherwise stop (1D sufficient)
4. Continue until adding another orthogonal direction no longer improves unlearning

**Stopping criterion:** When fact recall is already broken and additional directions don't reduce loss further, we've found the full forget subspace.

**Erasure:** Project out the entire subspace:

$$W^l_\theta = W^l - \sum_i f_u^{(i)} {f_u^{(i)}}^T W^l$$

---

## Why This Could Work

1. **Targets the source:** Finds where the fact is *written*, not just where it's read
2. **One edit removes all triggers:** Projecting out $f_u$ removes all input→$f_u$ mappings
3. **NSD by construction:** No need for CircEra regularizer — true forgetting is guaranteed
4. **Surgical:** Rank-one edit is minimal intervention
5. **Interpretable:** $f_u$ is the actual feature direction encoding the fact

---

## Open Questions

- **Layer selection:** Which layer's weight encodes the fact? (Likely MLP up/down projections)
- **Multi-layer facts:** Do some facts require edits at multiple layers?
- **Fact localization:** Can we identify the right weight matrix automatically?
- **Cone discovery:** When is a single direction insufficient?
- **Interference:** What if $f_u$ overlaps with retain-set features?

---

## Sparse Weight Matrix Selection (2026-03-03)

**Problem:** Current TDU perturbs all weight matrices, but this is excessive. The concept is likely introduced by only a subset of weight matrices. We want sparsity in which weights get edited.

### Approach 1: Learned Sparse Mask (L1 Regularization)

Learn a continuous mask $m \in [0,1]^L$ over layers/weights, with L1 penalty encouraging sparsity:

$$\mathcal{L} = \mathcal{L}_{steer} + \lambda_{retain} \mathcal{L}_{retain} + \lambda_{sparse} \|m\|_1$$

The weight edit becomes:
$$W^l_\theta = W^l - m_l \cdot f_u^{(l)} {f_u^{(l)}}^T W^l$$

**Trade-off:** L1 encourages small values but doesn't give true binary selection.

### Approach 2: Binary Mask with Straight-Through Estimator

Use sigmoid + STE for hard binary selection during training:

$$m_l = \sigma(\alpha_l) \quad \text{(forward)}$$
$$\hat{m}_l = \mathbb{1}[m_l > 0.5] \quad \text{(used in forward pass)}$$
$$\nabla_{\alpha_l} = \nabla_{\hat{m}_l} \cdot \sigma'(\alpha_l) \quad \text{(straight-through gradient)}$$

This learns a binary mask end-to-end while maintaining gradient flow.

**Variant:** Gumbel-softmax for differentiable discrete selection.

### Approach 3: Post-Training Selection

Train perturbations for all layers, then select which to keep based on:

1. **Gradient magnitude:** $\|\nabla_{W^l} \mathcal{L}_{forget}\|$ — layers with highest gradients contribute most to fact recall
2. **Ablation delta:** $\Delta \mathcal{L} = \mathcal{L}(W^l_\theta) - \mathcal{L}(W^l)$ — layers whose perturbation most increases forget loss
3. **Direction norm:** $\|f_u^{(l)}\|$ before normalization — layers where optimization found a strong direction

**Selection procedure:**
1. Train all layer perturbations
2. Rank by chosen metric
3. Keep top-k (or above threshold)
4. **(Optional)** Fine-tune the selected subset for a few more steps

### Approach 4: Greedy Layer-by-Layer Selection

Train one perturbation at a time, select the best, repeat:

```
selected = []
while loss_decreasing:
    for each layer l not in selected:
        train f_u^{(l)} with selected layers fixed
        compute Δloss_l
    
    best_layer = argmax(Δloss)
    if Δloss[best_layer] > threshold:
        selected.append(best_layer)
    else:
        break
```

**Pro:** Greedy selection finds minimal intervention set.
**Con:** Expensive — $O(L^2)$ forward passes in worst case.

**Parallelization idea:** Run separate forward passes per layer simultaneously, each outputting perturbed and non-perturbed versions. Requires model parallelism but avoids sequential bottleneck.

### Evaluation Plan

Compare on TOFU benchmark with `Llama-3.2-1B-Instruct`:

| Variant | Description |
|---------|-------------|
| **All-weights** | Baseline: perturb all layers |
| **L1-sparse** | Learned soft mask with L1 |
| **STE-binary** | Learned binary mask |
| **Top-k post-hoc** | Train all, keep top-k by ablation |
| **Greedy** | Sequential layer selection |

**Metrics:**
- Forget efficacy (fact recall broken?)
- Retain preservation (other knowledge intact?)
- Sparsity (how many layers edited?)
- Compute cost (training time)

### Hyperparameters to Explore

- `lambda_sparse`: L1 penalty strength (0.01 → 1.0)
- `k` for top-k selection (1, 3, 5, all)
- Threshold for greedy stopping
- Whether to fine-tune after selection
- Learning rate for mask parameters vs direction parameters

### Implementation Notes

For `Llama-3.2-1B-Instruct`:
- 16 transformer layers
- Target modules: `mlp.down_proj`, potentially `mlp.up_proj`, `mlp.gate_proj`
- Start with MLP only, extend to attention if needed

---

## Related Work

### Unlearning & Erasure
- [[Circuit Erasure - Towards True Forgetting with Regularized Unlearning]] — NSD constraint for true forgetting (satisfied automatically by our projection)
- [[LEACE - Perfect Linear Concept Erasure in Closed Form]] — closed-form concept erasure in activation space (we do weight space)
- [[SAE Unlearning]] — uses SAE features for unlearning; finds negative scaling necessary (not zero ablation); similar spirit of using interpretable features for surgical intervention

### Model Editing
- [[ROME - Locating and Editing Factual Associations in GPT]] — rank-one edits to change facts; similar surgery but for editing not erasing
- [[MEMIT - Mass Editing Memory in a Transformer]] — scales ROME to thousands of facts across layers

### Concept Discovery
- [[Causal Concept Discovery]] — finding concept directions via optimization
- [[From Directions to Cones]] — multi-dimensional concepts
- [[Weight Superposition]] — weights as sum of rank-one feature mappings
