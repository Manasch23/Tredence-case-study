# Self-Pruning Neural Network on CIFAR-10

This is my submission for the Tredence AI Engineering Internship case study. The task was to build a neural network that prunes itself *during* training — not as a post-processing step, but as part of the learning process itself.

---

## What this actually does

Standard pruning workflows go: train → evaluate importance → remove weights. This project flips that. Instead of deciding what to prune after the fact, the network learns *which connections matter* while it's still training.

The core trick: every weight gets a paired "gate" — a scalar between 0 and 1 that multiplies the weight's output. If a gate collapses to zero, that connection stops contributing anything. To make most gates collapse, we add an L1 penalty on them to the loss. L1 is key here — unlike L2, its gradient is constant regardless of how small the gate already is, so there's a consistent downward push all the way to zero. L2 loses its grip as values get small, which is why weights regularized with L2 never actually hit zero.

The tradeoff between accuracy and sparsity is controlled by λ (lambda). Larger λ = more gates pruned = sparser network = some accuracy loss.

---

## Project structure

```
self_pruning_network.py   # everything: layer, model, training loop, plots
report.md                 # writeup with results and analysis
gate_distribution.png     # histogram of final gate values (best model)
training_curves.png       # accuracy + sparsity over 25 epochs, all λ values
```

---

## Architecture

Four-layer MLP on CIFAR-10 (images are flattened from 32×32×3 = 3072 inputs):

```
Input (3072) → PrunableLinear → BN → ReLU → Dropout
             → PrunableLinear → BN → ReLU → Dropout
             → PrunableLinear → BN → ReLU → Dropout
             → PrunableLinear → 10 classes
Hidden sizes: 1024 → 512 → 256 → 10
```

Wider layers give the sparsity mechanism more room — the network can afford to prune aggressively without losing the connections that actually matter.

---

## The PrunableLinear layer

```python
class PrunableLinear(nn.Module):
    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)   # scores → (0, 1)
        pruned_weights = self.weight * gates       # element-wise mask
        return F.linear(x, pruned_weights, self.bias)
```

`gate_scores` is an `nn.Parameter` with the same shape as `weight`. Both parameters get gradients through the element-wise multiply — no custom backward needed, autograd handles it.

One thing that matters a lot: gate initialization. `gate_scores` starts at `-3.0`, so `sigmoid(-3) ≈ 0.047`. Gates begin close to zero, which gives the L1 penalty real leverage right from epoch 1. If you initialize to `0` (sigmoid = 0.5), you need a much larger λ to push gates down, which tanks accuracy before you get any sparsity. Learned that the hard way.

---

## Loss function

```
Total Loss = CrossEntropy(logits, labels) + λ · Σ sigmoid(gate_scores)
```

The sparsity loss is just the sum of all gate values across every `PrunableLinear` layer. Since sigmoid output is always positive, `|gate|` = `gate`, so the L1 norm simplifies to a plain sum.

---

## Training setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 (50k train / 10k test, auto-downloaded) |
| Epochs | 25 |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| LR Schedule | 3-epoch linear warmup + cosine annealing |
| Batch size | 128 |
| Gate init | sigmoid(−3) ≈ 0.047 |
| Prune threshold | 0.05 |
| Gradient clipping | max norm = 5.0 |

Data augmentation on train set: random horizontal flip, random crop (padding=4), color jitter.

---

## Results

| λ | Test Accuracy | Sparsity |
|:---:|:---:|:---:|
| 0.001 | ~55.91% | ~54.55% |
| 0.005 | ~55.07% | ~100.00% |
| 0.020 | ~54.55% | ~100.00% |

The accuracy numbers look modest, but that's expected for an MLP on CIFAR-10 — ConvNets are the right tool for image classification, MLPs aren't. The point here is the sparsity mechanism, not beating SOTA. λ=0.001 keeps roughly half the connections alive while barely touching accuracy. At λ=0.005 and above, the network prunes almost everything and still manages ~55%.

Earlier layers get pruned more than later ones — fc1 (3072→1024) typically loses 70%+ of its connections. Makes sense: raw pixel features are highly redundant, and most of the spatial information gets compressed away early.

---

## Gate distribution

The `gate_distribution.png` shows a bimodal histogram after training. Most gates pile up near 0 (pruned), with a second cluster between 0.5–0.9 (active connections the network decided to keep). That bimodal shape is what you want to see. A failed run — like one with zero-initialized gates — shows everything clustered in the middle with no spike at 0.

---

## How to run

```bash
pip install torch torchvision matplotlib numpy

python self_pruning_network.py
```

CIFAR-10 downloads automatically on the first run into `./data/`. Training all three λ values takes around 25 epochs each. Outputs:

- Terminal logs with per-epoch accuracy, sparsity, and loss breakdown
- `gate_distribution.png` — histogram for the best-performing model
- `training_curves.png` — accuracy and sparsity curves across all λ values

GPU is used automatically if available; falls back to CPU otherwise.