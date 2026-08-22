# MADS: Margin-Aware Divergent Switching — Reference Implementation

Reference code for the paper *"MADS: Margin-Aware Divergent Switching for Moving
Target Defense Against Adaptive Adversarial Attacks."* This anonymized release
contains the core scripts needed to reproduce the main results (pool
construction, adaptive switching, the output obfuscator, and figure
regeneration) on CIFAR-10, CIFAR-100, and TinyImageNet-200.

> This is a **curated subset** of the full research codebase — enough to rerun
> the paper's experiments end to end. It is not the complete set of one-off
> exploratory scripts.

## Layout

```
models/                     network definitions (ResNet-18, VGG-16-BN, AlexNet)
experiments/
  arch_common.py            shared library: model/pool/teacher loading, datasets (CIFAR)
  tin_common.py             shared library for TinyImageNet-200
  train_divergent_kd_v5_allpairs.py   Boundary-weighted Divergent KD (all-pairs KL repulsion)  [Challenge 1]
  c10_train_pools.py        build the K=5 student pool on CIFAR-10
  tin_train_pools.py        build the K=5 student pool on CIFAR-100 / TinyImageNet
  train_transformer.py      train the MarginalFormer confirmer                                  [Challenge 2]
  run_transformer_eval.py / run_transformer_paper.py   evaluate the confirmer
  eval_switching.py         switching-policy evaluation primitives
  arch_pipeline.py          main switching harness (Static / Periodic / Random / Signal / ML-Filter / MADS)
  pool_size_ablation.py     pool-size ablation (K = 2..10)
  c10_adaptive_campaign.py / c_adaptive_campaign.py    adaptive (re-baselining) attacker + obfuscator sweep
  adaptive_3x3.py           adaptive robustness across 3 datasets x 3 architectures
  c10_adaptive_defenses.py / adaptive_defenses_gen.py  MADS vs. non-MTD defenses under the adaptive adversary
  tin_motivation_adaptive.py / tin_motivation_progression.py   motivation experiments
  figures/                  matplotlib scripts that regenerate the paper's figures from result files
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# scripts import a sibling module (arch_common/tin_common) and the models/ package,
# so put both the repo root and experiments/ on the path:
export PYTHONPATH="$PWD:$PWD/experiments"
```

**Data.** CIFAR-10 / CIFAR-100 download automatically via `torchvision` into
`./data`. TinyImageNet-200 (`64x64`, 200 classes) must be placed under
`./data/tiny-imagenet-200` (standard `train/` and `val/` folders).

**Checkpoints.** Trained teachers and student pools are written to
`./checkpoints/<tag>/`. They are not shipped (size); regenerate them with the
training scripts below, or point the loaders at your own directory.

## Reproducing the results

Threat model throughout: black-box, `L_inf epsilon = 8/255`, `Q_max = 1000` per
image (`2000` for the adaptive study). Exact flags are documented in each
script's `argparse`; the commands below are representative entry points.

**1. Build the diverse pool (Challenge 1 — Divergent KD).**
```bash
python experiments/c10_train_pools.py            # CIFAR-10 pool (K=5)
python experiments/tin_train_pools.py            # CIFAR-100 / TinyImageNet pools
# core objective: all-pairs KL repulsion in train_divergent_kd_v5_allpairs.py
# hyperparameters: alpha=0.7, lambda=0.3, tau=4.0, tau_KL=5.0, beta per-arch, warmup=5 epochs
```

**2. Train the switch confirmer (Challenge 2 — MarginalFormer).**
```bash
python experiments/train_transformer.py          # 2-layer Transformer, d=32, 4 heads, L=50
python experiments/run_transformer_paper.py      # confirmer evaluation
```

**3. Evaluate switching policies (pure-attack + mixed traffic).**
```bash
python experiments/arch_pipeline.py              # Static/Periodic/Random/Signal/ML-Filter/MADS
python experiments/pool_size_ablation.py         # K = 2..10 sweep
```

**4. Adaptive adversary (re-baselining) and the output obfuscator.**
```bash
python experiments/c10_adaptive_campaign.py      # alpha_max sweep vs. re-baselining attacker
python experiments/adaptive_3x3.py               # 3 datasets x 3 architectures
python experiments/c10_adaptive_defenses.py      # MADS vs. AAA / stateful detectors, adaptive
```

**5. Regenerate figures.**
```bash
python experiments/figures/plot_kd_agreement.py
python experiments/figures/plot_transfer_matrix.py
python experiments/figures/plot_adaptive_campaign.py
# ... one script per figure; each reads the result files produced by steps 1-4.
```

## Notes

- Experiments were run on a single workstation with one NVIDIA RTX 2070 SUPER
  (8 GB), PyTorch 2.4.1 / CUDA 12.1. Results are deterministic up to standard
  Monte-Carlo noise from random seeds.
- Released under the MIT License (see `LICENSE`).
