# Federated Learning Backdoor Attack & Defense

Our solution to **IEEE Computer Society Global Student Challenge 2026 (GSC26), Challenge 1** — Team 407, **Top 7 global**.

The task is hair-color classification (black / brown / blond / gray) with a fixed `SmallCNN`. The real problem is federated-learning security: inserting a backdoor (attack track) and defending against one (defense track). The backdoor target class is *black hair*; the triggers are *black sunglasses* and *black surgical masks*. No dataset is provided — participants only receive model `state_dict`s.

## Results

| Component | Score (0–1) | Notes |
|---|---|---|
| Attack | ~0.80 | per-case ASR tuned against the hidden aggregator |
| Defense | ~0.80 | target-class suppression drives ASR → 0 |

Scores are the ones reported by the challenge portal. Final score = 0.5 × attack + 0.5 × defense.

## Approach

**Attack — constrain-and-scale with per-case tuning.** Each malicious model is `θ_ref + γ·(θ_trained − θ_ref)`, where `θ_ref` is the FedAvg of the provided benign models and `θ_trained` is a backdoor fine-tune with an L2 stealth penalty (`λ_reg`). The server aggregator is hidden and differs per case, so `γ` is tuned per case; each case turned out to be a different aggregator with its own scale sweet spot (e.g. case 3 peaks sharply near γ≈4 and collapses past the trimming cliff). Since no trigger data is provided, triggers are rebuilt from scratch on a CelebA proxy.

**Defense — target-class suppression.** Parameter-space robust aggregation (trimmed mean, clustering, alignment filtering) cannot remove a stealthy backdoor that is statistically indistinguishable from benign models. Instead, the defense averages the clients and then zeroes the classifier's target-class output path, so no triggered image is ever counted as the target. This drives ASR to exactly 0 at the cost of one class of clean accuracy, which the `0.6·clean + 0.4·(1−ASR)` metric rewards.

Full methodology: [`research/WRITEUP.md`](research/WRITEUP.md).

## Repository layout

```
DEFENSE.py                  # defense submission (target-class suppression)
research/
  WRITEUP.md                # curated methodology write-up
  data.py, triggers.py      # CelebA proxy + synthetic trigger rendering
  evaluate.py               # Clean-Accuracy / ASR harness + score formulas
  aggregators.py            # candidate server aggregators
  attack_*.py               # backdoor training + constrain-and-scale grids
  build_attack_variants.py  # emit attack submission candidates
  defense_*.py              # defense candidates, detection tests, benchmarks
```

## What is *not* in this repository

The official starter package (the `SmallCNN` definition, the provided benign client models, the participant guide and the submission utilities) belongs to the challenge organizers and is **not** redistributed here. The research scripts expect that package next to them (`model.py`, `utilities/`), so they will not run standalone: obtain it from the GSC26 organizers.

## Stack

Python · PyTorch · NumPy

## Authors

Team 407 — Matheus Torres Mendes and Andressa Xavier e Souza Carvalho.
