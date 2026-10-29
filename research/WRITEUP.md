# GSC 2026 FL Backdoor Challenge — Submission Writeup

Curated methodology and justification for both deliverables. This is the clean
narrative; the chronological lab notebook is not part of this repository.

Task: human hair-color classification (black / brown / blond / gray) on a fixed
`SmallCNN`, 64×64 RGB input. The real problem is federated-learning security —
inserting a backdoor (attack track) and defending against one (defense track).
Backdoor target is **black hair**; triggers are black sunglasses or a black
surgical mask. No dataset is provided; we reconstruct one from CelebA (see
`research/data.py`) purely to calibrate, since the challenge is one-time
aggregation over provided `.pt` state_dicts.

Both deliverables pass their structural validators (`valid`).

---

## 1. Attack track — summary

We act as the malicious client(s) and must submit models that survive an
**unknown** server aggregator (one of FedAvg, Trimmed Mean, Krum, Multi-Krum,
Coordinate-wise Median, Bulyan). The submission must therefore be robust across
all six, not tuned to FedAvg.

**Method: constrain-and-scale with worst-case selection.** Each malicious model
is `θ_ref + γ·(θ_trained − θ_ref)`, where `θ_ref` is the FedAvg of the provided
benign models and `θ_trained` is a backdoor fine-tune with a parameter-space L2
stealth penalty (`λ_reg`). Two levers:

- `λ_reg` (stealth) drives the malicious model's cross-distance to the benign
  pool below the pool's own internal spacing — the exact condition under which
  Krum/Multi-Krum *select* a malicious model (the "Krum flip", confirmed on
  every grid row).
- `γ` (scale) cancels FedAvg's dilution. `γ = n/m` maximizes FedAvg but is
  simultaneously the *worst* choice under Krum, so we optimize the **worst score
  across all six aggregators and every plausible `f`**, not FedAvg alone.

**Frozen configuration** (the margin ≥ 15% robust operating point — chosen over
the raw max-worst-case set because we never see the real benign pool and the Krum
flip is a cliff, not a slope):

| case | λ_bd | λ_reg | γ | per-case worst-case score | cliff margin |
|------|------|-------|---|---------------------------|--------------|
| 1    | 2.0  | 200   | 1.5 | 53.2 | +18.6% |
| 2    | 2.0  | 100   | 1.0 | 68.4 | +21.7% |
| 3    | 1.0  | 200   | 2.5 | 57.0 | +15.3% |

Predicted mean attack score **59.5** (on our proxy; the real trigger geometry is
unconfirmed, so this predicts rather than guarantees the portal score). All
malicious copies within a case are identical (within-malicious L2 = 0), the
Krum-flip precondition, free under every synthesis aggregator.

Regenerate: `research/build_attack_submission.py` → `attack/create_attack_submission.py`
→ `attack/validate_attack_submission.py` (prints `valid`).

---

## 2. Defense track — summary

We act as the server-side defender: implement `robust_aggregation(num_models,
models) -> state_dict`. Only Case 1 is visible; the function must generalize.
Malicious clients are at most ⌊N/3⌋ of any case (exact count unknown).

**Method: coordinate-wise trimmed mean, trim = ⌊N/3⌋.** Trimming the worst-case
malicious count from both ends of every coordinate bounds any Byzantine
minority's influence within the guaranteed bound, without knowing which or how
many clients are malicious. It is *synthesis-based*, so — unlike Krum/Multi-Krum
— it can never be captured into returning a single attacker model verbatim
(Krum was catastrophic as a defense, −3.8 score on the visible case). Benign
`SmallCNN` clients are nearly collinear (pairwise cosine ~0.9999), so trimming
benign coordinates barely moves the mean: the clean-accuracy premium is inside
the measurement noise floor.

**Validation.** `research/defense_benchmark.py` scores defenses against the
attacker they must actually stop — the FedAvg-optimal one (`λ_reg=0`, `γ=n/m`),
which drives FedAvg ASR to 98–99%. Our defense pulls ASR back to the benign-only
free floor in all three cases:

| case | trigger | free-floor ASR | FedAvg ASR | our defense ASR (residual lift) |
|------|---------|----------------|------------|----------------------------------|
| 1 | sunglasses | 46.3% | 98.1% | 47.1% (+0.8%) |
| 2 | mask       | 76.3% | 99.1% | 78.1% (+1.8%) |
| 3 | sunglasses | 52.0% | 99.4% | 53.7% (+1.6%) |

Every sufficiently-strong robust aggregator converges to the same free floor; the
top-8 candidates span only 0.5 in mean score (inside the ~1pp noise floor), so no
alternative beats trimmed-mean-at-⌊N/3⌋ significantly. We keep it for the cleanest
provable guarantee, determinism, and — unlike `largest_cluster_mean` (nominally
top) — no reliance on the malicious models forming a separable cluster, which a
stealthy attacker breaks by design.

Validate: `defense/test_defense_submission.py` (prints `valid`).

**The free-ASR floor is an unbeatable floor.** No legal defense can push ASR
below the benign-only aggregate: that residual is the trigger's *benign*
correlation (a black mask/sunglasses genuinely darkens and occludes the image,
and benign models legitimately associate dark regions with black hair), not a
backdoor, and removing it would require training data we are forbidden from
using. We are already at that optimum; there is no defense score to gain past it.

---

## 3. A scoring artifact on the surgical-mask case (Case 2)

The defense scoring rule is

    Defense Case Score = [0.6 · Clean Accuracy + 0.4 · (1 − ASR)] × 100

with clean accuracy weighted 1.5× the backdoor term. Our benchmark surfaces a
genuine tension in this rule: **on Case 2, the no-defense FedAvg baseline
out-scores the correct defense — even though FedAvg leaves the backdoor fully
active (99.1% ASR) and the defense neutralizes it (78.1% ≈ free floor).**

Measured, FedAvg-optimal attack, Case 2 (mask):

| aggregator | clean | ASR | Case Score |
|------------|-------|-----|-----------|
| FedAvg (no defense) | 69.8% | **99.1%** | **42.2** |
| our defense (trim ⌊25/3⌋=8) | 47.7% | 78.1% | 37.4 |

The defense loses by 4.9 points. Decomposing the score difference against the
rule's two terms:

- **ASR term (weight 0.4):** removing the backdoor reclaims
  `0.4 · (99.1 − 78.1) = +8.4` points.
- **Clean term (weight 0.6):** the defense *loses* `0.6 · (69.8 − 47.7) = −13.3`
  points of clean accuracy.
- **Net: −13.3 + 8.4 = −4.9.** The clean-accuracy penalty outweighs the entire
  backdoor-removal reward.

### Why this happens — two compounding causes

1. **The mask trigger's free-ASR floor is very high (76.3%).** The ASR term is
   bounded below by the free floor no aggregator can beat, so the *maximum*
   reward available for neutralizing the backdoor is only
   `0.4 · (99.1 − 76.3) = 9.1` points. Whatever we do, the ASR term cannot pay
   much. (Contrast the sunglasses cases 1 and 3, floors 46.3%/52.0%, where the
   available ASR reward is ~20 points and the defense wins by +2.8/+2.5.)

2. **The FedAvg-optimal malicious model is itself a strong clean classifier.**
   Our benchmark attacker is `θ_ref + (n/m)·(θ_trained − θ_ref)` with a
   well-trained `θ_trained` (it carries `L_clean`), amplified so it dominates the
   FedAvg average. It pulls *both* ASR and **clean accuracy** up — FedAvg's clean
   accuracy (69.8%) sits well above the benign-only floor (47.3%). Removing the
   attacker therefore drops clean accuracy back to the honest benign level, and
   under a 0.6 weight that drop is expensive.

The high floor caps the reward; the amplified-clean-classifier attacker maximizes
the penalty. Together they invert the scoring on this one case.

### Honest scope of the finding — why we still submit the defense

This is a property of **our proxy**, not a prediction that FedAvg will beat our
defense on the hidden Case 2:

- The inflated clean accuracy is specific to our *FedAvg-optimal* attack
  construction, where the malicious model is a deliberately strong clean
  classifier. A realistic attacker's malicious models generally *degrade* clean
  accuracy, in which case the defense strictly wins (it removes both the backdoor
  and a clean-accuracy drag).
- The real mask trigger's free floor is unknown; 76.3% is our synthetic mask on
  a CelebA subset.
- The defense can never do *worse* than return the honest benign aggregate. Its
  downside is bounded; the artifact only means the *upside* is small when the
  attacker happens to be a strong clean classifier and the trigger floor is high.

So the artifact is a real limitation of the challenge's scoring rule — it can, in
a specific corner, reward leaving a backdoor in place — but it is not a reason to
withhold a principled robust aggregator. On the majority of plausible attack
shapes the defense wins, and it is never harmful. We flag it here for
transparency and as a caveat on interpreting the Case 2 defense score.

---

## 4. Key limitations

- **No provided data.** All attack/defense scores are measured on a reconstructed
  CelebA proxy with a *synthetic* trigger whose exact geometry the guide does not
  specify. Numbers predict, they do not guarantee, the portal result.
- **Class-index order assumption** (`0=black … 3=gray`) is unconfirmed by
  organizers; if attack results look scrambled across classes this is the first
  thing to check (`research/data.py`).
- **Measurement resolution.** The defense eval set (~6500 images) gives a score
  std of ~1pp; defense-candidate differences below that are not distinguishable,
  which is why the choice among strong robust aggregators is made on principle
  (provable guarantee, determinism, capture-resistance), not on their nearly-tied
  measured scores.
