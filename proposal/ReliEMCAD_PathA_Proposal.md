# ReliEMCAD: Reliability-Gated Lightweight Test-Time Adaptation for Robust Medical Image Segmentation Under Domain Shift

**Path A — Lab Paper Extension / Methodological Enhancement**  
**Applicant:** Satti Praveena  
**Base paper:** EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation (CVPR 2024)

## Abstract

Efficient medical image segmentation is attractive for deployment, but computational efficiency alone does not guarantee reliability when image statistics change across hospitals, scanners, acquisition protocols, or patient populations. EMCAD offers an efficient multi-scale attention decoder and is therefore a strong backbone for studying this deployment gap. I propose **ReliEMCAD**, a reliability-gated test-time adaptation framework that preserves EMCAD's efficiency while selectively adapting to previously unseen target data. Instead of adapting on every incoming image, ReliEMCAD estimates feature shift and predictive uncertainty, then chooses among three actions: freeze the model, perform low-cost adaptation on a small parameter subset, or flag the case as unreliable and avoid self-training. The central hypothesis is that conditional adaptation is safer and more efficient than unconditional adaptation for lightweight medical segmentation. The project will evaluate natural and synthetic domain shifts using Dice, HD95, Jaccard, ASD, calibration, harmful-adaptation rate, memory use, and adaptation latency.

## 1. Motivation and Significance

Medical segmentation systems can experience distribution shift after deployment because of scanner vendors, reconstruction pipelines, image contrast, acquisition settings, disease prevalence, and local clinical practice. A model that performs well on a source benchmark can therefore become overconfident or unstable at a new site. This problem is important for efficient models because lightweight systems are often intended for resource-constrained clinical settings where frequent centralized retraining is difficult.

EMCAD directly addresses computational efficiency through multi-scale depth-wise convolution and efficient channel, spatial, and grouped gated attention. The proposed project extends this line by asking whether the same efficiency philosophy can be preserved during target-domain adaptation and reliability estimation.

### Research question

**Can reliability-gated, parameter-efficient test-time adaptation improve EMCAD under unseen domain shifts while reducing harmful updates, latency, and memory cost compared with unconditional test-time adaptation?**

### Hypotheses

- **H1:** Reliability gating will improve average target Dice relative to source-only inference and reduce worst-case degradation relative to unconditional TTA.
- **H2:** Updating only normalization/affine terms or compact low-rank adapters will recover most of the attainable TTA gain with substantially lower latency and memory.
- **H3:** Combining feature-shift magnitude with predictive uncertainty will better predict harmful updates than either signal alone.

## 2. Problem Definition and Related-Work Gap

Let a source model f(theta) be trained on labeled source data and deployed on unlabeled target images from an unknown or changing distribution. Source data may be unavailable at test time. The system must produce a segmentation while deciding whether and how to adapt using target-time signals only.

The main risk is **negative adaptation**: a target-time update intended to improve robustness may reinforce an incorrect pseudo-label, collapse to a trivial prediction, or overfit to a transient shift. General test-time adaptation methods such as entropy minimization can work well, but adapting every batch is not automatically safe in medical imaging, where batches may be tiny and class prevalence can vary sharply.

Recent medical TTA methods have introduced uncertainty guidance, source-friendly sample selection, shape priors, teacher-student consistency, and single-image continual adaptation. SicTTA, for example, selects source-friendly target samples to support adaptation. Uncertainty- and shape-aware CTTA uses uncertainty-weighted pseudo-labels and shape information. Recent large-scale benchmarking also indicates that no single TTA family dominates across modalities and that some methods degrade under severe inter-center or inter-device shift.

### Gap 1 — Adaptation is usually treated as a method, not a decision

Most TTA pipelines focus on how to update a model after adaptation has already been chosen. ReliEMCAD places a small reliability controller before the update and treats **whether to adapt** as an explicit decision.

### Gap 2 — Efficiency is rarely evaluated end to end

A model can use an efficient backbone but become expensive at test time through full-network gradients, multiple augmentations, ensembles, or teacher models. ReliEMCAD will therefore report adaptation latency, memory, and number of trainable test-time parameters together with segmentation quality.

### Gap 3 — Reliability and referral are usually secondary outputs

ReliEMCAD converts uncertainty into behavior. The system will choose among **freeze**, **adapt**, and **refer/flag**, allowing direct evaluation of reliability rather than using confidence only as an internal optimization weight.

## 3. Proposed Method

### 3.1 EMCAD base model

The released EMCAD PVTv2-B2 configuration is used as the source segmentation model. Its source checkpoint remains immutable as theta_0. Test-time changes are applied only to a temporary theta_t so the model can roll back or reset when reliability deteriorates.

### 3.2 Shift and uncertainty estimator

For each target image, ReliEMCAD computes two signals:

1. **Feature shift:** normalized distance between target intermediate feature statistics and frozen source reference statistics.
2. **Predictive uncertainty:** output entropy plus disagreement under two inexpensive weak perturbations such as horizontal flip and intensity jitter.

The estimator is intentionally lightweight so it does not eliminate EMCAD's efficiency advantage.

### 3.3 Three-way reliability gate

The gate maps feature-shift magnitude and uncertainty to three states:

- **Low shift / low uncertainty:** freeze model and infer normally.
- **Moderate shift / stable prediction:** allow selective adaptation.
- **High uncertainty / unstable prediction:** do not self-adapt; return the segmentation with a reliability flag for review.

Thresholds will first be calibrated on source-held-out corruptions and then evaluated without target labels.

### 3.4 Parameter-efficient adaptation

Two variants will be compared:

- **ReliEMCAD-Norm:** update normalization affine parameters and/or running statistics only.
- **ReliEMCAD-LoRA:** insert compact low-rank adapters into selected pointwise projections in late encoder/decoder stages.

The adaptation objective is:

**L = L_entropy + lambda_c L_consistency + lambda_a L_anchor**

where entropy minimization is restricted to confident regions, consistency enforces agreement across weak perturbations, and the anchor term limits parameter drift away from theta_0.

### 3.5 Update acceptance and rollback

After a short gradient step, the system re-evaluates the target image. The update is accepted only if uncertainty decreases without a substantial increase in prediction disagreement or implausible mask statistics. Otherwise the temporary parameters are rolled back.

### 3.6 Memory and reset policy

A small FIFO memory stores compact feature summaries from previously accepted target cases rather than raw medical images. A cumulative drift score monitors parameter movement. If drift exceeds a threshold or uncertainty rises for several consecutive cases, the adapted parameter subset resets to theta_0.

## 4. Experimental Plan

### 4.1 Step-1 verification baseline

The screening workflow uses the authors' official EMCAD repository, released Synapse test split, and released trained Synapse checkpoint. The evaluator reproduces the released 9-class label mapping and computes Dice, HD95, Jaccard, and ASD for every class and case. It records the upstream repository commit, checkpoint SHA-256, environment, raw metrics, and execution log. This is the direct execution baseline that ReliEMCAD extends. The final measured numbers will come from the automated experiment evidence bundle; no result is invented in this proposal.

### 4.2 Domain-shift protocol

- **Natural shift:** source checkpoint from one dataset/site evaluated on a related external dataset without using target labels for adaptation.
- **Controlled scanner/intensity shifts:** gamma, contrast, Gaussian/Rician noise, blur, resolution loss, and bias-field changes at test time.
- **Continual shift:** sequential target domains with alternating or increasing severity to test drift, reset, and recovery.

### 4.3 Baselines

- Source-only EMCAD.
- Entropy-minimization TTA on normalization parameters (TENT-style).
- A continual TTA baseline using teacher consistency or stochastic restoration where compatible.
- A reproducible medical TTA reference such as SicTTA on the selected 2D benchmark.
- ReliEMCAD without gating to isolate the value of the reliability decision.

### 4.4 Metrics

**Segmentation:** Dice, Jaccard/mIoU, HD95, ASD.  
**Reliability:** pixel ECE, uncertainty-error AUROC, selective Dice vs. coverage.  
**Safety:** harmful-adaptation rate = fraction of adapted cases where Dice decreases relative to source-only inference.  
**Efficiency:** trainable test-time parameters, peak memory, latency, throughput.

### 4.5 Ablations

- Feature shift only vs. uncertainty only vs. combined gating.
- Norm-only vs. decoder adapters vs. late encoder + decoder adapters.
- No rollback vs. uncertainty rollback vs. uncertainty + disagreement rollback.
- No feature memory vs. FIFO memory with different lengths.
- No reset vs. cumulative-drift reset.

## 5. Expected Outcomes

The intended outcome is a reproducible EMCAD extension that improves robustness without sacrificing the lightweight design philosophy. The most important result is not a small average Dice increase by itself. A successful method should also reduce the frequency of harmful updates, improve calibration/referral behavior, and keep test-time compute small.

Expected outputs include:

- Reliability-gated parameter-efficient EMCAD extension.
- Explicit adapt/freeze/refer decision logs.
- Cross-domain robustness and efficiency benchmark tables.
- Controlled shift generation and continual adaptation scripts.
- Reproducibility package containing source commit, checkpoints, commands, raw metrics, and analysis.

## 6. Risks and Mitigation

**Poor uncertainty calibration under severe shift:** combine uncertainty with feature-shift signals and evaluate selective-risk curves instead of trusting one score.  
**Dataset-specific gains:** evaluate multiple shifts and report negative results.  
**Adaptation too slow:** restrict final method to normalization or compact decoder adapters and minimize test-time perturbations.  
**Pseudo-label collapse:** use source anchoring, rollback, and reset-to-source logic.

## 7. 12-Week Execution Plan

| Weeks | Milestone |
|---|---|
| 1–2 | Lock EMCAD verification baseline; implement shift generator and source-only/TTA baselines. |
| 3–4 | Implement reliability features and gate; calibrate on held-out source corruptions. |
| 5–6 | Implement Norm and low-rank adapters; add rollback and source anchoring. |
| 7–8 | Run natural and controlled cross-domain experiments; collect reliability and efficiency metrics. |
| 9–10 | Run ablations, continual-shift experiments, failure analysis, and referral evaluation. |
| 11–12 | Finalize figures, reproducibility package, manuscript, and follow-up experiments. |

## References

1. M. M. Rahman, M. Munir, and R. Marculescu, “EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation,” CVPR, 2024.
2. D. Wang et al., “Tent: Fully Test-Time Adaptation by Entropy Minimization,” ICLR, 2021.
3. Q. Wang et al., “Continual Test-Time Domain Adaptation,” CVPR, 2022.
4. S. Niu et al., “Towards Stable Test-Time Adaptation in Dynamic Wild World,” ICLR, 2023.
5. L. Yuan et al., “Robust Test-Time Adaptation in Dynamic Scenarios,” CVPR, 2023.
6. “SicTTA: Single Image Continual Test Time Adaptation for Medical Image Segmentation,” Medical Image Analysis, 2025.
7. “Improving Cross-Domain Generalizability of Medical Image Segmentation Using Uncertainty and Shape-Aware Continual Test-Time Domain Adaptation,” Medical Image Analysis, 2024/2025.
8. J. Wu et al., “SAM-aware Test-time Adaptation for Universal Medical Image Segmentation,” arXiv:2506.05221, 2025.
9. W. Yu et al., “A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation (MedSeg-TTA),” arXiv:2512.02497, 2025.
10. X. Ma et al., “Test-time Generative Augmentation for Medical Image Segmentation,” Medical Image Analysis, 2026.

> The final submission distinguishes prior reported results from applicant-measured screening results. Step-1 measured values are inserted only after the automated evaluator finishes.