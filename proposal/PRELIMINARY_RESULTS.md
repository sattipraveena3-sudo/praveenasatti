# ReliEMCAD Preliminary Evidence

## Controlled single-slice domain-shift pilot

This result is **preliminary evidence only**. It is not a volume-level, dataset-level, or benchmark-level claim and it does not demonstrate an adaptation gain.

- Base model: authors' released PVT-EMCAD-B2 checkpoint
- Official Synapse test volume: `case0008`
- Slice selection: slice 127, automatically selected as the slice with the largest labeled foreground area in the first official test volume; it was not selected based on model performance
- Foreground pixels: 44,652
- Present classes: spleen, liver, stomach, aorta
- Controlled shift: per-slice min-max gamma intensity transform, gamma = 1.8
- Checkpoint SHA-256: `eba3d7ab1b07db87bb9c59c16d6651df86927aaac705b0a2f97440f8d61b3bcc`

| Metric | Clean | Gamma-shifted | Change |
|---|---:|---:|---:|
| Present-class mean Dice | 91.77% | 90.17% | -1.60 pp |
| Predictive entropy | 0.0246 | 0.0328 | +0.0081 |
| Clean-vs-shift prediction stability Dice | - | 95.77% | - |

### Per-class Dice

| Class | Clean | Gamma-shifted | Change |
|---|---:|---:|---:|
| Spleen | 93.42% | 93.22% | -0.20 pp |
| Liver | 96.15% | 93.25% | -2.90 pp |
| Stomach | 87.20% | 81.60% | -5.59 pp |
| Aorta | 90.32% | 92.62% | +2.29 pp |

## Interpretation

The controlled shift reduced present-class mean Dice while increasing predictive entropy, but the effect was heterogeneous across anatomy. Stomach and liver degraded, spleen was nearly stable, and aorta improved. This is consistent with the motivation for ReliEMCAD's reliability gate: detecting distribution shift should not automatically trigger adaptation. The decision should also use uncertainty and prediction stability, and it should permit freezing or rejecting adaptation when an update may be unnecessary or harmful.

## Guardrail

This pilot uses one automatically selected 2D slice. The full-volume EMCAD checkpoint verification is a separate Step 1 experiment. Larger controlled-corruption and cross-domain experiments are required before making any robustness or adaptation-performance claim.
