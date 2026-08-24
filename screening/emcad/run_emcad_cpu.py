#!/usr/bin/env python3
"""CPU-capable reproduction evaluator for the official SLDGroup/EMCAD Synapse checkpoint.

This script intentionally performs inference only. It uses the authors' released trained
weights, the released Synapse test split, and the same label remapping / per-class metric
aggregation used by the official test code. It writes machine-readable evidence and a
short screening report without inventing missing results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from medpy.metric import binary as mb
from scipy.ndimage import zoom

CLASS_NAMES = [
    "spleen", "right kidney", "left kidney", "gallbladder",
    "pancreas", "liver", "stomach", "aorta",
]
PAPER_DICE = 0.8363
PAPER_HD95 = 15.68
PAPER_MIOU = 0.7465


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_per_case(pred: np.ndarray, gt: np.ndarray):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        return (
            float(mb.dc(pred, gt)),
            float(mb.hd95(pred, gt)),
            float(mb.jc(pred, gt)),
            float(mb.assd(pred, gt)),
        )
    if pred.sum() > 0 and gt.sum() == 0:
        # Matches the released EMCAD utility.
        return 1.0, 0.0, 1.0, 0.0
    return 0.0, 0.0, 0.0, 0.0


def remap_label(label: np.ndarray) -> np.ndarray:
    label = label.copy()
    label[label == 5] = 0
    label[label == 9] = 0
    label[label == 10] = 0
    label[label == 12] = 0
    label[label == 13] = 0
    label[label == 11] = 5
    return label


def find_data_root(search_root: Path, cases: list[str]) -> Path:
    first = f"{cases[0]}.npy.h5"
    candidates = list(search_root.rglob(first))
    for p in candidates:
        parent = p.parent
        if all((parent / f"{c}.npy.h5").exists() for c in cases):
            return parent
    raise FileNotFoundError(
        f"Could not locate a directory containing all {len(cases)} Synapse test volumes under {search_root}"
    )


def checkpoint_score(p: Path) -> tuple[int, int]:
    s = str(p).lower()
    score = 0
    score += 12 if p.name.lower() == "best.pth" else 0
    score += 8 if "pvt_v2_b2" in s or "pvtv2_b2" in s else 0
    score += 6 if "emcad" in s else 0
    score += 5 if "synapse" in s else 0
    score += 2 if "run1" in s or "run_1" in s else 0
    return score, -len(s)


def find_checkpoint(root: Path) -> Path:
    files = list(root.rglob("*.pth")) + list(root.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pth/.pt checkpoint found under {root}")
    ranked = sorted(files, key=checkpoint_score, reverse=True)
    print("Checkpoint candidates (highest score first):")
    for p in ranked[:10]:
        print(" -", checkpoint_score(p)[0], p)
    return ranked[0]


def clean_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    cleaned = {}
    for k, v in obj.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def evaluate(args):
    emcad_root = args.emcad_root.resolve()
    sys.path.insert(0, str(emcad_root))
    from lib.networks import EMCADNet  # noqa: E402

    list_file = emcad_root / "lists" / "lists_Synapse" / "test_vol.txt"
    cases = [x.strip() for x in list_file.read_text().splitlines() if x.strip()]
    data_root = find_data_root(args.data_search.resolve(), cases)
    checkpoint = find_checkpoint(args.weights_root.resolve())

    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(os.cpu_count() or 2, 4)))
    torch.manual_seed(2222)
    np.random.seed(2222)

    model = EMCADNet(
        num_classes=9,
        kernel_sizes=[1, 3, 5],
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        lgag_ks=3,
        activation="relu6",
        encoder="pvt_v2_b2",
        pretrain=False,
        pretrained_dir="",
    ).to(device)

    raw = torch.load(checkpoint, map_location=device)
    state = clean_state_dict(raw)
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if len(missing) > 10:
        raise RuntimeError(f"Checkpoint/model mismatch: {len(missing)} missing keys; sample={missing[:10]}")
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.eval()

    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    case_summaries = []
    all_case_metrics = []
    start = time.time()

    with torch.inference_mode():
        for case_idx, case in enumerate(cases, 1):
            case_start = time.time()
            fp = data_root / f"{case}.npy.h5"
            with h5py.File(fp, "r") as h5:
                image = h5["image"][:]
                label = remap_label(h5["label"][:])

            prediction = np.zeros_like(label)
            for ind in range(image.shape[0]):
                sl = image[ind]
                x, y = sl.shape
                if (x, y) != (224, 224):
                    resized = zoom(sl, (224 / x, 224 / y), order=3)
                else:
                    resized = sl
                inp = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
                outputs = model(inp)
                logits = outputs[-1]
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
                if (x, y) != (224, 224):
                    pred = zoom(pred, (x / 224, y / 224), order=0)
                prediction[ind] = pred

            metrics = []
            for class_id, class_name in enumerate(CLASS_NAMES, 1):
                vals = metric_per_case(prediction == class_id, label == class_id)
                metrics.append(vals)
                rows.append({
                    "case": case,
                    "class_id": class_id,
                    "class_name": class_name,
                    "dice": vals[0],
                    "hd95": vals[1],
                    "jaccard": vals[2],
                    "asd": vals[3],
                })
            arr = np.asarray(metrics, dtype=float)
            all_case_metrics.append(arr)
            summary = {
                "case": case,
                "mean_dice": float(arr[:, 0].mean()),
                "mean_hd95": float(arr[:, 1].mean()),
                "mean_jaccard": float(arr[:, 2].mean()),
                "mean_asd": float(arr[:, 3].mean()),
                "slices": int(image.shape[0]),
                "seconds": round(time.time() - case_start, 2),
            }
            case_summaries.append(summary)
            print(
                f"[{case_idx:02d}/{len(cases)}] {case}: Dice={summary['mean_dice']:.4f} "
                f"HD95={summary['mean_hd95']:.4f} Jaccard={summary['mean_jaccard']:.4f} "
                f"({summary['seconds']:.1f}s)"
            )

    cube = np.asarray(all_case_metrics, dtype=float)  # cases x classes x metrics
    class_means = cube.mean(axis=0)
    overall = class_means.mean(axis=0)
    elapsed = time.time() - start

    class_results = []
    for i, name in enumerate(CLASS_NAMES):
        class_results.append({
            "class_id": i + 1,
            "class_name": name,
            "dice": float(class_means[i, 0]),
            "hd95": float(class_means[i, 1]),
            "jaccard": float(class_means[i, 2]),
            "asd": float(class_means[i, 3]),
        })

    metrics = {
        "project": "EMCAD Synapse inference reproduction",
        "paper": "EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation (CVPR 2024)",
        "official_repo": "https://github.com/SLDGroup/EMCAD",
        "official_repo_commit": git_commit(emcad_root),
        "device": str(device),
        "torch_version": torch.__version__,
        "num_test_cases": len(cases),
        "data_root": str(data_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "missing_checkpoint_keys": missing,
        "unexpected_checkpoint_keys": unexpected,
        "observed": {
            "mean_dice": float(overall[0]),
            "mean_hd95": float(overall[1]),
            "mean_jaccard": float(overall[2]),
            "mean_asd": float(overall[3]),
        },
        "paper_reference": {
            "mean_dice": PAPER_DICE,
            "mean_hd95": PAPER_HD95,
            "mean_jaccard": PAPER_MIOU,
            "note": "PVT-EMCAD-B2 with ImageNet pretraining; paper values are averaged over five runs.",
        },
        "absolute_dice_gap": float(abs(overall[0] - PAPER_DICE)),
        "elapsed_seconds": round(elapsed, 2),
        "class_results": class_results,
        "case_results": case_summaries,
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (outdir / "per_case_class_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = f"""# Step 1 — EMCAD Experimental Verification\n\n**Applicant:** Satti Praveena  \n**Paper:** EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation (CVPR 2024)  \n**Evaluation:** Official released PVT-EMCAD-B2 trained checkpoint on the official Synapse test split  \n**Execution device:** CPU (GitHub-hosted runner)\n\n## Reproducibility setup\n\n- Official repository commit: `{metrics['official_repo_commit']}`\n- Test volumes: {len(cases)}\n- Checkpoint SHA-256: `{metrics['checkpoint_sha256']}`\n- Model parameters instantiated: {metrics['model_parameters']:,}\n- Inference input size: 224×224\n- Evaluation follows the released 9-class Synapse label mapping and class/case aggregation.\n\n## Observed results\n\n| Metric | Observed | Paper reference |\n|---|---:|---:|\n| Mean Dice | {overall[0]*100:.2f}% | {PAPER_DICE*100:.2f}% |\n| Mean HD95 | {overall[1]:.2f} | {PAPER_HD95:.2f} |\n| Mean Jaccard / mIoU | {overall[2]*100:.2f}% | {PAPER_MIOU*100:.2f}% |\n| Mean ASD | {overall[3]:.2f} | — |\n\nThe paper reference corresponds to PVT-EMCAD-B2 with ImageNet pretraining and reports results averaged across five runs. This execution verifies the released trained checkpoint rather than retraining five models. The absolute Dice difference from the paper reference is {abs(overall[0]-PAPER_DICE)*100:.2f} percentage points.\n\n## Per-class results\n\n| Class | Dice | HD95 | Jaccard | ASD |\n|---|---:|---:|---:|---:|\n"""
    for r in class_results:
        report += f"| {r['class_name']} | {r['dice']*100:.2f}% | {r['hd95']:.2f} | {r['jaccard']*100:.2f}% | {r['asd']:.2f} |\n"
    report += f"""\n## Technical observations\n\n1. The current official repository provides trained Synapse weights separately from the source tree, so checkpoint provenance is recorded by SHA-256.\n2. The released inference code assumes CUDA through direct `.cuda()` calls. This reproduction preserves the network and metrics but moves tensors to CPU explicitly so it can execute on a standard hosted runner.\n3. The evaluation code remaps the original 14-label Synapse annotations to the released 9-class setting; using a different label convention would make the metrics incomparable.\n4. This is a checkpoint-verification experiment, not a claim of independent five-run retraining. Raw case/class metrics are included for auditability.\n\n## Runtime\n\nTotal measured inference/evaluation time: **{elapsed/60:.1f} minutes**.\n\n## Evidence files\n\n- `metrics.json` — machine-readable run metadata and aggregate metrics\n- `per_case_class_metrics.csv` — raw per-case, per-class metrics\n- `execution.log` — console log from environment setup and inference\n\n"""
    (outdir / "STEP1_EMCAD_REPORT.md").write_text(report)
    print("FINAL", json.dumps(metrics["observed"], indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emcad-root", type=Path, required=True)
    p.add_argument("--data-search", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
