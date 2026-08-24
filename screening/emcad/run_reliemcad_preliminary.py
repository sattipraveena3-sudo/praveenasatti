#!/usr/bin/env python3
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

CLASSES = ["spleen", "right kidney", "left kidney", "gallbladder", "pancreas", "liver", "stomach", "aorta"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap(label: np.ndarray) -> np.ndarray:
    label = label.copy()
    for k in [5, 9, 10, 12, 13]:
        label[label == k] = 0
    label[label == 11] = 5
    return label


def metric(pred: np.ndarray, gt: np.ndarray):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        return float(mb.dc(pred, gt)), float(mb.hd95(pred, gt)), float(mb.jc(pred, gt)), float(mb.assd(pred, gt))
    if pred.sum() > 0 and gt.sum() == 0:
        return 1.0, 0.0, 1.0, 0.0
    return 0.0, 0.0, 0.0, 0.0


def dice_between(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.astype(bool)
    b = b.astype(bool)
    if not a.any() and not b.any():
        return None
    denom = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / max(1, denom))


def gamma_shift(sl: np.ndarray, gamma: float) -> np.ndarray:
    sl = sl.astype(np.float32, copy=False)
    lo = float(np.min(sl))
    hi = float(np.max(sl))
    if hi - lo < 1e-8:
        return sl.copy()
    x = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
    y = np.power(x, gamma)
    return (lo + y * (hi - lo)).astype(np.float32)


def find_data(root: Path, case: str) -> Path:
    candidates = list(root.rglob(case + ".npy.h5"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {case}.npy.h5 under {root}")
    return candidates[0]


def ck_score(p: Path):
    s = str(p).lower()
    score = 0
    score += 12 if p.name.lower() == "best.pth" else 0
    score += 8 if "pvt_v2_b2" in s or "pvtv2_b2" in s else 0
    score += 6 if "emcad" in s else 0
    score += 5 if "synapse" in s else 0
    score += 2 if "run1" in s or "run_1" in s else 0
    return score, -len(s)


def find_ck(root: Path) -> Path:
    files = list(root.rglob("*.pth")) + list(root.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    return sorted(files, key=ck_score, reverse=True)[0]


def clean_state(obj):
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def mean_metrics(pred: np.ndarray, label: np.ndarray):
    rows = []
    vals = []
    for class_id, name in enumerate(CLASSES, 1):
        m = metric(pred == class_id, label == class_id)
        vals.append(m)
        rows.append({
            "class_id": class_id,
            "class_name": name,
            "dice": m[0],
            "hd95": m[1],
            "jaccard": m[2],
            "asd": m[3],
        })
    arr = np.asarray(vals, dtype=float)
    return {
        "mean_dice": float(arr[:, 0].mean()),
        "mean_hd95": float(arr[:, 1].mean()),
        "mean_jaccard": float(arr[:, 2].mean()),
        "mean_asd": float(arr[:, 3].mean()),
    }, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emcad-root", type=Path, required=True)
    p.add_argument("--data-search", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gamma", type=float, default=1.8)
    p.add_argument("--case-index", type=int, default=0)
    args = p.parse_args()

    root = args.emcad_root.resolve()
    sys.path.insert(0, str(root))
    from lib.networks import EMCADNet

    cases = [x.strip() for x in (root / "lists/lists_Synapse/test_vol.txt").read_text().splitlines() if x.strip()]
    if not cases:
        raise RuntimeError("Official Synapse test list is empty")
    case = cases[args.case_index % len(cases)]
    data_file = find_data(args.data_search.resolve(), case)
    checkpoint = find_ck(args.weights_root.resolve())

    torch.set_num_threads(max(1, min(os.cpu_count() or 2, 8)))
    torch.manual_seed(2222)
    np.random.seed(2222)
    device = torch.device("cpu")

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
    incompatible = model.load_state_dict(clean_state(torch.load(checkpoint, map_location=device)), strict=False)
    if len(incompatible.missing_keys) > 10:
        raise RuntimeError(f"Checkpoint mismatch: {len(incompatible.missing_keys)} missing keys")
    model.eval()

    with h5py.File(data_file, "r") as h5:
        image = h5["image"][:]
        label = remap(h5["label"][:])

    pred_clean = np.zeros_like(label)
    pred_shift = np.zeros_like(label)
    entropy_clean_sum = 0.0
    entropy_shift_sum = 0.0
    entropy_pixels = 0
    start = time.time()

    with torch.inference_mode():
        for b0 in range(0, image.shape[0], args.batch_size):
            b1 = min(image.shape[0], b0 + args.batch_size)
            clean_batch = []
            shift_batch = []
            shapes = []
            for i in range(b0, b1):
                sl = image[i].astype(np.float32)
                shifted = gamma_shift(sl, args.gamma)
                x, y = sl.shape
                shapes.append((x, y))
                if (x, y) != (224, 224):
                    clean_batch.append(zoom(sl, (224 / x, 224 / y), order=3))
                    shift_batch.append(zoom(shifted, (224 / x, 224 / y), order=3))
                else:
                    clean_batch.append(sl)
                    shift_batch.append(shifted)

            clean_np = np.stack(clean_batch)
            shift_np = np.stack(shift_batch)
            combo = np.concatenate([clean_np, shift_np], axis=0)
            inp = torch.from_numpy(combo).unsqueeze(1).float().to(device)
            logits = model(inp)[-1]
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
            n = b1 - b0
            clean_ent = ent[:n]
            shift_ent = ent[n:]
            entropy_clean_sum += float(clean_ent.sum().item())
            entropy_shift_sum += float(shift_ent.sum().item())
            entropy_pixels += int(clean_ent.numel())

            pc = predictions[:n]
            ps = predictions[n:]
            for j, i in enumerate(range(b0, b1)):
                x, y = shapes[j]
                clean_pred = pc[j]
                shift_pred = ps[j]
                if (x, y) != (224, 224):
                    clean_pred = zoom(clean_pred, (x / 224, y / 224), order=0)
                    shift_pred = zoom(shift_pred, (x / 224, y / 224), order=0)
                pred_clean[i] = clean_pred
                pred_shift[i] = shift_pred
            print(f"processed slices {b0}:{b1} / {image.shape[0]}", flush=True)

    clean_metrics, clean_rows = mean_metrics(pred_clean, label)
    shift_metrics, shift_rows = mean_metrics(pred_shift, label)
    stability_vals = []
    for class_id in range(1, 9):
        d = dice_between(pred_clean == class_id, pred_shift == class_id)
        if d is not None:
            stability_vals.append(d)
    stability = float(np.mean(stability_vals)) if stability_vals else 1.0

    ent_clean = entropy_clean_sum / max(1, entropy_pixels)
    ent_shift = entropy_shift_sum / max(1, entropy_pixels)
    elapsed = time.time() - start

    result = {
        "experiment": "ReliEMCAD preliminary domain-shift stress test",
        "scope": "single official Synapse test volume; preliminary evidence only",
        "case": case,
        "slices": int(image.shape[0]),
        "synthetic_shift": {"type": "per-slice min-max gamma intensity transform", "gamma": args.gamma},
        "official_repo_commit": git_commit(root),
        "checkpoint_sha256": sha256(checkpoint),
        "batch_size": args.batch_size,
        "device": "cpu",
        "clean": {**clean_metrics, "predictive_entropy": float(ent_clean)},
        "shifted": {**shift_metrics, "predictive_entropy": float(ent_shift)},
        "delta_shift_minus_clean": {
            "mean_dice": float(shift_metrics["mean_dice"] - clean_metrics["mean_dice"]),
            "mean_hd95": float(shift_metrics["mean_hd95"] - clean_metrics["mean_hd95"]),
            "mean_jaccard": float(shift_metrics["mean_jaccard"] - clean_metrics["mean_jaccard"]),
            "mean_asd": float(shift_metrics["mean_asd"] - clean_metrics["mean_asd"]),
            "predictive_entropy": float(ent_shift - ent_clean),
        },
        "clean_vs_shift_prediction_stability_dice": stability,
        "elapsed_seconds": round(elapsed, 2),
        "interpretation_guardrail": "This is a single-volume controlled stress test. It motivates the reliability-gating hypothesis but is not a benchmark-level claim and does not evaluate the proposed adaptation method yet.",
    }

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "preliminary_reliemcad.json").write_text(json.dumps(result, indent=2))
    with (out / "preliminary_per_class.csv").open("w", newline="") as f:
        fields = ["condition", "class_id", "class_name", "dice", "hd95", "jaccard", "asd"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for condition, rows in [("clean", clean_rows), ("gamma_shift", shift_rows)]:
            for r in rows:
                w.writerow({"condition": condition, **r})

    md = f"""# ReliEMCAD Preliminary Result - Controlled Domain-Shift Stress Test

This is a **single-volume preliminary experiment**, not a benchmark-level result.

- Synapse case: `{case}` ({image.shape[0]} slices)
- Base model: authors' released PVT-EMCAD-B2 checkpoint
- Controlled shift: per-slice gamma intensity transform, gamma={args.gamma}
- Device: CPU, batched inference

| Metric | Clean | Shifted | Shift - Clean |
|---|---:|---:|---:|
| Mean Dice | {100*clean_metrics['mean_dice']:.2f}% | {100*shift_metrics['mean_dice']:.2f}% | {100*(shift_metrics['mean_dice']-clean_metrics['mean_dice']):+.2f} pp |
| Mean HD95 | {clean_metrics['mean_hd95']:.2f} | {shift_metrics['mean_hd95']:.2f} | {shift_metrics['mean_hd95']-clean_metrics['mean_hd95']:+.2f} |
| Mean Jaccard | {100*clean_metrics['mean_jaccard']:.2f}% | {100*shift_metrics['mean_jaccard']:.2f}% | {100*(shift_metrics['mean_jaccard']-clean_metrics['mean_jaccard']):+.2f} pp |
| Predictive entropy | {ent_clean:.4f} | {ent_shift:.4f} | {ent_shift-ent_clean:+.4f} |

Clean-vs-shift prediction stability Dice: **{stability:.4f}**.

Interpretation: this controlled stress test measures whether an intensity-domain shift changes segmentation quality and prediction confidence/consistency. It provides preliminary motivation for a reliability gate, but it does not yet demonstrate an adaptation gain.
"""
    (out / "PRELIMINARY_RESULTS.md").write_text(md)
    print("PRELIMINARY_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
