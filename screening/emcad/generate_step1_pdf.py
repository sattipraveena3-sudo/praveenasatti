#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def f2(x: float) -> str:
    return f"{float(x):.2f}"


def build(metrics_path: Path, output_path: Path) -> None:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Measured metrics file not found: {metrics_path}")

    m = json.loads(metrics_path.read_text())
    required = ["observed", "paper_reference", "class_results", "checkpoint_sha256"]
    missing = [k for k in required if k not in m]
    if missing:
        raise ValueError(f"metrics.json is incomplete; missing keys: {missing}")

    obs = m["observed"]
    paper = m["paper_reference"]
    classes = m["class_results"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCompact",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9.4,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
        spaceAfter=10,
    )
    h1 = ParagraphStyle(
        "H1Compact",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=5,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "BodyCompact",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.2,
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=8.4,
        leading=10.5,
    )
    note = ParagraphStyle(
        "Note",
        parent=body,
        fontSize=8.8,
        leading=11.2,
        backColor=colors.HexColor("#F5F7FA"),
        borderColor=colors.HexColor("#D9DEE7"),
        borderWidth=0.6,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=7,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Step 1 - EMCAD Experimental Verification",
        author="Satti Praveena",
    )

    story = []
    story.append(Paragraph("Step 1 - EMCAD Experimental Verification", title))
    story.append(
        Paragraph(
            "Applicant: Satti Praveena | Paper: EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation (CVPR 2024)",
            subtitle,
        )
    )
    story.append(Paragraph("1. Experimental objective", h1))
    story.append(
        Paragraph(
            "This experiment verifies the authors' publicly released PVT-EMCAD-B2 checkpoint on the released Synapse test split. It is an inference-level reproducibility check, not a claim of independent five-run retraining. The goal is to confirm that the released model, label mapping, and evaluation pipeline can be executed end to end while retaining auditable evidence.",
            body,
        )
    )
    story.append(Paragraph("2. Reproducibility setup", h1))
    setup = [
        ["Item", "Recorded value"],
        ["Official repository commit", str(m.get("official_repo_commit", "unknown"))],
        ["Execution device", str(m.get("device", "cpu"))],
        ["PyTorch", str(m.get("torch_version", "unknown"))],
        ["Test volumes", str(m.get("num_test_cases", "unknown"))],
        ["Batch size", str(m.get("batch_size", "n/a"))],
        ["Model parameters", f"{int(m.get('model_parameters', 0)):,}"],
        ["Checkpoint SHA-256", str(m["checkpoint_sha256"])],
    ]
    t = Table(setup, colWidths=[45 * mm, 125 * mm], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.7),
                ("LEADING", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BFC7D2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 5))
    story.append(Paragraph("3. Aggregate observed results", h1))
    summary = [
        ["Metric", "Observed", "Paper reference"],
        ["Mean Dice", pct(obs["mean_dice"]), pct(paper["mean_dice"])],
        ["Mean HD95", f2(obs["mean_hd95"]), f2(paper["mean_hd95"])],
        ["Mean Jaccard / mIoU", pct(obs["mean_jaccard"]), pct(paper["mean_jaccard"])],
        ["Mean ASD", f2(obs["mean_asd"]), "-"],
    ]
    t2 = Table(summary, colWidths=[70 * mm, 45 * mm, 55 * mm], repeatRows=1)
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFC7D2")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t2)
    gap = 100.0 * abs(float(obs["mean_dice"]) - float(paper["mean_dice"]))
    story.append(
        Paragraph(
            f"The absolute Dice difference from the paper reference is {gap:.2f} percentage points. The paper value is an averaged reported result, whereas this execution evaluates one released checkpoint. Therefore, a non-zero gap is reported as an observed reproducibility difference rather than treated as evidence of model failure.",
            note,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("4. Per-class results", h1))
    rows = [["Class", "Dice", "HD95", "Jaccard", "ASD"]]
    for r in classes:
        rows.append(
            [
                str(r["class_name"]),
                pct(r["dice"]),
                f2(r["hd95"]),
                pct(r["jaccard"]),
                f2(r["asd"]),
            ]
        )
    tc = Table(rows, colWidths=[55 * mm, 30 * mm, 30 * mm, 30 * mm, 25 * mm], repeatRows=1)
    tc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFC7D2")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(tc)
    story.append(Spacer(1, 8))
    story.append(Paragraph("5. Technical observations", h1))
    observations = [
        "The released checkpoint is stored separately from the source tree; its SHA-256 is recorded to make the evaluated artifact identifiable.",
        "The released test code assumes CUDA. This reproduction changes tensor placement and batches inference on CPU, while preserving the network, checkpoint, prediction rule, class mapping, and metric definitions.",
        "Synapse annotations are remapped to the released 9-class evaluation setting before metrics are computed. Using a different label convention would make direct comparison invalid.",
        "Per-case and per-class metrics are retained alongside the aggregate values so the result can be independently audited rather than relying on a single headline number.",
    ]
    for i, text in enumerate(observations, 1):
        story.append(Paragraph(f"<b>{i}.</b> {text}", body))

    story.append(Paragraph("6. Runtime and evidence", h1))
    elapsed = float(m.get("elapsed_seconds", 0.0))
    story.append(
        Paragraph(
            f"Measured inference/evaluation runtime: <b>{elapsed/60.0:.1f} minutes</b>. The evidence bundle contains the machine-readable metrics JSON, raw case/class CSV, console execution log, environment record, official repository commit, and this report.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("7. Interpretation and reproducibility limits", h1))
    story.append(
        Paragraph(
            "This execution provides evidence that the released EMCAD checkpoint can be run on the stated Synapse split with the released label convention and produces measurable segmentation performance. It does not reproduce the training process, optimizer trajectory, random-seed variation, or the paper's five-run averaging. Those claims would require independent training runs under the paper's complete training protocol.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The experiment therefore separates two questions: checkpoint verifiability and full training reproducibility. Step 1 answers the former with raw artifacts and explicitly avoids presenting the checkpoint result as a multi-run retraining reproduction.",
            body,
        )
    )
    story.append(Paragraph("8. Connection to the proposed Step 2 research", h1))
    story.append(
        Paragraph(
            "The verified EMCAD baseline motivates the proposed ReliEMCAD extension. EMCAD targets efficient segmentation, while the proposed project asks how that efficiency can be preserved under deployment-time distribution shift. The Step 2 proposal introduces a reliability gate that decides whether to freeze, selectively adapt, or reject adaptation based on shift and uncertainty evidence, rather than adapting every target input unconditionally.",
            body,
        )
    )
    story.append(Paragraph("9. Submission evidence checklist", h1))
    checklist = [
        ["Evidence", "Purpose"],
        ["metrics.json", "Aggregate metrics, run metadata, checkpoint provenance"],
        ["per_case_class_metrics.csv", "Raw per-case and per-class values"],
        ["execution.log", "Console record of checkpoint loading and inference"],
        ["environment.txt", "Runner, Python, PyTorch, CUDA availability"],
        ["official_commit.txt", "Exact official EMCAD source revision"],
        ["STEP1_EMCAD_REPORT.pdf", "2-3 page technical screening report"],
    ]
    te = Table(checklist, colWidths=[55 * mm, 115 * mm], repeatRows=1)
    te.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFC7D2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(te)
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Conclusion: the submission reports only measured values produced by the automated run and clearly distinguishes those observations from the paper's published reference values.",
            note,
        )
    )

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(15 * mm, 8 * mm, "Satti Praveena - EMCAD screening reproduction")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.metrics, args.output)
