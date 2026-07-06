#!/usr/bin/env python
"""Generate the thesis result figures (vector PDFs) from the experiment JSONs.

Each figure function is independent and skipped (with a warning) if its input JSON does
not exist yet, so the script can be re-run as results come in.

Run (CPU)::

    .venv/bin/python experiments/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("../artefacts/thesis/res/03_research/results")
EVAL = Path("experiments/outputs/eval")

# Fixed categorical colors (validated palette; identity is constant across figures).
BLUE = "#2a78d6"     # the full / default method
YELLOW = "#eda100"   # the milder alternative (gating)
RED = "#e34948"      # the naive / disabled variant
VIOLET = "#4a3aa7"   # secondary ablation arm
AQUA = "#1baf7a"
GRAY = "#52514e"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
})


def _load(path: Path):
    if not path.exists():
        print(f"[skip] missing {path}")
        return None
    return json.loads(path.read_text())


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    p = (OUT / name).with_suffix(".png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved] {p}")


def _series(d, key):
    return np.array([r.get(key, np.nan) for r in d["frames"]], dtype=float)


# ------------------------------------------------------------------ bg subtraction


def fig_bgsub_sweep():
    d = _load(Path("experiments/outputs/bgsub/bgsub_benchmark.json"))
    if d is None:
        return
    sw = d["sweep"]
    thr = np.array(sw["thresholds"])
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.55), constrained_layout=True)

    ax = axes[0]
    for key, color, label in (("iou", BLUE, "IoU"), ("precision", AQUA, "precision"),
                              ("recall", VIOLET, "recall")):
        ax.plot(thr, [m[key] for m in sw["with_morph"]], color=color, marker="o", ms=3, label=label)
        ax.plot(thr, [m[key] for m in sw["no_morph"]], color=color, marker="o", ms=3,
                linestyle="--", alpha=0.45)
    ax.axvline(0.06, color=GRAY, linestyle=":", linewidth=1.1)
    ax.text(0.066, 0.32, "chosen $\\tau_s$", color=GRAY, fontsize=7, rotation=90, va="bottom", ha="left")
    ax.set_xlabel("change threshold $\\tau_s$")
    ax.set_ylabel("mask quality vs. ground truth")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    ax.set_title("(a) threshold sweep", fontsize=9.5)

    es = d.get("erosion_sweep")
    ax = axes[1]
    if es is not None:
        ee = np.array(es["erosions"])
        for key, color, label in (("iou", BLUE, "IoU"), ("precision", AQUA, "precision"),
                                  ("recall", VIOLET, "recall")):
            ax.plot(ee, [m[key] for m in es["metrics"]], color=color, marker="o", ms=3, label=label)
        ax.axvline(2, color=GRAY, linestyle=":", linewidth=1.1)
        ax.text(2.12, 1.005, "chosen", color=GRAY, fontsize=7.5, va="top")
        ax.set_xlabel("erosion steps ($3{\\times}3$)")
        ax.set_ylim(0.55, 1.02)
        ax.legend(frameon=False, loc="lower left", fontsize=7)
        ax.set_title("(b) erosion sweep at $\\tau_s = 0.06$", fontsize=9.5)

    ax = axes[2]
    cams = sorted(d["per_frame"].keys())
    colors = [BLUE, AQUA, YELLOW, VIOLET]
    for c, col in zip(cams, colors):
        iou = [m["iou"] for m in d["per_frame"][c]]
        ax.plot(np.arange(1, len(iou) + 1), iou, color=col, label=c, linewidth=1.1)
    ax.set_xlabel("frame")
    ax.set_ylabel("mask IoU")
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=7)
    ax.set_title("(c) per-frame IoU, chosen setting", fontsize=9.5)
    _save(fig, "bgsub_quality.png")


# ------------------------------------------------------------------ hull init


def fig_hull_init():
    d = _load(Path("experiments/outputs/hull_init/hull_init_benchmark.json"))
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), constrained_layout=True)
    arms = (("hull", BLUE, "visual-hull seed"), ("random", RED, "random seed (same budget)"))
    d50 = _load(Path("experiments/outputs/hull_init_50k/hull_init_benchmark.json"))

    ax = axes[0]
    for arm, color, label in arms:
        ax.plot(d[arm]["steps"], d[arm]["object_psnr"], color=color, label=label)
    if d50 is not None:
        ax.plot(d50["random"]["steps"], d50["random"]["object_psnr"], color=RED, linestyle="--",
                alpha=0.6, label="random seed ($30\\times$ budget)")
    ax.set_xlabel("frame-0 optimization step")
    ax.set_ylabel("object PSNR [dB]")
    ax.set_title("(a) reconstruction quality")

    ax = axes[1]
    for arm, color, label in arms:
        ax.plot(d[arm]["steps"], d[arm]["num_gaussians"], color=color, label=label)
    if d50 is not None:
        ax.plot(d50["random"]["steps"], d50["random"]["num_gaussians"], color=RED, linestyle="--",
                alpha=0.6, label="random seed ($30\\times$ budget)")
        ax.set_yscale("log")
    ax.set_xlabel("frame-0 optimization step")
    ax.set_ylabel("number of Gaussians")
    ax.set_title("(b) primitive count")

    # One shared legend below both panels (the line identities are common to (a) and (b)).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=7.5, loc="outside lower center")
    _save(fig, "hull_init.pdf")


# ------------------------------------------------------------------ velocity


def fig_velocity():
    dt = _load(Path("experiments/outputs/velocity_target/velocity_convergence_summary.json"))
    dp = _load(Path("experiments/outputs/velocity_plateau/velocity_convergence_summary.json"))
    if dt is None:
        return
    ncols = 2 if dp is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.3, 2.9), constrained_layout=True)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    nov = np.array(dt["results"]["no_velocity"]["steps"])
    vel = np.array(dt["results"]["velocity"]["steps"])
    bins = np.linspace(0, max(nov.max(), vel.max()) * 1.03, 26)
    ax.hist(nov, bins=bins, color=RED, alpha=0.6, label="without velocity", edgecolor="white")
    ax.hist(vel, bins=bins, color=BLUE, alpha=0.7, label="with velocity", edgecolor="white")
    for v, c in ((nov.mean(), RED), (vel.mean(), BLUE)):
        ax.axvline(v, color=c, linestyle="--", linewidth=1.6)
    ax.text(nov.mean() + 6, ax.get_ylim()[1] * 0.93, f"mean {nov.mean():.0f}", color=RED, fontsize=8.5)
    ax.text(vel.mean() + 6, ax.get_ylim()[1] * 0.8, f"mean {vel.mean():.0f}", color=BLUE, fontsize=8.5)
    ax.set_xlabel("steps to 20 dB object PSNR")
    ax.set_ylabel("number of frames")
    ax.set_title("(a) steps to a fixed quality target")

    if dp is not None:
        ax = axes[1]
        novq = np.array(dp["results"]["no_velocity"]["psnr"])
        velq = np.array(dp["results"]["velocity"]["psnr"])
        bins = np.linspace(min(novq.min(), velq.min()) - 0.5, max(novq.max(), velq.max()) + 0.5, 22)
        ax.hist(novq, bins=bins, color=RED, alpha=0.6, label="without velocity", edgecolor="white")
        ax.hist(velq, bins=bins, color=BLUE, alpha=0.7, label="with velocity", edgecolor="white")
        for v, c in ((novq.mean(), RED), (velq.mean(), BLUE)):
            ax.axvline(v, color=c, linestyle="--", linewidth=1.6)
        ax.set_xlabel("converged object PSNR [dB]")
        ax.set_ylabel("number of frames")
        ax.set_title("(b) converged quality per frame")

    # One shared legend below the panels (colors are common to (a) and (b)).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="outside lower center")
    _save(fig, "velocity_convergence.pdf")


# ------------------------------------------------------------------ snapping


def fig_snapping():
    runs = [("abl150_default", BLUE, "hull snapping (default)"),
            ("abl150_gating", YELLOW, "velocity gating"),
            ("abl150_nosnap", RED, "no snapping, no gating")]
    ds = [( _load(EVAL / f"{r}.json"), c, l) for r, c, l in runs]
    if any(d is None for d, _, _ in ds):
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.75), constrained_layout=True)
    for key, ax, ylabel, title, logy in (
        ("object_ssim", axes[0], "object SSIM", "(a) structural quality", False),
        ("radius_max", axes[1], "max. distance from centroid", "(b) cloud extent (runaway)", False),
        ("leak", axes[2], "background leak (mean $\\hat A$)", "(c) background leak", True),
    ):
        for d, color, label in ds:
            y = _series(d, key)
            ax.plot(np.arange(len(y)), y, color=color, label=label, linewidth=1.3)
        ax.set_xlabel("frame")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.5)
        if logy:
            ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.09), fontsize=8.5)
    _save(fig, "snapping_curves.pdf")


# ------------------------------------------------------------------ masking


def fig_masking():
    runs = [("abl150_default", BLUE, "masked + silhouette (default)"),
            ("abl150_fullimage", RED, "naive full-image loss"),
            ("abl150_frzop_nosnap", VIOLET, "silhouette, frozen opacity")]
    ds = [( _load(EVAL / f"{r}.json"), c, l) for r, c, l in runs]
    if any(d is None for d, _, _ in ds):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.85), constrained_layout=True)
    ax = axes[0]
    for d, color, label in ds:
        y = _series(d, "leak")
        ax.plot(np.arange(len(y)), np.maximum(y, 1e-6), color=color, label=label, linewidth=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("frame")
    ax.set_ylabel("background leak (mean $\\hat A$)")
    ax.set_title("(a) opacity leaked outside the object", fontsize=9.5)

    ax = axes[1]
    for d, color, label in ds:
        y = _series(d, "coverage")
        ax.plot(np.arange(len(y)), y, color=color, label=label, linewidth=1.3)
    ax.set_xlabel("frame")
    ax.set_ylabel("object-core coverage")
    ax.set_title("(b) object solidity", fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.09), fontsize=8.5)
    _save(fig, "masking_curves.pdf")


# ------------------------------------------------------------------ colour freezing


def fig_freecolors():
    d_def = _load(EVAL / "abl150_default.json")
    d_free = _load(EVAL / "abl150_freecolors.json")
    if d_def is None or d_free is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    ax = axes[0]
    for d, color, label in ((d_free, RED, "colours trainable"), (d_def, BLUE, "colours frozen (default)")):
        y = _series(d, "color_drift")
        ax.plot(np.arange(len(y)), y, color=color, label=label)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean $|f_{\\mathrm{dc}} - f_{\\mathrm{dc}}(0)|$")
    ax.legend(frameon=False)
    ax.set_title("(a) appearance drift")

    ax = axes[1]
    for d, color, label in ((d_free, RED, "colours trainable"), (d_def, BLUE, "colours frozen (default)")):
        y = _series(d, "object_psnr")
        ax.plot(np.arange(len(y)), y, color=color, label=label)
    ax.set_xlabel("frame")
    ax.set_ylabel("object PSNR [dB]")
    ax.set_title("(b) reconstruction quality")
    _save(fig, "freecolors_curves.pdf")


# ------------------------------------------------------------------ full sequence


def fig_full_sequence():
    d = _load(EVAL / "full150.json")
    if d is None:
        return
    ceil = _load(EVAL / "input_ceiling.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    ax = axes[0]
    psnr = _series(d, "object_psnr")
    ax.plot(np.arange(len(psnr)), psnr, color=BLUE, label="reconstruction")
    if ceil is not None:
        cp = np.array([r["object_psnr"] for r in ceil["frames"]])
        ax.plot(np.arange(len(cp)), cp, color=GRAY, linestyle="--", linewidth=1.2,
                label="segmented input (reference)")
    ax.set_xlabel("frame")
    ax.set_ylabel("object PSNR vs. ground truth [dB]")
    ax.set_title("(a) per-frame quality (150 frames)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ssim = _series(d, "object_ssim")
    cov = _series(d, "coverage")
    ax.plot(np.arange(len(ssim)), ssim, color=AQUA, label="object SSIM")
    ax.plot(np.arange(len(cov)), cov, color=VIOLET, label="core coverage")
    ax.set_xlabel("frame")
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(b) structure and solidity")
    _save(fig, "full_sequence.pdf")


# ------------------------------------------------------------------ compression


def fig_compression():
    d = _load(Path("experiments/outputs/compression/compression_benchmark.json"))
    if d is None:
        return
    pts = d["sweep"]  # list of {quality, ratio, psnr}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    ax = axes[0]
    ratios = [p["ratio"] for p in pts]
    psnrs = [p["render_psnr"] for p in pts]
    ax.plot(ratios, psnrs, color=BLUE, marker="o", ms=4)
    for p in pts:
        if abs(p["quality"] - 0.999) < 1e-9:
            ax.scatter([p["ratio"]], [p["render_psnr"]], s=70, facecolors="none",
                       edgecolors=RED, linewidths=1.6, zorder=5)
            ax.annotate(f"default $q$={p['quality']}\n{p['ratio']:.1f}x, {p['render_psnr']:.1f} dB",
                        (p["ratio"], p["render_psnr"]), textcoords="offset points",
                        xytext=(10, 6), fontsize=8, color=RED)
    ax.set_xlabel("compression ratio ($\\times$)")
    ax.set_ylabel("render PSNR vs. original [dB]")
    ax.set_title("(a) rate--distortion")

    ax = axes[1]
    per_attr = d.get("per_attribute_K")  # {attr: K} at default quality
    if per_attr:
        names = list(per_attr.keys())
        ks = [per_attr[n] for n in names]
        ypos = np.arange(len(names))
        ax.barh(ypos, ks, color=BLUE, height=0.62)
        ax.set_yticks(ypos, names, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("retained DCT coefficients $K$ (of %d)" % d.get("num_frames", 150))
        ax.set_title("(b) coefficient budget at $q$ = 0.999")
        ax.grid(axis="x")
    _save(fig, "compression.pdf")


if __name__ == "__main__":
    fig_bgsub_sweep()
    fig_hull_init()
    fig_velocity()
    fig_snapping()
    fig_masking()
    fig_freecolors()
    fig_full_sequence()
    fig_compression()
