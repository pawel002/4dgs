#!/usr/bin/env python
"""Generate the thesis figures for the *improvement* experiments (PNG, 300 dpi).

Covers the three mechanisms added on top of the measured v2 pipeline:

  * shadow-aware hysteresis segmentation   (fig: seg_variants, seg_montage)
  * distance-field (margin) hull snapping  (fig: snap_margin)
  * temporal attribute-smoothness prior    (fig: smoothness)
  * end-to-end comparison                  (fig: improved_summary)

Each figure is skipped with a warning if its inputs are missing, so the script can be
re-run as results come in.

Run (CPU)::

    .venv/bin/python experiments/make_improvement_figures.py
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
RUNS = Path("splats/thesis")

BLUE = "#2a78d6"     # improved / default method
YELLOW = "#eda100"   # intermediate variant
RED = "#e34948"      # baseline / naive
VIOLET = "#4a3aa7"   # secondary arm
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


def _stats(name: str):
    p = RUNS / name / "temporal-splatfacto" / "imp" / "temporal_frames" / "temporal_stats.json"
    return _load(p)


# ------------------------------------------------------------------ segmentation


def fig_seg_variants():
    d = _load(Path("experiments/outputs/bgsub/seg_variants.json"))
    if d is None:
        return
    res = d["results"]

    def pick(variant, **kw):
        for r in res:
            if r["variant"] == variant and all(abs(r[k] - v) < 1e-9 for k, v in kw.items()):
                return r
        return None

    base = pick("mean", thr=0.06)
    mx = pick("max", thr=0.08)
    noconn = pick("shadow-noconn", thr=0.1, thr_lo=0.01)
    hyst = pick("hyst+shadow", thr=0.1, thr_lo=0.01, chroma_thr=0.04)
    hyst_only = pick("hyst", thr=0.1, thr_lo=0.01)
    arms = [
        ("single $\\tau$\n(production)", base, RED),
        ("max-\nchannel", mx, YELLOW),
        ("hysteresis\n(no veto)", hyst_only, GRAY),
        ("veto (no\nconnectivity)", noconn, VIOLET),
        ("shadow-aware\nhysteresis", hyst, BLUE),
    ]
    arms = [(lbl, r, c) for lbl, r, c in arms if r is not None]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    x = np.arange(len(arms))
    w = 0.27
    for off, key, color, label in ((-w, "iou", BLUE, "IoU"), (0, "precision", AQUA, "precision"),
                                   (w, "recall", VIOLET, "recall")):
        ax.bar(x + off, [r[key] for _, r, _ in arms], width=w, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _, _ in arms], fontsize=6.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("mask quality vs. ground truth")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7)
    ax.set_title("(a) segmentation variants", fontsize=9.5)

    ax = axes[1]
    for label, r, color in ((lbl, r, c) for lbl, r, c in arms if lbl.startswith(("single", "shadow-aware"))):
        pf = r["per_frame_iou"]
        t = np.array(sorted(int(k) for k in pf))
        ax.plot(t, [pf[str(k)] for k in t], color=color, marker="o", ms=2.5,
                label=label.replace("\n", " "))
    ax.axvspan(0.5, 20.5, color=YELLOW, alpha=0.12, lw=0)
    ax.text(10.5, 0.4, "camouflage\nwindow", color=GRAY, fontsize=7, ha="center")
    ax.set_xlabel("frame")
    ax.set_ylabel("mask IoU")
    ax.set_ylim(0.35, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.set_title("(b) per-frame IoU", fontsize=9.5)
    _save(fig, "seg_variants")


def fig_seg_montage():
    """Qualitative: the camouflage frame under both segmenters (camera static1)."""
    import torch

    from experiments.seg_variants_benchmark import Config as SegCfg
    from experiments.seg_variants_benchmark import load_rgb, make_mask

    dataset = Path("../output/dataset")
    gt_dir = Path("../output/black-bg-frames")
    cam, t = "static1", 3
    if not (dataset / cam / "images" / f"{t:04d}.png").exists():
        print("[skip] seg_montage: dataset images not found")
        return
    cfg = SegCfg()
    device = "cpu"
    img = load_rgb(dataset / cam / "images" / f"{t:04d}.png", device)
    plate = load_rgb(dataset / cam / "images" / "0000.png", device)
    gt = load_rgb(gt_dir / cam / "images" / f"{t:04d}.png", device)
    diff = (img - plate).abs().mean(-1)
    m_base = make_mask("mean", img, plate, 0.06, 0.0, cfg)
    m_hyst = make_mask("hyst+shadow", img, plate, 0.1, 0.01, cfg, 0.04)
    gt_mask = (gt.mean(-1) > 0.02).float()

    # Crop to the subject bbox (from the GT mask, padded).
    ys, xs = torch.nonzero(gt_mask, as_tuple=True)
    pad = 60
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, img.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, img.shape[1])

    panels = [
        (img[y0:y1, x0:x1].numpy(), "input frame"),
        (np.clip(diff[y0:y1, x0:x1].numpy() * 6.0, 0, 1), "change magnitude $d$ ($\\times 6$)"),
        ((img * m_base[..., None])[y0:y1, x0:x1].numpy(), "single threshold"),
        ((img * m_hyst[..., None])[y0:y1, x0:x1].numpy(), "shadow-aware hysteresis"),
        (gt[y0:y1, x0:x1].numpy(), "ground truth"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8, 2.3), constrained_layout=True)
    for ax, (im, title) in zip(axes, panels):
        if im.ndim == 2:
            ax.imshow(im, cmap="magma", vmin=0, vmax=1)
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=7.5)
        ax.axis("off")
        ax.grid(False)
    _save(fig, "seg_montage_improved")


# ------------------------------------------------------------------ snapping


def fig_snap_margin():
    """Flagged-vs-snapped per frame across the arms + structural quality."""
    arms = [
        ("imp_snapfield", "margin, single-threshold seg", YELLOW),
        ("imp_seg", "no margin, hysteresis seg", VIOLET),
        ("imp_full", "margin + hysteresis seg", BLUE),
    ]
    stats = [(label, _stats(name), color) for name, label, color in arms]
    if all(s is None for _, s, _ in stats):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)

    ax = axes[0]
    for label, s, color in stats:
        if s is None:
            continue
        frames = sorted(int(k) for k in s["frames"])
        flagged = np.array([s["frames"][str(f)].get("flagged", np.nan) for f in frames], dtype=float)
        ax.plot(frames, flagged, color=color, label=label, linewidth=1.2)
    ax.set_xlabel("frame")
    ax.set_ylabel("gaussians flagged by strict test")
    ax.set_title("(a) strict-test nominations", fontsize=9.5)

    ax = axes[1]
    for label, s, color in stats:
        if s is None:
            continue
        frames = sorted(int(k) for k in s["frames"])
        snapped = np.array([s["frames"][str(f)].get("snapped", np.nan) for f in frames], dtype=float)
        ax.plot(frames, snapped, color=color, label=label, linewidth=1.2)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("gaussians actually snapped")
    ax.set_title("(b) teleports applied", fontsize=9.5)

    # One shared legend below both panels (the line colors are common to (a) and (b)).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=7.5, loc="outside lower center")
    _save(fig, "snap_margin")


# ------------------------------------------------------------------ smoothness


def fig_smoothness():
    jit_base = _load(Path("experiments/outputs/jitter/full150.json"))
    jit_smooth = _load(Path("experiments/outputs/jitter/imp_smooth.json"))
    comp_base = _load(Path("experiments/outputs/compression/compression_benchmark.json"))
    comp_smooth = _load(Path("experiments/outputs/compression/imp_smooth.json"))
    if jit_base is None or jit_smooth is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})

    ax = axes[0]
    groups = ["opacity", "scale", "rotation", "position"]
    x = np.arange(len(groups)) * 1.6  # extra horizontal spacing so tick labels don't collide
    w = 0.55
    b = [jit_base["summary"][g] for g in groups]
    s = [jit_smooth["summary"][g] for g in groups]
    ax.bar(x - w / 2, b, width=w, color=RED, label="baseline")
    ax.bar(x + w / 2, s, width=w, color=BLUE, label="smoothness prior")
    for xi, (bv, sv) in zip(x, zip(b, s)):
        ratio = bv / max(sv, 1e-12)
        if ratio > 1.5:
            ax.text(xi + w / 2, sv * 1.3, f"$\\times${ratio:.0f}", ha="center", fontsize=6.5, color=GRAY)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=6.5)
    ax.set_xlim(x[0] - 1.0, x[-1] + 1.0)
    ax.set_ylabel("mean $|\\Delta|$ per frame")
    ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("(a) attribute jitter", fontsize=9)

    # (b) the pilot lambda selection: SSIM cost vs compression gain.
    ax = axes[1]
    lams, ssims, ratios = [0.0], [], []
    ref_eval = _load(EVAL / "abl150_default.json")
    if ref_eval is not None:
        ssims.append(float(np.nanmean([r.get("object_ssim", np.nan) for r in ref_eval["frames"][:40]])))
    ref_comp = _load(Path("experiments/outputs/compression/lambda0_40f.json"))
    if ref_comp is not None:
        ratios.append(next(p["ratio"] for p in ref_comp["sweep"] if abs(p["quality"] - 0.999) < 1e-9))
    for lam, tag in ((0.3, "03"), (1.0, "10"), (3.0, "30")):
        ev = _load(EVAL / f"pilot_smooth{tag}.json")
        cp = _load(Path(f"experiments/outputs/compression/pilot_smooth{tag}.json"))
        if ev is None or cp is None:
            continue
        lams.append(lam)
        ssims.append(ev["summary"]["object_ssim"]["mean"])
        ratios.append(next(p["ratio"] for p in cp["sweep"] if abs(p["quality"] - 0.999) < 1e-9))
    if len(lams) == len(ssims) == len(ratios) and len(lams) > 1:
        ax.plot(lams, ssims, color=VIOLET, marker="o", ms=3.5, label="object SSIM")
        ax.set_xlabel("$\\lambda_{\\mathrm{s}}$")
        ax.set_ylabel("object SSIM", color=VIOLET)
        ax.tick_params(axis="y", labelcolor=VIOLET)
        ax2 = ax.twinx()
        ax2.plot(lams, ratios, color=AQUA, marker="s", ms=3.5, label="ratio")
        ax2.set_ylabel("ratio at $q=0.999$ ($\\times$)", color=AQUA)
        ax2.tick_params(axis="y", labelcolor=AQUA)
        ax2.grid(False)
        ax2.spines.top.set_visible(False)
        ax.axvline(0.3, color=GRAY, linestyle=":", linewidth=1.1)
        ax.text(0.42, min(ssims) + 0.002, "chosen", color=GRAY, fontsize=7)
    ax.set_title("(b) pilot: choosing $\\lambda_{\\mathrm{s}}$", fontsize=9)

    ax = axes[2]
    if comp_base is not None and comp_smooth is not None:
        for d, color, label in ((comp_base, RED, "baseline"), (comp_smooth, BLUE, "smoothness prior")):
            sw = d["sweep"]
            ax.plot([p["ratio"] for p in sw], [p["render_psnr"] for p in sw], color=color,
                    marker="o", ms=3.5, label=label)
            for p in sw:
                if abs(p["quality"] - 0.999) < 1e-9:
                    ax.scatter([p["ratio"]], [p["render_psnr"]], s=70, facecolors="none",
                               edgecolors=color, linewidths=1.4)
        ax.set_xlabel("compression ratio ($\\times$)")
        ax.set_ylabel("render PSNR (dB)")
        ax.legend(frameon=False, fontsize=6.5, title="circled: $q=0.999$", title_fontsize=6)
        ax.set_title("(c) rate--distortion, 150 frames", fontsize=9)
    _save(fig, "smoothness")


# ------------------------------------------------------------------ budget


def _run_minutes(name: str):
    """Wall-clock of a run in minutes: config.yml ctime -> last ply mtime."""
    run = RUNS / name / "temporal-splatfacto" / "imp"
    cfgf = run / "config.yml"
    last = run / "temporal_frames" / "00149.ply"
    if not (cfgf.exists() and last.exists()):
        return None
    return (last.stat().st_mtime - cfgf.stat().st_mtime) / 60.0


def fig_budget():
    conv = _load(Path("experiments/outputs/hull_init/frame0_convergence_hyst.json"))
    arms = [
        ("imp_full", "$10\\,000\\,/\\,3000$", BLUE),
        ("imp_budget_5k1000", "$5000\\,/\\,1000$", AQUA),
        ("imp_budget_5k500", "$5000\\,/\\,500$", VIOLET),
        ("imp_budget_3k300", "$3000\\,/\\,300$", YELLOW),
    ]
    evals = [(label, _load(EVAL / f"{name}.json"), _run_minutes(name), color)
             for name, label, color in arms]
    have_evals = any(e is not None for _, e, _, _ in evals)
    if conv is None and not have_evals:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)

    ax = axes[0]
    if conv is not None:
        h = conv["hull"]
        ax.plot(h["steps"], h["object_psnr"], color=BLUE, linewidth=1.4)
        for n0, style in ((10000, "--"), (5000, ":"), (3000, ":")):
            ax.axvline(n0, color=GRAY, linestyle=style, linewidth=1.0)
            ax.text(n0, ax.get_ylim()[0] + 2, f"$N_0{{=}}{n0}$", rotation=90,
                    fontsize=6.5, color=GRAY, ha="right", va="bottom")
        ax.set_xlabel("frame-0 step")
        ax.set_ylabel("object PSNR (dB)")
        ax.set_title("(a) frame-0 convergence", fontsize=9.5)

    ax = axes[1]
    if have_evals:
        for label, d, minutes, color in evals:
            if d is None:
                continue
            x = minutes if minutes is not None else np.nan
            ax.scatter([x], [d["summary"]["object_ssim"]["mean"]], color=color, s=42,
                       label=label, zorder=3)
        ax.set_xlabel("wall-clock (minutes, 150 frames)")
        ax.set_ylabel("object SSIM")
        ax.legend(frameon=False, fontsize=7, title="$N_0\\,/\\,N_t$", title_fontsize=7)
        ax.set_title("(b) quality vs. cost", fontsize=9.5)
    _save(fig, "budget_sweep")

    if have_evals:
        print("\n=== budget table ===")
        for label, d, minutes, _ in evals:
            if d is None:
                continue
            s = d["summary"]
            print(f"{label:22s} {minutes and f'{minutes:5.1f} min' or '   ?     '}  "
                  f"PSNR={s['object_psnr']['mean']:.2f}  SSIM={s['object_ssim']['mean']:.3f}  "
                  f"cov={s['coverage']['mean']:.3f}  leak={s['leak']['mean']:.2e}")


# ------------------------------------------------------------------ end-to-end


def fig_improved_summary():
    base = _load(EVAL / "full150.json")
    full = _load(EVAL / "imp_full.json")
    ceil = _load(EVAL / "input_ceiling.json")
    if base is None or full is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(_series(base, "object_psnr"), color=RED, label="v2 pipeline", linewidth=1.2)
    ax.plot(_series(full, "object_psnr"), color=BLUE, label="improved", linewidth=1.2)
    if ceil is not None:
        ax.plot(_series(ceil, "object_psnr"), color=GRAY, linestyle="--", linewidth=1.0,
                label="segmented-input reference")
    ax.set_xlabel("frame")
    ax.set_ylabel("object PSNR (dB)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(a) per-frame object PSNR", fontsize=9.5)

    ax = axes[1]
    ax.plot(_series(base, "object_ssim"), color=RED, label="v2 pipeline", linewidth=1.2)
    ax.plot(_series(full, "object_ssim"), color=BLUE, label="improved", linewidth=1.2)
    ax.set_xlabel("frame")
    ax.set_ylabel("object SSIM")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(b) per-frame object SSIM", fontsize=9.5)
    _save(fig, "improved_summary")

    # LaTeX-ready headline numbers.
    def summ(d, k):
        return d["summary"][k]["mean"]

    arms = [("full150 (v2 baseline)", base)]
    for name in ("imp_seg", "imp_snapfield", "imp_smooth", "imp_full"):
        d = _load(EVAL / f"{name}.json")
        if d is not None:
            arms.append((name, d))
    print("\n=== headline table ===")
    for name, d in arms:
        print(f"{name:28s} PSNR={summ(d, 'object_psnr'):.2f}  SSIM={summ(d, 'object_ssim'):.3f}  "
              f"cov={summ(d, 'coverage'):.3f}  leak={summ(d, 'leak'):.2e}  "
              f"extent(last)={d['summary']['radius_max']['last']:.2f}")


if __name__ == "__main__":
    fig_seg_variants()
    fig_seg_montage()
    fig_snap_margin()
    fig_smoothness()
    fig_budget()
    fig_improved_summary()
