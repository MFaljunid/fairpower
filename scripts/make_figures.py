"""Figures for the paper, drawn from the JSON files of the runs in results/.

    py scripts/make_figures.py
    py scripts/make_figures.py --out paper/figures

For every setting the LATEST matching run folder is used (folder names start with a timestamp).
Figures (PDF, sized for a single LNCS text column, 12.2 cm):
  fig_noise.pdf  (a) Phi_N@10 of every training seed, no intervention, per dataset / protocol / model;
                 (b) changes in the gender gap reported in published studies against the paired MDE of
                     the same protocol (range over the available models).
  fig_power.pdf  power to detect the observed test gap against users per group, leave-one-out (m = 1),
                 ML-100K and ML-1M, with the 0.8 target.
  fig_mde.pdf    MDE relative to overall NDCG@10 against held-out items per user m.
Settings without a run folder are skipped and listed in the console output.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 12.2 / 2.54                       # LNCS text width in inches
plt.rcParams.update({"axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})

# (label, dataset, folder suffixes tried in order, protocol key used in CLAIMS)
NOISE_SETTINGS = [
    ("ML-100K\nrandom\niALS", "ml-100k", ["_noise_ials_random_gender", "_noise_random_gender"], "random"),
    ("ML-100K\nrandom\nBPR-MF", "ml-100k", ["_noise_bpr_random_gender"], "random"),
    ("ML-100K\nLOO\niALS", "ml-100k", ["_noise_ials_loo_gender", "_noise_loo_gender"], "loo"),
    ("ML-1M\nrandom\niALS", "ml-1m", ["_noise_ials_random_gender", "_noise_random_gender"], "random"),
    ("ML-1M\nrandom\nBPR-MF", "ml-1m", ["_noise_bpr_random_gender"], "random"),
    ("ML-1M\ntemporal 20%\nBPR-MF", "ml-1m", ["_noise_bpr_temporal20_gender"], "temporal20"),
]

# Reported changes in the gender gap (NDCG@10): label, dataset, protocol, |change|
CLAIMS = [
    ("FairDgcl vs. Graphair, ML-100K", "ml-100k", "random", 0.0024),
    ("FairDgcl vs. LightGCN, ML-100K", "ml-100k", "random", 0.0059),
    ("FairDgcl vs. Graphair, ML-1M", "ml-1m", "random", 0.0062),
    ("FairDgcl vs. LightGCN, ML-1M", "ml-1m", "random", 0.0139),
    ("Boratto et al., smallest, ML-1M", "ml-1m", "temporal20", 0.002),
    ("Boratto et al., largest, ML-1M", "ml-1m", "temporal20", 0.008),
]

# Okabe-Ito colour-blind safe palette; line styles and markers are kept so the figures also print in grey
BLUE, ORANGE, GREEN, RED, PURPLE, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"
OK_COL, BAD_COL = "#1B7837", "#B2182B"     # claim above / below the noise floor (distinct from dataset colours)
COMPARISONS = [("activity Q4 vs Q1", "Activity Q4 vs. Q1", "-", "o", BLUE),
               ("activity Q4 vs Q3", "Activity Q4 vs. Q3", "--", "s", ORANGE),
               ("gender M vs F", "Gender M vs. F", ":", "^", PURPLE)]
SETTING_COLOR = {"ml-100k": BLUE, "ml-1m": ORANGE}


def latest(dataset, suffixes, fname):
    base = ROOT / "results" / dataset
    for suf in suffixes:
        hits = sorted(p for p in base.glob(f"*{suf}") if (p / fname).exists() and p.name.endswith(suf))
        if hits:
            return json.loads((hits[-1] / fname).read_text(encoding="utf-8")), hits[-1].name
    return None, None


def fig_noise(out, log):
    rows, mde_by = [], {}
    for label, ds, sufs, prot in NOISE_SETTINGS:
        res, name = latest(ds, sufs, "noise.json")
        if res is None:
            log(f"  noise: no run for {label.replace(chr(10), ' ')} (skipped)")
            continue
        log(f"  noise: {label.replace(chr(10), ' ')} <- {name}")
        k = res["by_k"]["10"]
        rows.append((label, k["phi"], ds))
        pm = [c["paired_mde"] for c in k["seed_changes"]]
        lo, hi = mde_by.get((ds, prot), (float("inf"), 0.0))
        mde_by[(ds, prot)] = (min(lo, min(pm)), max(hi, max(pm)))
    if not rows:
        return
    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 4.8), gridspec_kw={"height_ratios": [1.1, 1]})
    for i, (label, phis, ds) in enumerate(rows):
        c = SETTING_COLOR[ds]
        a.vlines(i, min(phis), max(phis), color=c, alpha=0.35, lw=7, zorder=2)
        a.scatter([i] * len(phis), phis, s=22, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        a.text(i, max(phis), f"  {max(phis) - min(phis):.4f}", fontsize=5.5, va="bottom", ha="center", color=c)
    from matplotlib.lines import Line2D
    a.legend([Line2D([], [], marker="o", ls="", color=SETTING_COLOR[d]) for d in ("ml-100k", "ml-1m")],
             ["MovieLens-100K", "MovieLens-1M"], frameon=False, loc="upper right")
    a.set_xticks(range(len(rows)))
    a.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    a.set_ylabel(r"$\Phi_N$@10 (gender)")
    a.set_title("(a) Gender gap of each training seed, no intervention", fontsize=8)
    claims = [c for c in CLAIMS if (c[1], c[2]) in mde_by]
    for j, (label, ds, prot, val) in enumerate(claims):
        lo, hi = mde_by[(ds, prot)]
        b.hlines(j, lo, hi, color=SETTING_COLOR[ds], alpha=0.3, lw=7, zorder=1)
        ok = val >= lo
        b.scatter(val, j, marker="D" if ok else "X", s=34, color=OK_COL if ok else BAD_COL, edgecolor="black",
                  linewidth=0.5, zorder=3)
    b.set_yticks(range(len(claims)))
    b.set_yticklabels([c[0] for c in claims], fontsize=6.5)
    b.invert_yaxis()
    b.set_xlim(left=0)
    b.set_xlabel(r"|change in $\Phi_N$@10|")
    b.set_title("(b) Reported change vs. paired MDE of the protocol (bar)", fontsize=8)
    fig.legend([Line2D([], [], marker="D", ls="", color=OK_COL, markeredgecolor="black"),
                Line2D([], [], marker="X", ls="", color=BAD_COL, markeredgecolor="black"),
                Line2D([], [], lw=6, color="0.6", alpha=0.5)],
               ["detectable change", "not detectable", "paired MDE range"],
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig_noise.pdf")
    plt.close(fig)
    log(f"  -> {out / 'fig_noise.pdf'} (bar in (b): paired MDE range over available models; "
        f"green diamond: change at or above the smallest paired MDE, red cross: below it)")


def fig_power(out, log):
    data = []
    for ds, title in (("ml-100k", "ML-100K"), ("ml-1m", "ML-1M")):
        res, name = latest(ds, ["_power_ials"], "power.json")
        if res is None:
            log(f"  power: no run for {ds} (skipped)")
            continue
        log(f"  power: {ds} <- {name}")
        data.append((title, res))
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(WIDTH, 2.2), sharey=True, squeeze=False)
    for ax, (title, res) in zip(axes[0], data):
        for key, lab, ls, mk, col in COMPARISONS:
            pc = res["1"]["comparisons"][key]["power_curve"]
            ax.plot([p["n"] for p in pc], [p["power"] for p in pc], ls=ls, marker=mk, ms=4, lw=1.5, color=col, label=lab)
        ax.axhline(0.8, color=RED, lw=0.9, ls="-.")
        ax.text(1.0, 0.81, "target 0.8", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6, color=RED)
        ax.set_xscale("log")
        ax.set_xticks([100, 250, 500, 1000, 2500])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Users per group")
        ax.set_title(f"{title}, leave-one-out", fontsize=8)
    axes[0][0].set_ylabel("Power (observed gap)")
    axes[0][0].legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "fig_power.pdf")
    plt.close(fig)
    log(f"  -> {out / 'fig_power.pdf'}")


def fig_mde(out, log):
    data = []
    for ds, title in (("ml-100k", "ML-100K"), ("ml-1m", "ML-1M")):
        res, _ = latest(ds, ["_power_ials"], "power.json")
        if res is not None:
            data.append((title, res))
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(WIDTH, 2.0), sharey=True, squeeze=False)
    for ax, (title, res) in zip(axes[0], data):
        ms = sorted(int(m) for m in res)
        for key, lab, ls, mk, col in COMPARISONS:
            ax.plot(ms, [100 * res[str(m)]["comparisons"][key]["mde_rel"] for m in ms], ls=ls, marker=mk,
                    ms=4, lw=1.5, color=col, label=lab)
        ax.set_xticks(ms)
        ax.set_xlabel("Held-out items per user $m$")
        ax.set_title(title, fontsize=8)
    axes[0][0].set_ylabel("MDE (% of NDCG@10)")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_mde.pdf")
    plt.close(fig)
    log(f"  -> {out / 'fig_mde.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_noise, fig_power, fig_mde):
        f(out, print)
    sys.exit(0)