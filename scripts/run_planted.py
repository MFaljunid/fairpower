"""Sensitivity of the minimum detectable gap to the simulation assumptions (reviewer comment 3).

    py scripts/run_planted.py --dataset ml-100k
    py scripts/run_planted.py --dataset ml-1m
    py scripts/run_planted.py --dataset gowalla

For each comparison (as in run_power.py), under leave-one-out (m = 1 by default), a gap of size
k x MDE is planted for k in {0, 0.5, 1, 1.5, 2} and the empirical power is compared with the power the
normal approximation predicts (analytic: 0.05 at k = 0 and 0.80 at k = 1 by construction).
The gap is planted
  - with two schemes: "mixture" (more users get a hit) and "scale" (existing hits move up the list);
  - starting from the observed values of either group (base a or base b), i.e. two score distributions.
k = 0 measures the false positive rate of the test (nominal 0.05).
The per-user test NDCG and group labels are also saved (yhat.npz) so later analyses need no retraining.
Outputs: results/<dataset>/<timestamp>_planted/ (output.txt, config.yaml, planted.json, yhat.npz)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairpower.data import ROOT, k_core, read_ratings  # noqa: E402
from fairpower.power import analytic_power, mde, planted_power  # noqa: E402
from fairpower.runlog import Run  # noqa: E402
from fairpower.splits import evaluate_multi, temporal_multi  # noqa: E402
from run_power import comparisons, tune  # noqa: E402

KS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCHEMES = ["mixture", "scale"]


def main(dataset, cfg, m=1, log=print, save_dir=None):
    k_eval, seed, reps = cfg["eval"]["k"], cfg["seed"], cfg["power"]["reps"]
    d = cfg["datasets"][dataset]
    df = read_ratings(dataset, d["raw_dir"])
    if d.get("core"):
        df = k_core(df, d["core"], d["core"])
    split = temporal_multi(df, n_holdout=m, min_train=3, tie_break=cfg.get("tie_break", "random"), seed=seed)
    log(f"{dataset}: m = {m}, users={split.n_users} items={split.n_items}; planted gaps k x MDE, "
        f"k in {KS}; schemes {SCHEMES}; base distribution = group a or group b; {reps} replicates")
    mdl, hp = tune("ials", split, cfg, k_eval, log)
    y, _ = evaluate_multi(mdl, split, "test", k_eval)
    log(f"  overall test NDCG@{k_eval} {y.mean():.4f}")
    res = {"m": m, "hp": hp, "overall": float(y.mean()), "comparisons": {}}
    saved = {"y": y}
    for name, g, a, b in comparisons(dataset, split, d["raw_dir"]):
        saved["g_" + name.replace(" ", "_")] = g
        n = int(min((g == a).sum(), (g == b).sum()))
        out = {}
        log(f"\n  == {name}: n = {n} users per group")
        log(f"  {'base':<6}{'scheme':<9}{'MDE':>8}" + "".join(f"{'k=' + format(k, 'g'):>8}" for k in KS))
        for base_name, base in (("a", a), ("b", b)):
            md = mde(y, g, base, base, n_a=n, n_b=n)
            ana = [analytic_power(k * md, md) for k in KS]
            log(f"  {base_name:<6}{'analytic':<9}{md:>8.4f}" + "".join(f"{p:>8.2f}" for p in ana))
            for sch in SCHEMES:
                emp = [planted_power(y, g, base, n, k * md, reps, seed, sch)[0] for k in KS]
                log(f"  {base_name:<6}{sch:<9}{'':>8}" + "".join(
                    f"{p:>8.2f}" if p is not None else f"{'n.r.':>8}" for p in emp))
                out[f"{base_name}_{sch}"] = emp
            out[f"{base_name}_analytic"] = ana
            out[f"{base_name}_mde"] = md
        res["comparisons"][name] = dict(out, n=n)
    if save_dir is not None:
        np.savez_compressed(Path(save_dir) / "yhat.npz", **saved)
    log("\n  n.r. = gap not reachable with the mixture scheme (more than all zeros would have to flip)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m", "gowalla"])
    ap.add_argument("--holdout", type=int, default=1)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    with Run(cfg, "planted") as run:
        out = main(args.dataset, cfg, args.holdout, save_dir=run.path("."))
        run.path("planted.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")