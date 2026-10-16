"""RQ1: are user-group gaps detectable? Minimum detectable gap (MDE) by dataset, grouping and the number
of held-out items per user.

    py scripts/run_power.py --dataset ml-1m
    py scripts/run_power.py --dataset ml-1m --holdout 1,3,5 --model ials

Protocol: temporal hold-out of m items per user (m = 1 is leveldown's leave-one-out), random
tie-breaking, full ranking, NDCG@10. For comparability across m, every m uses the SAME users: those
with at least 2 * max(m) + 3 interactions. The model is tuned on validation NDCG@10 for each m.
Comparisons (gap = group b - group a, test NDCG@10):
  activity: Q4 (most active) vs Q1, and Q4 vs Q3 (adjacent quartiles)
  gender (MovieLens only): M vs F
For each: gap with bootstrap 95% CI and permutation p, MDE at alpha 0.05 / power 0.8, |gap| / MDE,
empirical power to detect the observed gap at several group sizes, and the planted-gap power at the
MDE (a check of the normal approximation on the real, zero-inflated metric).
Outputs: results/<dataset>/<timestamp>_power/ (output.txt, config.yaml, power.json)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fairpower.data import ROOT, activity_groups, k_core, read_gender, read_ratings  # noqa: E402
from fairpower.models import EASE, IALS, ItemPop  # noqa: E402
from fairpower.power import gap_test, mde, planted_power, power_curve  # noqa: E402
from fairpower.runlog import Run  # noqa: E402
from fairpower.splits import evaluate_multi, temporal_multi  # noqa: E402

MODELS = {"ials": IALS, "ease": EASE}


def grid(d):
    keys = list(d)
    return [dict(zip(keys, v)) for v in itertools.product(*(d[k] for k in keys))]


def tune(name, split, cfg, k, log):
    if name == "itempop":
        return ItemPop().fit(split.train, split.n_items), {}
    best = None
    for hp in grid(cfg["grids"][name]):
        m = MODELS[name](**hp).fit(split.train, split.n_items)
        v = evaluate_multi(m, split, "val", k)[0].mean()
        log(f"    {name} {hp}: val NDCG@{k} {v:.4f}")
        if best is None or v > best[0]:
            best = (v, hp, m)
    log(f"    -> selected {name} {best[1]} (val NDCG@{k} {best[0]:.4f})")
    return best[2], best[1]


def comparisons(dataset, split, raw_dir):
    q, _ = activity_groups(split.train)
    out = [("activity Q4 vs Q1", q, 0, 3), ("activity Q4 vs Q3", q, 2, 3)]
    if dataset in ("ml-100k", "ml-1m"):
        gmap = read_gender(dataset, raw_dir)
        gl = np.array([1 if gmap[int(u)] == "M" else 0 for u in split.users])
        out.append(("gender M vs F", gl, 0, 1))
    return out


def main(dataset, cfg, holdouts, model, log=print):
    k, B, seed = cfg["eval"]["k"], cfg["eval"]["bootstrap"], cfg["seed"]
    d = cfg["datasets"][dataset]
    df = read_ratings(dataset, d["raw_dir"])
    if d.get("core"):
        df = k_core(df, d["core"], d["core"])
    need = 2 * max(holdouts) + 3
    cnt = df.groupby("user").size()
    df = df[df["user"].isin(cnt[cnt >= need].index)]
    log(f"{dataset}: users with >= {need} interactions kept for every m (same users across m); "
        f"model {model}; FULL RANKING, NDCG@{k}")
    res = {}
    for m in holdouts:
        split = temporal_multi(df, n_holdout=m, min_train=3, tie_break=cfg.get("tie_break", "random"), seed=seed)
        log(f"\n######## m = {m} held-out items per user: users={split.n_users} items={split.n_items}")
        mdl, hp = tune(model, split, cfg, k, log)
        y, _ = evaluate_multi(mdl, split, "test", k)
        log(f"  overall test NDCG@{k} {y.mean():.4f}")
        res[str(m)] = {"hp": hp, "overall": float(y.mean()), "comparisons": {}}
        for name, g, a, b in comparisons(dataset, split, d["raw_dir"]):
            t = gap_test(y, g, a, b, B, seed)
            md = mde(y, g, a, b)
            sizes = [s for s in (100, 250, 500, 1000, 2500) if s < min(t["n_a"], t["n_b"])] + \
                    [min(t["n_a"], t["n_b"])]
            pc = power_curve(y, g, a, b, sizes, reps=cfg["power"]["reps"], seed=seed)
            n_pl = min(t["n_a"], t["n_b"])
            md_eq = mde(y, g, a, a, n_a=n_pl, n_b=n_pl)     # MDE for n_pl per group, both drawn from group a
            pl, q = planted_power(y, g, a, n_pl, md_eq, reps=cfg["power"]["reps"], seed=seed)
            ratio = abs(t["gap"]) / md
            log(f"  {name:<20} n={t['n_a']}/{t['n_b']}  means {t['mean_a']:.4f} / {t['mean_b']:.4f}  "
                f"gap {t['gap']:+.4f} [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] p={t['p']:.3f}  "
                f"MDE {md:.4f} ({md / y.mean():.0%} of overall)  |gap|/MDE {ratio:.2f}")
            log("      power to detect the observed gap by group size: " +
                "  ".join(f"n={r['n']}: {r['power']:.2f}" for r in pc))
            log(f"      planted gap = MDE ({md_eq:.4f}) at n={n_pl} per group: empirical power "
                + (f"{pl:.2f} (q={q:.3f}; nominal 0.80)" if pl is not None else "not reachable"))
            res[str(m)]["comparisons"][name] = dict(t, mde=md, mde_rel=md / y.mean(), ratio=ratio,
                                                    power_curve=pc, planted_n=n_pl, planted_mde=md_eq,
                                                    planted_power_at_mde=pl)
    log("\n== SUMMARY: MDE (relative to overall NDCG) and |gap|/MDE")
    names = list(res[str(holdouts[0])]["comparisons"])
    log(f"  {'comparison':<20}" + "".join(f"{'m=' + str(m):>24}" for m in holdouts))
    for nm in names:
        log(f"  {nm:<20}" + "".join(
            f"{res[str(m)]['comparisons'][nm]['mde']:>9.4f} ({res[str(m)]['comparisons'][nm]['mde_rel']:>4.0%}) "
            f"r={res[str(m)]['comparisons'][nm]['ratio']:>4.2f}" for m in holdouts))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m", "gowalla"])
    ap.add_argument("--holdout", default="1,3,5")
    ap.add_argument("--model", default="ials", choices=["ials", "ease", "itempop"])
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    hs = [int(x) for x in args.holdout.split(",")]
    with Run(cfg, f"power_{args.model}") as run:
        out = main(args.dataset, cfg, hs, args.model)
        run.path("power.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
