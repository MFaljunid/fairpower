"""RQ2: how large must a reported change in a group gap be to exceed the noise of the protocol?

    py scripts/run_noise_floor.py --dataset ml-100k --protocol random
    py scripts/run_noise_floor.py --dataset ml-1m --protocol random --ks 10,20
    py scripts/run_noise_floor.py --dataset ml-1m --protocol loo
    py scripts/run_noise_floor.py --dataset ml-1m --protocol random --model bpr
    py scripts/run_noise_floor.py --dataset ml-100k --protocol random --model bpr --vary both

Protocols:
  random : random split of all interactions 80/10/10 (e.g. FairDgcl, TKDE / arXiv 2410.17555)
  loo    : temporal leave-one-out (leveldown protocol)
  temporal20 : per user, the most recent 20% -> test, the most recent 10% of the rest -> validation
           (Boratto et al., ECIR 2022, consumer fairness reproducibility study)
The same model (iALS or BPR-MF, tuned once on validation NDCG@10 with seed 0) is trained with several
seeds and NO fairness intervention. iALS has closed-form updates, so its seed only changes the
initialization; BPR-MF is trained by SGD with sampled negatives, like most models in the fairness
literature (LightGCN, GCCF, NCF), and gives the more realistic noise floor. Any change in the gap between two seeds is noise. Reported per K, for the absolute gap
Phi = |mean NDCG of group 1 - mean NDCG of group 0| (Phi_N in the fairness literature):
  - Phi for every seed, and its mean, SD and range across seeds (training noise);
  - the MDE of the gap level from user sampling (alpha 0.05, power 0.8);
  - the paired MDE of a change in the gap between two models on the same users (seed 0 vs each other
    seed), i.e. the smallest change of the gap a paper could detect with a paired test.
Users without a test item (possible under the random split) are left out.
Outputs: results/<dataset>/<timestamp>_noise_<protocol>_<groups>/ (output.txt, config.yaml, noise.json)
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
from fairpower.models import BPRMF, IALS  # noqa: E402
from fairpower.power import mde  # noqa: E402
from fairpower.runlog import Run  # noqa: E402
from fairpower.splits import evaluate_multi, random_split, temporal_multi, temporal_ratio  # noqa: E402


def grid(d):
    keys = list(d)
    return [dict(zip(keys, v)) for v in itertools.product(*(d[k] for k in keys))]


MODELS = {"ials": IALS, "bpr": BPRMF}


def make_split(df, protocol, cfg, split_seed):
    tb = cfg.get("tie_break", "random")
    if protocol == "random":
        return random_split(df, tuple(cfg["noise"]["ratios"]), seed=split_seed)
    if protocol == "temporal20":
        return temporal_ratio(df, 0.2, 0.1, min_train=3, tie_break=tb, seed=split_seed)
    return temporal_multi(df, n_holdout=1, min_train=3, tie_break=tb, seed=split_seed)


def group_labels(split, grouping, dataset, raw_dir):
    """Returns (g, names); g = 0 / 1 for the two compared groups, -1 for users outside both."""
    if grouping == "gender":
        gmap = read_gender(dataset, raw_dir)
        return np.array([1 if gmap[int(u)] == "M" else 0 for u in split.users]), ["F", "M"]
    q, _ = activity_groups(split.train)
    return np.where(q == 3, 1, np.where(q == 0, 0, -1)), ["Q1", "Q4"]


def main(dataset, cfg, protocol, grouping, seeds, ks, model="ials", vary="train", log=print):
    """vary = "train": fixed split, model seed s (training noise only; paired MDE between seeds);
              "split": data split seed varies with s, model seed 0 (split noise only);
              "both" : split seed and model seed both vary with s (total noise).
    The split seed of run s is cfg seed + s, so s = 0 always reproduces the default split.
    Under temporal protocols the split varies only through random timestamp tie-breaking."""
    if vary not in ("train", "split", "both"):
        raise ValueError(vary)
    cls = MODELS[model]
    base_seed = cfg["seed"]
    d = cfg["datasets"][dataset]
    df = read_ratings(dataset, d["raw_dir"])
    if d.get("core"):
        df = k_core(df, d["core"], d["core"])
    split = make_split(df, protocol, cfg, base_seed)
    g, names = group_labels(split, grouping, dataset, d["raw_dir"])
    n_test = np.array([len(t) for t in split.test])
    log(f"{dataset}: protocol={protocol}, groups={grouping} ({names[0]}={int((g == 0).sum())}, "
        f"{names[1]}={int((g == 1).sum())}), users={split.n_users}, items={split.n_items}, "
        f"test items per user: mean {n_test.mean():.1f}, median {np.median(n_test):.0f}, "
        f"users without test items {int((n_test == 0).sum())}; varying: {vary}")

    log(f"\n== tuning {model} on validation NDCG@10 (default split, model seed 0)")
    best = None
    for hp in grid(cfg["grids"][model]):
        m = cls(**hp, seed=0).fit(split.train, split.n_items)
        v = np.nanmean(evaluate_multi(m, split, "val", 10)[0])
        log(f"  {model} {hp}: val NDCG@10 {v:.4f}")
        if best is None or v > best[0]:
            best = (v, hp)
    hp = best[1]
    log(f"  -> selected {hp}")

    runs = []                                   # per run s: {k: (y, g)}
    for s in seeds:
        sp = split if vary == "train" else make_split(df, protocol, cfg, base_seed + int(s))
        gs = g if vary == "train" else group_labels(sp, grouping, dataset, d["raw_dir"])[0]
        mseed = int(s) if vary in ("train", "both") else 0
        m = cls(**hp, seed=mseed).fit(sp.train, sp.n_items)
        runs.append({k: (evaluate_multi(m, sp, "test", k)[0], gs) for k in ks})

    res = {"model": model, "vary": vary, "hp": hp, "n_users": int(split.n_users),
           "test_items_mean": float(n_test.mean()), "by_k": {}}
    for k in ks:
        phis, overall, level = [], [], []
        for r in runs:
            y, gs = r[k]
            ok = ~np.isnan(y) & (gs >= 0)
            y, gg = y[ok], gs[ok]
            phis.append(abs(y[gg == 1].mean() - y[gg == 0].mean()))
            overall.append(float(y.mean()))
            level.append(mde(y, gg, 0, 1))
        chg = []
        if vary == "train":                     # same users and test items: paired comparison is valid
            y0, g0 = runs[0][k]
            ok = ~np.isnan(y0) & (g0 >= 0)
            for i in range(1, len(seeds)):
                dd = runs[i][k][0][ok] - y0[ok]
                chg.append({"seed": int(seeds[i]), "phi_change": phis[i] - phis[0],
                            "paired_mde": mde(dd, g0[ok], 0, 1)})
        log(f"\n== K = {k}")
        log(f"  NDCG@{k} per run: " + " ".join(f"{x:.4f}" for x in overall))
        log(f"  Phi_N@{k} per run: " + " ".join(f"{x:.4f}" for x in phis))
        log(f"  across runs: mean {np.mean(phis):.4f}, SD {np.std(phis, ddof=1):.4f}, "
            f"range {min(phis):.4f} to {max(phis):.4f} (max - min {max(phis) - min(phis):.4f})")
        log(f"  MDE of the gap level (user sampling): {np.mean(level):.4f}"
            + ("" if vary == "train" else f" (mean over splits, {min(level):.4f} to {max(level):.4f})"))
        if chg:
            log("  seed 0 vs seed s, no intervention: " + "; ".join(
                f"s={c['seed']}: Phi change {c['phi_change']:+.4f}, paired MDE {c['paired_mde']:.4f}" for c in chg))
        else:
            log("  paired MDE not computed: the test items differ between splits")
        res["by_k"][str(k)] = {"overall": overall, "phi": phis, "phi_sd": float(np.std(phis, ddof=1)),
                               "phi_range": float(max(phis) - min(phis)), "level_mde": float(np.mean(level)),
                               "seed_changes": chg}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m", "gowalla"])
    ap.add_argument("--protocol", default="random", choices=["random", "loo", "temporal20"])
    ap.add_argument("--groups", default="gender", choices=["gender", "activity"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ks", default="10,20")
    ap.add_argument("--model", default="ials", choices=["ials", "bpr"])
    ap.add_argument("--vary", default="train", choices=["train", "split", "both"],
                    help="train: training seed only (default); split: data split only; both: total noise")
    args = ap.parse_args()
    if args.groups == "gender" and args.dataset == "gowalla":
        raise SystemExit("Gowalla has no gender attribute")
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    tag = f"noise_{args.model}_{args.protocol}_{args.groups}" + ("" if args.vary == "train" else f"_vary{args.vary}")
    with Run(cfg, tag) as run:
        out = main(args.dataset, cfg, args.protocol, args.groups,
                   [int(x) for x in args.seeds.split(",")], [int(x) for x in args.ks.split(",")], args.model, args.vary)
        run.path("noise.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")