"""RQ3 / RQ4: is the setting of a fairness intervention chosen on validation also the best on test, and
can the change in the gap it produces be detected at all?

    py scripts/run_selection.py --dataset ml-100k
    py scripts/run_selection.py --dataset ml-1m --holdout 1,3 --groups gender

Intervention: user reweighting of the disadvantaged group in iALS (as in the leveldown audit),
w in config selection.weights (w = 1 is the base model). Groups: activity quartiles (advantaged and
disadvantaged = best and worst quartile of the base model on VALIDATION) or gender.
Selection rule (leveldown): smallest |validation gap| among settings whose overall validation NDCG@10 is
>= 95% of the base model's. The same rule applied to TEST gives the test-optimal setting.

Reported for each m (held-out items per user; same users for every m):
  agreement   : does the validation choice equal the test choice? Observed, and the share of B user
                bootstrap resamples (users drawn jointly for validation and test) in which they agree;
  regret      : test |gap| of the validation choice minus test |gap| of the test choice;
  rank corr   : Spearman correlation between validation and test gaps across settings;
  gap change  : for every w, the paired change of the test gap against the base model,
                d_u = NDCG_w(u) - NDCG_base(u), change = mean_dis(d) - mean_adv(d), its bootstrap 95% CI
                and the paired MDE of the change (alpha 0.05, power 0.8).
Outputs: results/<dataset>/<timestamp>_selection_<groups>/ (output.txt, config.yaml, selection.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairpower.data import ROOT, activity_groups, k_core, read_gender, read_ratings  # noqa: E402
from fairpower.models import IALS  # noqa: E402
from fairpower.power import gap_test, mde  # noqa: E402
from fairpower.runlog import Run  # noqa: E402
from fairpower.splits import evaluate_multi, temporal_multi  # noqa: E402
from run_power import tune  # noqa: E402


def select(overall, gaps, budget=0.95):
    """Index of the setting with the smallest |gap| among those with overall >= budget * overall[0]."""
    ok = [i for i in range(len(overall)) if overall[i] >= budget * overall[0]]
    return min(ok, key=lambda i: abs(gaps[i]))


def stats(V, T, adv, dis, users=None):
    """Per setting: overall and gap (dis - adv) on validation and test, optionally on a user resample."""
    if users is not None:
        V, T, adv, dis = V[:, users], T[:, users], adv[users], dis[users]
    vo, to = V.mean(axis=1), T.mean(axis=1)
    vg = V[:, dis].mean(axis=1) - V[:, adv].mean(axis=1)
    tg = T[:, dis].mean(axis=1) - T[:, adv].mean(axis=1)
    return vo, vg, to, tg


def main(dataset, cfg, holdouts, grouping, log=print):
    k, B, seed = cfg["eval"]["k"], cfg["eval"]["bootstrap"], cfg["seed"]
    weights = [float(w) for w in cfg["selection"]["weights"]]
    d = cfg["datasets"][dataset]
    df = read_ratings(dataset, d["raw_dir"])
    if d.get("core"):
        df = k_core(df, d["core"], d["core"])
    need = 2 * max(holdouts) + 3
    cnt = df.groupby("user").size()
    df = df[df["user"].isin(cnt[cnt >= need].index)]
    log(f"{dataset}: groups={grouping}, iALS reweighting w in {weights}, same users for every m "
        f"(>= {need} interactions); FULL RANKING, NDCG@{k}")
    res = {}
    for m in holdouts:
        split = temporal_multi(df, n_holdout=m, min_train=3, tie_break=cfg.get("tie_break", "random"), seed=seed)
        log(f"\n######## m = {m}: users={split.n_users} items={split.n_items}")
        base, hp = tune("ials", split, cfg, k, log)
        v0, _ = evaluate_multi(base, split, "val", k)
        if grouping == "activity":
            g, _ = activity_groups(split.train)
            gv = [v0[g == q].mean() for q in range(4)]
            a, b = int(np.argmax(gv)), int(np.argmin(gv))
        else:
            gmap = read_gender(dataset, d["raw_dir"])
            g = np.array([1 if gmap[int(u)] == "M" else 0 for u in split.users])
            gv = [v0[g == q].mean() for q in range(2)]
            a, b = int(np.argmax(gv)), int(np.argmin(gv))
        adv, dis = g == a, g == b
        log(f"  advantaged group {a}, disadvantaged group {b} (validation means {[round(float(x), 4) for x in gv]})")
        V, T = [], []
        for w in weights:
            mdl = base if w == 1.0 else IALS(**hp, user_weights=np.where(dis, w, 1.0)).fit(split.train, split.n_items)
            V.append(evaluate_multi(mdl, split, "val", k)[0])
            T.append(evaluate_multi(mdl, split, "test", k)[0])
        V, T = np.array(V), np.array(T)
        vo, vg, to, tg = stats(V, T, adv, dis)
        sv, st = select(vo, vg), select(to, tg)
        rho = spearmanr(vg, tg).correlation
        log(f"  {'w':>5}{'val all':>9}{'val gap':>9}{'test all':>9}{'test gap':>9}"
            f"{'change':>9}{'95% CI':>20}{'MDE chg':>9}{'|chg|/MDE':>10}")
        rows = []
        for i, w in enumerate(weights):
            dd = T[i] - T[0]
            gt = gap_test(dd, g, a, b, B, seed)
            md = mde(dd, g, a, b) if i > 0 else float("nan")
            ratio = abs(gt["gap"]) / md if i > 0 and md > 0 else float("nan")
            log(f"  {w:>5g}{vo[i]:>9.4f}{vg[i]:>+9.4f}{to[i]:>9.4f}{tg[i]:>+9.4f}{gt['gap']:>+9.4f}"
                f"  [{gt['ci'][0]:+.4f}, {gt['ci'][1]:+.4f}]{md:>9.4f}{ratio:>10.2f}")
            rows.append({"w": w, "val_all": float(vo[i]), "val_gap": float(vg[i]), "test_all": float(to[i]),
                         "test_gap": float(tg[i]), "gap_change": gt["gap"], "gap_change_ci": gt["ci"],
                         "mde_change": md, "ratio": ratio})
        rng = np.random.default_rng(seed)
        agree, regret = 0, []
        for _ in range(B):
            u = rng.integers(0, split.n_users, split.n_users)
            bvo, bvg, bto, btg = stats(V, T, adv, dis, u)
            i, j = select(bvo, bvg), select(bto, btg)
            agree += i == j
            regret.append(abs(btg[i]) - abs(btg[j]))
        log(f"  validation choice w={weights[sv]:g}, test choice w={weights[st]:g} -> "
            f"{'AGREE' if sv == st else 'DISAGREE'}; regret {abs(tg[sv]) - abs(tg[st]):+.4f}; "
            f"Spearman(val gap, test gap) {rho:+.2f}")
        log(f"  bootstrap over users ({B}): agreement {agree / B:.0%}, mean regret {np.mean(regret):+.4f} "
            f"(95th pct {np.percentile(regret, 95):+.4f})")
        res[str(m)] = {"hp": hp, "adv": a, "dis": b, "rows": rows, "val_choice": weights[sv],
                       "test_choice": weights[st], "spearman": float(rho), "boot_agreement": agree / B,
                       "boot_regret_mean": float(np.mean(regret)), "boot_regret_p95": float(np.percentile(regret, 95))}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m", "gowalla"])
    ap.add_argument("--holdout", default="1,3")
    ap.add_argument("--groups", default="activity", choices=["activity", "gender"])
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    if args.groups == "gender" and args.dataset == "gowalla":
        raise SystemExit("Gowalla has no gender attribute")
    hs = [int(x) for x in args.holdout.split(",")]
    with Run(cfg, f"selection_{args.groups}") as run:
        out = main(args.dataset, cfg, hs, args.groups)
        run.path("selection.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
