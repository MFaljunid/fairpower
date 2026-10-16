"""Statistical power of user-group gaps in offline recommender evaluation.

All functions take per-user metric values y (e.g. NDCG@10) and integer group labels g, and compare
group b with group a (gap = mean_b - mean_a, the sign convention of leveldown).

mde            : minimum detectable gap (two-sided test at level alpha with the given power), normal
                 approximation with the two groups' sample variances (Welch).
gap_test       : observed gap, stratified bootstrap 95% CI and two-sided permutation p-value.
power_curve    : for each group size n, the share of bootstrap samples (n users per group, drawn with
                 replacement from each group) in which the 95% CI of the gap excludes zero, i.e. the
                 empirical power to detect the gap observed on the full data, plus the analytic MDE at n.
planted_power  : power to detect a planted gap delta of known size, keeping the zero-inflated shape of
                 top-K metrics (see the function).
"""
import numpy as np
from scipy.stats import norm


def _two(y, g, a, b):
    y, g = np.asarray(y, float), np.asarray(g)
    return y[g == a], y[g == b]


def mde(y, g, a, b, alpha=0.05, power=0.8, n_a=None, n_b=None):
    ya, yb = _two(y, g, a, b)
    na = len(ya) if n_a is None else n_a
    nb = len(yb) if n_b is None else n_b
    se = np.sqrt(ya.var(ddof=1) / na + yb.var(ddof=1) / nb)
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def gap_test(y, g, a, b, B=1000, seed=0):
    ya, yb = _two(y, g, a, b)
    gap = yb.mean() - ya.mean()
    rng = np.random.default_rng(seed)
    bs = np.array([yb[rng.integers(0, len(yb), len(yb))].mean() - ya[rng.integers(0, len(ya), len(ya))].mean()
                   for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pooled = np.concatenate([ya, yb])
    na = len(ya)
    perm = np.empty(B)
    for i in range(B):
        p = rng.permutation(pooled)
        perm[i] = p[na:].mean() - p[:na].mean()
    pval = (1 + (np.abs(perm) >= abs(gap)).sum()) / (B + 1)
    return {"gap": float(gap), "ci": [float(lo), float(hi)], "p": float(pval),
            "n_a": int(na), "n_b": int(len(yb)), "mean_a": float(ya.mean()), "mean_b": float(yb.mean())}


def _ci_excludes_zero(xa, xb):
    d = xb.mean() - xa.mean()
    se = np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
    return abs(d) > 1.959964 * se


def power_curve(y, g, a, b, sizes, reps=500, seed=0):
    ya, yb = _two(y, g, a, b)
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        hits = sum(_ci_excludes_zero(rng.choice(ya, n, replace=True), rng.choice(yb, n, replace=True))
                   for _ in range(reps))
        out.append({"n": int(n), "power": hits / reps, "mde": mde(y, g, a, b, n_a=n, n_b=n)})
    return out


def planted_power(y, g, a, n, delta, reps=500, seed=0):
    """Empirical power to detect a planted gap of exactly `delta` (in expectation) with n users per group.
    Both samples are drawn from group a's observed values. In the second sample each zero value is
    replaced, with probability q, by a value drawn from the positive values of group a, where
    q = delta / (share of zeros * mean positive value). This keeps the zero-inflated shape of top-K
    metrics (a location shift would not). Returns (power, q); power is None if delta is not reachable."""
    ya = np.asarray(y, float)[np.asarray(g) == a]
    pos = ya[ya > 0]
    p0 = (ya == 0).mean()
    if len(pos) == 0 or p0 == 0:
        return None, None
    q = delta / (p0 * pos.mean())
    if q > 1:
        return None, float(q)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        xa = rng.choice(ya, n, replace=True)
        xb = rng.choice(ya, n, replace=True)
        flip = (xb == 0) & (rng.random(n) < q)
        xb[flip] = rng.choice(pos, int(flip.sum()), replace=True)
        hits += _ci_excludes_zero(xa, xb)
    return hits / reps, float(q)
