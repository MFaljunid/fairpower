import numpy as np
import pandas as pd

from fairpower.data import temporal_loo
from fairpower.models import IALS, evaluate
from fairpower.power import gap_test, mde, planted_power, power_curve
from fairpower.splits import evaluate_multi, temporal_multi


def synth_df(n_users=150, n_items=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        n = int(rng.integers(12, 60))
        for t, it in enumerate(rng.choice(n_items, n, replace=False)):
            rows.append((u + 1, int(it) + 1, 4, int(t // 2)))      # pairs share a timestamp -> ties
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


def test_one_item_split_equals_leveldown_loo():
    df = synth_df()
    a = temporal_loo(df, min_interactions=5, tie_break="random", seed=42)
    b = temporal_multi(df, n_holdout=1, min_train=3, tie_break="random", seed=42)
    assert a.n_users == b.n_users and a.n_items == b.n_items
    for u in range(a.n_users):
        assert np.array_equal(a.train[u], b.train[u])
        assert a.val[u] == b.val[u][0] and a.test[u] == b.test[u][0]


def test_multi_split_is_temporal_and_disjoint():
    df = synth_df()
    s = temporal_multi(df, n_holdout=3)
    for u in range(s.n_users):
        assert len(s.val[u]) == 3 and len(s.test[u]) == 3 and len(s.train[u]) >= 3
        assert not set(s.train[u]) & set(s.val[u]) and not set(s.val[u]) & set(s.test[u])


def test_one_item_ndcg_equals_leveldown_evaluate():
    df = synth_df()
    a = temporal_loo(df, min_interactions=5, tie_break="random", seed=42)
    b = temporal_multi(df, n_holdout=1, tie_break="random", seed=42)
    m = IALS(dim=8, alpha=5.0, reg=10.0, iters=3).fit(a.train, a.n_items)
    for which in ("val", "test"):
        _, nd_a = evaluate(m, a, which, 10)
        nd_b, _ = evaluate_multi(m, b, which, 10)
        assert np.allclose(nd_a, nd_b)


def test_ties_are_pessimistic():
    class Const:
        def scores(self, users):
            return np.zeros((len(users), 20))
    from fairpower.splits import MultiSplit
    s = MultiSplit(1, 20, [np.array([0, 1])], [np.array([2])], [np.array([3])])
    nd, rc = evaluate_multi(Const(), s, "test", k=10)
    assert nd[0] == 0.0 and rc[0] == 0.0         # all scores equal -> relevant item ranked last


def test_mde_matches_formula_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    y = rng.random(2000)
    g = np.repeat([0, 1], 1000)
    se = np.sqrt(y[:1000].var(ddof=1) / 1000 + y[1000:].var(ddof=1) / 1000)
    assert np.isclose(mde(y, g, 0, 1), (1.959964 + 0.841621) * se, rtol=1e-4)
    assert mde(y, g, 0, 1, n_a=4000, n_b=4000) < mde(y, g, 0, 1)


def test_gap_test_null_and_real_gap():
    rng = np.random.default_rng(1)
    y0 = rng.random(4000)
    g = np.repeat([0, 1], 2000)
    r = gap_test(y0, g, 0, 1, B=500)
    assert r["ci"][0] < 0 < r["ci"][1] and r["p"] > 0.05
    y1 = y0 + 0.1 * (g == 1)
    r = gap_test(y1, g, 0, 1, B=500)
    assert r["ci"][0] > 0 and r["p"] < 0.01 and np.isclose(r["gap"], y1[2000:].mean() - y1[:2000].mean())


def test_power_curve_increases_with_n():
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.random(3000), rng.random(3000) + 0.03])
    g = np.repeat([0, 1], 3000)
    pc = power_curve(y, g, 0, 1, [100, 3000], reps=300)
    assert pc[0]["power"] < pc[1]["power"] and pc[1]["power"] > 0.8


def test_planted_power_size_and_power():
    rng = np.random.default_rng(3)
    y = np.where(rng.random(5000) < 0.1, rng.random(5000), 0.0)      # zero-inflated like NDCG@10
    g = np.zeros(5000, int)
    p_null, _ = planted_power(y, g, 0, 1000, 0.0, reps=400)
    p_big, q = planted_power(y, g, 0, 1000, 0.03, reps=400)
    assert p_null < 0.10                     # close to the nominal 5% false positive rate
    assert p_big > 0.8 and 0 < q <= 1
    assert planted_power(y, g, 0, 1000, 0.5, reps=10)[0] is None     # not reachable


def test_random_split_partitions_all_interactions():
    from fairpower.splits import random_split
    df = synth_df()
    s = random_split(df, (0.8, 0.1, 0.1), seed=1)
    tot = sum(len(s.train[u]) + len(s.val[u]) + len(s.test[u]) for u in range(s.n_users))
    assert tot == len(df)
    frac = sum(len(t) for t in s.test) / tot
    assert 0.07 < frac < 0.13
    for u in range(s.n_users):
        assert len(s.train[u]) >= 1
        assert not set(s.train[u]) & set(s.test[u]) and not set(s.val[u]) & set(s.test[u])


def test_users_without_test_items_are_nan():
    from fairpower.splits import MultiSplit
    class Const:
        def scores(self, users):
            return np.arange(20, dtype=float)[None, :].repeat(len(users), 0)
    s = MultiSplit(2, 20, [np.array([0]), np.array([1])], [np.array([], int), np.array([2])],
                   [np.array([19]), np.array([], int)])
    nd, rc = evaluate_multi(Const(), s, "test", k=10)
    assert nd[0] == 1.0 and np.isnan(nd[1])


def test_temporal_ratio_holds_out_recent_fraction():
    from fairpower.splits import temporal_ratio
    df = synth_df()
    s = temporal_ratio(df, 0.2, 0.1, seed=42)
    sizes = df.groupby("user").size().sort_index().to_numpy()
    for u in range(s.n_users):
        n = len(s.train[u]) + len(s.val[u]) + len(s.test[u])
        assert len(s.test[u]) == max(1, int(round(0.2 * n)))
        assert not set(s.train[u]) & set(s.test[u]) and not set(s.val[u]) & set(s.test[u])
    assert len(s.test) == len(sizes)
