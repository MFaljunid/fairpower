"""Temporal hold-out with m items per user, and full-ranking evaluation with several relevant items.

temporal_multi(df, n_holdout=m): per user, interactions sorted by time (ties broken exactly as in
data.temporal_loo); the last m -> test, the m before -> validation, the rest -> train. Users need at
least 2m + min_train interactions. With m = 1 the split is identical to data.temporal_loo.

evaluate_multi: per-user NDCG@k and Recall@k over the held-out items, full catalog ranking, training
items and the other held-out set excluded, ties broken pessimistically (a relevant item is placed after
non-relevant items with the same score). With one held-out item it equals models.evaluate.
Users with no held-out item in the evaluated set (possible under random_split) get NaN.

temporal_ratio(df, test_frac=0.2, val_frac=0.1): per user, the most recent test_frac of the
interactions -> test, the most recent val_frac of the remaining ones -> validation (the protocol of
Boratto et al., ECIR 2022); ties broken as in temporal_multi.

random_split(df, ratios=(0.8, 0.1, 0.1)): the protocol of many fairness papers, a random split of
ALL interactions (not per user) into train / validation / test; returns a MultiSplit whose val[u] and
test[u] may hold any number of items, including none.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MultiSplit:
    n_users: int
    n_items: int
    train: list        # train[u] = item array (time order)
    val: list          # val[u] = item array (m items)
    test: list         # test[u] = item array (m items)
    users: np.ndarray = None


def temporal_multi(df, n_holdout=1, min_train=3, tie_break="random", seed=42):
    m = int(n_holdout)
    need = 2 * m + min_train
    counts = df.groupby("user").size()
    df = df[df["user"].isin(counts[counts >= need].index)]
    umap = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    imap = {it: j for j, it in enumerate(sorted(df["item"].unique()))}
    df = df.assign(u=df["user"].map(umap), i=df["item"].map(imap))
    if tie_break == "item":
        df = df.sort_values(["u", "ts", "i"])
    elif tie_break == "random":
        rnd = np.random.default_rng(seed).random(len(df))
        df = df.assign(_r=rnd).sort_values(["u", "ts", "_r"]).drop(columns="_r")
    else:
        raise ValueError(f"unknown tie_break {tie_break}")
    train, val, test = [], [], []
    for _, g in df.groupby("u", sort=True):
        items = g["i"].to_numpy(int)
        train.append(items[:-2 * m])
        val.append(items[-2 * m:-m])
        test.append(items[-m:])
    users = np.array(sorted(umap, key=umap.get))
    return MultiSplit(len(umap), len(imap), train, val, test, users)


def evaluate_multi(model, split, which="test", k=10, chunk=512):
    """Per-user NDCG@k and Recall@k (Recall = hits / min(m, k)). Returns (ndcg, recall) arrays."""
    n = split.n_users
    nd, rc = np.zeros(n), np.zeros(n)
    target = split.test if which == "test" else split.val
    other = split.val if which == "test" else split.test
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for s in range(0, n, chunk):
        users = np.arange(s, min(s + chunk, n))
        S = model.scores(users)
        for j, u in enumerate(users):
            sc = S[j].copy()
            sc[np.asarray(split.train[u], int)] = -np.inf
            sc[np.asarray(other[u], int)] = -np.inf
            rel = np.zeros(len(sc), int)
            rel[np.asarray(target[u], int)] = 1
            order = np.lexsort((rel, -sc))[:k]          # by score desc, relevant after equal-score items
            hits = rel[order]
            m = len(target[u])
            if m == 0:
                nd[u] = rc[u] = np.nan
                continue
            ideal = disc[:min(m, k)].sum()
            nd[u] = (hits * disc[:len(hits)]).sum() / ideal
            rc[u] = hits.sum() / min(m, k)
    return nd, rc


def random_split(df, ratios=(0.8, 0.1, 0.1), min_train=1, seed=42):
    """Random split of all interactions into train / validation / test with the given ratios.
    Users with fewer than min_train training interactions are dropped."""
    rng = np.random.default_rng(seed)
    r = rng.random(len(df))
    part = np.where(r < ratios[0], 0, np.where(r < ratios[0] + ratios[1], 1, 2))
    df = df.assign(_p=part)
    keep = df[df["_p"] == 0].groupby("user").size()
    df = df[df["user"].isin(keep[keep >= min_train].index)]
    umap = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    imap = {it: j for j, it in enumerate(sorted(df["item"].unique()))}
    df = df.assign(u=df["user"].map(umap), i=df["item"].map(imap))
    n = len(umap)
    train, val, test = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for u, i, p in zip(df["u"].to_numpy(), df["i"].to_numpy(), df["_p"].to_numpy()):
        (train, val, test)[p][u].append(i)
    conv = lambda L: [np.array(x, int) for x in L]
    users = np.array(sorted(umap, key=umap.get))
    return MultiSplit(n, len(imap), conv(train), conv(val), conv(test), users)


def temporal_ratio(df, test_frac=0.2, val_frac=0.1, min_train=3, tie_break="random", seed=42):
    """Per-user temporal hold-out of a fraction of each user's interactions (at least one item each)."""
    umap = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    imap = {it: j for j, it in enumerate(sorted(df["item"].unique()))}
    df = df.assign(u=df["user"].map(umap), i=df["item"].map(imap))
    if tie_break == "item":
        df = df.sort_values(["u", "ts", "i"])
    elif tie_break == "random":
        rnd = np.random.default_rng(seed).random(len(df))
        df = df.assign(_r=rnd).sort_values(["u", "ts", "_r"]).drop(columns="_r")
    else:
        raise ValueError(f"unknown tie_break {tie_break}")
    train, val, test, users = [], [], [], []
    inv = np.array(sorted(umap, key=umap.get))
    for u, g in df.groupby("u", sort=True):
        items = g["i"].to_numpy(int)
        n_test = max(1, int(round(test_frac * len(items))))
        rest = items[:-n_test]
        n_val = max(1, int(round(val_frac * len(rest))))
        if len(rest) - n_val < min_train:
            continue
        train.append(rest[:-n_val]); val.append(rest[-n_val:]); test.append(items[-n_test:]); users.append(inv[u])
    return MultiSplit(len(train), len(imap), train, val, test, np.array(users))