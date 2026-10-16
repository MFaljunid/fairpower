"""MovieLens loaders and the temporal leave-one-out protocol.

Every rating is an implicit positive (as in NCF and fair-fed-ncf). Per user, interactions are
sorted by timestamp: last -> test item, second last -> validation item, rest -> train.
Timestamp ties: tie_break="item" (sort by item id; original L1 runs) or "random" (seeded random order
within equal timestamps, so the held-out item is not systematically the highest item id).
Users with fewer than `min_interactions` are dropped.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_ratings(name, raw_dir):
    raw = _path(raw_dir)
    if name == "ml-100k":
        f = raw / "u.data"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found (MovieLens-100K u.data)")
        df = pd.read_csv(f, sep="\t", names=["user", "item", "rating", "ts"])
    elif name == "ml-1m":
        f = raw / "ratings.dat"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found (MovieLens-1M ratings.dat)")
        df = pd.read_csv(f, sep="::", engine="python", names=["user", "item", "rating", "ts"])
    elif name == "gowalla":
        cands = [raw / "loc-gowalla_totalCheckins.txt", raw / "loc-gowalla_totalCheckins.txt.gz"]
        f = next((c for c in cands if c.exists()), None)
        if f is None:
            raise FileNotFoundError(f"{cands[0]} (.gz) not found (SNAP Gowalla check-ins)")
        df = pd.read_csv(f, sep="\t", header=None, names=["user", "time", "lat", "lon", "item"],
                         usecols=["user", "time", "item"])
        t = pd.to_datetime(df["time"], utc=True)
        df["ts"] = ((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")
        df = df.drop(columns="time")
        df = df.sort_values("ts").drop_duplicates(["user", "item"], keep="first")   # first visit only
        df["rating"] = 1
    else:
        raise ValueError(f"unknown dataset {name}")
    return df


def k_core(df, min_user, min_item):
    """Iteratively keep users with >= min_user and items with >= min_item interactions."""
    while True:
        uc = df.groupby("user").size()
        ic = df.groupby("item").size()
        keep = df["user"].isin(uc[uc >= min_user].index) & df["item"].isin(ic[ic >= min_item].index)
        if keep.all():
            return df
        df = df[keep]


@dataclass
class Split:
    n_users: int
    n_items: int
    train: list        # train[u] = item array
    val: np.ndarray    # val[u] = item id
    test: np.ndarray   # test[u] = item id
    users: np.ndarray = None   # users[u] = original user id (for joining user attributes)


def temporal_loo(df, min_interactions=5, tie_break="item", seed=42):
    counts = df.groupby("user").size()
    df = df[df["user"].isin(counts[counts >= min_interactions].index)]
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
        train.append(items[:-2])
        val.append(items[-2])
        test.append(items[-1])
    users = np.array(sorted(umap, key=umap.get))
    return Split(len(umap), len(imap), train, np.array(val), np.array(test), users)


def load(name, raw_dir, min_interactions=5, tie_break="item", seed=42, core=None):
    df = read_ratings(name, raw_dir)
    if core:
        df = k_core(df, core, core)
    return temporal_loo(df, min_interactions, tie_break, seed)


def tie_stats(df, min_interactions=5):
    """Per user: number of interactions sharing the LAST timestamp (1 = no tie)."""
    counts = df.groupby("user").size()
    df = df[df["user"].isin(counts[counts >= min_interactions].index)]
    last = df.groupby("user")["ts"].transform("max")
    return df[df["ts"] == last].groupby("user").size()


def activity_groups(train, n_groups=4):
    """Quartiles of #train items, rank-based so groups are equal-sized (0 = least active)."""
    n = np.array([len(t) for t in train])
    ranks = pd.Series(n).rank(method="first").to_numpy()
    return np.minimum((ranks - 1) * n_groups // len(n), n_groups - 1).astype(int), n


def read_gender(name, raw_dir):
    """{original user id: 'M' | 'F'} from u.user (ML-100K) or users.dat (ML-1M)."""
    raw = _path(raw_dir)
    if name == "ml-100k":
        f = raw / "u.user"
        df = pd.read_csv(f, sep="|", names=["user", "age", "gender", "occupation", "zip"])
    elif name == "ml-1m":
        f = raw / "users.dat"
        df = pd.read_csv(f, sep="::", engine="python", names=["user", "gender", "age", "occupation", "zip"])
    else:
        raise ValueError(f"no gender file for {name}")
    if not f.exists():
        raise FileNotFoundError(f"{f} not found")
    return dict(zip(df["user"].astype(int), df["gender"].astype(str)))


def group_labels(split, kind, name=None, raw_dir=None):
    """kind='activity': quartiles of #train (0 = least active); kind='gender': 0 = F, 1 = M.
    Returns (labels array, list of group names)."""
    if kind == "activity":
        g, _ = activity_groups(split.train)
        return g, [f"act{q}" for q in range(4)]
    if kind == "gender":
        gmap = read_gender(name, raw_dir)
        lab = np.array([1 if gmap[int(u)] == "M" else 0 for u in split.users])
        return lab, ["F", "M"]
    raise ValueError(f"unknown grouping {kind}")