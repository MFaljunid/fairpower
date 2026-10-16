"""Recommenders (NumPy / SciPy) and full-ranking evaluation of one held-out item per user."""
import numpy as np
import scipy.sparse as sp


def interaction_matrix(lists, n_items):
    """Dense 0/1 matrix (small datasets only; see sparse_interactions for large ones)."""
    X = np.zeros((len(lists), n_items))
    for u, it in enumerate(lists):
        X[u, it] = 1.0
    return X


def sparse_interactions(lists, n_items):
    rows = np.repeat(np.arange(len(lists)), [len(t) for t in lists])
    cols = np.concatenate([np.asarray(t, int) for t in lists]) if len(lists) else np.array([], int)
    return sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(lists), n_items))


def item_popularity(lists, n_items):
    return np.bincount(np.concatenate([np.asarray(t, int) for t in lists]), minlength=n_items).astype(float)


class ItemPop:
    def fit(self, lists, n_items):
        self.pop = item_popularity(lists, n_items)
        return self

    def scores(self, users):
        return np.tile(self.pop, (len(users), 1))


class IALS:
    """Implicit ALS (Hu et al. 2008): conf 1 + alpha for observed items, preference 1; all items.
    Optional user weights w_u (fairness reweighting): objective sum_u w_u sum_i c_ui (p_ui - x_u.y_i)^2
    + reg (||X||^2 + ||Y||^2). w = 1 for every user is the standard model."""

    def __init__(self, dim=64, alpha=5.0, reg=10.0, iters=10, seed=0, user_weights=None):
        self.dim, self.alpha, self.reg, self.iters, self.seed = dim, alpha, reg, iters, seed
        self.user_weights = user_weights

    def fit(self, lists, n_items):
        rng = np.random.default_rng(self.seed)
        n_users = len(lists)
        w = np.ones(n_users) if self.user_weights is None else np.asarray(self.user_weights, float)
        self.U = rng.normal(0, 0.01, (n_users, self.dim))
        self.V = rng.normal(0, 0.01, (n_items, self.dim))
        by_item = [[] for _ in range(n_items)]
        for u, it in enumerate(lists):
            for i in it:
                by_item[i].append(u)
        by_item = [np.array(b, int) for b in by_item]
        I = np.eye(self.dim)
        for _ in range(self.iters):
            G = self.V.T @ self.V
            for u, idx in enumerate(lists):          # user step: weight acts as reg / w_u
                if len(idx) == 0:
                    continue
                F = self.V[idx]
                self.U[u] = np.linalg.solve(w[u] * (G + self.alpha * F.T @ F) + self.reg * I,
                                            w[u] * (1 + self.alpha) * F.sum(axis=0))
            Gw = (self.U * w[:, None]).T @ self.U   # item step: weighted over all users
            for i, idx in enumerate(by_item):
                if len(idx) == 0:
                    self.V[i] = 0.0
                    continue
                F, wi = self.U[idx], w[idx][:, None]
                self.V[i] = np.linalg.solve(Gw + self.alpha * (F * wi).T @ F + self.reg * I,
                                            (1 + self.alpha) * (F * wi).sum(axis=0))
        return self

    def scores(self, users):
        return self.U[users] @ self.V.T


class EASE:
    """EASE (Steck 2019): B = I - P diag(1/diag(P)), P = (X^T W X + lam I)^{-1}, zero diagonal.
    Optional user weights W = diag(w_u) (fairness reweighting); w = 1 is the standard model."""

    def __init__(self, lam=200.0, user_weights=None):
        self.lam, self.user_weights = lam, user_weights

    def fit(self, lists, n_items):
        self.X = sparse_interactions(lists, n_items)
        w = np.ones(len(lists)) if self.user_weights is None else np.asarray(self.user_weights, float)
        G = (self.X.T @ sp.diags(w) @ self.X).toarray()
        P = np.linalg.inv(G + self.lam * np.eye(n_items))
        self.B = -P / np.diag(P)[None, :]
        np.fill_diagonal(self.B, 0.0)
        return self

    def scores(self, users):
        return np.asarray(self.X[users] @ self.B)


class ItemKNN:
    """Item-based kNN with cosine similarity (Sarwar et al. 2001; strong baseline per Ferrari Dacrema
    et al. 2019). Scalable (sparse). Optional user weights: co-occurrences weighted by w_u."""

    def __init__(self, k=100, shrink=0.0, user_weights=None):
        self.k, self.shrink, self.user_weights = int(k), float(shrink), user_weights

    def fit(self, lists, n_items):
        self.X = sparse_interactions(lists, n_items)
        w = np.ones(len(lists)) if self.user_weights is None else np.asarray(self.user_weights, float)
        C = (self.X.T @ sp.diags(w) @ self.X).tocsr()
        norms = np.sqrt(np.asarray(C.diagonal()))
        C.setdiag(0.0)
        C.eliminate_zeros()
        rows, cols, vals = [], [], []
        for i in range(n_items):
            a, b = C.indptr[i], C.indptr[i + 1]
            if a == b:
                continue
            j = C.indices[a:b]
            v = C.data[a:b] / (norms[i] * norms[j] + self.shrink + 1e-12)
            if len(v) > self.k:
                top = np.argpartition(-v, self.k)[:self.k]
                j, v = j[top], v[top]
            rows.append(np.full(len(j), i)); cols.append(j); vals.append(v)
        rows = np.concatenate(rows) if rows else np.array([], int)
        cols = np.concatenate(cols) if cols else np.array([], int)
        vals = np.concatenate(vals) if vals else np.array([])
        # score(u, j) = sum over the user's items i of sim(i, j) for j among i's neighbours
        self.S = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
        return self

    def scores(self, users):
        return (self.X[users] @ self.S).toarray()


def rank_of_target(scores, target, exclude):
    """0-based rank of `target` among items not in `exclude` (ties counted pessimistically: items with
    equal score other than the target rank ahead of it)."""
    s = scores.copy()
    s[np.asarray(exclude, int)] = -np.inf
    t = s[target]
    return int((s > t).sum() + (s == t).sum() - 1)


def evaluate(model, split, which="test", k=10, chunk=512):
    """Per-user HR@k and NDCG@k for the held-out item. Exclusions: test -> train + val item;
    val -> train + test item (known positives are never counted as misses)."""
    n = split.n_users
    hr, nd = np.zeros(n), np.zeros(n)
    target = split.test if which == "test" else split.val
    other = split.val if which == "test" else split.test
    for s in range(0, n, chunk):
        users = np.arange(s, min(s + chunk, n))
        S = model.scores(users)
        for j, u in enumerate(users):
            r = rank_of_target(S[j], target[u], np.append(split.train[u], other[u]))
            if r < k:
                hr[u] = 1.0
                nd[u] = 1.0 / np.log2(r + 2)
    return hr, nd


class BPRMF:
    """BPR matrix factorisation (Rendle et al. 2009), NumPy SGD, one uniform negative per positive.

    Optional in-processing fairness regulariser (gap penalty on the training loss of two groups):
        L = mean_s BPR_s + gap_lambda * | L_a - L_b |,   L_g = mean BPR loss of group g's samples
    computed per mini-batch. Its gradient equals the BPR gradient with per-sample weights
        w_s = 1 + gap_lambda * sign(L_a - L_b) * B * ([s in a]/n_a - [s in b]/n_b)
    (B = batch size, n_g = group samples in the batch): the higher-loss group's samples are up-weighted and
    the lower-loss group's samples DOWN-weighted, possibly below zero (gradient ascent on its loss).
    The sign uses exponential moving averages of the two group losses (less noisy than one batch).
    gap_groups = (labels per user, group a, group b)."""

    def __init__(self, dim=64, lr=0.05, reg=1e-4, epochs=20, batch=4096, seed=0,
                 gap_lambda=0.0, gap_groups=None):
        self.dim, self.lr, self.reg, self.epochs, self.batch, self.seed = dim, lr, reg, int(epochs), int(batch), seed
        self.gap_lambda, self.gap_groups = float(gap_lambda), gap_groups

    def fit(self, lists, n_items):
        rng = np.random.default_rng(self.seed)
        n_users = len(lists)
        self.U = rng.normal(0, 0.1, (n_users, self.dim))
        self.V = rng.normal(0, 0.1, (n_items, self.dim))
        users = np.repeat(np.arange(n_users), [len(t) for t in lists])
        items = np.concatenate([np.asarray(t, int) for t in lists])
        if self.gap_lambda > 0:
            labels, ga, gb = self.gap_groups
            labels = np.asarray(labels)
            ema_a = ema_b = None
        for _ in range(self.epochs):
            order = rng.permutation(len(users))
            for s in range(0, len(order), self.batch):
                b = order[s:s + self.batch]
                u, i = users[b], items[b]
                j = rng.integers(0, n_items, len(b))
                P, Qi, Qj = self.U[u], self.V[i], self.V[j]
                x = np.einsum("ij,ij->i", P, Qi - Qj)
                g = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))          # sigma(-x) = -dloss/dx
                w = np.ones(len(b))
                if self.gap_lambda > 0:
                    lab = labels[u]
                    ma, mb = lab == ga, lab == gb
                    if ma.any() and mb.any():
                        loss = np.logaddexp(0.0, -x)                     # -log sigma(x)
                        la, lb = loss[ma].mean(), loss[mb].mean()
                        ema_a = la if ema_a is None else 0.9 * ema_a + 0.1 * la
                        ema_b = lb if ema_b is None else 0.9 * ema_b + 0.1 * lb
                        sgn = np.sign(ema_a - ema_b)
                        w = w + self.gap_lambda * sgn * len(b) * (ma / ma.sum() - mb / mb.sum())
                coef = (w * g)[:, None]
                dU = coef * (Qi - Qj) - self.reg * P
                dVi = coef * P - self.reg * Qi
                dVj = -coef * P - self.reg * Qj
                np.add.at(self.U, u, self.lr * dU)
                np.add.at(self.V, i, self.lr * dVi)
                np.add.at(self.V, j, self.lr * dVj)
        return self

    def scores(self, users):
        return self.U[users] @ self.V.T


def evaluate_sampled(model, split, which="test", k=10, n_neg=99, seed=0, chunk=512):
    """Sampled protocol: the held-out item is ranked against n_neg random items the user has not
    interacted with (train, val or test). Ties pessimistic. Returns per-user HR@k, NDCG@k."""
    rng = np.random.default_rng(seed)
    n = split.n_users
    hr, nd = np.zeros(n), np.zeros(n)
    target = split.test if which == "test" else split.val
    for s in range(0, n, chunk):
        users = np.arange(s, min(s + chunk, n))
        S = model.scores(users)
        for j, u in enumerate(users):
            seen = set(split.train[u].tolist()) | {int(split.val[u]), int(split.test[u])}
            negs = []
            while len(negs) < n_neg:
                cand = rng.integers(0, split.n_items, 2 * n_neg)
                negs.extend(int(c) for c in cand if int(c) not in seen)
            negs = np.array(negs[:n_neg])
            t = S[j, target[u]]
            r = int((S[j, negs] >= t).sum())
            if r < k:
                hr[u] = 1.0
                nd[u] = 1.0 / np.log2(r + 2)
    return hr, nd