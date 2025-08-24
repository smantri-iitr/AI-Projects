# pip install pandas networkx numpy
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ===========================
# Utilities
# ===========================

def _minmax_scale(d: Dict, clip_zero: bool = True) -> Dict:
    """Min-max scale dict values to [0,1]. If all equal, return zeros."""
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if math.isclose(vmin, vmax):
        return {k: 0.0 for k in d}
    scaled = {k: (float(v) - vmin) / (vmax - vmin) for k, v in d.items()}
    if clip_zero:
        for k in scaled:
            if scaled[k] < 0:
                scaled[k] = 0.0
    return scaled


def _prefixed(node_id: str | int, kind: str) -> str:
    """Create a namespaced node id for bipartite graph."""
    return f"{kind}:{node_id}"


def _largest_eigenvalue_estimate(G: nx.Graph) -> float:
    """
    Estimate largest eigenvalue magnitude of adjacency matrix.
    Tries scipy.sparse eigs; falls back to power iteration on dense matrix.
    """
    # Try sparse if available
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        A = nx.to_scipy_sparse_array(G, format="csr", weight="weight", dtype=float)
        # k=1 largest real part
        try:
            vals = spla.eigs(A, k=1, which="LR", return_eigenvectors=False, maxiter=5000)
        except Exception:
            # fallback: use magnitude largest
            vals = spla.eigs(A, k=1, which="LM", return_eigenvectors=False, maxiter=5000)
        lam = float(np.abs(vals[0]))
        if lam <= 0 or math.isnan(lam) or math.isinf(lam):
            raise ValueError("invalid eigen estimate")
        return lam
    except Exception:
        # Dense power iteration (fine for small/medium graphs)
        A = nx.to_numpy_array(G, dtype=float, weight="weight")
        n = A.shape[0]
        if n == 0:
            return 1.0
        x = np.ones(n) / math.sqrt(n)
        for _ in range(100):
            y = A @ x
            norm = np.linalg.norm(y)
            if norm == 0:
                return 1.0
            x = y / norm
        lam = float(np.linalg.norm(A @ x) / (np.linalg.norm(x) + 1e-12))
        return lam if lam > 0 else 1.0


# ===========================
# Core: Build graph + Katz
# ===========================

def build_bipartite_graph(ratings: pd.DataFrame,
                          user_col: str = "user_id",
                          item_col: str = "movie_id",
                          rating_col: str = "rating") -> nx.Graph:
    """
    Build an undirected bipartite graph G with:
      - user nodes labeled 'u:<user_id>'
      - movie nodes labeled 'm:<movie_id>'
      - edges weighted by rating (or 1.0 if rating column missing)
    """
    if rating_col not in ratings.columns:
        temp = ratings.copy()
        temp[rating_col] = 1.0
        ratings = temp

    G = nx.Graph()
    # Add nodes with bipartite attribute
    users = ratings[user_col].unique().tolist()
    movies = ratings[item_col].unique().tolist()

    G.add_nodes_from((_prefixed(u, "u"), {"bipartite": 0}) for u in users)
    G.add_nodes_from((_prefixed(m, "m"), {"bipartite": 1}) for m in movies)

    # Add edges
    for row in ratings.itertuples(index=False):
        u = _prefixed(getattr(row, user_col), "u")
        m = _prefixed(getattr(row, item_col), "m")
        w = float(getattr(row, rating_col))
        # If multiple rows exist, accumulate weight
        if G.has_edge(u, m):
            G[u][m]["weight"] += w
        else:
            G.add_edge(u, m, weight=w)

    return G


def katz_centrality_for_movies(G: nx.Graph,
                               alpha: float | None = None,
                               beta: float = 1.0,
                               tol: float = 1.0e-6,
                               max_iter: int = 1000) -> Dict[str, float]:
    """
    Compute Katz centrality on the bipartite graph and return scores for movie nodes only.
    If alpha is None, choose alpha = 0.85 / lambda_max where lambda_max is estimated.
    """
    if alpha is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam = _largest_eigenvalue_estimate(G)
        if lam <= 0:
            lam = 1.0
        alpha = 0.85 / lam

    # Use numpy-based solver (fast & stable for moderate graphs)
    kc = nx.katz_centrality_numpy(G, alpha=alpha, beta=beta, weight="weight", normalized=False)
    # Keep only movies
    movie_scores = {n: float(s) for n, s in kc.items() if str(n).startswith("m:")}
    return _minmax_scale(movie_scores)


# ===========================
# Reward Factor (Neighborhood CF)
# ===========================

@dataclass
class RewardConfig:
    similarity: str = "cosine"     # 'cosine' or 'jaccard' (on rated sets)
    min_overlap: int = 2            # minimum common movies to compute similarity
    shrinkage: float = 10.0         # shrinkage term for noisy small overlaps
    center_by_user_mean: bool = True
    clip_range: Tuple[float, float] = (1.0, 5.0)  # expected rating range


def _user_item_dicts(ratings: pd.DataFrame,
                     user_col: str = "user_id",
                     item_col: str = "movie_id",
                     rating_col: str = "rating"):
    """Build helper dicts for quick lookup."""
    # ratings by user: user -> {movie: rating}
    by_user: Dict = {}
    # ratings by movie: movie -> {user: rating}
    by_item: Dict = {}
    # user mean rating
    user_mean: Dict = {}

    for row in ratings.itertuples(index=False):
        u = getattr(row, user_col)
        i = getattr(row, item_col)
        r = float(getattr(row, rating_col))
        by_user.setdefault(u, {})[i] = r
        by_item.setdefault(i, {})[u] = r

    for u, d in by_user.items():
        user_mean[u] = float(np.mean(list(d.values()))) if d else 0.0

    return by_user, by_item, user_mean


def _cosine_sim(a: Dict, b: Dict, center_a=0.0, center_b=0.0) -> Tuple[float, int]:
    """Cosine similarity for two sparse dict vectors over their union; returns (sim, overlap_count)."""
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0, 0
    va = np.array([a[k] - center_a for k in common], dtype=float)
    vb = np.array([b[k] - center_b for k in common], dtype=float)
    num = float(va @ vb)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12
    return (num / denom), len(common)


def _jaccard_sim(set_a: set, set_b: set) -> Tuple[float, int]:
    inter = set_a & set_b
    union = set_a | set_b
    return (len(inter) / (len(union) + 1e-12), len(inter))


def reward_factor_for_user(ratings: pd.DataFrame,
                           user_id,
                           cfg: RewardConfig = RewardConfig(),
                           user_col: str = "user_id",
                           item_col: str = "movie_id",
                           rating_col: str = "rating") -> Dict:
    """
    Compute a similarity-weighted reward score for each *candidate* movie
    (i.e., not yet rated by user_id). Returns min-max scaled dict {movie_node: reward in [0,1]}.
    """
    by_user, by_item, user_mean = _user_item_dicts(ratings, user_col, item_col, rating_col)

    if user_id not in by_user:
        # Cold-start: no ratings → reward is empty (handled by recommender with fallback)
        return {}

    target_r = by_user[user_id]
    target_mean = user_mean[user_id] if cfg.center_by_user_mean else 0.0

    # Compute similarities with other users
    sims: Dict = {}
    overlaps: Dict = {}
    tgt_set = set(target_r.keys())

    for v, v_r in by_user.items():
        if v == user_id:
            continue
        if cfg.similarity == "cosine":
            sim, ov = _cosine_sim(target_r, v_r,
                                  center_a=target_mean if cfg.center_by_user_mean else 0.0,
                                  center_b=user_mean[v] if cfg.center_by_user_mean else 0.0)
        else:
            sim, ov = _jaccard_sim(tgt_set, set(v_r.keys()))
        if ov >= cfg.min_overlap and sim > 0:
            # Shrinkage to temper small overlaps
            sim_adj = (ov / (ov + cfg.shrinkage)) * sim
            sims[v] = sim_adj
            overlaps[v] = ov

    # Candidate movies = items not rated by target user
    all_items = set(by_item.keys())
    seen = set(target_r.keys())
    candidates = all_items - seen

    reward_raw: Dict = {}
    lo, hi = cfg.clip_range
    for m in candidates:
        num = 0.0
        den = 0.0
        # aggregate neighbor ratings
        for v, sim in sims.items():
            if m in by_user[v]:
                rv = by_user[v][m]
                if cfg.center_by_user_mean:
                    rv = rv - user_mean[v] + target_mean  # translate neighbor to target user's scale
                # Clip to reasonable bounds
                rv = float(np.clip(rv, lo, hi))
                num += sim * rv
                den += abs(sim)
        if den > 1e-12:
            reward_raw[_prefixed(m, "m")] = num / den

    # If empty (no neighbors or no candidate overlap), fallback to global movie means
    if not reward_raw:
        global_means = {m: float(np.mean(list(us.values()))) for m, us in by_item.items() if m in candidates and us}
        reward_raw = {_prefixed(m, "m"): v for m, v in global_means.items()}

    return _minmax_scale(reward_raw)


# ===========================
# Recommender: Katz + Reward
# ===========================

@dataclass
class RecommenderConfig:
    lambda_katz: float = 0.5   # weight on Katz (0..1). 1.0 = pure Katz, 0.0 = pure Reward
    top_n: int = 10
    require_reward: bool = False  # if True, drop movies without reward signal


def recommend_for_user(ratings: pd.DataFrame,
                       user_id,
                       rec_cfg: RecommenderConfig = RecommenderConfig(),
                       reward_cfg: RewardConfig = RewardConfig(),
                       user_col: str = "user_id",
                       item_col: str = "movie_id",
                       rating_col: str = "rating") -> pd.DataFrame:
    """
    Generate recommendations for a given user by combining:
      score = λ * Katz(movie) + (1-λ) * Reward(user, movie)
    Returns a dataframe: movie_id, katz, reward, score (sorted desc), top_n rows.
    """
    # Build graph + Katz
    G = build_bipartite_graph(ratings, user_col, item_col, rating_col)
    katz = katz_centrality_for_movies(G)  # already min-max scaled

    # Reward for this user
    reward = reward_factor_for_user(ratings, user_id, cfg=reward_cfg,
                                    user_col=user_col, item_col=item_col, rating_col=rating_col)

    # Candidate set = movies not yet rated by user
    user_seen = set(ratings.loc[ratings[user_col] == user_id, item_col].unique().tolist())
    all_movies = set(ratings[item_col].unique().tolist())
    candidates = all_movies - user_seen

    rows = []
    for m in candidates:
        m_node = _prefixed(m, "m")
        katz_m = katz.get(m_node, 0.0)
        reward_m = reward.get(m_node, np.nan)  # may be missing if no neighbor signal
        if rec_cfg.require_reward and (m_node not in reward):
            continue
        # If reward missing, back off to Katz only
        if np.isnan(reward_m):
            score = katz_m
            reward_m = 0.0
            lam = 1.0
        else:
            lam = rec_cfg.lambda_katz
            score = lam * katz_m + (1.0 - lam) * reward_m
        rows.append((m, katz_m, reward_m, score, lam))

    recs = pd.DataFrame(rows, columns=["movie_id", "katz", "reward", "score", "lambda_used"])
    recs = recs.sort_values("score", ascending=False).reset_index(drop=True)
    return recs.head(rec_cfg.top_n)


# ===========================
# Demo / Example
# ===========================

if __name__ == "__main__":
    # Tiny synthetic dataset just to show usage.
    # Replace with your real data: columns [user_id, movie_id, rating]
    data = [
        # user 1
        (1, "A", 5), (1, "B", 4), (1, "C", 4),
        # user 2
        (2, "A", 5), (2, "B", 3), (2, "D", 4),
        # user 3
        (3, "B", 5), (3, "C", 3), (3, "E", 4),
        # user 4
        (4, "C", 5), (4, "D", 4), (4, "E", 2),
        # user 5 (sparser)
        (5, "A", 4), (5, "E", 5),
    ]
    df = pd.DataFrame(data, columns=["user_id", "movie_id", "rating"])

    # Configure weights
    rec_cfg = RecommenderConfig(lambda_katz=0.5, top_n=5)
    reward_cfg = RewardConfig(similarity="cosine", min_overlap=1, shrinkage=5.0)

    user = 5
    recs = recommend_for_user(df, user_id=user, rec_cfg=rec_cfg, reward_cfg=reward_cfg)
    print(f"Top recommendations for user {user}:")
    print(recs)
