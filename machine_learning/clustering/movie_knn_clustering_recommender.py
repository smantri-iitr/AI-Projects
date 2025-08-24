# -------------------------------------------------------------
# Clustering users via KNN on preference similarity (cosine),
# then recommending movies using weighted user-based CF
# restricted to each user's cluster.
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
from scipy import sparse


# -----------------------------
# Data loading / preparation
# -----------------------------

def load_ratings_and_movies(
    ratings_path: Optional[str] = "ratings.csv",
    movies_path: Optional[str] = "movies.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ratings and movies. If files not found, returns a small synthetic dataset.
    """
    rpath = Path(ratings_path) if ratings_path else None
    mpath = Path(movies_path) if movies_path else None

    if rpath and rpath.exists() and mpath and mpath.exists():
        ratings = pd.read_csv(rpath)
        movies = pd.read_csv(mpath)
        return ratings, movies

    # --- Synthetic fallback so the script runs anywhere ---
    rng = np.random.default_rng(42)
    users = np.arange(1, 26)         # 25 users
    movie_ids = np.arange(100, 151)  # 50 movies
    rows = []
    # Make 3 latent taste groups to produce meaningful clusters
    for u in users:
        if u <= 8:
            liked = movie_ids[:15]
        elif u <= 16:
            liked = movie_ids[15:30]
        else:
            liked = movie_ids[30:]
        # each user rates 15-25 movies
        rated = rng.choice(movie_ids, size=rng.integers(15, 26), replace=False)
        for m in rated:
            base = 4.3 if m in liked else 2.9
            rating = np.clip(rng.normal(base, 0.7), 0.5, 5.0)
            rows.append((u, m, round(float(rating)*2)/2, 0))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    movies = pd.DataFrame(
        {"movieId": movie_ids, "title": [f"Movie {i}" for i in movie_ids], "genres": ["Synthetic"]*len(movie_ids)}
    )
    return ratings, movies


def build_user_item_matrix(
    ratings: pd.DataFrame
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int], List[int], List[int]]:
    """
    Build dense user-item matrix R (users x items) with zeros for missing entries.
    Returns:
      R, user_to_idx, item_to_idx, idx_to_user, idx_to_item
    """
    users = np.sort(ratings["userId"].unique())
    items = np.sort(ratings["movieId"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {m: j for j, m in enumerate(items)}

    R = np.zeros((len(users), len(items)), dtype=np.float32)
    for row in ratings.itertuples(index=False):
        i = user_to_idx[row.userId]
        j = item_to_idx[row.movieId]
        R[i, j] = row.rating
    return R, user_to_idx, item_to_idx, list(users), list(items)


def mean_center_by_user(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean-center each user's ratings (ignoring zeros). Zeros stay zero.
    Returns centered matrix and per-user means.
    """
    R_centered = R.copy().astype(np.float32)
    user_means = np.zeros(R.shape[0], dtype=np.float32)
    for i in range(R.shape[0]):
        nz = R[i] > 0
        if np.any(nz):
            mu = R[i, nz].mean()
            user_means[i] = mu
            R_centered[i, nz] = R[i, nz] - mu
        else:
            user_means[i] = 0.0
    return R_centered, user_means


# -----------------------------
# Clustering via KNN graph
# -----------------------------

def knn_clusters(
    X: np.ndarray,
    n_neighbors: int = 8,
) -> Tuple[np.ndarray, sparse.csr_matrix, int]:
    """
    Build a KNN graph on X (rows are users), using cosine distance, then
    symmetrize and take connected components as clusters.

    Returns:
      labels: array of shape (n_users,)
      adj:    symmetric adjacency (CSR)
      n_comp: number of clusters
    """
    # Use cosine distance (1 - cosine similarity)
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, max(1, X.shape[0]-1)),
                            metric="cosine", algorithm="brute")
    nbrs.fit(X)
    # Connectivity graph (unweighted)
    adj = nbrs.kneighbors_graph(mode="connectivity", include_self=False)
    # Symmetrize to make an undirected graph
    adj = adj.maximum(adj.T)

    # If the graph is too sparse (users with no edges), lightly densify by
    # connecting exact duplicates (cosine distance == 0) if any
    if (adj.sum(axis=1) == 0).A.ravel().any():
        sims = cosine_similarity(X)
        # form an edge where similarity == 1 (distance 0), excluding self
        dup_edges = sparse.csr_matrix((sims >= 0.9999).astype(int))  # boolean to int
        dup_edges.setdiag(0)
        dup_edges.eliminate_zeros()
        adj = adj.maximum(dup_edges)

    n_comp, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    return labels, adj, n_comp


# -----------------------------
# Recommendations (within cluster)
# -----------------------------

def recommend_for_user(
    user_id: int,
    R: np.ndarray,
    R_centered: np.ndarray,
    user_means: np.ndarray,
    labels: np.ndarray,
    idx_users: List[int],
    idx_items: List[int],
    top_n: int = 10,
    min_neighbor_ratings: int = 2,
    k_sim_neighbors: int = 20,
) -> List[Tuple[int, float]]:
    """
    Recommend top-N movies for a given user (by original user_id):
    1) Restrict neighbors to user's cluster
    2) Compute cosine similarity to neighbors using centered ratings
    3) Predict rating for each unrated item with weighted mean of neighbor deviations
       and add back the target user's mean.

    Returns a list of (movieId, predicted_score) sorted descending.
    """
    user_to_idx = {u: i for i, u in enumerate(idx_users)}
    if user_id not in user_to_idx:
        raise ValueError(f"user_id {user_id} not found.")
    uidx = user_to_idx[user_id]

    # Users in same cluster (excluding self)
    cluster_id = labels[uidx]
    cluster_users = np.where(labels == cluster_id)[0]
    cluster_users = cluster_users[cluster_users != uidx]
    if cluster_users.size == 0:
        # fallback: use all users
        cluster_users = np.setdiff1d(np.arange(len(idx_users)), np.array([uidx]))

    # Similarities to cluster peers (cosine on centered ratings)
    uvec = R_centered[uidx:uidx+1]
    sims = cosine_similarity(uvec, R_centered[cluster_users])[0]  # shape: (n_cluster-1,)
    # keep top-k most similar (positive) neighbors
    sim_order = np.argsort(-sims)
    keep = sim_order[:min(k_sim_neighbors, sim_order.size)]
    nbr_idxs = cluster_users[keep]
    nbr_sims = sims[keep]

    # Items the user hasn't rated
    user_row = R[uidx]
    unrated = np.where(user_row == 0)[0]
    if unrated.size == 0:
        return []  # nothing to recommend

    # Build predictions
    preds = []
    # Precompute neighbor mask of items they rated (non-zero)
    nbr_matrix = R_centered[nbr_idxs]          # centered ratings for neighbors
    nbr_mask = (R[nbr_idxs] > 0).astype(np.float32)

    # For each item, use only neighbors who rated it
    for j in unrated:
        rated_mask = nbr_mask[:, j] > 0
        if rated_mask.sum() < min_neighbor_ratings:
            continue
        neigh_devs = nbr_matrix[rated_mask, j]
        neigh_sims = nbr_sims[rated_mask]
        denom = np.abs(neigh_sims).sum()
        if denom <= 1e-8:
            continue
        # predicted deviation from user's mean, then add back user mean
        pred = (neigh_sims @ neigh_devs) / denom + user_means[uidx]
        preds.append((idx_items[j], float(pred)))

    # Sort by predicted score (desc) and return top-n
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:top_n]


# -----------------------------
# Convenience wrapper
# -----------------------------

def build_and_recommend(
    ratings_path: Optional[str] = "ratings.csv",
    movies_path: Optional[str] = "movies.csv",
    n_neighbors_graph: int = 8,
    k_sim_neighbors: int = 25,
    top_n: int = 10,
    user_id_to_demo: Optional[int] = None,
):
    ratings, movies = load_ratings_and_movies(ratings_path, movies_path)

    # Build matrix
    R, user_to_idx, item_to_idx, idx_users, idx_items = build_user_item_matrix(ratings)

    # Center by user mean
    R_centered, user_means = mean_center_by_user(R)

    # KNN graph -> connected components as clusters
    labels, adj, n_comp = knn_clusters(R_centered, n_neighbors=n_neighbors_graph)
    print(f"[info] Users: {len(idx_users)}, Items: {len(idx_items)}, Clusters found: {n_comp}")

    # Show cluster sizes
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    print("[info] Cluster sizes:")
    print(cluster_sizes.to_string())

    # Demo: pick a user (first user if not specified)
    if user_id_to_demo is None:
        user_id_to_demo = idx_users[0]

    preds = recommend_for_user(
        user_id=user_id_to_demo,
        R=R,
        R_centered=R_centered,
        user_means=user_means,
        labels=labels,
        idx_users=idx_users,
        idx_items=idx_items,
        top_n=top_n,
        k_sim_neighbors=k_sim_neighbors,
    )

    if not preds:
        print(f"\nNo recommendations for user {user_id_to_demo} (maybe they rated everything).")
        return

    # Join with movie titles if available
    recs_df = pd.DataFrame(preds, columns=["movieId", "pred_score"])
    if "movieId" in movies.columns and "title" in movies.columns:
        recs_df = recs_df.merge(movies[["movieId", "title"]], on="movieId", how="left")
        recs_df = recs_df[["movieId", "title", "pred_score"]]

    print(f"\nTop {len(recs_df)} recommendations for user {user_id_to_demo}:")
    print(recs_df.to_string(index=False, formatters={"pred_score": "{:.3f}".format}))


# -----------------------------
# Run example
# -----------------------------

if __name__ == "__main__":
    # If you have MovieLens CSVs in the working dir, they'll be used.
    # Otherwise, a synthetic dataset is generated so this runs out of the box.
    build_and_recommend(
        ratings_path="ratings.csv",   # change to your paths if needed
        movies_path="movies.csv",
        n_neighbors_graph=8,          # K in the KNN graph for clustering
        k_sim_neighbors=25,           # top-K most similar neighbors for prediction
        top_n=10,                     # number of recommendations to print
        user_id_to_demo=None          # set a specific userId to demo if you want
    )
