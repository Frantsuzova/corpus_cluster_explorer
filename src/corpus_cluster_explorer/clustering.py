import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def evaluate_k_range(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[list[int], list[float]]:
    valid_k: list[int] = []
    scores: list[float] = []

    for k in range(k_min, k_max + 1):
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        valid_k.append(k)
        scores.append(float(score))

    return valid_k, scores


def run_kmeans(
    X: np.ndarray,
    candidate_tokens: list[str],
    token_counts,
    model: Word2Vec,
    n_clusters: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[int, str]]:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20
    )
    labels = kmeans.fit_predict(X)

    df_clusters = pd.DataFrame({
        "token": candidate_tokens,
        "cluster": labels,
        "freq": [token_counts[t] for t in candidate_tokens]
    })

    centers = kmeans.cluster_centers_
    df_clusters["dist_to_center"] = [
        np.linalg.norm(model.wv[token] - centers[cluster])
        for token, cluster in zip(candidate_tokens, labels)
    ]

    cluster_labels: dict[int, str] = {}
    for cluster_id in range(n_clusters):
        cluster_df = df_clusters[df_clusters["cluster"] == cluster_id].copy()
        top_freq = cluster_df.sort_values("freq", ascending=False).head(5)
        cluster_labels[cluster_id] = ", ".join(top_freq["token"].tolist())

    return df_clusters, labels, cluster_labels


def project_pca(X: np.ndarray, random_state: int) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X)
