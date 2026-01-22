from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunConfig:
    outdir: Path
    random_state: int = 13


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def load_unlabelled() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_iris(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.target
    return X, y


def cluster_with_kmeans(X: pd.DataFrame, k: int, random_state: int) -> Dict[str, Any]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = model.fit_predict(Xs)

    sil = silhouette_score(Xs, labels)
    return {"k": int(k), "silhouette": float(sil), "inertia": float(model.inertia_)}


def pca_projection(X: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=13)
    Z = pca.fit_transform(Xs)
    return Z, pca


def plot_projection(Z: np.ndarray, y: pd.Series, outpath: Path) -> None:
    fig = plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=y.to_numpy())
    plt.title("13UNIT: PCA projection (Iris)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="13UNIT unsupervised learning lab")
    p.add_argument("--outdir", default="output", help="Output directory")
    p.add_argument("--demo", action="store_true", help="Run a short demo")
    p.add_argument("--k", type=int, default=3, help="Number of clusters for k-means")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(outdir=Path(args.outdir))
    ensure_outdir(cfg.outdir)

    X, y = load_unlabelled()
    summary: Dict[str, Any] = {"unit": "13", "task": "unsupervised", "kmeans": cluster_with_kmeans(X, args.k, cfg.random_state)}

    Z, pca = pca_projection(X, n_components=2)
    summary["pca_explained_variance_ratio"] = [float(v) for v in pca.explained_variance_ratio_]
    plot_projection(Z, y, cfg.outdir / "pca_projection.png")

    (cfg.outdir / "unsupervised.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
