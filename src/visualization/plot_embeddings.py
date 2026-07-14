from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_pca_embeddings(embeddings: np.ndarray, labels: np.ndarray, output_path: str | Path) -> None:
    coords = PCA(n_components=2).fit_transform(embeddings)
    fig, ax = plt.subplots()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="coolwarm", s=8)
    fig.colorbar(scatter, ax=ax)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
