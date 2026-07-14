from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_attention_heatmap(attention: np.ndarray, output_path: str | Path, title: str = "attention") -> None:
    fig, ax = plt.subplots()
    sns.heatmap(attention, cmap="viridis", ax=ax)
    ax.set_title(title)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
