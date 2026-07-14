from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm: list[list[int]], output_path: str | Path) -> None:
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
