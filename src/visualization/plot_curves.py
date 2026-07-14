from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots()
    for col in [c for c in df.columns if c.endswith("loss")]:
        ax.plot(df["epoch"], df[col], label=col)
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
