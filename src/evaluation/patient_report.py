from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_reports(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    return summary
