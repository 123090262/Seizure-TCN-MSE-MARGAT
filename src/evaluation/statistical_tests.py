from __future__ import annotations

import numpy as np
from scipy import stats


def paired_ttest(a: list[float], b: list[float]) -> dict[str, float]:
    result = stats.ttest_rel(np.asarray(a), np.asarray(b), nan_policy="omit")
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}
