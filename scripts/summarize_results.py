from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out_csv", default="outputs/metrics/summary.csv")
    args = parser.parse_args()
    rows = []
    for path in Path(args.outputs).rglob("report.json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["run_dir"] = str(path.parent)
        rows.append(obj)
    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
