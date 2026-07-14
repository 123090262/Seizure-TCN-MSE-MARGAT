from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    print(f"Use src.visualization.plot_attention helpers with saved aux tensors from {args.run_dir}.")


if __name__ == "__main__":
    main()
