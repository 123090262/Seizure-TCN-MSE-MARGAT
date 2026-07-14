# Window sweep configurations

| Config | Window | Ictal overlap | Interictal overlap | Output tag |
|---|---:|---:|---:|---|
| `win1s_ov50.yaml` | 1 s / 256 samples | 50% | 50% | `win1s_ov50` |
| `win2s_ov50.yaml` | 2 s / 512 samples | 50% | 50% | `win2s_ov50` |

Both configs inherit `../exp_mixed_5fold.yaml`. They only override window
length, both overlap rates, and the data/split/output partition paths. The
preprocessed EEG arrays remain shared and are not regenerated.

Use `python scripts/run_window_sweep.py --generate_only` to prepare both data
partitions and split manifests, or omit `--generate_only` to train both runs.
