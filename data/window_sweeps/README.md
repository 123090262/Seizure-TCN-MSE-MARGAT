# Window sweep data

Generated window variants are isolated below this directory:

```text
win1s_ov50/
  metadata.csv
  window_index.csv
  dataset_manifest.json
  splits/mixed_5fold/fold_*.json
win2s_ov50/
  metadata.csv
  window_index.csv
  dataset_manifest.json
  splits/mixed_5fold/fold_*.json
```

The large preprocessed EEG arrays remain shared read-only through
`data/processed/chbmit`. Run `scripts/prepare_window_variant.py` or
`scripts/run_window_sweep.py` to populate these ignored runtime files.
