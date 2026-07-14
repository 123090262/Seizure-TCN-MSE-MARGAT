# Seizure-TCN-MSE-MARGAT

Research implementation for window-level seizure detection on CHB-MIT bipolar
scalp EEG. The proposed model combines a strong TCN expert, an adaptive
multi-scale encoder (MSE), and a montage-aware residual graph attention module
(MARGAT).

## Project relationship and data

This repository is parallel to `../Seizure-Detection`. It keeps its own source,
configs, tests, outputs, checkpoints, and reports. To avoid duplicating roughly
100 GB of EEG data, these paths are symbolic links to the original project:

```text
data/CHBMIT
data/processed
data/splits
```

Consequently, window contents and split manifests are directly comparable with
the original CNN1D, TCN, and MS-DSTGAT experiments. Outputs are never shared.

## Proposed architecture

```text
Raw bipolar EEG [B,C,T]
├── TCN expert
│   └── dilated residual temporal blocks → z_tcn
└── Adaptive multi-scale encoder (MSE)
    ├── channel-independent kernels 7 / 15 / 31
    ├── per-window scale attention
    └── channel node features
        └── MARGAT
            ├── signed bipolar line-graph topology
            ├── shared-electrode structural edges
            ├── absolute-Pearson conditioned dynamic residual edges
            └── graph attention pooling → z_graph

z = z_tcn + gate(x) * graph_scale_correction
z → classifier
```

The residual correction and its gate are initialized at zero and `0.02`,
respectively. At initialization the prediction path is therefore effectively
the TCN, while MSE and MARGAT learn through auxiliary expert supervision. This
is intended to keep a weak graph branch from immediately degrading the strong
temporal model.

### Montage-aware graph

Every CHB-MIT bipolar derivation is treated as a directed edge of the underlying
10-20 electrode graph and as a node in a line graph. Two channel nodes receive
a structural edge when their derivations share an electrode. The edge relation
also records whether the shared electrode has the same or opposite subtraction
sign. Dynamic non-structural edges are continuously gated and sparsity
regularized; no hard top-k operation is used.

## Training protocol

The default protocol matches the original project:

- four-second, 18-channel windows at 256 Hz;
- pooled stratified mixed five-fold CV;
- exact duplicate windows kept in one split group;
- train-only per-channel z-score normalization;
- independently balanced train/validation/test sets;
- AdamW, cosine schedule, and validation-Accuracy checkpoint selection;
- best checkpoint restored before test evaluation;
- timestamped output directories with overwrite protection.

## Run the proposed model

```bash
python scripts/run_mixed_5fold.py \
  --config configs/exp_mixed_5fold.yaml \
  --model_config configs/model_tcn_mse_margat.yaml
```

## Run unchanged baselines

```bash
python scripts/run_mixed_5fold.py \
  --config configs/exp_mixed_5fold.yaml \
  --model_config configs/model_baseline_tcn.yaml

python scripts/run_mixed_5fold.py \
  --config configs/exp_mixed_5fold.yaml \
  --model_config configs/model_baseline_cnn1d.yaml
```

## Evaluate an existing run

```bash
python scripts/evaluate.py --run_dir outputs/mixed_5fold/tcn_mse_margat/<timestamp>
```

The evaluator rebuilds the saved configuration, restores each fold's
`best.pt`, recalculates test metrics, preserves the previous report, and updates
the five-fold summary.

## Ablation suite

The suite first measures progressive gains:

```text
TCN only
MSE only
TCN + MSE
TCN + MSE + MARGAT (full)
```

It then isolates adaptive scale routing, temporal pooling, montage topology,
signed orientation, dynamic edges, graph depth, sparsity, safe residual fusion,
and auxiliary expert supervision.

```bash
python scripts/run_ablation_suite.py --run_name margat_ablation_v1
```

Resume an interrupted suite:

```bash
python scripts/run_ablation_suite.py --run_name margat_ablation_v1 --resume
```

Run only the high-level progressive experiments first:

```bash
python scripts/run_ablation_suite.py \
  --run_name margat_ablation_v1 \
  --experiments full tcn_only mse_only tcn_mse_no_graph
```

The generated `ablation_summary.csv` uses paired folds and defines:

```text
module_effect_accuracy = full_accuracy - ablation_accuracy
```

Positive values support the removed module; negative values indicate that its
removal improved Accuracy. Automatic verdicts are screening aids, not
significance claims, and promising findings should be repeated across seeds.

## Other protocols

```bash
python scripts/run_intra_10fold.py \
  --config configs/exp_intra_10fold.yaml \
  --model_config configs/model_tcn_mse_margat.yaml

python scripts/run_lopo.py \
  --config configs/exp_lopo.yaml \
  --model_config configs/model_tcn_mse_margat.yaml
```

## Window-length and overlap sweep

Two isolated mixed-five-fold variants compare the existing four-second
protocol with one-second and two-second windows. Both seizure and non-seizure
windows use 50% overlap; preprocessing, exclusion interval, balancing,
training, model, and evaluation settings remain unchanged.

```text
configs/window_sweeps/win1s_ov50.yaml
configs/window_sweeps/win2s_ov50.yaml

data/window_sweeps/win1s_ov50/...
data/window_sweeps/win2s_ov50/...

outputs/window_sweeps/win1s_ov50/mixed_5fold/<model>/win1s_ov50_<timestamp>/...
outputs/window_sweeps/win2s_ov50/mixed_5fold/<model>/win2s_ov50_<timestamp>/...
```

The variant preparation step reuses the existing preprocessed EEG arrays and
only creates separate metadata snapshots, window indices, manifests, and
split files. Existing matching data are reused; a mismatched target is never
overwritten without the explicit `--overwrite_data` option.

Prepare both window indices without training:

```bash
python scripts/run_window_sweep.py --prepare_only
```

Prepare data and generate both five-fold split sets without training:

```bash
python scripts/run_window_sweep.py --generate_only
```

Run the complete two-variant training sweep:

```bash
python scripts/run_window_sweep.py
```

Each trained run stores a `run_manifest.json`, the fully resolved fold config,
and a partition-tagged microsecond timestamp. This prevents the 1s, 2s, and
existing 4s outputs from being mixed or overwritten.

## Subject-global normalization branch

The **全局归一化** branch compares the default fold-specific train-only Z-score
with per-subject global Z-score statistics. For each subject, the new branch
computes per-channel mean and standard deviation from all of that subject's
preprocessed continuous EEG records. It then uses the subject's own statistics
for train, validation, and test windows. All model, split, training, and
evaluation settings remain inherited from `exp_mixed_5fold.yaml`.

```bash
python scripts/run_mixed_5fold.py \
  --config configs/normalization/subject_global.yaml \
  --model_config configs/model_tcn_mse_margat.yaml
```

The first fold creates and caches subject statistics under
`data/normalization/subject_global/`; later folds reuse the exact cache. Results
are isolated under `outputs/normalization/subject_global/` and use a
`subject_global_<timestamp>` run name. This experiment is intentionally
transductive because unlabeled test-subject signals contribute normalization
statistics; it is not a train-only generalization protocol.

The same branch provides three directly comparable configs for both full-model
training and the unchanged 14-variant ablation suite:

```text
configs/normalization/subject_global.yaml
configs/normalization/subject_global_win1s_ov50.yaml
configs/normalization/subject_global_win2s_ov50.yaml
```

## Tests

```bash
pytest
```

Tests cover split leakage guards, normalization, preprocessing, checkpoint
protocol, all ablation forward paths, multi-scale routing, bipolar graph signs,
dynamic/structural graph attention, and finite end-to-end gradients.
