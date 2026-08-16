# LOPO Mixed-Window Validation Design

## Goal

Add an isolated LOPO experiment variant that follows the GLWA paper's validation
construction: hold out one complete CHB-MIT case for testing, pool the remaining
23 cases, and draw a stratified random 10% of their windows for validation. The
variant must improve specificity-oriented model selection without changing the
existing strict LOPO baseline or any mixed-10-fold behavior.

## Scope and preserved behavior

- Keep `configs/lopo.yaml` unchanged as the existing case-disjoint validation
  baseline.
- Keep `configs/mixed_10fold.yaml`, its split behavior, cached artifacts, and run
  semantics unchanged.
- Add `configs/lopo_mixedval.yaml` for the new experiment. It reuses
  `/workspace/output/prepared_lopo_24case`; no EDF preprocessing or window index
  rebuild is required.
- Continue treating the experiment as 24-case leave-one-case-out. The existing
  `chb17a`/`chb17b`/`chb17c` to `chb17` mapping remains enabled only for LOPO
  artifacts.
- Do not claim strict unseen-person evaluation because `chb01` and `chb21` are
  two cases from the same subject.

## Split design

The new configuration sets:

```yaml
split:
  protocol: lopo
  validation_strategy: mixed_windows
  val_fraction: 0.10
  train_negative_ratio: 2.0
  val_negative_ratio: 1.0
  test_negative_ratio: 1.0
```

For each outer fold:

1. Select every window belonging to the requested test case as the test pool.
2. Pool every window from the other 23 cases.
3. Use a seeded, label-stratified random split to assign 10% of the pooled
   windows to validation and 90% to training.
4. Balance each already-isolated pool independently. Retain all seizure windows,
   then sample at most 2 non-seizure windows per seizure window in training and
   at most 1 non-seizure window per seizure window in validation and testing.
5. Never balance the full catalog before the outer test case is removed.

The configuration value `validation_strategy: case_holdout` remains the default
for existing LOPO behavior. Existing `split.balance_ratio` remains a supported
fallback so old configs and checkpoints continue to load.

The mixed-window validation choice intentionally permits windows from the same
case, record, and neighboring time region to appear in both training and
validation, matching the paper's wording. Documentation and run metadata must
label this as a weaker validation protocol and not as patient-independent
validation.

## Threshold and checkpoint selection

The new configuration sets:

```yaml
train:
  threshold_metric: balanced_accuracy
```

At each epoch, use validation data only to select the probability threshold that
maximizes

```text
balanced_accuracy = (sensitivity + specificity) / 2
```

Use the balanced accuracy at that threshold for early stopping and best-checkpoint
selection. Preserve the existing `f1` threshold and checkpoint behavior as the
default for old configurations.

Resolve ties deterministically by choosing the highest threshold among tied
balanced-accuracy candidates. This favors fewer false positives without changing
the primary objective. The saved checkpoint must record the selected threshold,
threshold metric, validation metrics, and best selection score.

## Diagnostics and reproducibility

Each run must log and save:

- training, validation, and test case IDs;
- pre-balance and post-balance positive and negative window counts per split;
- validation positive-probability quantiles for both classes;
- selected threshold and threshold metric;
- the existing full merged configuration and environment metadata.

The diagnostics must not save raw EEG or all per-window probabilities. They are
small JSON-compatible summaries intended to explain extremely low thresholds and
class-distribution changes.

## Validation and failure handling

- Reject unknown validation strategies and non-positive negative ratios with
  clear `ValueError` messages.
- Require every train, validation, and test split to contain both classes before
  and after balancing.
- Add tests showing that the test case never enters the pooled train/validation
  data.
- Add tests showing that mixed-window validation is stratified, reproducible, and
  approximately 10% of the remaining windows before balancing.
- Add tests for independent train/validation/test ratios.
- Add tests for balanced-accuracy threshold selection and deterministic ties.
- Retain regression tests for the current case-holdout LOPO and mixed-10-fold
  split behavior.

## Experiment sequence

After implementation and local tests, run only `chb01` and `chb02` as development
folds first. Compare the existing baseline with the new variant using accuracy,
balanced accuracy, sensitivity, specificity, precision, F1, AUROC, AUPRC,
threshold, and confusion matrix. The intended improvement is higher specificity
and precision while retaining useful sensitivity; a smaller threshold value by
itself is not a failure if validation and test classification metrics improve.

Freeze the configuration before launching all 24 folds. Keep the development
runs and final runs in distinct timestamped directories, retain every fold, and
report the 24-fold mean and sample standard deviation. A natural-distribution
continuous-test protocol and event-level post-processing are separate future
work because the GLWA paper does not provide enough detail to reproduce them
unambiguously.
