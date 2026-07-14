# Normalization strategy branch

The following configs define the experimental branch **全局归一化**:

| Config | Window protocol | Full-model and ablation output partition |
|---|---|---|
| `subject_global.yaml` | current 4s baseline | `win4s_baseline` |
| `subject_global_win1s_ov50.yaml` | 1s, 50% overlap | `win1s_ov50` |
| `subject_global_win2s_ov50.yaml` | 2s, 50% overlap | `win2s_ov50` |

The baseline (`normalization.scope: train_only`) fits one per-channel mean and
standard deviation from each fold's training windows, then applies those
statistics to validation and test windows.

The new branch (`normalization.scope: subject_global`) instead fits one
per-channel mean and standard deviation for each subject using all of that
subject's preprocessed continuous EEG records. Every train/validation/test
window is normalized with its own subject's statistics. The cached statistics
are isolated under `data/normalization/subject_global/` and reused by all five
folds and all three window protocols.

Every config can be passed to either `scripts/run_mixed_5fold.py` for the full
model or `scripts/run_ablation_suite.py` for the same 14-experiment ablation
manifest used on `main`.

This is intentionally a transductive normalization experiment because test
subjects contribute unlabeled signal statistics. It must not be described as
train-only normalization.
