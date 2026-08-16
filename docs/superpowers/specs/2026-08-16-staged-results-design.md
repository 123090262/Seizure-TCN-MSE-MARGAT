# Staged Results Experiment Suite Design

## 1. Objective

Create an isolated `staged results/` experiment suite for a time-limited stage
presentation. The suite will use only fold 0 of the existing mixed 10-fold
protocol. It will support:

- module-level ablations;
- internal innovation ablations;
- bipolar-lead contribution heatmaps;
- every attention mechanism exposed by TCN-MSE-MARGAT;
- PCA and t-SNE embedding plots;
- sample-level error analysis.

The stage results are preliminary presentation evidence. They are not a
replacement for all-fold or patient-independent paper experiments.

## 2. Isolation Boundary

The suite must not modify or depend on mutations to the current experiment
code or outputs.

- Do not edit top-level `src/`, `configs/`, or current run directories.
- Keep all new source, configurations, tests, and documentation under
  `staged results/`.
- Read the existing prepared continuous EEG arrays, window catalogs,
  manifest, and normalization artifacts without copying the approximately
  61 GiB record cache.
- Do not call the existing `create_dataloaders` entry point because it writes
  split artifacts under the shared prepared directory.
- Build fold-0 indices in memory through a local read-only data adapter.
- Write checkpoints and generated artifacts only under
  `/workspace/output/staged_results/`.
- Reject any protocol other than `mixed_10fold` and any fold other than 0.

The local data adapter may reuse stable read-only primitives from `src.data`,
such as dataset classes and deterministic split logic, but owns loader
construction and metadata collection so it never writes shared split files.

## 3. Directory Layout

```text
staged results/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- base.yaml
|   |-- mixed_fold0.yaml
|   |-- analysis.yaml
|   |-- module_ablation/
|   |   |-- full.yaml
|   |   |-- tcn_only.yaml
|   |   |-- graph_only.yaml
|   |   `-- tcn_mse_no_graph.yaml
|   `-- internal_ablation/
|       |-- mse_single_15.yaml
|       |-- mse_equal_scales.yaml
|       |-- graph_no_priors.yaml
|       |-- graph_no_dynamic.yaml
|       `-- fusion_add.yaml
|-- staged_results/
|   |-- __init__.py
|   |-- data.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   `-- analyze.py
`-- tests/
    |-- test_data.py
    |-- test_model.py
    `-- test_analysis.py
```

Runtime artifacts use this separate hierarchy:

```text
/workspace/output/staged_results/
|-- runs/
|-- figures/
|   |-- heatmap/
|   |-- attention/
|   `-- embeddings/
`-- tables/
    |-- ablation/
    `-- errors/
```

## 4. Configuration

Configuration files are merged from left to right. `base.yaml` contains shared
data, model, training, and output settings. `mixed_fold0.yaml` fixes the
protocol and fold. Each ablation overlay changes only the relevant model
switches.

The effective configuration is saved in every run directory. The training
entry point validates the following before loading data:

- `split.protocol == "mixed_10fold"`;
- `split.fold == 0`;
- the output root is `/workspace/output/staged_results` or a descendant;
- AMP is disabled by default for the staged runs because prior evidence found
  FP16 forward overflow while FP32 remained finite.

## 5. Model Interface

The local model preserves the full model's parameter names where practical so
that a compatible existing full-model `best.pt` can be loaded for inference.
Variant behavior is controlled by a compact configuration rather than by
separate model classes.

The full forward pass returns logits by default. An optional analysis path
returns logits and a dictionary containing:

- `temporal_embedding`: `[B, 128]`;
- `temporal_attention`: `[B, T]`;
- `scale_attention`: `[B, 18, 3]`;
- `graph_attention`: `[B, 4, 18, 18]` for the default four heads;
- `graph_pool_attention`: `[B, 18]`;
- `graph_correction`: `[B, 128]`;
- `fusion_gate`: `[B, 128]`;
- `fused_embedding`: `[B, 128]`.

The analysis path must not change full-model logits.

## 6. Ablation Definitions

### 6.1 Module-Level Ablations

| Variant | Active path | Question |
|---|---|---|
| `full` | TCN + MSE + MARGAT + gated fusion | Reference model |
| `tcn_only` | TCN classifier | Does the graph branch add value? |
| `graph_only` | MSE + MARGAT classifier | Does the temporal branch add value? |
| `tcn_mse_no_graph` | TCN + pooled MSE | Does relational graph modeling add value beyond channel encoding? |

Each branch-only or no-graph variant uses a small classification/fusion head
appropriate to its available representation. Results must include parameter
counts so capacity differences remain visible.

### 6.2 Internal Innovation Ablations

| Variant | Change from full model | Question |
|---|---|---|
| `mse_single_15` | one kernel of size 15 | Is multi-scale encoding useful? |
| `mse_equal_scales` | kernels 7/15/31 with equal averaging | Is learned scale weighting useful? |
| `graph_no_priors` | learned QKV attention without structural, orientation, dynamic, or soft-edge priors | Do graph priors help? |
| `graph_no_dynamic` | remove dynamic signal correlation | Does sample-specific correlation help? |
| `fusion_add` | direct residual addition without learned gate | Does feature-wise gated fusion help? |

All ablations use the same fold-0 indices, seed, negative sampling, validation
threshold selection, epoch budget, and early stopping rule. They are labelled
as fold-0 preliminary results in tables and figure captions.

## 7. Training and Evaluation

Only ablation variants require training. The training entry point:

1. loads independent staged configuration;
2. creates the deterministic fold-0 loaders without writing shared artifacts;
3. trains in FP32 by default with finite checks for inputs, logits, loss,
   gradients, parameters, and optimizer state;
4. records train loss, validation loss, validation metrics, threshold, and
   learning rate for every epoch;
5. saves `last.pt`, `best.pt`, `history.json`, `config.yaml`, environment data,
   and diagnostics under the staged output root;
6. selects checkpoints and thresholds using validation data only;
7. evaluates the selected `best.pt` on fold-0 test data.

## 8. Analysis Tasks

Except for ablations, all analysis tasks load a compatible full-model
`best.pt`. They run with `model.eval()` and `torch.inference_mode()` and do not
create an optimizer or update weights.

`analyze.py` exposes five subcommands:

```text
attention
heatmap
embeddings
errors
all
```

`all` performs one ordinary inference pass to collect predictions,
embeddings, and attention tensors, then reuses those arrays for downstream
plots. Occlusion heatmaps require additional forward passes.

### 8.1 Attention Visualization

Generate and retain the underlying arrays for:

- mean temporal attention profiles by class;
- `18 x 3` MSE scale-selection heatmaps;
- one `18 x 18` graph-attention matrix per head;
- ictal-minus-interictal graph-attention differences;
- bipolar-lead graph-pooling weights;
- fusion-gate distributions.

Attention is described as model weighting, not causal explanation or clinical
functional connectivity.

### 8.2 Bipolar-Lead Contribution Heatmap

Use channel occlusion because it is class-specific, simple, and requires no
additional dependency:

```text
importance_c = p_seizure(x) - p_seizure(x with channel c set to zero)
```

Zero is the normalized-channel mean. The analysis uses a deterministic,
configurable sample cap and reports the selected counts.

Produce:

- per-lead contribution tables;
- interictal, ictal, and ictal-minus-interictal summaries;
- a bipolar montage edge map;
- a lead-midpoint interpolated head map with an explicit non-localization
  disclaimer;
- raw contribution arrays.

The output is named a bipolar-lead contribution map, not ECAM or seizure-focus
localization.

### 8.3 PCA and t-SNE

Extract temporal, graph-correction, and fused embeddings. Use deterministic,
class-balanced sample selection. Fit PCA and t-SNE with settings saved in the
analysis metadata. Plot labels by color and optionally patients by marker.

PCA and t-SNE are descriptive embedding analyses and are not presented as
evidence of patient-independent generalization.

### 8.4 Error Analysis

Save sample-level predictions with:

- patient/case ID;
- record ID;
- window start sample and seconds;
- true label;
- seizure probability;
- predicted label;
- error type (`TP`, `TN`, `FP`, or `FN`).

Produce a confusion matrix, per-case error summary, probability histograms,
and tables of the highest-confidence false positives and false negatives.

## 9. Resource Use

- Ablation training occupies the GPU for full training runs.
- Ordinary attention, embedding, and error extraction uses roughly one test
  inference pass.
- PCA and t-SNE run on CPU after feature extraction.
- Eighteen-channel occlusion uses the original pass plus one pass per channel;
  GPU is recommended but no gradients or optimizer are needed.
- Analysis batch size and sample caps are configurable to bound GPU memory and
  runtime.

## 10. Failure Handling

The suite fails with explicit messages when:

- protocol or fold is not the staged fold-0 setting;
- output paths escape the staged output root;
- prepared data or the checkpoint is missing;
- checkpoint configuration or parameter shapes are incompatible;
- required labels, probabilities, or attention arrays are empty/non-finite;
- t-SNE receives too few samples for the configured perplexity;
- an analysis attempts to overwrite an existing artifact directory without an
  explicit overwrite flag.

## 11. Verification

Tests use synthetic data and temporary directories where possible. They cover:

- rejection of non-fold-0 and non-mixed configurations;
- no writes to the shared prepared split directory;
- forward and backward shape checks for every ablation;
- equality of full-model logits with and without analysis details;
- expected attention and embedding tensor shapes;
- compatible full checkpoint loading;
- deterministic sampling;
- occlusion importance shape and finite values;
- PCA/t-SNE input validation;
- error-table classification of TP/TN/FP/FN;
- output-root containment.

Local verification includes compile checks and the staged unit tests. Real
data, checkpoint, tensor, GPU, latency, and memory validation must run in the
server PyTorch environment because the local environment may not provide a
compatible Torch installation or the prepared CHB-MIT artifacts.

## 12. Presentation Boundary

Every staged result table and figure caption states:

> Preliminary mixed 10-fold fold-0 result for stage presentation; not an
> all-fold or unseen-patient generalization estimate.

No stage output is described as final paper evidence, clinical performance,
ECAM, seizure-focus localization, or causal explanation.
