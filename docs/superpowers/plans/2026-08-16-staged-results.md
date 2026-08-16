# Staged Results Experiment Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated fold-0 mixed-10 experiment suite for ablation training and checkpoint-only attention, bipolar-lead contribution, embedding, and error analyses.

**Architecture:** A self-contained Python package lives under `staged results/` and reads the existing prepared EEG artifacts without writing to them. A configuration-driven model implements the full architecture and nine staged variants, while a single analysis CLI collects reusable inference arrays and renders task-specific outputs under `/workspace/output/staged_results/`.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy, scikit-learn, matplotlib, PyYAML, pytest.

## Global Constraints

- Do not modify top-level `src/`, `configs/`, existing run directories, or prepared data artifacts.
- Use only `split.protocol=mixed_10fold` and `split.fold=0`.
- Read the existing prepared records, manifest, normalization, and window catalogs without copying the 61 GiB cache.
- Write runtime artifacts only below `/workspace/output/staged_results/`.
- Default staged training to FP32 (`train.amp=false`) and reject non-finite tensors.
- Label every result as preliminary fold-0 stage-presentation evidence.
- Use validation data only for checkpoint and threshold selection.
- Keep dependencies limited to the existing scientific Python stack plus matplotlib.

---

## File Map

- `staged results/README.md`: commands, output descriptions, and scientific limits.
- `staged results/requirements.txt`: minimal plotting/runtime additions.
- `staged results/configs/base.yaml`: shared data, model, training, and output settings.
- `staged results/configs/mixed_fold0.yaml`: the only allowed protocol and fold.
- `staged results/configs/analysis.yaml`: sampling, PCA, t-SNE, and plot settings.
- `staged results/configs/module_ablation/*.yaml`: four module-level variants.
- `staged results/configs/internal_ablation/*.yaml`: five internal variants.
- `staged results/staged_results/data.py`: config merge, fold-0 split, read-only datasets, metadata.
- `staged results/staged_results/model.py`: full model, ablations, and analysis details.
- `staged results/staged_results/evaluate.py`: prediction, thresholding, and metrics.
- `staged results/staged_results/train.py`: finite FP32 training and checkpoint history.
- `staged results/staged_results/analyze.py`: collection, attention, heatmap, embedding, and error CLIs.
- `staged results/tests/test_data.py`: isolation and deterministic split tests.
- `staged results/tests/test_model.py`: variant, shape, and checkpoint-compatibility tests.
- `staged results/tests/test_train.py`: finite checks and history tests.
- `staged results/tests/test_analysis.py`: sampling, occlusion, embedding, and error tests.

---

### Task 1: Isolated Configuration and Read-Only Fold-0 Data

**Files:**
- Create: `staged results/staged_results/__init__.py`
- Create: `staged results/staged_results/data.py`
- Create: `staged results/configs/base.yaml`
- Create: `staged results/configs/mixed_fold0.yaml`
- Create: `staged results/configs/analysis.yaml`
- Test: `staged results/tests/test_data.py`

**Interfaces:**
- Produces: `load_config(paths: Sequence[Path]) -> dict[str, Any]`
- Produces: `validate_staged_config(config: dict[str, Any]) -> None`
- Produces: `make_fold0_indices(labels: np.ndarray, seed: int, folds: int, val_fraction: float, balance_ratio: float) -> dict[str, np.ndarray]`
- Produces: `create_loaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], dict[str, Any]]`
- Produces: datasets yielding `(inputs: Tensor, target: Tensor, sample_index: Tensor)` and exposing `metadata(index: int) -> dict[str, Any]`

- [ ] **Step 1: Write failing isolation and split tests**

```python
def test_rejects_non_fold_zero(tmp_path):
    config = staged_config(tmp_path)
    config["split"]["fold"] = 1
    with pytest.raises(ValueError, match="fold 0"):
        validate_staged_config(config)


def test_fold0_split_is_deterministic():
    labels = np.tile(np.array([0, 1], dtype=np.int64), 100)
    first = make_fold0_indices(labels, 42, 10, 0.1, 1.0)
    second = make_fold0_indices(labels, 42, 10, 0.1, 1.0)
    assert all(np.array_equal(first[name], second[name]) for name in first)


def test_loader_creation_does_not_create_shared_split_files(prepared_fixture):
    before = sorted(prepared_fixture.rglob("*"))
    create_loaders(prepared_fixture.config)
    after = sorted(prepared_fixture.rglob("*"))
    assert after == before
```

- [ ] **Step 2: Run the tests and verify they fail**

Run from `staged results/`:

```powershell
python -m pytest tests/test_data.py -v
```

Expected: import failure because `staged_results.data` does not exist.

- [ ] **Step 3: Implement config validation and deterministic split creation**

```python
def validate_staged_config(config: dict[str, Any]) -> None:
    split = config["split"]
    if split["protocol"] != "mixed_10fold" or int(split["fold"]) != 0:
        raise ValueError("Staged results require mixed_10fold fold 0")
    output_root = Path(config["train"]["output_root"])
    required = Path("/workspace/output/staged_results")
    if output_root != required and required not in output_root.parents:
        raise ValueError(f"Output root must stay below {required}")


def make_fold0_indices(labels, seed, folds, val_fraction, balance_ratio):
    all_indices = np.arange(len(labels))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    train_val, test = next(splitter.split(all_indices, labels))
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=labels[train_val],
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    return {
        name: balance_indices(indices, labels, balance_ratio, rng)
        for name, indices in {"train": train, "val": val, "test": test}.items()
    }
```

Implement local lazy and optional in-memory datasets by reading `manifest.json`, `normalization.npz`, and `windows/2s.npz`. Loader construction must not call `mkdir`, `np.savez`, or `write_text` under the prepared root.

- [ ] **Step 4: Add independent base and fold overlays**

`configs/mixed_fold0.yaml` must contain exactly the protocol restriction:

```yaml
split:
  protocol: mixed_10fold
  fold: 0
  folds: 10
  val_fraction: 0.1
  balance_ratio: 1.0
```

`configs/base.yaml` must use `/workspace/output/staged_results`, two-second windows, seed 42, and `amp: false`.

- [ ] **Step 5: Run data tests**

```powershell
python -m pytest tests/test_data.py -v
```

Expected: all tests pass and fixture prepared directories remain unchanged.

- [ ] **Step 6: Commit**

```powershell
git add -- "staged results/staged_results/__init__.py" "staged results/staged_results/data.py" "staged results/configs" "staged results/tests/test_data.py"
git commit -m "feat: add isolated staged fold data"
```

---

### Task 2: Configurable Model Variants and Analysis Details

**Files:**
- Create: `staged results/staged_results/model.py`
- Create: `staged results/configs/module_ablation/full.yaml`
- Create: `staged results/configs/module_ablation/tcn_only.yaml`
- Create: `staged results/configs/module_ablation/graph_only.yaml`
- Create: `staged results/configs/module_ablation/tcn_mse_no_graph.yaml`
- Create: `staged results/configs/internal_ablation/mse_single_15.yaml`
- Create: `staged results/configs/internal_ablation/mse_equal_scales.yaml`
- Create: `staged results/configs/internal_ablation/graph_no_priors.yaml`
- Create: `staged results/configs/internal_ablation/graph_no_dynamic.yaml`
- Create: `staged results/configs/internal_ablation/fusion_add.yaml`
- Test: `staged results/tests/test_model.py`

**Interfaces:**
- Consumes: merged config from Task 1.
- Produces: `build_model(config: dict[str, Any]) -> TCNMSEMARGAT`
- Produces: `TCNMSEMARGAT.forward(inputs: Tensor, return_details: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]`
- Produces details keys defined in the approved design.

- [ ] **Step 1: Write failing tests for all variants and details**

```python
@pytest.mark.parametrize(
    "variant",
    [
        "full", "tcn_only", "graph_only", "tcn_mse_no_graph",
        "mse_single_15", "mse_equal_scales", "graph_no_priors",
        "graph_no_dynamic", "fusion_add",
    ],
)
def test_variant_returns_two_logits(variant):
    model = small_model(variant)
    assert model(torch.randn(2, 18, 512)).shape == (2, 2)


def test_full_details_have_expected_shapes():
    model = small_model("full").eval()
    logits, details = model(torch.randn(2, 18, 512), return_details=True)
    assert logits.shape == (2, 2)
    assert details["temporal_attention"].shape == (2, 512)
    assert details["scale_attention"].shape == (2, 18, 3)
    assert details["graph_attention"].shape == (2, 4, 18, 18)
    assert details["graph_pool_attention"].shape == (2, 18)


def test_analysis_path_does_not_change_logits():
    model = small_model("full").eval()
    inputs = torch.randn(2, 18, 512)
    plain = model(inputs)
    detailed, _ = model(inputs, return_details=True)
    torch.testing.assert_close(plain, detailed)
```

- [ ] **Step 2: Run model tests and verify failure**

```powershell
python -m pytest tests/test_model.py -v
```

Expected: import failure because the staged model is absent.

- [ ] **Step 3: Implement the full architecture with explicit feature returns**

Keep full-model parameter names compatible with the top-level checkpoint. Submodules return weights only when requested:

```python
def forward(self, inputs: Tensor, return_details: bool = False):
    temporal, temporal_details = self.tcn(inputs, return_details=True)
    nodes, scale_attention = self.mse(inputs, return_attention=True)
    graph, graph_details = self.graph(nodes, inputs, return_details=True)
    correction = self.correction(torch.cat([nodes.mean(dim=1), graph], dim=-1))
    gate = torch.sigmoid(self.gate(torch.cat([temporal, correction], dim=-1)))
    fused = self.norm(temporal + gate * correction)
    logits = self.classifier(fused)
    if not return_details:
        return logits
    return logits, {
        "temporal_embedding": temporal,
        "temporal_attention": temporal_details["attention"],
        "scale_attention": scale_attention,
        "graph_attention": graph_details["attention"],
        "graph_pool_attention": graph_details["pool"],
        "graph_correction": correction,
        "fusion_gate": gate,
        "fused_embedding": fused,
    }
```

Variant branches must explicitly handle every name in `SUPPORTED_VARIANTS` and
raise `ValueError` for any other name. Unused modules must not silently
influence logits.

- [ ] **Step 4: Add one-field variant overlays**

Each YAML contains only its variant or internal switch. Example:

```yaml
model:
  variant: graph_no_dynamic
```

- [ ] **Step 5: Add full checkpoint compatibility test**

Construct the top-level and staged full models with the same small dimensions, load the top-level state dict into the staged model with `strict=True`, and compare logits in evaluation mode.

- [ ] **Step 6: Run model tests**

```powershell
python -m pytest tests/test_model.py -v
```

Expected: all variant, gradient, detail-shape, and compatibility tests pass.

- [ ] **Step 7: Commit**

```powershell
git add -- "staged results/staged_results/model.py" "staged results/configs/module_ablation" "staged results/configs/internal_ablation" "staged results/tests/test_model.py"
git commit -m "feat: add staged ablation models"
```

---

### Task 3: FP32 Training, History, and Fold-0 Evaluation

**Files:**
- Create: `staged results/staged_results/evaluate.py`
- Create: `staged results/staged_results/train.py`
- Test: `staged results/tests/test_train.py`

**Interfaces:**
- Consumes: `create_loaders`, `build_model`.
- Produces: `ensure_finite(name: str, tensor: Tensor) -> None`
- Produces: `evaluate_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]`
- Produces: `select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float`
- Produces: `binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]`
- Produces checkpoint fields: model, optimizer, epoch, config, threshold, validation metrics, split summary, and history.

- [ ] **Step 1: Write failing finite and history tests**

```python
def test_ensure_finite_names_bad_tensor():
    with pytest.raises(FloatingPointError, match="logits"):
        ensure_finite("logits", torch.tensor([float("inf")]))


def test_validation_loss_is_recorded(tiny_training_fixture):
    history = run_training(tiny_training_fixture.config)
    assert set(history[0]) >= {
        "epoch", "train_loss", "val_loss", "val_f1", "val_auprc",
        "val_auroc", "threshold", "learning_rate",
    }
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
python -m pytest tests/test_train.py -v
```

Expected: imports fail because train/evaluate modules are absent.

- [ ] **Step 3: Implement evaluation and validation loss**

```python
@torch.inference_mode()
def evaluate_loader(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    total_loss = 0.0
    for inputs, targets, _ in loader:
        logits = model(inputs.to(device))
        ensure_finite("validation logits", logits)
        targets = targets.to(device)
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
        probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy())
        labels.append(targets.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities), total_loss / len(loader.dataset)
```

- [ ] **Step 4: Implement finite FP32 training and atomic history writes**

Reject `train.amp=true`. Check inputs, logits, loss, gradients, and parameters. Write history to a temporary file and replace `history.json` after every epoch. Save `best.pt` only when validation F1 improves.

- [ ] **Step 5: Run training tests**

```powershell
python -m pytest tests/test_train.py -v
```

Expected: synthetic one-epoch training passes; non-finite regression tests fail safely.

- [ ] **Step 6: Commit**

```powershell
git add -- "staged results/staged_results/evaluate.py" "staged results/staged_results/train.py" "staged results/tests/test_train.py"
git commit -m "feat: train staged fold ablations"
```

---

### Task 4: Shared Inference Collection and Error Analysis

**Files:**
- Create: `staged results/staged_results/analyze.py`
- Test: `staged results/tests/test_analysis.py`

**Interfaces:**
- Produces: `deterministic_sample(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray`
- Produces: `collect_analysis(model, loader, device, indices: set[int] | None) -> dict[str, np.ndarray]`
- Produces: `classify_errors(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray`
- Produces: `write_error_outputs(collection: dict[str, np.ndarray], output_dir: Path) -> None`

- [ ] **Step 1: Write failing sampling and error tests**

```python
def test_deterministic_sample_balances_classes():
    labels = np.array([0] * 20 + [1] * 10)
    chosen = deterministic_sample(labels, per_class=5, seed=42)
    assert np.bincount(labels[chosen]).tolist() == [5, 5]


def test_classify_errors():
    result = classify_errors(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert result.tolist() == ["TN", "FP", "FN", "TP"]
```

- [ ] **Step 2: Run the focused tests and verify failure**

```powershell
python -m pytest tests/test_analysis.py -k "sample or errors" -v
```

- [ ] **Step 3: Implement one-pass collection**

Collect probabilities, labels, sample indices, metadata columns, embeddings, and all attention arrays. Reject missing or non-finite arrays before saving `analysis_arrays.npz` and `analysis_metadata.json`.

- [ ] **Step 4: Implement error CSV and plots without pandas**

Use `csv.DictWriter` for sample predictions and summaries. Generate confusion matrix, probability histogram, and top-confidence FP/FN tables. Every figure includes the preliminary fold-0 subtitle.

- [ ] **Step 5: Run analysis tests**

```powershell
python -m pytest tests/test_analysis.py -k "sample or errors or collection" -v
```

- [ ] **Step 6: Commit**

```powershell
git add -- "staged results/staged_results/analyze.py" "staged results/tests/test_analysis.py"
git commit -m "feat: add staged inference and errors"
```

---

### Task 5: Attention Visualization

**Files:**
- Modify: `staged results/staged_results/analyze.py`
- Modify: `staged results/tests/test_analysis.py`

**Interfaces:**
- Consumes: arrays from `collect_analysis`.
- Produces: `plot_attention(collection: dict[str, np.ndarray], channels: Sequence[str], output_dir: Path) -> list[Path]`

- [ ] **Step 1: Write failing aggregation and output tests**

```python
def test_attention_aggregation_preserves_expected_axes(synthetic_collection):
    summary = summarize_attention(synthetic_collection)
    assert summary["scale_ictal"].shape == (18, 3)
    assert summary["graph_difference"].shape == (4, 18, 18)
    assert summary["pool_ictal"].shape == (18,)
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest tests/test_analysis.py -k attention -v
```

- [ ] **Step 3: Implement class summaries and plots**

Create temporal class profiles, `18 x 3` scale heatmaps, four graph-head matrices, ictal-minus-interictal matrices, pooling-weight plots, and fusion-gate distributions. Save CSV/NPZ summaries beside PNG files and use a consistent ictal/interictal palette.

- [ ] **Step 4: Run attention tests**

```powershell
python -m pytest tests/test_analysis.py -k attention -v
```

- [ ] **Step 5: Commit**

```powershell
git add -- "staged results/staged_results/analyze.py" "staged results/tests/test_analysis.py"
git commit -m "feat: visualize staged attention"
```

---

### Task 6: Bipolar-Lead Occlusion and Head Maps

**Files:**
- Modify: `staged results/staged_results/analyze.py`
- Modify: `staged results/tests/test_analysis.py`

**Interfaces:**
- Produces: `occlusion_importance(model, inputs: Tensor, device: torch.device) -> Tensor` with shape `[B, 18]`.
- Produces: `plot_lead_contributions(values: np.ndarray, channels: Sequence[str], output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write failing occlusion tests**

```python
def test_occlusion_returns_one_finite_value_per_lead(small_full_model):
    inputs = torch.randn(3, 18, 64)
    values = occlusion_importance(small_full_model.eval(), inputs, torch.device("cpu"))
    assert values.shape == (3, 18)
    assert torch.isfinite(values).all()
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest tests/test_analysis.py -k occlusion -v
```

- [ ] **Step 3: Implement inference-only occlusion**

```python
@torch.inference_mode()
def occlusion_importance(model, inputs, device):
    inputs = inputs.to(device)
    baseline = model(inputs).softmax(-1)[:, 1]
    values = []
    for channel in range(inputs.shape[1]):
        occluded = inputs.clone()
        occluded[:, channel] = 0.0
        probability = model(occluded).softmax(-1)[:, 1]
        values.append(baseline - probability)
    return torch.stack(values, dim=1).cpu()
```

- [ ] **Step 4: Implement montage and midpoint plots**

Use documented 10-20 two-dimensional coordinates. Draw bipolar leads as colored edges and interpolate only lead-midpoint values for the head map. Put `Model contribution; not clinical focus localization` directly under the title.

- [ ] **Step 5: Run occlusion and plotting tests**

```powershell
python -m pytest tests/test_analysis.py -k "occlusion or lead" -v
```

- [ ] **Step 6: Commit**

```powershell
git add -- "staged results/staged_results/analyze.py" "staged results/tests/test_analysis.py"
git commit -m "feat: add bipolar lead contribution maps"
```

---

### Task 7: PCA and t-SNE Embedding Analysis

**Files:**
- Modify: `staged results/staged_results/analyze.py`
- Modify: `staged results/tests/test_analysis.py`

**Interfaces:**
- Produces: `reduce_embeddings(features: np.ndarray, labels: np.ndarray, seed: int, perplexity: float) -> dict[str, np.ndarray]`.
- Produces: `plot_embeddings(reductions: dict[str, np.ndarray], labels: np.ndarray, patients: np.ndarray, output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_tsne_rejects_too_few_samples():
    with pytest.raises(ValueError, match="perplexity"):
        reduce_embeddings(np.ones((5, 8)), np.array([0, 0, 1, 1, 1]), 42, 5.0)


def test_reduction_returns_two_dimensions():
    rng = np.random.default_rng(42)
    result = reduce_embeddings(rng.normal(size=(40, 8)), np.tile([0, 1], 20), 42, 5.0)
    assert result["pca"].shape == (40, 2)
    assert result["tsne"].shape == (40, 2)
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest tests/test_analysis.py -k "tsne or reduction" -v
```

- [ ] **Step 3: Implement standardized PCA and deterministic t-SNE**

Fit `StandardScaler`, `PCA(n_components=2, random_state=seed)`, and `TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity)`. Save coordinates and exact settings.

- [ ] **Step 4: Plot temporal, correction, and fused embeddings**

Use color for class and marker shape for a bounded set of patient IDs. Add the descriptive-only caption to every image.

- [ ] **Step 5: Run embedding tests**

```powershell
python -m pytest tests/test_analysis.py -k "tsne or reduction or embedding" -v
```

- [ ] **Step 6: Commit**

```powershell
git add -- "staged results/staged_results/analyze.py" "staged results/tests/test_analysis.py"
git commit -m "feat: add staged embedding analysis"
```

---

### Task 8: CLI, Documentation, and End-to-End Verification

**Files:**
- Create: `staged results/README.md`
- Create: `staged results/requirements.txt`
- Modify: `staged results/staged_results/train.py`
- Modify: `staged results/staged_results/evaluate.py`
- Modify: `staged results/staged_results/analyze.py`
- Modify: `staged results/tests/test_analysis.py`

**Interfaces:**
- Produces commands `python -m staged_results.train`, `python -m staged_results.evaluate`, and `python -m staged_results.analyze {attention,heatmap,embeddings,errors,all}` when run from `staged results/`.

- [ ] **Step 1: Write CLI help and overwrite-safety tests**

```python
def test_existing_analysis_directory_requires_overwrite(tmp_path):
    output = tmp_path / "analysis"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_output_directory(output, overwrite=False)
```

- [ ] **Step 2: Run the full staged test suite before CLI completion**

```powershell
python -m pytest -q
```

Expected: CLI-specific test fails.

- [ ] **Step 3: Implement argparse CLIs and output containment**

Training accepts multiple config overlays and no fold argument. Evaluation and analysis require `--checkpoint`; analysis also accepts `--device`, `--output-dir`, and `--overwrite`. Resolve paths and reject outputs outside the staged root.

- [ ] **Step 4: Write README commands**

Document exact quoted working-directory commands, one command per ablation, existing full-checkpoint analysis, output files, GPU expectations, and the fold-0 scientific disclaimer. Do not include a shell loop.

- [ ] **Step 5: Run static and unit verification**

From `staged results/`:

```powershell
python -m compileall -q staged_results tests
python -m pytest -q
python -m staged_results.train --help
python -m staged_results.evaluate --help
python -m staged_results.analyze --help
```

Expected: compile succeeds, all tests pass, and all help commands exit 0.

- [ ] **Step 6: Inspect repository isolation**

```powershell
git status --short
git diff --check
```

Expected: implementation changes exist only under `staged results/`, plus the approved design and plan documents; pre-existing user files remain untouched.

- [ ] **Step 7: Commit final integration**

```powershell
git add -- "staged results"
git commit -m "feat: add isolated staged results suite"
```

- [ ] **Step 8: Record unavailable real-data verification**

If local Torch or prepared artifacts are unavailable, report that real fold-0 training, checkpoint loading, GPU attention extraction, latency, memory, and final figures remain server-side validation tasks. Provide the exact first server smoke command from the README rather than claiming success.
