# LOPO Mixed-Window Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated 24-case LOPO variant with paper-aligned mixed-window 10% validation, split-specific negative ratios, balanced-accuracy threshold selection, and reproducible diagnostics without changing the existing LOPO baseline or mixed-10-fold behavior.

**Architecture:** Extend the compact `src/data.py` split API with optional LOPO validation strategy and per-split ratios while preserving all existing positional calls and defaults. Add a general validation-threshold selector in `src/evaluate.py`, then have `src/train.py` use the configured selection metric and persist compact split/probability diagnostics. A new overlay config enables every new behavior; old configs continue through their current paths.

**Tech Stack:** Python 3.10+, NumPy, scikit-learn, PyTorch, PyYAML, pytest.

## Global Constraints

- Keep `configs/lopo.yaml` unchanged as the case-disjoint LOPO baseline.
- Keep `configs/mixed_10fold.yaml`, its exact fold-zero indices, cached artifacts, and run semantics unchanged.
- Reuse `/workspace/output/prepared_lopo_24case`; do not rebuild EDF records or window catalogs.
- Keep `chb17a`/`chb17b`/`chb17c` mapped to `chb17` only in the existing LOPO artifact path.
- Split the held-out test case before validation sampling or balancing.
- Select epochs and thresholds with validation data only; never use test labels or probabilities for model selection.
- Preserve old `split.balance_ratio` and F1-based checkpoint behavior when new keys are absent.
- Do not add dependencies or reorganize the four-file project.
- Treat the new variant as 24-case leave-one-case-out, not strict unseen-person LOPO, because `chb01` and `chb21` are the same subject.

---

### Task 1: Mixed-window LOPO split and independent class ratios

**Files:**
- Modify: `tests/test_data.py:241-342`
- Modify: `src/data.py:67-135`

**Interfaces:**
- Consumes: existing `make_split(labels, patients, protocol, split_id, seed, folds, val_fraction, balance_ratio)` calls.
- Produces: backward-compatible `make_split(..., *, validation_strategy="case_holdout", negative_ratios=None)` returning `dict[str, np.ndarray]`.
- Produces: private `_make_split_with_summary(...) -> tuple[dict[str, np.ndarray], dict[str, Any]]` for Task 3.
- Produces: summary shape `{"validation_strategy": str, "splits": {name: {"case_ids": list[str], "pre_balance": counts, "post_balance": counts}}}`.

- [ ] **Step 1: Write a failing test for paper-aligned 10% validation**

Add a test with hand-built balanced labels so balancing does not obscure the raw 10% size:

```python
def test_lopo_mixed_window_validation_is_stratified_reproducible_and_test_isolated() -> None:
    patients = np.repeat(["chb01", "chb02", "chb03", "chb04"], 20)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 40)

    first = make_split(
        labels,
        patients,
        "lopo",
        "chb04",
        42,
        10,
        0.10,
        1.0,
        validation_strategy="mixed_windows",
    )
    second = make_split(
        labels,
        patients,
        "lopo",
        "chb04",
        42,
        10,
        0.10,
        1.0,
        validation_strategy="mixed_windows",
    )

    assert np.array_equal(first["val"], second["val"])
    assert len(first["val"]) == 6
    assert np.count_nonzero(labels[first["val"]] == 0) == 3
    assert np.count_nonzero(labels[first["val"]] == 1) == 3
    assert set(patients[first["test"]]) == {"chb04"}
    assert "chb04" not in set(patients[first["train"]])
    assert "chb04" not in set(patients[first["val"]])
    assert set(patients[first["train"]]) & set(patients[first["val"]])
    assert not set(first["train"]) & set(first["val"])
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
python -m pytest tests/test_data.py::test_lopo_mixed_window_validation_is_stratified_reproducible_and_test_isolated -q
```

Expected: FAIL with `TypeError: make_split() got an unexpected keyword argument 'validation_strategy'`.

- [ ] **Step 3: Add failing tests for independent negative ratios and validation errors**

```python
def test_lopo_supports_independent_negative_ratios() -> None:
    patients = np.repeat(["chb01", "chb02", "chb03", "chb04"], 50)
    labels = np.tile(np.array([0, 0, 0, 0, 1], dtype=np.int64), 40)

    split = make_split(
        labels,
        patients,
        "lopo",
        "chb04",
        42,
        10,
        0.10,
        1.0,
        validation_strategy="mixed_windows",
        negative_ratios={"train": 2.0, "val": 1.0, "test": 1.0},
    )

    for name, expected_ratio in {"train": 2, "val": 1, "test": 1}.items():
        split_labels = labels[split[name]]
        positives = np.count_nonzero(split_labels == 1)
        negatives = np.count_nonzero(split_labels == 0)
        assert negatives == positives * expected_ratio


@pytest.mark.parametrize(
    ("validation_strategy", "negative_ratios", "message"),
    [
        ("unknown", None, "Unsupported LOPO validation strategy"),
        ("mixed_windows", {"train": 0.0, "val": 1.0, "test": 1.0}, "train negative ratio"),
    ],
)
def test_lopo_rejects_invalid_split_options(
    validation_strategy: str,
    negative_ratios: dict[str, float] | None,
    message: str,
) -> None:
    labels = np.tile(np.array([0, 0, 1], dtype=np.int64), 20)
    patients = np.repeat(["chb01", "chb02", "chb03", "chb04"], 15)

    with pytest.raises(ValueError, match=message):
        make_split(
            labels,
            patients,
            "lopo",
            "chb04",
            42,
            10,
            0.10,
            1.0,
            validation_strategy=validation_strategy,
            negative_ratios=negative_ratios,
        )
```

- [ ] **Step 4: Run the ratio/error tests and verify RED**

Run:

```powershell
python -m pytest tests/test_data.py -k "independent_negative or invalid_split" -q
```

Expected: FAIL because the new keyword arguments and validation do not exist.

- [ ] **Step 5: Implement raw-pool splitting, independent balancing, and summary generation**

In `src/data.py`, keep `_balance` focused on one split and add explicit ratio validation:

```python
def _balance(
    indices: np.ndarray,
    labels: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if ratio <= 0:
        raise ValueError(f"Negative ratio must be positive, got {ratio}")
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Every split must contain both seizure and non-seizure windows")
    wanted = min(len(negative), max(1, int(round(len(positive) * ratio))))
    chosen = rng.choice(negative, size=wanted, replace=False)
    return rng.permutation(np.concatenate([positive, chosen])).astype(np.int64)
```

Add helpers with these exact contracts:

```python
def _class_counts(indices: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    subset = labels[indices]
    positive = int(np.count_nonzero(subset == 1))
    negative = int(np.count_nonzero(subset == 0))
    return {"positive": positive, "negative": negative, "total": positive + negative}


def _negative_ratios(
    balance_ratio: float,
    ratios: dict[str, float] | None,
) -> dict[str, float]:
    resolved = {
        name: float((ratios or {}).get(name, balance_ratio))
        for name in ("train", "val", "test")
    }
    for name, ratio in resolved.items():
        if ratio <= 0:
            raise ValueError(f"{name} negative ratio must be positive, got {ratio}")
    return resolved
```

Implement `_make_split_with_summary` so `mixed_windows` first removes the test case, then uses a stratified `train_test_split`:

```python
remaining = all_indices[patients != test_patient]
if validation_strategy == "mixed_windows":
    train, val = train_test_split(
        remaining,
        test_size=val_fraction,
        stratify=labels[remaining],
        random_state=seed,
    )
elif validation_strategy == "case_holdout":
    remaining_patients = np.unique(patients[patients != test_patient])
    rng.shuffle(remaining_patients)
    count = min(
        len(remaining_patients) - 1,
        max(1, round(len(remaining_patients) * val_fraction)),
    )
    val_patients = remaining_patients[:count]
    val = all_indices[np.isin(patients, val_patients)]
    train = all_indices[
        (patients != test_patient) & ~np.isin(patients, val_patients)
    ]
else:
    raise ValueError(
        f"Unsupported LOPO validation strategy: {validation_strategy}"
    )
```

Build pre-balance and post-balance summaries from literal split names, sorted case IDs, and `_class_counts`. Keep `make_split` as a wrapper returning only balanced indices so existing callers remain unchanged:

```python
def make_split(
    labels: np.ndarray,
    patients: np.ndarray,
    protocol: str,
    split_id: int | str,
    seed: int,
    folds: int,
    val_fraction: float,
    balance_ratio: float,
    *,
    validation_strategy: str = "case_holdout",
    negative_ratios: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    split, _ = _make_split_with_summary(
        labels,
        patients,
        protocol,
        split_id,
        seed,
        folds,
        val_fraction,
        balance_ratio,
        validation_strategy=validation_strategy,
        negative_ratios=negative_ratios,
    )
    return split
```

- [ ] **Step 6: Run the focused split tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_data.py -k "mixed_window_validation or independent_negative or invalid_split or mixed_fold_zero or all_24_lopo" -q
```

Expected: all selected tests PASS, including the exact mixed-fold-zero regression.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/data.py tests/test_data.py
git commit -m "feat: add mixed-window LOPO validation split"
```

---

### Task 2: Balanced-accuracy threshold selection

**Files:**
- Modify: `tests/test_evaluate.py:1-24`
- Modify: `src/evaluate.py:12-68`

**Interfaces:**
- Consumes: `labels: np.ndarray`, `probabilities: np.ndarray` containing both binary classes.
- Produces: `select_threshold(labels, probabilities, metric="f1") -> float`.
- Preserves: `best_f1_threshold(labels, probabilities) -> float` as a compatibility wrapper.
- Extends: `binary_metrics(...)` with `balanced_accuracy`.

- [ ] **Step 1: Write failing tests for balanced accuracy and deterministic threshold ties**

```python
from src.evaluate import (
    best_f1_threshold,
    binary_metrics,
    select_threshold,
)


def test_metrics_include_balanced_accuracy() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = binary_metrics(labels, probabilities, threshold=0.5)

    assert metrics["balanced_accuracy"] == 0.75


def test_balanced_accuracy_threshold_uses_highest_threshold_on_tie() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.4, 0.9])

    threshold = select_threshold(labels, probabilities, "balanced_accuracy")

    assert threshold == 0.9


def test_threshold_selection_rejects_unknown_metric() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])

    with pytest.raises(ValueError, match="Unsupported threshold metric"):
        select_threshold(labels, probabilities, "accuracy")
```

Add `import pytest` at the top of the test file.

- [ ] **Step 2: Run the new evaluate tests and verify RED**

Run:

```powershell
python -m pytest tests/test_evaluate.py -q
```

Expected: collection FAIL because `select_threshold` does not exist.

- [ ] **Step 3: Implement threshold selection and balanced accuracy**

Add:

```python
def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
) -> float:
    """Select a deterministic decision threshold using validation data only."""
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        if len(thresholds) == 0:
            return 0.5
        scores = (
            2
            * precision[:-1]
            * recall[:-1]
            / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        )
        return float(thresholds[int(np.argmax(scores))])
    if metric == "balanced_accuracy":
        thresholds = np.unique(probabilities)
        scores = []
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(np.int64)
            tn, fp, fn, tp = confusion_matrix(
                labels, predictions, labels=[0, 1]
            ).ravel()
            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            scores.append((sensitivity + specificity) / 2)
        best = np.flatnonzero(np.isclose(scores, np.max(scores)))
        return float(thresholds[int(best[-1])])
    raise ValueError(f"Unsupported threshold metric: {metric}")


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Preserve the original public F1 threshold API."""
    return select_threshold(labels, probabilities, "f1")
```

In `binary_metrics`, add:

```python
"balanced_accuracy": float((sensitivity + specificity) / 2),
```

- [ ] **Step 4: Run evaluate tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_evaluate.py -q
```

Expected: all tests PASS, including the original `threshold == 0.45` assertion.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: select validation threshold by balanced accuracy"
```

---

### Task 3: Config propagation, isolated split cache, and split diagnostics

**Files:**
- Modify: `tests/test_data.py`
- Modify: `src/data.py:564-610`

**Interfaces:**
- Consumes: optional `split.validation_strategy`, `split.train_negative_ratio`, `split.val_negative_ratio`, and `split.test_negative_ratio`.
- Produces: `create_dataloaders_with_summary(config, split_id) -> tuple[dict[str, DataLoader], dict[str, Any]]`.
- Preserves: `create_dataloaders(config, split_id) -> dict[str, DataLoader]` for evaluation and existing code.
- Produces: distinct mixed-validation split cache directory containing the split configuration fingerprint.

- [ ] **Step 1: Write failing tests for config defaults and split cache isolation**

Add a focused helper-level test instead of mocking datasets:

```python
from src.data import _split_options, _split_variant_name


def test_split_options_preserve_legacy_defaults() -> None:
    strategy, ratios = _split_options({"balance_ratio": 1.0})

    assert strategy == "case_holdout"
    assert ratios == {"train": 1.0, "val": 1.0, "test": 1.0}


def test_mixed_validation_uses_distinct_split_variant_name() -> None:
    baseline = _split_variant_name(
        "lopo",
        42,
        {
            "balance_ratio": 1.0,
            "val_fraction": 0.15,
        },
    )
    mixed_validation = _split_variant_name(
        "lopo",
        42,
        {
            "balance_ratio": 1.0,
            "validation_strategy": "mixed_windows",
            "val_fraction": 0.10,
            "train_negative_ratio": 2.0,
            "val_negative_ratio": 1.0,
            "test_negative_ratio": 1.0,
        },
    )

    assert baseline == "lopo_seed42"
    assert mixed_validation.startswith("lopo_mixed_windows_")
    assert mixed_validation.endswith("_seed42")
    assert mixed_validation != baseline
```

- [ ] **Step 2: Run the helper tests and verify RED**

Run:

```powershell
python -m pytest tests/test_data.py -k "split_options or split_variant_name" -q
```

Expected: collection FAIL because `_split_options` and `_split_variant_name` do not exist.

- [ ] **Step 3: Implement config resolution and variant naming**

Add:

```python
def _split_options(split_config: dict[str, Any]) -> tuple[str, dict[str, float]]:
    strategy = str(split_config.get("validation_strategy", "case_holdout"))
    fallback = float(split_config["balance_ratio"])
    ratios = {
        "train": float(split_config.get("train_negative_ratio", fallback)),
        "val": float(split_config.get("val_negative_ratio", fallback)),
        "test": float(split_config.get("test_negative_ratio", fallback)),
    }
    return strategy, _negative_ratios(fallback, ratios)


def _split_variant_name(
    protocol: str,
    seed: int,
    split_config: dict[str, Any],
) -> str:
    strategy = str(split_config.get("validation_strategy", "case_holdout"))
    if strategy == "case_holdout" and not any(
        key in split_config
        for key in (
            "train_negative_ratio",
            "val_negative_ratio",
            "test_negative_ratio",
        )
    ):
        return f"{protocol}_seed{seed}"
    relevant = {
        key: split_config.get(key)
        for key in (
            "validation_strategy",
            "val_fraction",
            "balance_ratio",
            "train_negative_ratio",
            "val_negative_ratio",
            "test_negative_ratio",
        )
    }
    return f"{protocol}_{strategy}_{_fingerprint(relevant)}_seed{seed}"
```

Refactor dataloader construction into a private helper that calls `_make_split_with_summary`. Expose two stable public functions:

```python
def create_dataloaders(
    config: dict[str, Any], split_id: int | str
) -> dict[str, DataLoader]:
    loaders, _ = _create_dataloaders(config, split_id)
    return loaders


def create_dataloaders_with_summary(
    config: dict[str, Any], split_id: int | str
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    return _create_dataloaders(config, split_id)
```

Use `_split_variant_name(...)` for the split directory so the old path stays exact and the new configuration cannot overwrite it. Save the compact summary next to the `.npz` file as `<split_id>.json` with UTF-8 JSON and indentation.

- [ ] **Step 4: Run data tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_data.py -q
```

Expected: all data tests PASS, including exact mixed-fold-zero indices and all 24 baseline LOPO folds.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/data.py tests/test_data.py
git commit -m "feat: persist isolated LOPO split diagnostics"
```

---

### Task 4: Training selection metric and probability diagnostics

**Files:**
- Modify: `tests/test_train.py:1-30`
- Modify: `src/train.py:20-183`

**Interfaces:**
- Consumes: `create_dataloaders_with_summary`, `select_threshold`, and `train.threshold_metric`.
- Produces: `probability_summary(labels, probabilities) -> dict[str, dict[str, float]]`.
- Produces: checkpoints containing `selection_metric`, `selection_score`, `best_score`, and `validation_probability_summary` while retaining `best_f1`.
- Produces: `diagnostics.json` in the run directory for the current best checkpoint.

- [ ] **Step 1: Write failing tests for probability summaries and selection score**

```python
from src.train import probability_summary, seed_everything, selection_score, train_epoch


def test_probability_summary_reports_each_class_without_raw_predictions() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

    summary = probability_summary(labels, probabilities)

    assert summary["negative"]["min"] == 0.1
    assert summary["negative"]["median"] == 0.2
    assert summary["negative"]["max"] == 0.3
    assert summary["positive"]["min"] == 0.6
    assert summary["positive"]["median"] == 0.8
    assert summary["positive"]["max"] == 0.9
    assert "probabilities" not in summary


def test_selection_score_uses_configured_metric() -> None:
    metrics = {"f1": 0.81, "balanced_accuracy": 0.87}

    assert selection_score(metrics, "f1") == 0.81
    assert selection_score(metrics, "balanced_accuracy") == 0.87


def test_selection_score_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported selection metric"):
        selection_score({"f1": 0.81}, "accuracy")
```

Add `import pytest` and extend the `src.train` import.

- [ ] **Step 2: Run train tests and verify RED**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: collection FAIL because the two helpers do not exist.

- [ ] **Step 3: Implement compact probability and selection helpers**

```python
def probability_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for label, name in ((0, "negative"), (1, "positive")):
        values = probabilities[labels == label]
        if len(values) == 0:
            raise ValueError(f"Validation data has no {name} samples")
        result[name] = {
            "min": float(np.min(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    return result


def selection_score(metrics: dict[str, Any], metric: str) -> float:
    if metric not in {"f1", "balanced_accuracy"}:
        raise ValueError(f"Unsupported selection metric: {metric}")
    return float(metrics[metric])
```

- [ ] **Step 4: Update the training loop with backward-compatible checkpoint state**

Change imports to use `create_dataloaders_with_summary` and `select_threshold`. Resolve the metric once:

```python
selection_metric = str(config["train"].get("threshold_metric", "f1"))
loaders, split_summary = create_dataloaders_with_summary(config, split_id)
LOGGER.info("Split summary: %s", json.dumps(split_summary, sort_keys=True))
best_score = -1.0
```

Resume old and new checkpoints safely:

```python
best_score = float(resume.get("best_score", resume.get("best_f1", -1.0)))
```

At each epoch:

```python
threshold = select_threshold(val_labels, val_probabilities, selection_metric)
metrics = binary_metrics(val_labels, val_probabilities, threshold)
score = selection_score(metrics, selection_metric)
summary = probability_summary(val_labels, val_probabilities)
improved = score > best_score
best_score = max(best_score, score)
```

Save these fields in `state`:

```python
"selection_metric": selection_metric,
"selection_score": score,
"best_score": best_score,
"best_f1": max(float(state_best_f1), float(metrics["f1"])),
"split_summary": split_summary,
"validation_probability_summary": summary,
```

Maintain a separate `best_f1` variable for compatibility rather than reading `state` while constructing it. When improved, write `best.pt` and:

```python
diagnostics = {
    "split_id": split_id,
    "split": split_summary,
    "best_epoch": epoch,
    "selection_metric": selection_metric,
    "selection_score": score,
    "threshold": threshold,
    "validation_metrics": metrics,
    "validation_probability_summary": summary,
}
(run_dir / "diagnostics.json").write_text(
    json.dumps(diagnostics, indent=2), encoding="utf-8"
)
LOGGER.info(
    "Best validation probability summary: %s",
    json.dumps(summary, sort_keys=True),
)
```

Log the generic selection name and score instead of labeling every run `val_f1`:

```python
LOGGER.info(
    "epoch=%d loss=%.4f val_%s=%.4f threshold=%.6f",
    epoch,
    train_loss,
    selection_metric,
    score,
    threshold,
)
```

- [ ] **Step 5: Run train and evaluate tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_train.py tests/test_evaluate.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/train.py tests/test_train.py
git commit -m "feat: select LOPO checkpoints by balanced accuracy"
```

---

### Task 5: New overlay configuration and reproducible usage documentation

**Files:**
- Create: `configs/lopo_mixedval.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: new configuration keys implemented in Tasks 1-4.
- Produces: server command using `configs/base.yaml configs/lopo_mixedval.yaml configs/window_2s.yaml`.

- [ ] **Step 1: Add the isolated configuration**

Create exactly:

```yaml
data:
  prepared_dir: /workspace/output/prepared_lopo_24case
  merge_chb17: true

split:
  protocol: lopo
  validation_strategy: mixed_windows
  val_fraction: 0.10
  train_negative_ratio: 2.0
  val_negative_ratio: 1.0
  test_negative_ratio: 1.0

train:
  threshold_metric: balanced_accuracy
```

- [ ] **Step 2: Document the scientific boundary and pilot commands**

In the LOPO README section, add a subsection explaining:

- held-out test case remains fully isolated;
- validation is a stratified 10% random window split from the other 23 cases;
- train and validation may share cases and neighboring windows, so this is weaker than `configs/lopo.yaml`;
- training uses 2:1 negatives while validation/test remain 1:1;
- threshold and epoch selection maximize validation balanced accuracy;
- the existing mixed-10-fold files and prepared artifacts are untouched.

Provide these commands:

```bash
lab-submit pytorch 8 bash -lc "cd /workspace/project && /usr/bin/python -m src.train configs/base.yaml configs/lopo_mixedval.yaml configs/window_2s.yaml --fold chb01"
lab-submit pytorch 8 bash -lc "cd /workspace/project && /usr/bin/python -m src.train configs/base.yaml configs/lopo_mixedval.yaml configs/window_2s.yaml --fold chb02"
```

Do not document a 24-fold submission until the two development folds have been reviewed and the configuration frozen.

- [ ] **Step 3: Validate merged configuration values**

Run:

```powershell
python -c "from src.data import load_config; c=load_config(['configs/base.yaml','configs/lopo_mixedval.yaml','configs/window_2s.yaml']); print(c['split']); print(c['train']['threshold_metric']); print(c['data']['prepared_dir'])"
```

Expected output includes:

```text
'validation_strategy': 'mixed_windows'
'val_fraction': 0.1
'train_negative_ratio': 2.0
'val_negative_ratio': 1.0
'test_negative_ratio': 1.0
balanced_accuracy
/workspace/output/prepared_lopo_24case
```

- [ ] **Step 4: Commit Task 5**

```powershell
git add configs/lopo_mixedval.yaml README.md
git commit -m "docs: add mixed-validation LOPO pilot workflow"
```

---

### Task 6: Full regression and source audit

**Files:**
- Verify: `src/data.py`
- Verify: `src/evaluate.py`
- Verify: `src/train.py`
- Verify: `configs/lopo_mixedval.yaml`
- Verify: `README.md`
- Verify: `tests/`

**Interfaces:**
- Consumes: all deliverables from Tasks 1-5.
- Produces: fresh local evidence that syntax, the full test suite, legacy config behavior, and the new merged config are valid.

- [ ] **Step 1: Compile source files**

Run:

```powershell
python -m compileall -q src
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: every test passes; no test is skipped or relaxed to obtain a green run.

- [ ] **Step 3: Verify old configuration defaults through real config loading**

Run:

```powershell
python -c "from src.data import load_config,_split_options,_split_variant_name; c=load_config(['configs/base.yaml','configs/lopo.yaml','configs/window_2s.yaml']); print(_split_options(c['split'])); print(_split_variant_name(c['split']['protocol'],c['seed'],c['split']))"
```

Expected:

```text
('case_holdout', {'train': 1.0, 'val': 1.0, 'test': 1.0})
lopo_seed42
```

- [ ] **Step 4: Inspect the final diff for unintended mixed-10-fold or artifact changes**

Run:

```powershell
git diff --check HEAD~5..HEAD
git status --short
```

Confirm that no prepared data, outputs, checkpoints, local literature PDFs, `fp32_probe.py`, or unrelated untracked files were added.

- [ ] **Step 5: Record server verification still required**

The local checkout cannot prove real CHB-MIT split sizes, RTX 4090 execution, or performance improvement. After uploading only `configs`, `src`, `tests`, `docs`, and `README.md`, run on the server:

```bash
cd /workspace/project
/usr/bin/python -m pytest -q
/usr/bin/python -m src.data configs/base.yaml configs/lopo_mixedval.yaml configs/window_2s.yaml --check-lopo --windows 2
```

Then submit only the `chb01` and `chb02` development folds from Task 5. Do not claim improved LOPO performance until both new `test_metrics.json` files and `diagnostics.json` files have been inspected against the existing baseline.

- [ ] **Step 6: Commit any verification-only documentation correction if needed**

If verification reveals a documentation mismatch, edit only that mismatch and run the relevant command again before committing:

```powershell
git add README.md docs/superpowers/plans/2026-08-16-lopo-mixed-validation.md
git commit -m "docs: align LOPO mixed-validation verification"
```

If no correction is needed, do not create an empty commit.
