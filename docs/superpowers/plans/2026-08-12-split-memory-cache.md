# Split Memory Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize only the active train, validation, and test windows in RAM so GPU training no longer performs random mmap reads every epoch.

**Architecture:** Keep `EEGWindowDataset` as the single source of window extraction and normalization truth. Add one small in-memory Dataset that fills a contiguous `float32` tensor in record/time read order while storing each sample at its original split position; `create_dataloaders` selects it from configuration and forces zero workers in RAM mode.

**Tech Stack:** Python 3.10, NumPy, PyTorch, pytest, YAML.

## Global Constraints

- Keep the four-file Python structure; do not add a new production module or dependency.
- Do not change split membership, labels, balancing, patient-global transductive normalization, model, optimizer, threshold selection, or checkpoint structure.
- The in-memory path must preserve the exact logical sample order and `float32` values returned by the mmap path.
- Missing `train.cache_in_memory` must retain the current mmap behavior.
- Ten mixed-CV folds will be submitted as ten independent scheduler jobs, not one serial job.

---

### Task 1: Equivalent in-memory Dataset

**Files:**
- Modify: `tests/test_cache_smoke.py`
- Modify: `src/data.py`

**Interfaces:**
- Consumes: `EEGWindowDataset(config: dict[str, Any], indices: np.ndarray)`.
- Produces: `InMemoryEEGWindowDataset(source: EEGWindowDataset, split_name: str)` implementing `__len__` and `__getitem__`.

- [ ] **Step 1: Add a minimal indexed-cache test helper**

Add `torch` and the two Dataset imports, then create a fixture independent of seizure-window generation:

```python
import torch

from src.data import (
    EEGWindowDataset,
    InMemoryEEGWindowDataset,
    create_dataloaders,
    prepare_windows,
)


def _indexed_cache_config(tmp_path: Path) -> dict:
    prepared = tmp_path / "indexed"
    records = prepared / "records"
    windows = prepared / "windows"
    records.mkdir(parents=True)
    windows.mkdir()
    signal = np.arange(18 * 800, dtype=np.float32).reshape(18, 800)
    record_path = records / "chb01_01.npy"
    np.save(record_path, signal)
    manifest = {
        "records": [
            {
                "name": "chb01_01.edf",
                "patient": "chb01",
                "path": str(record_path),
                "samples": 800,
                "seizures_seconds": [],
            }
        ]
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(
        prepared / "normalization.npz",
        chb01_mean=np.zeros(18, dtype=np.float32),
        chb01_std=np.ones(18, dtype=np.float32),
    )
    np.savez(
        windows / "2s.npz",
        record=np.zeros(40, dtype=np.int32),
        start=np.arange(40, dtype=np.int64) * 20,
        label=np.tile(np.array([0, 1], dtype=np.uint8), 20),
    )
    return {
        "seed": 42,
        "data": {
            "prepared_dir": str(prepared),
            "sample_rate": 10,
            "window_seconds": 2,
        },
        "split": {
            "protocol": "mixed_10fold",
            "folds": 2,
            "val_fraction": 0.25,
            "balance_ratio": 1.0,
        },
        "train": {"batch_size": 4, "num_workers": 2, "device": "cpu"},
    }
```

- [ ] **Step 2: Write the failing equivalence and order test**

Add a test that constructs a lazy Dataset with deliberately non-monotonic catalog indices, materializes it, and checks every tensor and label:

```python
def test_in_memory_dataset_matches_mmap_values_and_split_order(tmp_path: Path) -> None:
    config = _indexed_cache_config(tmp_path)
    indices = np.array([3, 0, 2, 1], dtype=np.int64)
    lazy = EEGWindowDataset(config, indices)
    cached = InMemoryEEGWindowDataset(lazy, "train")

    assert len(cached) == len(lazy)
    for index in range(len(lazy)):
        expected_input, expected_label = lazy[index]
        actual_input, actual_label = cached[index]
        torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
        assert actual_input.dtype == torch.float32
        assert actual_label.item() == expected_label.item()
```

- [ ] **Step 3: Run the new test and verify RED**

Run:

```powershell
python -m pytest tests/test_cache_smoke.py::test_in_memory_dataset_matches_mmap_values_and_split_order -q
```

Expected: collection fails because `InMemoryEEGWindowDataset` does not exist.

- [ ] **Step 4: Implement the minimal in-memory Dataset**

In `src/data.py`, import `time`, then add `InMemoryEEGWindowDataset` immediately after `EEGWindowDataset`:

```python
class InMemoryEEGWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Materialize one split while preserving its logical sample order."""

    def __init__(self, source: EEGWindowDataset, split_name: str):
        shape = (len(source), len(next(iter(source.means.values()))), source.size)
        gib = np.prod(shape, dtype=np.int64) * np.dtype(np.float32).itemsize / 1024**3
        LOGGER.info("Materializing %s: %d windows, %.2f GiB", split_name, len(source), gib)
        started = time.perf_counter()
        try:
            inputs = np.empty(shape, dtype=np.float32)
        except MemoryError as error:
            raise MemoryError(
                f"Cannot materialize {split_name} ({gib:.2f} GiB); "
                "set train.cache_in_memory to false"
            ) from error
        order = np.lexsort((source.starts, source.record_ids))
        for position in order:
            inputs[int(position)] = source[int(position)][0].numpy()
        self.inputs = torch.from_numpy(inputs)
        self.labels = torch.from_numpy(source.labels.astype(np.int64, copy=True))
        LOGGER.info("Materialized %s in %.1f seconds", split_name, time.perf_counter() - started)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index]
```

Keep the implementation within this class; do not introduce cache managers or persistence layers.

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```powershell
python -m pytest tests/test_cache_smoke.py::test_in_memory_dataset_matches_mmap_values_and_split_order -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit the Dataset change**

```powershell
git add src/data.py tests/test_cache_smoke.py
git commit -m "feat: add split memory dataset"
```

### Task 2: Select RAM loading through configuration

**Files:**
- Modify: `tests/test_cache_smoke.py`
- Modify: `src/data.py`
- Modify: `configs/base.yaml`

**Interfaces:**
- Consumes: `InMemoryEEGWindowDataset(source, split_name)` from Task 1.
- Produces: existing `create_dataloaders(config, split_id) -> dict[str, DataLoader]`, with optional `train.cache_in_memory: bool`.

- [ ] **Step 1: Write the failing DataLoader selection test**

Add:

```python
def test_create_dataloaders_uses_zero_worker_memory_cache(tmp_path: Path) -> None:
    config = _indexed_cache_config(tmp_path)
    config["train"]["cache_in_memory"] = True

    loaders = create_dataloaders(config, 0)

    assert all(loader.num_workers == 0 for loader in loaders.values())
    assert all(isinstance(loader.dataset, InMemoryEEGWindowDataset) for loader in loaders.values())
    inputs, labels = next(iter(loaders["train"]))
    assert inputs.dtype == torch.float32
    assert labels.dtype == torch.long
```

Add a separate fallback test:

```python
def test_create_dataloaders_defaults_to_mmap_dataset(tmp_path: Path) -> None:
    config = _indexed_cache_config(tmp_path)

    loaders = create_dataloaders(config, 0)

    assert isinstance(loaders["train"].dataset, EEGWindowDataset)
    assert loaders["train"].num_workers == 2
```

- [ ] **Step 2: Run both cache tests and verify RED**

Run:

```powershell
python -m pytest tests/test_cache_smoke.py -q
```

Expected: the new selection test fails because `create_dataloaders` still always creates `EEGWindowDataset` with configured workers.

- [ ] **Step 3: Implement configuration selection with no duplicate loader logic**

In `create_dataloaders`, read the option safely and build each Dataset through one local expression:

```python
cache_in_memory = bool(config["train"].get("cache_in_memory", False))
workers = 0 if cache_in_memory else int(config["train"]["num_workers"])

datasets = {}
for name, indices in split.items():
    source = EEGWindowDataset(config, indices)
    datasets[name] = (
        InMemoryEEGWindowDataset(source, name) if cache_in_memory else source
    )
```

Build the existing DataLoader dictionary from `datasets.items()`. Preserve `shuffle=name == "train"`, batch size, CUDA pinning, and `persistent_workers=workers > 0`.

Add to `configs/base.yaml`:

```yaml
train:
  cache_in_memory: true
```

- [ ] **Step 4: Run cache tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_cache_smoke.py -q
```

Expected: all cache smoke tests pass.

- [ ] **Step 5: Run all data and training regression tests**

Run:

```powershell
python -m pytest tests/test_data.py tests/test_train.py tests/test_evaluate.py -q
```

Expected: all selected regression tests pass.

- [ ] **Step 6: Commit the integration**

```powershell
git add src/data.py tests/test_cache_smoke.py configs/base.yaml
git commit -m "perf: load active split windows into memory"
```

### Task 3: Document, verify, and prepare server profiling

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `train.cache_in_memory` and unchanged `src.train --fold` interface.
- Produces: reproducible server commands for profiling and ten independent folds.

- [ ] **Step 1: Update the README**

Document these exact facts:

- only the active split is materialized; the 61 GiB continuous-record cache is not loaded;
- 2-second fold 0 measured about 1.13 GiB for train/val/test under a 32 GiB container limit;
- initialization reads in record/time order and later epochs use RAM;
- set `train.cache_in_memory: false` to restore mmap loading;
- do not increase workers in RAM mode because the implementation intentionally forces zero workers;
- mixed 10-fold uses ten independent scheduler jobs.

Add the submission loop:

```bash
for fold in {0..9}; do
  lab-submit pytorch 8 bash -lc "cd /workspace/project && python3 -m src.train configs/base.yaml configs/mixed_10fold.yaml configs/window_2s.yaml --fold $fold"
done
```

State that this loop is only run after the real-data profiling gate passes.

- [ ] **Step 2: Run syntax and complete test verification**

Run:

```powershell
python -m compileall -q src tests
python -m pytest tests -q
```

Expected: compile command exits 0 and the complete suite passes with no failures.

- [ ] **Step 3: Inspect the final diff for scope and accidental generated files**

Run:

```powershell
git diff --check
git status --short
git diff -- src/data.py tests/test_cache_smoke.py configs/base.yaml README.md
```

Expected: no whitespace errors; only the planned source, test, configuration, and README changes are present, apart from previously untracked baseline files.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md
git commit -m "docs: explain split memory training"
```

- [ ] **Step 5: Run the server acceptance gate after upload**

Repeat the existing 30-batch timing with the updated code. Acceptance requires all of the following:

- `data_wait` is materially below the prior 1.1854 seconds per batch;
- tensors remain `[32, 18, 512]` with `float32` inputs and integer labels;
- forward and backward complete without CUDA or memory errors;
- peak host use remains below the 32 GiB container limit;
- only after these checks pass, submit folds 0 through 9 as independent jobs.
