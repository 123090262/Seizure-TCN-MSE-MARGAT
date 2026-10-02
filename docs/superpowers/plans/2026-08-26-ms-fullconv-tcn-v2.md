# MS-FullConv-TCN V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace only the A5 depthwise temporal branch with multi-scale full Conv1d blocks, BatchNorm, and single-head temporal attention.

**Architecture:** Preserve temporal length with a stride-one stem and five residual blocks using kernels `(3,7,15)` and dilations `2**level`. Each branch emits `max(16, hidden*3//8)` channels, the concatenated result is projected back to `hidden`, and one attention map pools time to `[B,hidden]`.

**Tech Stack:** Python, native PyTorch, pytest.

## Global Constraints

- Modify only temporal helper classes, `TCNBranch`, and `tests/test_model.py`.
- Preserve `lead_graph`, MSE, MARGAT, top-level fusion, classifier, training, data, and evaluation behavior.
- Do not restore local graph attention or introduce recurrent, Transformer, spectral, or auxiliary-loss components.

---

### Task 1: Specify V2 behavior

**Files:**
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `MultiScaleTemporalBlock` and `TCNBranch`.
- Produces: tests for full convolutions, branch dimension, BatchNorm, dilation schedule, single attention map, finite outputs, gradients, and absence of local graph state.

- [ ] Replace A5-specific tests with V2 expectations.
- [ ] Run the focused test and confirm failure because V2 symbols are absent.

### Task 2: Implement V2 temporal branch

**Files:**
- Modify: `src/model.py`

**Interfaces:**
- Produces: `MultiScaleTemporalBlock`, `SingleHeadTemporalAttentionPooling`, and `TCNBranch.forward(inputs) -> Tensor[B,hidden]`.

- [ ] Replace depthwise branches with full Conv1d branches using `branch_dim=max(16, hidden*3//8)`.
- [ ] Replace all temporal GroupNorm layers with BatchNorm1d.
- [ ] Replace four-head pooling with one `Conv1d(hidden,1,1)` attention map and weighted temporal sum.
- [ ] Preserve the public TCNMSEMARGAT constructor and forward path.

### Task 3: Verify scope and runtime

**Files:**
- Verify: `src/model.py`
- Verify: `tests/test_model.py`

**Interfaces:**
- Produces: protected-source fingerprints, test evidence, tensor shapes, and parameter comparison.

- [ ] Run `python -m compileall -q src tests`.
- [ ] Run `python -m pytest tests/test_model.py -q` where PyTorch is available.
- [ ] Confirm protected source hashes remain unchanged and no `local_*`, GroupNorm, or depthwise group setting remains in the temporal branch.
- [ ] Report source-derived parameters locally and request one server runtime verification command if local PyTorch remains unavailable.
