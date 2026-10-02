# A5 Temporal Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace only the local-GAT TCN branch with MS-DS-TCN, GroupNorm, and four-head temporal attention pooling while preserving the MSE-MARGAT path and gated fusion.

**Architecture:** Map `[B,C,T]` to `[B,H,T]` with a stride-one temporal stem. Apply `tcn_levels` residual multi-scale blocks using depthwise kernels `(3,7,15)` and dilations `2**level`, then pool four independent temporal attention heads and project `[B,4H]` to `[B,H]`.

**Tech Stack:** Python, native PyTorch, pytest.

## Global Constraints

- Modify only temporal-branch classes and their model tests.
- Keep `lead_graph`, `MultiScaleEncoder`, `LeadGraphAttention`, correction, gate, normalization, classifier, and top-level fusion logic unchanged.
- Use no BatchNorm, graph attention, recurrent layer, Transformer, spectral transform, new loss, or new dependency in the temporal branch.
- Preserve the public `TCNMSEMARGAT` construction pattern and `[B,2]` forward output.

---

### Task 1: Specify the A5 temporal branch

**Files:**
- Modify: `tests/test_model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `TCNBranch(channels, hidden, levels, dropout)` and `TCNMSEMARGAT(...)`.
- Produces: regression expectations for GroupNorm, depthwise multi-scale blocks, four-head attention maps, finite outputs, and removal of local graph fields.

- [ ] **Step 1: Replace local-GAT tests with failing A5 tests**

Assert output `[B,H]`, attention `[B,4,T]`, attention sums of one, three depthwise kernels per block, no BatchNorm, and no `local_*` graph state.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_model.py -q`

Expected: failures because A5 classes and multi-head attention do not yet exist. If PyTorch or pytest is unavailable, record the environment failure and use an AST assertion to confirm the old local-GAT implementation violates the new specification.

### Task 2: Implement the A5 temporal encoder

**Files:**
- Modify: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `make_group_norm(channels)`, `MultiScaleDSTemporalBlock`, `MultiHeadTemporalAttentionPooling`, and `TCNBranch.forward(inputs) -> Tensor[B,H]`.

- [ ] **Step 1: Implement normalization and the residual MS-DS-TCN block**

Use GroupNorm groups selected from `(8,4,2,1)`, three length-preserving depthwise convolutions with kernels `(3,7,15)`, concatenation, `Conv1d(3H,H,1)`, GroupNorm, GELU, and dropout before residual addition.

- [ ] **Step 2: Implement temporal attention pooling**

Generate `[B,4,T]` logits with `Conv1d(H,4,1)`, softmax over time, store detached weights, compute `[B,4,H]` pooled features with einsum, and finish with linear projection, LayerNorm, GELU, and dropout.

- [ ] **Step 3: Replace TCNBranch internals**

Use a `Conv1d(C,H,7,padding=3)` stem with GroupNorm and GELU, sequential A5 blocks with dilations `2**level`, and the four-head pooler. Remove all local graph fields and calls.

### Task 3: Verify scope and runtime behavior

**Files:**
- Verify: `src/model.py`
- Verify: `tests/test_model.py`

**Interfaces:**
- Consumes: completed A5 implementation.
- Produces: verification evidence and parameter counts when PyTorch is available.

- [ ] **Step 1: Run focused and full model tests**

Run: `python -m pytest tests/test_model.py -q`

Run: `python -m compileall -q src tests`

- [ ] **Step 2: Verify protected code and API**

Compare source fingerprints for `lead_graph`, `MultiScaleEncoder`, `LeadGraphAttention`, and the non-TCN portions of `TCNMSEMARGAT`. Confirm input windows of 256, 512, and 1024 samples still produce `[B,2]`.

- [ ] **Step 3: Report evidence honestly**

Report actual commands and outputs, attention shapes, model and temporal parameter counts if executable, checkpoint incompatibility, and any verification blocked by missing PyTorch.
