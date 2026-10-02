# GitHub Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a polished research-project README and selected presentation visuals to the requested GitHub repository.

**Architecture:** Store stable screenshots under `docs/assets/` and reference them from a focused README introduction. Keep training and evaluation behavior unchanged, and use `.gitignore` to separate source artifacts from local data and intermediate outputs.

**Tech Stack:** Markdown, PNG assets, Git, PyTorch project tests

## Global Constraints

- Present 98.12% mixed five-fold accuracy and 88.65% leave-one-case-out accuracy as user-supplied final report summaries.
- Do not claim local reproduction from absent fold-level result files.
- Do not commit literature PDFs, checkpoints, datasets, output directories, or temporary renders.

---

### Task 1: Showcase assets and README

**Files:**
- Create: `docs/assets/model-architecture.png`
- Create: `docs/assets/attention-visualization.png`
- Create: `docs/assets/feature-visualization.png`
- Modify: `README.md`

**Interfaces:**
- Consumes: rendered slides 4, 11, and 13 from the supplied progress-report presentation
- Produces: stable repository-relative image references used by `README.md`

- [ ] **Step 1: Copy the selected rendered slides into `docs/assets/`.**
- [ ] **Step 2: Rewrite the README opening with headline metrics, method summary, architecture, and visual results.**
- [ ] **Step 3: Verify every local Markdown image path exists.**

### Task 2: Repository hygiene and validation

**Files:**
- Create: `.gitignore`
- Test: `tests/`

**Interfaces:**
- Consumes: the completed source tree and README assets
- Produces: a reviewable Git commit without local data or temporary artifacts

- [ ] **Step 1: Add ignore rules for caches, outputs, checkpoints, temporary renders, and `文献/`.**
- [ ] **Step 2: Run `python -m compileall -q src` and `python -m pytest -q`.**
- [ ] **Step 3: Inspect `git status`, staged paths, and staged file sizes.**
- [ ] **Step 4: Commit the current project and showcase assets.**

### Task 3: GitHub publication

**Files:**
- Modify: Git remote configuration

**Interfaces:**
- Consumes: the verified local commit
- Produces: the repository's published default branch

- [ ] **Step 1: Inspect the target remote refs.**
- [ ] **Step 2: Configure `origin` to the requested URL.**
- [ ] **Step 3: Push the verified commit to `main`.**
- [ ] **Step 4: Confirm the remote `main` commit matches local `HEAD`.**
