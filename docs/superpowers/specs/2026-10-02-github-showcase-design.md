# GitHub Showcase Design

## Goal

Turn the current research repository into a concise GitHub showcase for academic outreach while preserving enough protocol context to avoid misleading readers.

## Scope

- Keep the existing Chinese README as the technical foundation.
- Lead with the supplied final summary metrics: 98.12% accuracy for mixed five-fold cross-validation and 88.65% accuracy for leave-one-case-out evaluation.
- Add the presentation's model architecture and representative interpretability figures as repository assets.
- Push source code, configuration, tests, scripts, documentation, and showcase assets to the requested GitHub repository.
- Exclude local render directories, experiment outputs, checkpoints, caches, and literature PDFs.

## Presentation

The README opens with the project purpose and two large headline metrics, followed by a compact metric table. It then presents the report-stage architecture figure and a small set of visualization figures before the existing protocol and reproduction sections.

The headline values are identified as final report summary values. Detailed fold-level artifacts are not claimed to exist in this checkout.

## Verification

- Confirm every README image path resolves in the repository.
- Run the existing test suite.
- Inspect the staged file list for accidental datasets, checkpoints, outputs, temporary renders, or literature PDFs.
- Verify the target remote and push result.
