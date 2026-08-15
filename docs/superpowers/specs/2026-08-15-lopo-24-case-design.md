# 24-Case LOPO Preparation Design

## Goal

Prepare a scientifically explicit 24-fold CHB-MIT leave-one-case-out
experiment by mapping `chb17a`, `chb17b`, and `chb17c` to the canonical case
ID `chb17`, rebuilding patient-dependent artifacts, and rejecting any prepared
dataset whose case IDs are not exactly `chb01` through `chb24`.

The experiment retains `chb01` and `chb21` as separate cases. Results must be
described as 24-case LOPO or leave-one-case-out because those two cases were
recorded from the same subject.

## Non-Goals and Compatibility

- Do not change the `mixed_10fold` split algorithm, configuration, fold count,
  sampling behavior, training behavior, checkpoints, or existing run outputs.
- Do not merge `chb01` and `chb21` in this experiment.
- Do not change the model, optimizer, loss, window labeling, filtering,
  balancing ratio, or evaluation metrics.
- Do not introduce a general alias configuration system for a single known
  CHB-MIT filename convention.
- Do not silently migrate old prepared artifacts.

Existing mixed-10-fold results remain valid under their original 26-ID,
window-level protocol and must be reported separately. New LOPO runs use newly
rebuilt 24-case artifacts and must not share split files with the prior mixed
runs.

## Design

### Canonical case identity

Add one small function in `src/data.py` that extracts a CHB-MIT case ID from an
EDF stem. It accepts ordinary stems such as `chb01_03` and the CHB17 stems
`chb17a_03`, `chb17b_57`, and `chb17c_02`. The three CHB17 variants all return
`chb17`; ordinary stems return their `chbNN` ID. Unrecognized stems retain the
current behavior: preparation logs a warning and skips the file.

Record cache paths and original EDF names remain unchanged. Only the
`patient` field stored in `manifest.json` and the key used to accumulate
normalization moments become canonical. Consequently all CHB17 records share
one mean and standard deviation pair.

### Cache invalidation and rebuild

Increase `CACHE_VERSION`. Because the version participates in the record
fingerprint, an old manifest cannot be silently reused. Without `--force`, the
existing mismatch raises the current explicit rebuild error. With `--force`,
the command rebuilds:

- `manifest.json` with canonical case IDs;
- `normalization.npz` with one mean/std pair per canonical case;
- each requested `windows/*.npz` catalog and its metadata;
- subsequent `splits/lopo_seed42/*.npz` files as folds are created.

The implementation must not delete or overwrite prior experiment run
directories. Operators should use a fresh prepared directory for the new LOPO
protocol when retaining the old mixed artifacts on the server.

### Strict 24-case validation

Define the expected case set as exactly `chb01` through `chb24`. After record
preparation, validate that the manifest contains this exact set. An error must
show missing and unexpected IDs so a partial mount or incorrect alias cannot
start an expensive experiment.

Validate normalization output against the same set: exactly one mean and one
standard-deviation array per case, with no `chb17a`, `chb17b`, or `chb17c`
keys. The real-data preflight must also inspect the selected window catalog and
confirm that every case contributes both seizure and non-seizure windows.

Synthetic unit tests and small fixture datasets are not required to contain
all 24 cases. The strict dataset check is called by the real preparation or
explicit preflight boundary, not unconditionally inside the generic split
function or dataset constructor.

### Split behavior

Leave `make_split()` unchanged. Its LOPO branch already assigns all windows
from the selected case to test, chooses validation by whole cases, and excludes
both from training. Canonical IDs ensure all CHB17 variants move together.

Leave the `mixed_10fold` branch byte-for-byte unchanged. Regression tests must
compare its deterministic split indices before and after this feature for the
existing synthetic fixture.

The 24 legal LOPO fold IDs are `chb01` through `chb24`. A preflight command or
helper prints these IDs and refuses any other prepared case set before jobs are
submitted.

## Error Handling

- Unknown EDF filename: warn and skip, preserving current behavior.
- Old cache fingerprint: raise and require an explicit `--force` rebuild.
- Missing case: raise with the missing IDs.
- Unexpected case, including any CHB17 suffix: raise with the unexpected IDs.
- Case missing seizure or non-seizure windows: raise with the affected case and
  class counts before training.
- Unknown LOPO fold ID: retain the existing `Unknown test patient` error.

## Tests

Use test-driven development for each behavior:

1. CHB17 stems map to `chb17`, while ordinary stems remain unchanged and
   invalid stems are rejected by the parser.
2. Canonical CHB17 records contribute to one normalization identity.
3. Exact 24-case validation passes only for `chb01` through `chb24` and reports
   missing or unexpected IDs clearly.
4. Normalization validation rejects suffixed CHB17 keys or missing arrays.
5. Catalog preflight rejects a case without both window classes.
6. All 24 LOPO folds keep the selected case out of train and validation.
7. The existing deterministic mixed-10-fold split test returns the same indices
   and remains disjoint and balanced.
8. Run the complete local test suite, syntax compilation, and `git diff
   --check`.

Server verification is separate from local unit tests. Before submitting the
24 jobs, the server must show the exact 24 case IDs, 48 normalization arrays,
both classes for every case in the chosen window catalog, and 24 legal fold
IDs.

## Documentation and Repository Hygiene

Update `README.md` with the 24-case terminology, rebuild requirement, server
preflight output, and the scientific limitation involving `chb01/chb21`.
Document that old mixed runs and new LOPO artifacts must be kept separate.

Add a focused `.gitignore` for Python caches, pytest caches, temporary files,
local artifacts, model checkpoints, and run outputs. Resolve the stale
`src/src/` duplicate without changing the authoritative top-level `src/`
modules. Stage only files belonging to this task; unrelated user files remain
untouched.

## Success Criteria

- Preparation maps all CHB17 EDF variants to `chb17`.
- Prepared case IDs equal `chb01` through `chb24`, with no suffixed CHB17 IDs.
- Normalization contains 24 mean and 24 standard-deviation arrays.
- Every selected case has seizure and non-seizure windows.
- Each of the 24 fold IDs produces patient-disjoint train, validation, and test
  partitions.
- Existing mixed-10-fold regression tests pass with unchanged expected indices.
- Documentation calls the experiment 24-case LOPO and states the
  `chb01/chb21` subject-identity limitation.
