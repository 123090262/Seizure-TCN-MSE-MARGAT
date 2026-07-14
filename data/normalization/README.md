# Normalization caches

The `subject_global/` cache is generated on the first training fold from the
existing `data/processed/chbmit` arrays. It stores one per-channel mean and
standard deviation for each subject and is shared read-only by subsequent
folds. The cache is ignored by version control and never overwrites the
train-only fold statistics saved inside existing output directories.
