# Reproducibility and non-destructive design

All clustering, annotation, and compaction steps in scOmnom are **non-destructive**:

* intermediate results are preserved
* no clustering step overwrites a previous result
* all decisions (parameters, metrics, diagnostics) are stored in the dataset

This design ensures that clustering results are fully reproducible, auditable, and comparable across runs.

---
