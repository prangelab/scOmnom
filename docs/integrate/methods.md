# Supported integration methods

Supported methods include:

* **scVI / scANVI** — variational autoencoders (recommended for large or complex datasets)
* **Harmony** — fast linear batch correction
* **Scanorama** — panoramic batch integration
* **BBKNN** — graph-based batch correction (baseline)

**Default methods:** all of the above.

Methods can be restricted explicitly:

```bash
scomnom integrate \
  --input-path adata.filtered.zarr \
  --batch-key sample_id \
  --methods scVI scANVI Harmony \
  --output-dir results/integrate/
```

---
