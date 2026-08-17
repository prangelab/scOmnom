# Supported integration methods

Supported methods include:

* **[scVI / scANVI](https://docs.scvi-tools.org/en/stable/)** — variational autoencoders via `scvi-tools` (recommended for large or complex datasets)
* **[Harmony](https://github.com/slowkow/harmonypy)** — fast linear batch correction via `harmonypy`
* **[Scanorama](https://github.com/brianhie/scanorama)** — panoramic batch integration
* **[BBKNN](https://bbknn.readthedocs.io/en/latest/)** — graph-based batch correction (baseline)

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
