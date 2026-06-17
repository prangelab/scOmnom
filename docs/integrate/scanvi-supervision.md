# Intelligent scANVI supervision (BISC)

When `scANVI` is enabled, `scOmnom` can generate supervision labels automatically instead of relying on a single user-chosen clustering.

The supervision step uses **BISC** on the **scVI latent space** with a structural-only sweep:

1. **Latent-space resolution sweep**
   A coarse, limited bandwith Leiden sweep is performed on the scVI latent space.

2. **Structural selection**
   Candidate resolutions are scored using stability, centroid-based silhouette, and tiny-cluster penalties.

3. **Parsimony**
   The lowest resolution near the top score is selected.

This produces labels used **only for scANVI supervision**, which will be replaced by downstream clustering of the final integrated embedding.

---
