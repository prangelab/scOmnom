# Reporting

scOmnom modules write self-contained HTML reports as part of the pipeline output layer. Reports collect run information and generated plots so each command run can be inspected without opening every figure file manually.

Reports are written under the module's figure output tree, usually in the same run folder as the figures they summarize. Exact names vary by module and output round, but follow the same convention as figures and tables: reruns create new output folders instead of overwriting earlier results.

See [Output Organization](output-organization.md) for how report folders relate to filesystem output rounds and AnnData clustering rounds.

---
