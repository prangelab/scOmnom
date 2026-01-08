# Contributing to scOmnom

Thank you for contributing to **scOmnom**. This project is developed collaboratively within the Prange Lab, with an emphasis on code quality, reproducibility, and maintainability.

The guidelines below are intentionally lightweight.

---

## General workflow

1. **Create a feature branch** from `main`
   ```bash
   git checkout -b my-feature
   ```

2. **Make your changes**
   - Keep commits focused and readable
   - Prefer small, logically grouped commits over large ones

3. **Open a pull request (PR)** into `main`
   - Describe *what* you changed and *why*
   - Reference relevant issues or discussions if applicable

4. **Wait for review**
   - At least one approval is required before merging
   - Address comments and requested changes

Direct pushes to `main` are intentionally restricted.

---

## Coding guidelines

- Follow **PEP8** style conventions
- Use **type hints** for public functions and configuration objects
- Prefer **explicit, readable code** over clever abstractions
- Keep module responsibilities narrow

Where applicable:
- Use `AnnData` / Scanpy conventions
- Avoid loading full matrices into memory unless explicitly intended

---

## Configuration and interfaces

- Public CLI options should remain **stable** whenever possible
- Changes to defaults or interfaces should be clearly documented in the PR
- Prefer extending existing config objects over introducing ad-hoc parameters

---

## Testing

- New features should include **basic tests** where feasible
- Tests should be lightweight and runnable on CPU
- Use small synthetic datasets for validation when possible

CI coverage is intentionally minimal at this stage, but tests help prevent regressions.

---

## Documentation

- Update `README.md` if user-facing behavior changes
- Keep documentation concise and factual
- Avoid documenting experimental or unstable features as part of the main workflow

---

## Scope and stability

Parts of the pipeline (e.g. clustering and annotation) are under active development.

If you are contributing to experimental modules:
- expect interfaces to change
- clearly mark assumptions and limitations in code or PR descriptions

---

## Questions

If you are unsure about design decisions, scope, or implementation details:
- open a draft PR
- or discuss informally before investing significant effort

Thanks for helping improve **scOmnom**.

