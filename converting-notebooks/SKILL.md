---
name: converting-notebooks
description: Converts Jupyter notebooks (.ipynb) to clean, standalone PEP 723 Python scripts runnable with `uv run`. Produces well-structured scripts with Google-style docstrings, type hints, and dated output filenames.
---
# Converting Notebooks Skill

Transforms notebook experiments into production-ready, self-executing Python scripts.

## Contents
- [Examples](examples.md)
    - Canonical output structure: PEP 723 header, constants, helper functions, `main()`.

## Output Conventions
- **Filename**: `YYYYMMDD_<name>.py` (today's date by default)
- **Saved outputs** (images, CSVs): `YYYYMMDD_<name>.png` / `.csv`
- **Output directory**: same as script unless `--output-dir` is specified

## Quick Checklist
1. `# /// script` block **first** (before module docstring).
2. Dependencies: list third-party packages; omit stdlib; skip version if unknown.
3. Module docstring: Google style, English; translate Japanese math comments.
4. Constants section: all hardcoded literals in `UPPER_SNAKE_CASE`.
5. Output helper `_out(name)` → prefixes every saved file with `YYYYMMDD_`.
6. `matplotlib.use("Agg")` immediately after `import matplotlib`.
7. Every function/class: Google docstring + full type hints.
8. `main() -> None` reads as a high-level narrative; delegate logic to helpers.
9. `print()` for result tables/metrics; `logging.info()` for pipeline progress.
10. Reproducibility: set `RANDOM_SEED` constant; call seed functions at top of `main()`.
