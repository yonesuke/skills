# uv Examples

Modern Python packaging and project management.

## 1. Scripts with Dependencies
Running self-contained scripts with `uv run`.

**Source:** [scripts/pep723_demo.py](scripts/pep723_demo.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "requests",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint

def main():
    # ... See script ...
    pass
```

## 2. Common Commands

```bash
# Initialize
uv init my-project

# Add/Remove Dependencies
uv add requests
uv remove requests
uv add --dev pytest

# Run
uv run python main.py
uv run pytest

# Tools
uv tool install ruff
uvx ruff check .
```
