# Polars Examples

Data manipulation with Polars (Lazy & Eager).

## 1. Wrangling & Aggregation
Demonstrates core workflow: Lazy API, Filtering, Columns, GroupBy, and Window Functions.

**Source:** [scripts/demo.py](scripts/demo.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["polars", "pandas", "numpy"]
# ///

import polars as pl
import numpy as np
import datetime

def main():
    print("Running Polars Data Wrangling Demo")

    # 1. Create DataFrame
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 1, 2],
        "category": ["A", "B", "A", "B", "A", "B"],
        "value": [10.5, 20.0, 15.5, 7.5, 12.0, 18.0],
        "date": [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(6)]
    })

    # 2. Lazy Operations / Method Chaining
    q = (
        df.lazy()
        .filter(pl.col("value") > 10)
        .with_columns(
            (pl.col("value") * 1.1).alias("value_with_tax"),
            pl.col("date").dt.month().alias("month")
        )
        .group_by("category")
        .agg(
            pl.len().alias("count"),
            pl.col("value").sum().alias("total_value"),
            pl.col("value_with_tax").mean().alias("avg_taxed_value")
        )
        .sort("total_value", descending=True)
    )

    # 3. Execution
    result = q.collect()
    print(result)

if __name__ == "__main__":
    main()
```
