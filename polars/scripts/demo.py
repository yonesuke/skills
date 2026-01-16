# /// script
# requires-python = ">=3.12"
# dependencies = ["polars", "numpy"]
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
    print(f"Original Schema: {df.schema}")

    # 2. Lazy Operations / Method Chaining
    # Usually starts with pl.scan_csv(...)
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

    # 3. Execution using collect()
    result = q.collect()
    print("\nResult Aggregation:")
    print(result)

    # 4. Window Functions
    df_window = df.with_columns(
        pl.col("value").mean().over("category").alias("category_mean")
    )
    print("\nWindow Function Result (First 3 rows):")
    print(df_window.head(3))

if __name__ == "__main__":
    main()
