---
name: polars
description: High-performance DataFrame library usage. Covers Lazy API, Wrangling, Aggregation.
---
# Polars Skill

Fast data manipulation in Python.

## Contents

- [Examples](examples.md)
    - Lazy API, GroupBy, Window Functions.

## Core Concepts

- **Lazy API**: `df.lazy()...collect()`. Preferred for performance (Query Optimization).
- **Expressions**: `pl.col("a") * 2`. Parallelizable logic.
- **Eager API**: `df.filter(...)`. Good for debugging.

## Usage

Use for all tabular data tasks unless `pandas` is strictly required by legacy dependencies.
