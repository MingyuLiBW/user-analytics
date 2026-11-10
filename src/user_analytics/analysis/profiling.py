"""Data profiling functions independent of Streamlit UI."""

import pandas as pd


def analyze_missing_values(dataframe):
    """Return per-column and whole-dataset missing-value metrics."""
    row_count, column_count = dataframe.shape
    missing_counts = dataframe.isna().sum()
    denominator = row_count if row_count else 1
    per_column = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_count": missing_counts.to_numpy(),
            "missing_percentage": (missing_counts / denominator * 100).to_numpy(),
        }
    )
    total_cells = row_count * column_count
    total_missing = int(missing_counts.sum())
    return {
        "per_column": per_column,
        "total_missing": total_missing,
        "total_cells": total_cells,
        "missing_percentage": (total_missing / total_cells * 100) if total_cells else 0.0,
    }


def analyze_duplicates(dataframe):
    """Count fully duplicated rows (excluding each row's first occurrence)."""
    duplicate_count = int(dataframe.duplicated(keep="first").sum())
    row_count = len(dataframe)
    return {
        "duplicate_count": duplicate_count,
        "row_count": row_count,
        "duplicate_percentage": duplicate_count / row_count * 100 if row_count else 0.0,
    }


def classify_column(series):
    """Classify a column while avoiding treating every object as categorical."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "Date/time"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "Numeric"
    if pd.api.types.is_bool_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype):
        return "Categorical"

    non_missing = series.dropna()
    if non_missing.empty:
        return "Other"

    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        parsed_dates = pd.to_datetime(non_missing, errors="coerce", format="mixed")
        if parsed_dates.notna().mean() >= 0.9:
            return "Date/time"
        unique_ratio = non_missing.nunique() / len(non_missing)
        if unique_ratio < 0.8:
            return "Categorical"
    return "Other"


def analyze_data_types(dataframe):
    """Return field names, Pandas dtypes, and inferred semantic groups."""
    return pd.DataFrame(
        [
            {
                "column": column,
                "pandas_dtype": str(dataframe[column].dtype),
                "category": classify_column(dataframe[column]),
            }
            for column in dataframe.columns
        ],
        columns=["column", "pandas_dtype", "category"],
    )
