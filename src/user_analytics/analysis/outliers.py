"""Outlier detection helpers."""

import pandas as pd


def analyze_outliers(dataframe, min_valid_values=2):
    """Count IQR outliers in numeric columns without modifying the input."""
    results = []
    row_count = len(dataframe)

    for column in dataframe.columns:
        series = dataframe[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        values = series.dropna()
        if len(values) < min_valid_values:
            results.append(
                {
                    "column": column,
                    "valid_values": len(values),
                    "lower_bound": None,
                    "upper_bound": None,
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                    "status": "Insufficient numeric values",
                }
            )
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            results.append(
                {
                    "column": column,
                    "valid_values": len(values),
                    "lower_bound": None if pd.isna(iqr) else q1,
                    "upper_bound": None if pd.isna(iqr) else q3,
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                    "status": "IQR is zero; outlier detection skipped",
                }
            )
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())
        results.append(
            {
                "column": column,
                "valid_values": len(values),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_count / len(values) * 100,
                "status": "Analyzed",
            }
        )

    return pd.DataFrame(
        results,
        columns=[
            "column", "valid_values", "lower_bound", "upper_bound",
            "outlier_count", "outlier_percentage", "status",
        ],
    )
