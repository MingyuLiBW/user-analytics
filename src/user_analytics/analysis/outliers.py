"""Outlier detection helpers."""

import numpy as np
import pandas as pd


def analyze_outliers(dataframe, min_valid_values=2):
    """Count IQR outliers from finite numeric values without changing input.

    Missing values and positive/negative infinities are excluded from IQR
    calculations and reported separately.
    """
    results = []
    for column in dataframe.columns:
        series = dataframe[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        numeric_values = pd.to_numeric(series, errors="coerce")
        numeric_array = numeric_values.to_numpy(dtype=float, na_value=np.nan)
        finite_mask = np.isfinite(numeric_array)
        values = numeric_values.iloc[finite_mask]
        missing_count = int(series.isna().sum())
        non_finite_count = int((series.notna().to_numpy() & ~finite_mask).sum())
        if len(values) < min_valid_values:
            status = (
                "No finite numeric values"
                if values.empty
                else "Insufficient finite numeric values"
            )
            results.append(
                {
                    "column": column,
                    "valid_values": len(values),
                    "missing_count": missing_count,
                    "non_finite_count": non_finite_count,
                    "lower_bound": None,
                    "upper_bound": None,
                    "outlier_count": None,
                    "outlier_percentage": None,
                    "status": status,
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
                    "missing_count": missing_count,
                    "non_finite_count": non_finite_count,
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
                "missing_count": missing_count,
                "non_finite_count": non_finite_count,
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
            "column", "valid_values", "missing_count", "non_finite_count",
            "lower_bound", "upper_bound",
            "outlier_count", "outlier_percentage", "status",
        ],
    )
