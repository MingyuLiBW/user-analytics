"""Descriptive and univariate analysis helpers."""

import numpy as np
import pandas as pd


def get_numeric_columns(dataframe):
    """Return numeric columns, excluding boolean fields."""
    return [
        column for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        and not pd.api.types.is_bool_dtype(dataframe[column].dtype)
    ]


def get_categorical_columns(dataframe):
    """Return non-numeric, non-datetime fields suitable for category summaries."""
    return [
        column for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        and not pd.api.types.is_datetime64_any_dtype(dataframe[column].dtype)
    ]


def describe_numeric(dataframe, columns=None):
    """Return requested numeric descriptive statistics without mutating input."""
    numeric_columns = get_numeric_columns(dataframe)
    selected = list(dataframe.columns) if columns is None else list(columns)
    selected = [
        column for column in selected
        if column in dataframe.columns
        and (column in numeric_columns or dataframe[column].isna().all())
    ]
    statistic_names = [
        "Count", "Missing Count", "Non-finite Count", "Mean", "Standard Deviation",
        "Minimum", "25%", "Median", "75%", "Maximum"
    ]
    if not selected:
        return pd.DataFrame(columns=statistic_names)

    rows = {}
    for column in selected:
        series = dataframe[column]
        if column not in numeric_columns and not series.isna().all():
            continue
        numeric_values = pd.to_numeric(series, errors="coerce")
        values_array = numeric_values.to_numpy(dtype=float, na_value=np.nan)
        finite_values = values_array[np.isfinite(values_array)]
        missing_count = int(series.isna().sum())
        non_finite_count = int((series.notna().to_numpy() & ~np.isfinite(values_array)).sum())
        description = pd.Series(finite_values, dtype="float64").describe() if finite_values.size else None
        rows[column] = [
            int(finite_values.size), missing_count, non_finite_count,
            description["mean"] if description is not None else float("nan"),
            description["std"] if description is not None else float("nan"),
            description["min"] if description is not None else float("nan"),
            description["25%"] if description is not None else float("nan"),
            description["50%"] if description is not None else float("nan"),
            description["75%"] if description is not None else float("nan"),
            description["max"] if description is not None else float("nan"),
        ]
    result = pd.DataFrame.from_dict(rows, orient="index", columns=statistic_names)
    result.index.name = "Field"
    return result


def describe_categorical(dataframe, columns=None):
    """Summarize unique values and mode frequency for non-numeric fields."""
    selected = get_categorical_columns(dataframe) if columns is None else list(columns)
    valid = set(get_categorical_columns(dataframe))
    rows = []
    for column in selected:
        if column not in valid:
            continue
        values = dataframe[column].dropna()
        counts = values.value_counts()
        rows.append(
            {
                "Field": column,
                "Unique Count": int(values.nunique()),
                "Most Frequent Value": counts.index[0] if not counts.empty else None,
                "Frequency of Most Frequent Value": int(counts.iloc[0]) if not counts.empty else 0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Field", "Unique Count", "Most Frequent Value", "Frequency of Most Frequent Value"],
    ).set_index("Field")


def prepare_numeric_series(dataframe, column):
    """Return a copy of valid numeric observations for univariate charts."""
    if column not in get_numeric_columns(dataframe):
        raise ValueError(f"'{column}' is not a numeric field.")
    series = dataframe[column]
    numeric_values = pd.to_numeric(series, errors="coerce")
    values_array = numeric_values.to_numpy(dtype=float, na_value=np.nan)
    return numeric_values.iloc[np.isfinite(values_array)].copy()


def prepare_category_frequencies(dataframe, column, top_n=None):
    """Return category counts and percentages, optionally grouping the tail as Other."""
    if column not in get_categorical_columns(dataframe):
        raise ValueError(f"'{column}' is not a categorical field.")

    values = dataframe[column].dropna()
    counts = values.value_counts()
    if top_n is not None and top_n > 0 and len(counts) > top_n:
        top_counts = counts.iloc[:top_n]
        other_count = int(counts.iloc[top_n:].sum())
        has_real_other = "Other" in counts.index
        counts = top_counts.copy()
        other_label = "Other"
        if has_real_other:
            other_label = "Other (remaining)"
            suffix = 2
            while other_label in counts.index or other_label in values.unique():
                other_label = f"Other (remaining) {suffix}"
                suffix += 1
        counts.loc[other_label] = other_count

    denominator = int(values.size)
    result = pd.DataFrame({"Category": counts.index, "Count": counts.to_numpy()})
    result["Percentage"] = result["Count"] / denominator * 100 if denominator else 0.0
    return result



def prepare_bivariate_data(dataframe, x_column, y_column):
    """Return a copy of pairwise-valid observations for two selected fields."""
    if x_column == y_column:
        raise ValueError("Choose different fields for X and Y.")
    if x_column not in dataframe.columns or y_column not in dataframe.columns:
        raise ValueError("Choose fields that exist in the dataset.")
    result = dataframe[[x_column, y_column]].dropna().copy()
    numeric_columns = [
        column for column in (x_column, y_column)
        if pd.api.types.is_numeric_dtype(dataframe[column].dtype)
    ]
    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        result = result.loc[np.isfinite(values)]
    return result


def prepare_categorical_numeric_data(dataframe, category_column, numeric_column):
    """Return pairwise-valid category/numeric observations for grouped plots."""
    if category_column == numeric_column:
        raise ValueError("Choose different fields for X and Y.")
    if category_column not in get_categorical_columns(dataframe):
        raise ValueError(f"'{category_column}' is not a categorical field.")
    if numeric_column not in get_numeric_columns(dataframe):
        raise ValueError(f"'{numeric_column}' is not a numeric field.")
    result = dataframe[[category_column, numeric_column]].dropna().copy()
    values = pd.to_numeric(result[numeric_column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return result.loc[np.isfinite(values)].copy()


def prepare_categorical_crosstab(dataframe, x_column, y_column, normalize=False):
    """Return pairwise category counts or row-wise proportions."""
    if x_column == y_column:
        raise ValueError("Choose different fields for X and Y.")
    categorical_columns = get_categorical_columns(dataframe)
    if x_column not in categorical_columns or y_column not in categorical_columns:
        raise ValueError("Both fields must be categorical.")
    valid_pairs = dataframe[[x_column, y_column]].dropna()
    return pd.crosstab(valid_pairs[x_column], valid_pairs[y_column], normalize="index" if normalize else False)


def calculate_correlation(dataframe, columns=None, method="pearson"):
    """Calculate Pearson or Spearman correlations on pairwise valid observations."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("Correlation method must be 'pearson' or 'spearman'.")
    numeric_columns = get_numeric_columns(dataframe)
    selected = numeric_columns if columns is None else [
        column for column in columns if column in numeric_columns
    ]
    if not selected:
        return pd.DataFrame()
    numeric_data = dataframe[selected].copy()
    for column in selected:
        values = pd.to_numeric(numeric_data[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        numeric_data.loc[~np.isfinite(values), column] = np.nan
    return numeric_data.corr(method=method, min_periods=1)
