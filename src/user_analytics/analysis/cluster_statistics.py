"""Exploratory statistical comparisons across precomputed clusters.

These tests describe associations in the supplied sample. When clusters were
created from the same features being tested, p-values are exploratory and are
not independent confirmatory evidence of population-level differences.
"""

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, chi2_contingency, f_oneway, kruskal, levene


NUMERIC_RESULT_COLUMNS = [
    "Variable", "Sample Count", "Non-finite Count", "Cluster Count",
    "Levene Statistic", "Levene P-value", "Variance Homogeneity Status", "Levene Reason",
    "ANOVA Statistic", "ANOVA P-value", "Eta-squared", "ANOVA Status", "ANOVA Reason",
    "Kruskal Statistic", "Kruskal P-value", "Epsilon-squared", "Kruskal Status", "Kruskal Reason",
]
CATEGORICAL_RESULT_COLUMNS = [
    "Variable", "Sample Count", "Cluster Count", "Category Count", "Degrees of Freedom",
    "Chi-square Statistic", "P-value", "Cramer's V", "Minimum Expected Frequency",
    "Expected Cells < 5 Proportion", "Status", "Reason", "Warning",
]


@dataclass(frozen=True)
class ClusterStatisticsResult:
    """Numeric and categorical cluster comparison tables."""

    numeric: pd.DataFrame
    categorical: pd.DataFrame


def _align_labels(dataframe, labels):
    """Align Series labels by exact index or array-like labels by row position."""
    if dataframe.empty:
        raise ValueError("Cluster statistics require a non-empty DataFrame.")
    if not dataframe.index.is_unique:
        raise ValueError("DataFrame index must be unique to align cluster labels.")
    if isinstance(labels, pd.Series):
        if len(labels) != len(dataframe):
            raise ValueError("Cluster label count must match the number of DataFrame rows.")
        if not labels.index.is_unique:
            raise ValueError("Cluster label index must be unique.")
        if not labels.index.difference(dataframe.index).empty or not dataframe.index.difference(labels.index).empty:
            raise ValueError("Cluster label index must match the DataFrame index exactly.")
        aligned = labels.loc[dataframe.index].copy()
    else:
        values = list(labels)
        if len(values) != len(dataframe):
            raise ValueError("Cluster label count must match the number of DataFrame rows.")
        aligned = pd.Series(values, index=dataframe.index.copy(), name="Cluster")
    if aligned.isna().any():
        raise ValueError("Cluster labels must not contain missing values.")
    return aligned


def _validate_columns(dataframe, numeric_columns, categorical_columns):
    """Validate selected fields and their declared data types."""
    numeric = list(numeric_columns or [])
    categorical = list(categorical_columns or [])
    for kind, columns in (("numeric", numeric), ("categorical", categorical)):
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate {kind} fields are not allowed.")
    missing = [column for column in numeric + categorical if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Selected field(s) not found: {missing}")
    invalid_numeric = [
        column for column in numeric
        if (
            (not pd.api.types.is_numeric_dtype(dataframe[column].dtype)
             and not dataframe[column].isna().all())
            or pd.api.types.is_bool_dtype(dataframe[column].dtype)
        )
    ]
    if invalid_numeric:
        raise ValueError(f"Selected numeric fields must have non-boolean numeric dtypes: {invalid_numeric}")
    invalid_categorical = [
        column for column in categorical
        if pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        or pd.api.types.is_datetime64_any_dtype(dataframe[column].dtype)
    ]
    if invalid_categorical:
        raise ValueError(f"Categorical fields must be non-numeric and non-date: {invalid_categorical}")
    overlap = set(numeric).intersection(categorical)
    if overlap:
        raise ValueError(f"Fields cannot be both numeric and categorical: {sorted(overlap)}")
    return numeric, categorical


def _finite_or_none(value):
    """Convert finite scalars to Python floats, representing invalid results as None."""
    value = float(value)
    return value if np.isfinite(value) else None


def compare_numeric_clusters(dataframe, labels, numeric_columns):
    """Compare each numeric field across clusters using ANOVA and Kruskal-Wallis.

    Each field independently excludes missing and non-finite observations.
    ``Sample Count`` counts finite observations used by the tests, and
    ``Non-finite Count`` records positive/negative infinities excluded.

    Levene's median-centered test is reported independently from ANOVA. A
    non-significant Levene result is not proof of equal variances. Ordinary
    one-way ANOVA still relies on the homogeneity-of-variance assumption;
    neither Levene nor ANOVA tests independence or normality. ANOVA status
    indicates calculation availability, not that its assumptions are satisfied.
    Cluster comparisons remain exploratory, especially when clusters were
    generated from the same features being tested. Kruskal-Wallis epsilon-squared
    uses the untruncated estimator (H - k + 1) / (N - k); a negative estimate is
    retained and does not mean the true explained proportion is negative.
    """
    aligned = _align_labels(dataframe, labels)
    numeric, _ = _validate_columns(dataframe, numeric_columns, [])
    rows = []
    cluster_order = list(pd.unique(aligned))

    for column in numeric:
        values = dataframe[column]
        numeric_values = pd.to_numeric(values, errors="coerce")
        finite_mask = values.notna() & np.isfinite(numeric_values)
        nonfinite_count = int((values.notna() & ~np.isfinite(numeric_values)).sum())
        valid = pd.DataFrame(
            {"value": values.loc[finite_mask].astype(float), "cluster": aligned.loc[finite_mask]},
            index=dataframe.index[finite_mask],
        )
        groups = [valid.loc[valid["cluster"] == cluster, "value"].to_numpy() for cluster in cluster_order]
        groups = [group for group in groups if len(group)]
        sample_count = int(sum(len(group) for group in groups))
        cluster_count = len(groups)
        all_values = np.concatenate(groups) if groups else np.array([], dtype=float)
        constant = sample_count > 0 and np.all(all_values == all_values[0])

        row = {
            "Variable": column,
            "Sample Count": sample_count,
            "Non-finite Count": nonfinite_count,
            "Cluster Count": cluster_count,
            "Levene Statistic": None,
            "Levene P-value": None,
            "Variance Homogeneity Status": "Unavailable",
            "Levene Reason": None,
            "ANOVA Statistic": None,
            "ANOVA P-value": None,
            "Eta-squared": None,
            "ANOVA Status": "Not calculated",
            "ANOVA Reason": None,
            "Kruskal Statistic": None,
            "Kruskal P-value": None,
            "Epsilon-squared": None,
            "Kruskal Status": "Not calculated",
            "Kruskal Reason": None,
        }
        if cluster_count >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    levene_result = levene(*groups, center="median")
                if np.isfinite(levene_result.statistic) and np.isfinite(levene_result.pvalue):
                    levene_status = (
                        "No significant evidence of unequal variances"
                        if levene_result.pvalue >= 0.05
                        else "Potential heterogeneity"
                    )
                    row.update({
                        "Levene Statistic": _finite_or_none(levene_result.statistic),
                        "Levene P-value": _finite_or_none(levene_result.pvalue),
                        "Variance Homogeneity Status": levene_status,
                        "Levene Reason": None,
                    })
                else:
                    row["Levene Reason"] = "Levene test returned a non-finite result."
            except (ValueError, FloatingPointError, ZeroDivisionError) as error:
                row["Levene Reason"] = str(error) or "Levene test could not be calculated."
        else:
            row["Levene Reason"] = "At least two clusters must have valid values for Levene's test."

        if cluster_count < 2:
            reason = "At least two clusters must have valid values."
            row.update({"ANOVA Status": "Unavailable", "ANOVA Reason": reason,
                        "Kruskal Status": "Unavailable", "Kruskal Reason": reason})
        elif constant:
            reason = "The variable is constant across all valid observations; test statistics are undefined."
            row.update({"ANOVA Status": "Unavailable", "ANOVA Reason": reason,
                        "Kruskal Status": "Unavailable", "Kruskal Reason": reason})
        else:
            if any(len(group) < 2 for group in groups):
                row["ANOVA Status"] = "Unavailable"
                row["ANOVA Reason"] = "Each cluster needs at least two valid observations for one-way ANOVA."
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConstantInputWarning)
                        test = f_oneway(*groups)
                    if np.isfinite(test.statistic) and np.isfinite(test.pvalue):
                        grand_mean = float(all_values.mean())
                        ss_total = float(((all_values - grand_mean) ** 2).sum())
                        ss_between = float(sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups))
                        eta_squared = ss_between / ss_total if ss_total > 0 else None
                        row.update({
                            "ANOVA Statistic": _finite_or_none(test.statistic),
                            "ANOVA P-value": _finite_or_none(test.pvalue),
                            "Eta-squared": _finite_or_none(eta_squared) if eta_squared is not None else None,
                            "ANOVA Status": "OK",
                            "ANOVA Reason": None,
                        })
                    else:
                        row["ANOVA Status"] = "Unavailable"
                        row["ANOVA Reason"] = "ANOVA returned a non-finite result."
                except (ValueError, FloatingPointError, ZeroDivisionError) as error:
                    row["ANOVA Status"] = "Unavailable"
                    row["ANOVA Reason"] = str(error) or "ANOVA could not be calculated."

            if sample_count <= cluster_count:
                row["Kruskal Status"] = "Unavailable"
                row["Kruskal Reason"] = "At least one additional valid observation beyond one per cluster is required."
            else:
                try:
                    test = kruskal(*groups)
                    if np.isfinite(test.statistic) and np.isfinite(test.pvalue):
                        epsilon_squared = (float(test.statistic) - cluster_count + 1) / (sample_count - cluster_count)
                        row.update({
                            "Kruskal Statistic": _finite_or_none(test.statistic),
                            "Kruskal P-value": _finite_or_none(test.pvalue),
                            "Epsilon-squared": _finite_or_none(epsilon_squared),
                            "Kruskal Status": "OK",
                            "Kruskal Reason": None,
                        })
                    else:
                        row["Kruskal Status"] = "Unavailable"
                        row["Kruskal Reason"] = "Kruskal-Wallis returned a non-finite result."
                except (ValueError, FloatingPointError, ZeroDivisionError) as error:
                    row["Kruskal Status"] = "Unavailable"
                    row["Kruskal Reason"] = str(error) or "Kruskal-Wallis could not be calculated."
        rows.append(row)

    return pd.DataFrame(rows, columns=NUMERIC_RESULT_COLUMNS)


def compare_categorical_clusters(dataframe, labels, categorical_columns):
    """Compare categorical distributions across clusters using Pearson chi-square.

    Missing values are excluded separately per field. Expected-frequency
    diagnostics are included because sparse tables weaken the usual asymptotic
    chi-square approximation. Cramer's V is the classical, uncorrected estimate
    and is not adjusted for finite-sample bias.
    """
    aligned = _align_labels(dataframe, labels)
    _, categorical = _validate_columns(dataframe, [], categorical_columns)
    rows = []
    for column in categorical:
        valid = pd.DataFrame(
            {"cluster": aligned, "category": dataframe[column]},
            index=dataframe.index,
        ).dropna(subset=["cluster", "category"])
        sample_count = len(valid)
        cluster_count = int(valid["cluster"].nunique())
        category_count = int(valid["category"].nunique())
        contingency = pd.crosstab(valid["cluster"], valid["category"])
        row = {
            "Variable": column,
            "Sample Count": sample_count,
            "Cluster Count": cluster_count,
            "Category Count": category_count,
            "Degrees of Freedom": None,
            "Chi-square Statistic": None,
            "P-value": None,
            "Cramer's V": None,
            "Minimum Expected Frequency": None,
            "Expected Cells < 5 Proportion": None,
            "Status": "Unavailable",
            "Reason": None,
            "Warning": None,
        }
        if cluster_count < 2:
            row["Reason"] = "At least two clusters must have valid values."
        elif category_count < 2:
            row["Reason"] = "At least two valid categories are required."
        else:
            try:
                chi2, p_value, degrees, expected = chi2_contingency(contingency.to_numpy(), correction=False)
                if not (np.isfinite(chi2) and np.isfinite(p_value) and np.isfinite(expected).all()):
                    row["Reason"] = "Chi-square test returned a non-finite result."
                else:
                    dimension = min(contingency.shape)
                    cramers_v = np.sqrt(chi2 / (sample_count * (dimension - 1))) if dimension > 1 else None
                    min_expected = float(expected.min())
                    small_proportion = float((expected < 5).mean())
                    warning = None
                    if min_expected < 1 or small_proportion > 0.2:
                        warning = (
                            "Small expected counts: the ordinary chi-square approximation may be unreliable. "
                            "Consider combining sparse categories or using a suitable exact/Monte Carlo method."
                        )
                    row.update({
                        "Degrees of Freedom": int(degrees),
                        "Chi-square Statistic": _finite_or_none(chi2),
                        "P-value": _finite_or_none(p_value),
                        "Cramer's V": _finite_or_none(cramers_v) if cramers_v is not None else None,
                        "Minimum Expected Frequency": min_expected,
                        "Expected Cells < 5 Proportion": small_proportion,
                        "Status": "OK",
                        "Reason": None,
                        "Warning": warning,
                    })
            except (ValueError, FloatingPointError, ZeroDivisionError) as error:
                row["Reason"] = str(error) or "Chi-square test could not be calculated."
        rows.append(row)
    return pd.DataFrame(rows, columns=CATEGORICAL_RESULT_COLUMNS)


def adjust_p_values(numeric_results, categorical_results):
    """Apply Benjamini-Hochberg FDR within separate test families.

    Numeric ANOVA p-values, numeric Kruskal-Wallis p-values, and categorical
    chi-square p-values are corrected independently. Invalid/missing p-values
    do not participate. Returns copies with adjusted-p-value columns.
    """
    numeric = numeric_results.copy(deep=True)
    categorical = categorical_results.copy(deep=True)
    for column in ("ANOVA P-value", "Kruskal P-value"):
        adjusted_column = column.replace("P-value", "Adjusted P-value")
        numeric[adjusted_column] = np.nan
        valid = pd.to_numeric(numeric[column], errors="coerce") if column in numeric else pd.Series(dtype=float)
        valid_mask = valid.notna() & np.isfinite(valid) & valid.between(0, 1)
        if valid_mask.any():
            p_values = valid.loc[valid_mask].to_numpy(dtype=float)
            order = np.argsort(p_values)
            ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
            ranked = np.minimum.accumulate(ranked[::-1])[::-1]
            adjusted = np.empty_like(ranked)
            adjusted[order] = np.minimum(ranked, 1.0)
            numeric.loc[valid_mask, adjusted_column] = adjusted

    categorical["Adjusted P-value"] = np.nan
    if "P-value" in categorical:
        valid = pd.to_numeric(categorical["P-value"], errors="coerce")
        valid_mask = valid.notna() & np.isfinite(valid) & valid.between(0, 1)
        if valid_mask.any():
            p_values = valid.loc[valid_mask].to_numpy(dtype=float)
            order = np.argsort(p_values)
            ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
            ranked = np.minimum.accumulate(ranked[::-1])[::-1]
            adjusted = np.empty_like(ranked)
            adjusted[order] = np.minimum(ranked, 1.0)
            categorical.loc[valid_mask, "Adjusted P-value"] = adjusted
    return numeric, categorical


def analyze_cluster_statistics(
    dataframe,
    labels,
    numeric_columns=None,
    categorical_columns=None,
):
    """Run exploratory numeric/categorical tests with family-wise FDR correction.

    If clusters were generated from these same features, results are descriptive
    and exploratory: p-values are not independent confirmatory evidence of
    population-level differences.
    """
    aligned = _align_labels(dataframe, labels)
    if numeric_columns is None:
        numeric_columns = [
            column for column in dataframe.columns
            if pd.api.types.is_numeric_dtype(dataframe[column].dtype)
            and not pd.api.types.is_bool_dtype(dataframe[column].dtype)
        ]
    if categorical_columns is None:
        categorical_columns = [
            column for column in dataframe.columns
            if not pd.api.types.is_numeric_dtype(dataframe[column].dtype)
            and not pd.api.types.is_datetime64_any_dtype(dataframe[column].dtype)
        ]
    numeric, categorical = _validate_columns(dataframe, numeric_columns, categorical_columns)
    numeric_results = compare_numeric_clusters(dataframe, aligned, numeric)
    categorical_results = compare_categorical_clusters(dataframe, aligned, categorical)
    numeric_results, categorical_results = adjust_p_values(numeric_results, categorical_results)
    return ClusterStatisticsResult(numeric=numeric_results, categorical=categorical_results)

