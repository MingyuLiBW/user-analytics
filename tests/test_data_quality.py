import pandas as pd
import pytest

from user_analytics.analysis.outliers import analyze_outliers
from user_analytics.analysis.profiling import (
    analyze_data_types,
    analyze_duplicates,
    analyze_missing_values,
)


def test_missing_value_counts_and_percentages():
    dataframe = pd.DataFrame({"a": [1, None, 3], "b": [None, "x", "y"]})

    result = analyze_missing_values(dataframe)

    assert result["total_missing"] == 2
    assert result["total_cells"] == 6
    assert result["missing_percentage"] == pytest.approx(100 / 3)
    assert result["per_column"]["missing_count"].tolist() == [1, 1]
    assert result["per_column"]["missing_percentage"].tolist() == pytest.approx([100 / 3, 100 / 3])


def test_duplicate_rows_use_full_row_match():
    dataframe = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "y"]})

    result = analyze_duplicates(dataframe)

    assert result["duplicate_count"] == 1
    assert result["duplicate_percentage"] == pytest.approx(100 / 3)


def test_numeric_categorical_and_object_classification():
    dataframe = pd.DataFrame(
        {
            "amount": [1, 2, 3],
            "group": ["a", "a", "b"],
            "description": ["first", "second", "third"],
        }
    )

    result = analyze_data_types(dataframe).set_index("column")["category"]

    assert result["amount"] == "Numeric"
    assert result["group"] == "Categorical"
    assert result["description"] == "Other"


def test_iqr_outliers_are_counted_without_changing_data():
    dataframe = pd.DataFrame({"score": [1, 2, 3, 4, 100], "group": ["a"] * 5})
    original = dataframe.copy(deep=True)

    result = analyze_outliers(dataframe).set_index("column").loc["score"]

    assert result["outlier_count"] == 1
    assert result["outlier_percentage"] == 20
    pd.testing.assert_frame_equal(dataframe, original)


def test_all_missing_column_is_handled():
    dataframe = pd.DataFrame({"empty": [None, None], "value": [5, 5]})

    missing = analyze_missing_values(dataframe)
    types = analyze_data_types(dataframe).set_index("column")
    outliers = analyze_outliers(dataframe).set_index("column")

    assert missing["per_column"].loc[0, "missing_count"] == 2
    assert types.loc["empty", "category"] == "Other"
    assert outliers.loc["value", "status"] == "IQR is zero; outlier detection skipped"
    assert "empty" not in outliers.index


def test_zero_iqr_is_skipped_and_input_is_unchanged():
    dataframe = pd.DataFrame({"constant": [5, 5, 5, 5, 100]})
    original = dataframe.copy(deep=True)

    result = analyze_outliers(dataframe).iloc[0]

    assert result["outlier_count"] == 0
    assert result["status"] == "IQR is zero; outlier detection skipped"
    pd.testing.assert_frame_equal(dataframe, original)
