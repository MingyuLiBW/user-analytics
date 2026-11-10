import pandas as pd
import pytest

from user_analytics.analysis.eda import (
    calculate_correlation,
    prepare_bivariate_data,
    prepare_categorical_crosstab,
    prepare_categorical_numeric_data,
)


def test_numeric_bivariate_data_drops_only_pairwise_missing_rows():
    dataframe = pd.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 2, 1], "z": ["a", "b", "c", "d"]})

    result = prepare_bivariate_data(dataframe, "x", "y")

    assert result.to_dict("records") == [{"x": 1.0, "y": 4.0}, {"x": 4.0, "y": 1.0}]
    assert len(dataframe) == 4


def test_category_numeric_data_is_pairwise_valid_and_preserves_input():
    dataframe = pd.DataFrame({"group": ["A", "A", None], "score": [1, None, 3]})
    original = dataframe.copy(deep=True)

    result = prepare_categorical_numeric_data(dataframe, "group", "score")

    assert result.to_dict("records") == [{"group": "A", "score": 1.0}]
    pd.testing.assert_frame_equal(dataframe, original)


def test_categorical_crosstab_counts_and_row_proportions():
    dataframe = pd.DataFrame({"x": ["A", "A", "A", "B", None], "y": ["u", "u", "v", "v", "u"]})

    counts = prepare_categorical_crosstab(dataframe, "x", "y")
    proportions = prepare_categorical_crosstab(dataframe, "x", "y", normalize=True)

    assert counts.loc["A", "u"] == 2
    assert counts.loc["B", "v"] == 1
    assert proportions.loc["A", "u"] == pytest.approx(2 / 3)
    assert proportions.loc["A"].sum() == pytest.approx(1)


def test_pearson_and_spearman_correlations():
    dataframe = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "z": [4, 3, 2, 1], "flag": [True] * 4})

    pearson = calculate_correlation(dataframe, method="pearson")
    spearman = calculate_correlation(dataframe, method="spearman")

    assert pearson.loc["x", "y"] == pytest.approx(1)
    assert pearson.loc["x", "z"] == pytest.approx(-1)
    assert spearman.loc["x", "y"] == pytest.approx(1)
    assert "flag" not in pearson.columns


def test_correlation_uses_pairwise_valid_values_and_constant_stays_nan():
    dataframe = pd.DataFrame({"x": [1, 2, None, 4], "y": [2, 4, 6, 8], "constant": [5, 5, 5, 5]})

    matrix = calculate_correlation(dataframe, method="pearson")

    assert matrix.loc["x", "y"] == pytest.approx(1)
    assert pd.isna(matrix.loc["constant", "x"])
    assert pd.isna(matrix.loc["constant", "constant"])


def test_single_numeric_field_returns_one_by_one_matrix():
    matrix = calculate_correlation(pd.DataFrame({"x": [1, 2, 3]}))

    assert matrix.shape == (1, 1)
    assert matrix.loc["x", "x"] == 1


def test_bivariate_and_correlation_helpers_do_not_modify_source():
    dataframe = pd.DataFrame({"x": [1, 2, None], "y": [3, 2, 1], "g": ["a", "b", "a"]})
    original = dataframe.copy(deep=True)

    prepare_bivariate_data(dataframe, "x", "y")
    prepare_categorical_numeric_data(dataframe, "g", "x")
    calculate_correlation(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_same_field_is_rejected_for_bivariate_data():
    dataframe = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(ValueError, match="different"):
        prepare_bivariate_data(dataframe, "x", "x")
