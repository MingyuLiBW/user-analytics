import pandas as pd
import pytest
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from unittest.mock import patch

from user_analytics.analysis.eda import (
    calculate_correlation,
    prepare_bivariate_data,
    prepare_categorical_crosstab,
    prepare_categorical_numeric_data,
)
from user_analytics.ui.pages.eda import _build_numeric_scatter


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


def test_numeric_bivariate_data_excludes_non_finite_pairs():
    dataframe = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, float("inf"), 5.0, None],
        "y": [2.0, 4.0, 6.0, 8.0, -float("inf"), 12.0],
        "group": ["a", "b", "c", "d", "e", "f"],
    })
    original = dataframe.copy(deep=True)

    pair = prepare_bivariate_data(dataframe, "x", "y")
    correlation = calculate_correlation(dataframe, columns=["x", "y"])

    assert pair.to_dict("records") == [
        {"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}, {"x": 3.0, "y": 6.0}
    ]
    assert correlation.loc["x", "y"] == pytest.approx(1)
    pd.testing.assert_frame_equal(dataframe, original)


def test_category_numeric_data_excludes_non_finite_numbers_only():
    dataframe = pd.DataFrame({"group": ["a", "b", "c", "inf"], "score": [1.0, float("inf"), 3.0, -float("inf")]})

    result = prepare_categorical_numeric_data(dataframe, "group", "score")

    assert result.to_dict("records") == [{"group": "a", "score": 1.0}, {"group": "c", "score": 3.0}]


def test_ols_scatter_builds_plotly_trendline_and_regular_scatter():
    dataframe = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 3.0, 5.0, 7.0]})

    trendline_figure, trendline_note = _build_numeric_scatter(
        dataframe, "x", "y", None, add_trendline=True
    )
    regular_figure, regular_note = _build_numeric_scatter(
        dataframe, "x", "y", None, add_trendline=False
    )

    assert callable(sm.OLS)
    assert trendline_note is None
    assert len(px.get_trendline_results(trendline_figure)) == 1
    assert regular_note is None
    assert len(px.get_trendline_results(regular_figure)) == 0


def test_ols_scatter_uses_pairwise_finite_data():
    dataframe = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, float("inf"), 4.0, None],
        "y": [1.0, 3.0, 5.0, 7.0, -float("inf"), 11.0],
    })
    pair_data = prepare_bivariate_data(dataframe, "x", "y")

    figure, note = _build_numeric_scatter(pair_data, "x", "y", None, add_trendline=True)

    assert len(pair_data) == 3
    assert note is None
    assert len(px.get_trendline_results(figure)) == 1


def test_ols_scatter_with_insufficient_data_falls_back_without_page_error():
    dataframe = pd.DataFrame({"x": [1.0], "y": [2.0]})

    figure, note = _build_numeric_scatter(dataframe, "x", "y", None, add_trendline=True)

    assert note is not None
    assert "at least two finite observations" in note
    assert len(figure.data) == 1
    assert len(px.get_trendline_results(figure)) == 0


def test_ols_scatter_checks_sample_sufficiency_within_color_groups():
    dataframe = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [1.0, 3.0, 5.0, 7.0],
        "group": ["single", "other", "other", "other"],
    })

    figure, note = _build_numeric_scatter(
        dataframe, "x", "y", "group", add_trendline=True
    )

    assert note is not None
    assert "each color group" in note
    assert len(px.get_trendline_results(figure)) == 0


@pytest.mark.parametrize(
    "fit_error",
    [ValueError("OLS input could not be fitted."), np.linalg.LinAlgError("Singular fit.")],
)
def test_ols_expected_fit_errors_fall_back_to_regular_scatter(fit_error):
    dataframe = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 3.0, 5.0]})
    plotly_scatter = px.scatter

    def scatter_with_failing_ols(*args, **kwargs):
        if kwargs.get("trendline") == "ols":
            raise fit_error
        return plotly_scatter(*args, **kwargs)

    with patch("user_analytics.ui.pages.eda.px.scatter", side_effect=scatter_with_failing_ols):
        figure, note = _build_numeric_scatter(
            dataframe, "x", "y", None, add_trendline=True
        )

    assert note == "The linear trendline could not be calculated; the scatter plot is shown without it."
    assert len(figure.data) == 1
    assert len(px.get_trendline_results(figure)) == 0


def test_ols_unexpected_scatter_error_is_not_silently_swallowed():
    dataframe = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 3.0, 5.0]})

    with patch(
        "user_analytics.ui.pages.eda.px.scatter",
        side_effect=RuntimeError("Unexpected Plotly programming error."),
    ):
        with pytest.raises(RuntimeError, match="Unexpected Plotly programming error"):
            _build_numeric_scatter(dataframe, "x", "y", None, add_trendline=True)
