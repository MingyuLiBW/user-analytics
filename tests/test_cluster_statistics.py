import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from user_analytics.analysis.cluster_statistics import (
    adjust_p_values,
    analyze_cluster_statistics,
    compare_categorical_clusters,
    compare_numeric_clusters,
)


def numeric_fixture():
    dataframe = pd.DataFrame(
        {
            "measure": [1.0, 2.0, 2.0, 3.0, 8.0, 9.0, 9.0, 10.0],
            "constant": [4.0] * 8,
            "group": ["a", "a", "b", "b", "a", "a", "b", "b"],
        },
        index=[10, 20, 30, 40, 50, 60, 70, 80],
    )
    labels = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=dataframe.index)
    return dataframe, labels


def test_numeric_anova_kruskal_and_effect_sizes():
    dataframe, labels = numeric_fixture()

    result = compare_numeric_clusters(dataframe, labels, ["measure"])
    row = result.iloc[0]

    assert row["ANOVA Status"] == "OK"
    assert row["Kruskal Status"] == "OK"
    assert row["ANOVA P-value"] < 0.05
    assert row["Kruskal P-value"] < 0.05
    assert 0 <= row["Eta-squared"] <= 1
    assert -1 <= row["Epsilon-squared"] <= 1
    assert row["Sample Count"] == 8
    assert row["Cluster Count"] == 2


def test_numeric_missing_values_are_removed_per_variable():
    dataframe = pd.DataFrame({"measure": [1.0, None, 3.0, 4.0, 8.0, None], "other": [None] * 6})
    labels = pd.Series([0, 0, 0, 1, 1, 1], index=dataframe.index)

    result = compare_numeric_clusters(dataframe, labels, ["measure"]).iloc[0]

    assert result["Sample Count"] == 4
    assert result["Cluster Count"] == 2


def test_constant_numeric_variable_returns_explained_unavailable_status():
    dataframe = pd.DataFrame({"constant": [2.0] * 6})
    labels = [0, 0, 0, 1, 1, 1]

    result = compare_numeric_clusters(dataframe, labels, ["constant"]).iloc[0]

    assert result["ANOVA Status"] == "Unavailable"
    assert result["Kruskal Status"] == "Unavailable"
    assert result["ANOVA P-value"] is None
    assert "constant" in result["ANOVA Reason"]


def test_numeric_insufficient_valid_cluster_has_status_not_nan():
    dataframe = pd.DataFrame({"x": [1.0, None, None, 2.0]})
    labels = [0, 0, 1, 1]

    row = compare_numeric_clusters(dataframe, labels, ["x"]).iloc[0]

    assert row["ANOVA Status"] == "Unavailable"
    assert row["Kruskal Status"] == "Unavailable"
    assert row["ANOVA P-value"] is None


def test_categorical_chi_square_cramers_v_and_expected_count_warning():
    dataframe = pd.DataFrame({"choice": ["a"] * 8 + ["b"] * 2 + ["a"] * 2 + ["b"] * 8})
    labels = pd.Series([0] * 10 + [1] * 10, index=dataframe.index)

    result = compare_categorical_clusters(dataframe, labels, ["choice"]).iloc[0]

    assert result["Status"] == "OK"
    assert result["Degrees of Freedom"] == 1
    assert result["Chi-square Statistic"] > 0
    assert 0 <= result["P-value"] <= 1
    assert 0 <= result["Cramer's V"] <= 1
    assert result["Minimum Expected Frequency"] == 5
    assert result["Expected Cells < 5 Proportion"] == 0


def test_categorical_small_expected_counts_have_warning():
    dataframe = pd.DataFrame({"kind": ["rare", "common", "common", "rare", "common", "common"]})
    labels = pd.Series([0, 0, 0, 1, 1, 1], index=dataframe.index)

    row = compare_categorical_clusters(dataframe, labels, ["kind"]).iloc[0]

    assert row["Status"] == "OK"
    assert row["Warning"] is not None
    assert row["Minimum Expected Frequency"] < 5


def test_categorical_single_category_and_single_cluster_are_unavailable():
    one_category = pd.DataFrame({"kind": ["x"] * 4})
    result = compare_categorical_clusters(one_category, [0, 0, 1, 1], ["kind"]).iloc[0]
    assert result["Status"] == "Unavailable"
    assert "two valid categories" in result["Reason"]

    one_cluster = pd.DataFrame({"kind": ["x", "y", "x"]})
    result = compare_categorical_clusters(one_cluster, [0, 0, 0], ["kind"]).iloc[0]
    assert result["Status"] == "Unavailable"
    assert "two clusters" in result["Reason"]


def test_fdr_adjustment_is_separate_for_each_test_family_and_skips_invalid():
    numeric = pd.DataFrame(
        {
            "ANOVA P-value": [0.01, 0.04, None],
            "Kruskal P-value": [0.03, 0.2, np.inf],
        }
    )
    categorical = pd.DataFrame({"P-value": [0.02, 0.5]})

    adjusted_numeric, adjusted_categorical = adjust_p_values(numeric, categorical)

    assert adjusted_numeric["ANOVA Adjusted P-value"].iloc[:2].tolist() == pytest.approx([0.02, 0.04])
    assert adjusted_numeric["Kruskal Adjusted P-value"].iloc[:2].tolist() == pytest.approx([0.06, 0.2])
    assert pd.isna(adjusted_numeric["ANOVA Adjusted P-value"].iloc[2])
    assert adjusted_categorical["Adjusted P-value"].tolist() == pytest.approx([0.04, 0.5])


def test_empty_fdr_inputs_return_empty_adjusted_columns():
    numeric, categorical = adjust_p_values(pd.DataFrame(), pd.DataFrame())

    assert numeric.empty and categorical.empty
    assert "ANOVA Adjusted P-value" in numeric
    assert "Adjusted P-value" in categorical


def test_unified_entry_aligns_series_and_array_labels_and_preserves_inputs():
    dataframe, labels = numeric_fixture()
    original = dataframe.copy(deep=True)
    original_labels = labels.copy(deep=True)

    result = analyze_cluster_statistics(
        dataframe,
        labels.iloc[::-1],
        numeric_columns=["measure"],
        categorical_columns=["group"],
    )
    positional = analyze_cluster_statistics(
        dataframe,
        labels.to_numpy(),
        numeric_columns=["measure"],
        categorical_columns=["group"],
    )

    assert result.numeric["ANOVA P-value"].iloc[0] == pytest.approx(positional.numeric["ANOVA P-value"].iloc[0])
    assert not result.categorical.empty
    pd.testing.assert_frame_equal(dataframe, original)
    pd.testing.assert_series_equal(labels, original_labels)


def test_invalid_inputs_fields_and_misaligned_series_raise_clear_errors():
    dataframe, labels = numeric_fixture()

    with pytest.raises(ValueError, match="non-empty"):
        analyze_cluster_statistics(pd.DataFrame(columns=["x"]), [])
    with pytest.raises(ValueError, match="label count"):
        analyze_cluster_statistics(dataframe, [0])
    with pytest.raises(ValueError, match="index must match"):
        analyze_cluster_statistics(dataframe, pd.Series(labels.to_numpy(), index=range(100, 108)))
    with pytest.raises(ValueError, match="unique"):
        analyze_cluster_statistics(dataframe, pd.Series(labels.to_numpy(), index=[1, 1, 2, 3, 4, 5, 6, 7]))
    with pytest.raises(ValueError, match="not found"):
        analyze_cluster_statistics(dataframe, labels, numeric_columns=["missing"], categorical_columns=[])
    with pytest.raises(ValueError, match="numeric fields"):
        analyze_cluster_statistics(dataframe, labels, numeric_columns=["group"], categorical_columns=[])


def test_no_valid_values_and_zero_variance_with_group_separation_are_explicit():
    dataframe = pd.DataFrame({"empty": pd.Series([None] * 6, dtype=float), "split": [0.0] * 3 + [1.0] * 3})
    labels = [0, 0, 0, 1, 1, 1]

    empty = compare_numeric_clusters(dataframe, labels, ["empty"]).iloc[0]
    split = compare_numeric_clusters(dataframe, labels, ["split"]).iloc[0]

    assert empty["Sample Count"] == 0
    assert empty["ANOVA Status"] == "Unavailable"
    assert empty["Kruskal Status"] == "Unavailable"
    assert split["ANOVA Status"] == "Unavailable"
    assert "non-finite" in split["ANOVA Reason"]
    assert split["Kruskal Status"] == "OK"



def test_fixed_anova_f_p_and_eta_squared_reference_values():
    # Hand calculation: group means are 2 and 5; SS_between=13.5,
    # SS_within=4, so F=13.5 and eta^2=13.5/(13.5+4)=27/35.
    dataframe = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    labels = [0, 0, 0, 1, 1, 1]

    row = compare_numeric_clusters(dataframe, labels, ["x"]).iloc[0]

    assert row["ANOVA Statistic"] == pytest.approx(13.5)
    assert row["ANOVA P-value"] == pytest.approx(0.02131164112875672)
    assert row["Eta-squared"] == pytest.approx(27 / 35)
    assert row["Variance Homogeneity Status"] == "No significant evidence of unequal variances"


def test_fixed_kruskal_ties_h_p_and_epsilon_reference_values():
    # Combined average ranks are [1, 3, 3, 3, 5.5, 5.5]. The tie
    # correction coefficient is 1 - sum(t^3-t)/(N^3-N) = 6/7;
    # the uncorrected H is 2.333333..., so corrected H=H/(6/7)=2.722222... .
    # This gives p=0.098960... and epsilon=(H-2+1)/(6-2).
    dataframe = pd.DataFrame({"x": [1.0, 2.0, 2.0, 2.0, 3.0, 3.0]})
    labels = [0, 0, 0, 1, 1, 1]

    row = compare_numeric_clusters(dataframe, labels, ["x"]).iloc[0]

    assert row["Kruskal Statistic"] == pytest.approx(2.7222222222222165)
    assert row["Kruskal P-value"] == pytest.approx(0.09896015401940612)
    assert row["Epsilon-squared"] == pytest.approx((2.7222222222222165 - 2 + 1) / 4)


def test_fixed_chi_square_expected_values_df_p_and_cramers_v():
    # Observed [[10,20],[20,10]], N=60; expected cells are 15,
    # chi-square=4*(25/15)=20/3, df=1, Cramer's V=sqrt((20/3)/60)=1/3.
    dataframe = pd.DataFrame({"choice": ["a"] * 10 + ["b"] * 20 + ["a"] * 20 + ["b"] * 10})
    labels = [0] * 30 + [1] * 30

    row = compare_categorical_clusters(dataframe, labels, ["choice"]).iloc[0]

    assert row["Chi-square Statistic"] == pytest.approx(20 / 3)
    assert row["Degrees of Freedom"] == 1
    assert row["P-value"] == pytest.approx(0.009823274507519235)
    assert row["Cramer's V"] == pytest.approx(1 / 3)
    assert row["Minimum Expected Frequency"] == pytest.approx(15)
    assert row["Expected Cells < 5 Proportion"] == 0


def test_levene_diagnostic_is_independent_from_anova_calculation_status():
    dataframe = pd.DataFrame({"x": [1.0, 2.0, 4.0, 3.0, 5.0, 9.0]})
    with patch(
        "user_analytics.analysis.cluster_statistics.f_oneway",
        side_effect=ValueError("ANOVA calculation unavailable for this test."),
    ):
        row = compare_numeric_clusters(dataframe, [0, 0, 0, 1, 1, 1], ["x"]).iloc[0]

    assert row["ANOVA Status"] == "Unavailable"
    assert row["Variance Homogeneity Status"] in {
        "No significant evidence of unequal variances", "Potential heterogeneity"
    }
    assert np.isfinite(row["Levene P-value"])
    assert row["Levene Reason"] is None


@pytest.mark.parametrize(
    "mock_result, expected_reason",
    [
        (ValueError("synthetic Levene failure"), "synthetic Levene failure"),
        (type("LeveneResult", (), {"statistic": np.nan, "pvalue": 0.2})(), "non-finite"),
        (type("LeveneResult", (), {"statistic": 1.2, "pvalue": np.inf})(), "non-finite"),
    ],
)
def test_levene_failure_is_unavailable_without_losing_anova(mock_result, expected_reason):
    dataframe = pd.DataFrame({"x": [1.0, 2.0, 4.0, 3.0, 5.0, 9.0]})
    behavior = {"side_effect": mock_result} if isinstance(mock_result, Exception) else {"return_value": mock_result}
    with patch("user_analytics.analysis.cluster_statistics.levene", **behavior):
        row = compare_numeric_clusters(dataframe, [0, 0, 0, 1, 1, 1], ["x"]).iloc[0]

    assert row["Variance Homogeneity Status"] == "Unavailable"
    assert isinstance(row["Levene Reason"], str) and expected_reason in row["Levene Reason"]
    assert row["ANOVA Status"] == "OK"
    assert np.isfinite(row["ANOVA Statistic"])
    assert np.isfinite(row["ANOVA P-value"])


def test_negative_epsilon_squared_is_retained_without_truncation():
    dataframe = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
    row = compare_numeric_clusters(dataframe, [0, 0, 0, 1, 1, 1], ["x"]).iloc[0]

    assert row["Kruskal Statistic"] == pytest.approx(0.0)
    assert row["Kruskal P-value"] == pytest.approx(1.0)
    assert row["Epsilon-squared"] == pytest.approx(-1 / 4)


def test_bh_boundaries_repeated_values_single_value_and_invalid_family():
    numeric = pd.DataFrame({
        "ANOVA P-value": [0.0, 0.0, 1.0, 0.4],
        "Kruskal P-value": [0.25, np.nan, np.inf, -0.1],
    })
    categorical = pd.DataFrame({"P-value": [np.nan, np.inf]})
    adjusted_numeric, adjusted_categorical = adjust_p_values(numeric, categorical)

    assert adjusted_numeric["ANOVA Adjusted P-value"].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.5333333333333333])
    assert adjusted_numeric.loc[0, "Kruskal Adjusted P-value"] == pytest.approx(0.25)
    assert adjusted_numeric["Kruskal Adjusted P-value"].iloc[1:].isna().all()
    assert adjusted_categorical["Adjusted P-value"].isna().all()

    single_numeric, _ = adjust_p_values(
        pd.DataFrame({"ANOVA P-value": [0.37]}), pd.DataFrame()
    )
    assert single_numeric.loc[0, "ANOVA Adjusted P-value"] == pytest.approx(0.37)


def test_chi_square_warning_threshold_exactly_twenty_percent_and_expected_one():
    # Expected table [[1,5,5,5,14],[1,5,5,5,14]] has 2/10 cells <5,
    # minimum 1, and cells exactly 5 are not counted as below five.
    categories = ["a"] * 2 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10 + ["e"] * 28
    dataframe = pd.DataFrame({"kind": categories})
    labels = [0] * 30 + [1] * 30
    row = compare_categorical_clusters(dataframe, labels, ["kind"]).iloc[0]

    assert row["Expected Cells < 5 Proportion"] == pytest.approx(0.2)
    assert row["Minimum Expected Frequency"] == pytest.approx(1.0)
    assert row["Warning"] is None


def test_chi_square_warning_above_twenty_percent_and_expected_five_boundary():
    # Change one column total to 8, yielding expected counts 4 and 5
    # in two columns and a <5 proportion above 20 percent.
    category_counts = [2, 8, 10, 10, 30]
    categories = sum(([f"c{i}"] * count for i, count in enumerate(category_counts)), [])
    dataframe = pd.DataFrame({"kind": categories})
    labels = [0] * 30 + [1] * 30
    row = compare_categorical_clusters(dataframe, labels, ["kind"]).iloc[0]

    assert row["Minimum Expected Frequency"] == pytest.approx(1.0)
    assert row["Expected Cells < 5 Proportion"] == pytest.approx(0.4)
    assert row["Warning"] is not None


def test_nonfinite_numeric_observations_are_excluded_and_reported():
    dataframe = pd.DataFrame({"x": [1.0, 2.0, np.inf, -np.inf, 4.0, 5.0, np.nan]})
    row = compare_numeric_clusters(dataframe, [0, 0, 0, 1, 1, 1, 0], ["x"]).iloc[0]

    assert row["Sample Count"] == 4
    assert row["Non-finite Count"] == 2
    assert row["Cluster Count"] == 2
    assert row["ANOVA Status"] == "OK"


def test_empty_numeric_and_categorical_results_keep_full_schema():
    dataframe, labels = numeric_fixture()
    numeric_empty = compare_numeric_clusters(dataframe, labels, [])
    numeric_full = compare_numeric_clusters(dataframe, labels, ["measure"])
    categorical_empty = compare_categorical_clusters(dataframe, labels, [])
    categorical_full = compare_categorical_clusters(dataframe, labels, ["group"])

    assert numeric_empty.empty
    assert numeric_empty.columns.tolist() == numeric_full.columns.tolist()
    assert categorical_empty.empty
    assert categorical_empty.columns.tolist() == categorical_full.columns.tolist()


def test_result_schema_and_unavailable_values_survive_fdr_entrypoint():
    dataframe = pd.DataFrame({
        "constant": [2.0] * 6,
        "single_category": ["only"] * 6,
    })
    result = analyze_cluster_statistics(
        dataframe, [0, 0, 0, 1, 1, 1],
        numeric_columns=["constant"], categorical_columns=["single_category"],
    )
    numeric = result.numeric.iloc[0]
    categorical = result.categorical.iloc[0]

    assert {
        "Variable", "Sample Count", "Levene Statistic", "Levene P-value",
        "Variance Homogeneity Status", "Levene Reason", "ANOVA Statistic",
        "ANOVA P-value", "Eta-squared", "ANOVA Status", "ANOVA Reason",
        "Kruskal Statistic", "Kruskal P-value", "Epsilon-squared",
        "Kruskal Status", "Kruskal Reason", "ANOVA Adjusted P-value",
        "Kruskal Adjusted P-value",
    }.issubset(result.numeric.columns)
    assert {
        "Variable", "Sample Count", "Cluster Count", "Category Count",
        "Degrees of Freedom", "Chi-square Statistic", "P-value", "Cramer's V",
        "Minimum Expected Frequency", "Expected Cells < 5 Proportion",
        "Status", "Reason", "Warning", "Adjusted P-value",
    }.issubset(result.categorical.columns)

    assert numeric["ANOVA Status"] == "Unavailable"
    assert isinstance(numeric["ANOVA Reason"], str)
    assert pd.isna(numeric["ANOVA Statistic"])
    assert pd.isna(numeric["ANOVA P-value"])
    assert pd.isna(numeric["ANOVA Adjusted P-value"])
    assert numeric["Kruskal Status"] == "Unavailable"
    assert isinstance(numeric["Kruskal Reason"], str)
    assert pd.isna(numeric["Kruskal Statistic"])
    assert pd.isna(numeric["Kruskal P-value"])
    assert pd.isna(numeric["Kruskal Adjusted P-value"])
    assert categorical["Status"] == "Unavailable"
    assert isinstance(categorical["Reason"], str)
    assert pd.isna(categorical["Chi-square Statistic"])
    assert pd.isna(categorical["P-value"])
    assert pd.isna(categorical["Adjusted P-value"])
