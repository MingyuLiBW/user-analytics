import pandas as pd
import pytest

from user_analytics.analysis.clustering import summarize_cluster_profiles


def fixture_data():
    dataframe = pd.DataFrame(
        {
            "spend": [10.0, 20.0, 30.0, None],
            "visits": [1.0, 3.0, 5.0, 7.0],
            "tier": ["A", "B", "A", None],
        },
        index=[11, 21, 41, 81],
    )
    labels = pd.Series([0, 0, 1, 1], index=dataframe.index, name="Cluster")
    return dataframe, labels


def test_numeric_profile_statistics_are_computed_from_original_values():
    dataframe, labels = fixture_data()

    result = summarize_cluster_profiles(dataframe, labels, numeric_columns=["spend"])
    cluster_zero = result.numeric.query("Cluster == 0 and Field == 'spend'").iloc[0]

    assert cluster_zero["Count"] == 2
    assert cluster_zero["Mean"] == 15
    assert cluster_zero["Median"] == 15
    assert cluster_zero["Standard Deviation"] == pytest.approx(7.0710678119)
    assert cluster_zero["Minimum"] == 10
    assert cluster_zero["Maximum"] == 20


def test_categorical_profile_counts_and_proportions_use_cluster_valid_denominator():
    dataframe, labels = fixture_data()

    result = summarize_cluster_profiles(dataframe, labels, numeric_columns=[], categorical_columns=["tier"])
    cluster_zero = result.categorical.query("Cluster == 0").set_index("Category")
    cluster_one = result.categorical.query("Cluster == 1").set_index("Category")

    assert cluster_zero["Count"].to_dict() == {"A": 1, "B": 1}
    assert cluster_zero["Proportion"].to_dict() == {"A": 0.5, "B": 0.5}
    assert cluster_one["Count"].to_dict() == {"A": 1}
    assert cluster_one["Proportion"].to_dict() == {"A": 1.0}


def test_clusters_are_summarized_separately():
    dataframe, labels = fixture_data()

    result = summarize_cluster_profiles(dataframe, labels, numeric_columns=["visits"], categorical_columns=[])
    means = result.numeric.set_index("Cluster")["Mean"]

    assert means[0] == 2
    assert means[1] == 6


def test_nonconsecutive_index_labels_align_by_index_even_when_labels_are_reordered():
    dataframe, labels = fixture_data()
    reordered_labels = labels.iloc[::-1]

    result = summarize_cluster_profiles(dataframe, reordered_labels, numeric_columns=["spend"], categorical_columns=[])

    means = result.numeric.set_index("Cluster")["Mean"]
    assert means[0] == 15
    assert means[1] == 30


def test_missing_numeric_and_categorical_values_are_omitted_per_field():
    dataframe, labels = fixture_data()

    result = summarize_cluster_profiles(dataframe, labels, numeric_columns=["spend"], categorical_columns=["tier"])
    cluster_one_spend = result.numeric.query("Cluster == 1 and Field == 'spend'").iloc[0]

    assert cluster_one_spend["Count"] == 1
    assert cluster_one_spend["Mean"] == 30
    assert result.categorical.query("Cluster == 1")["Proportion"].sum() == 1
    assert result.categorical["Category"].notna().all()


def test_cluster_field_with_all_missing_values_has_zero_count_and_no_categories():
    dataframe = pd.DataFrame({"score": [1.0, 2.0], "kind": [None, "x"]}, index=[4, 9])
    labels = pd.Series([0, 1], index=dataframe.index)

    result = summarize_cluster_profiles(dataframe, labels, numeric_columns=["score"], categorical_columns=["kind"])

    assert result.categorical.loc[result.categorical["Category"] == "x", "Count"].iloc[0] == 1
    assert result.categorical["Cluster"].tolist() == [1]
    all_missing_numeric = result.numeric.query("Cluster == 0").iloc[0]
    assert all_missing_numeric["Count"] == 1
    # Cluster 0's all-missing category emits no fabricated category row.
    assert result.categorical["Category"].tolist() == ["x"]


def test_empty_dataframe_raises_clear_error():
    dataframe = pd.DataFrame(columns=["score"])

    with pytest.raises(ValueError, match="empty DataFrame"):
        summarize_cluster_profiles(dataframe, pd.Series(dtype=int))


def test_label_length_and_index_mismatch_raise_clear_errors():
    dataframe, labels = fixture_data()

    with pytest.raises(ValueError, match="label count"):
        summarize_cluster_profiles(dataframe, labels.iloc[:-1])
    mismatched = pd.Series([0, 0, 1, 1], index=[101, 102, 103, 104])
    with pytest.raises(ValueError, match="index must match"):
        summarize_cluster_profiles(dataframe, mismatched)


def test_duplicate_dataframe_or_label_indices_are_rejected():
    dataframe, labels = fixture_data()
    duplicate_dataframe = dataframe.copy()
    duplicate_dataframe.index = [1, 1, 2, 3]
    with pytest.raises(ValueError, match="DataFrame index must be unique"):
        summarize_cluster_profiles(duplicate_dataframe, labels)

    duplicate_labels = pd.Series([0, 0, 1, 1], index=[11, 11, 41, 81])
    with pytest.raises(ValueError, match="label index must be unique"):
        summarize_cluster_profiles(dataframe, duplicate_labels)


def test_invalid_fields_and_numeric_type_are_rejected_and_inputs_unchanged():
    dataframe, labels = fixture_data()
    original = dataframe.copy(deep=True)
    original_labels = labels.copy(deep=True)

    with pytest.raises(ValueError, match="not found"):
        summarize_cluster_profiles(dataframe, labels, numeric_columns=["missing"])
    with pytest.raises(ValueError, match="numeric dtypes"):
        summarize_cluster_profiles(dataframe, labels, numeric_columns=["tier"])

    pd.testing.assert_frame_equal(dataframe, original)
    pd.testing.assert_series_equal(labels, original_labels)
