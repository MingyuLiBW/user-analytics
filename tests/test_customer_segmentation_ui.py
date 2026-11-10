from unittest.mock import patch
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from user_analytics.analysis import clustering
from user_analytics.analysis.clustering import fit_kmeans, prepare_clustering_data
from user_analytics.ui.pages.customer_segmentation import (
    build_cluster_statistics_tables,
    build_cluster_profile_tables,
    make_result_payload,
    prepare_categorical_statistical_tables,
    prepare_numeric_statistical_tables,
    render,
    result_matches_current,
)


def test_customer_segmentation_page_shows_upload_prompt_without_data():
    with patch("user_analytics.ui.pages.customer_segmentation.st.title") as title, patch(
        "user_analytics.ui.pages.customer_segmentation.st.info"
    ) as info:
        render(None)

    title.assert_called_once_with("Customer Segmentation")
    info.assert_called_once()
    assert "Upload a CSV file" in info.call_args.args[0]


def test_display_payload_keeps_cluster_labels_aligned_to_original_index():
    dataframe = pd.DataFrame(
        {"spend": [1.0, 1.2, 9.8, 10.0], "visits": [2.0, 2.2, 9.7, 10.1]},
        index=[101, 103, 205, 209],
    )

    payload = make_result_payload(dataframe, ["spend", "visits"], n_clusters=2, random_state=11)

    assert payload["sample_indices"].tolist() == dataframe.index.tolist()
    assert payload["labels"].index.tolist() == dataframe.index.tolist()
    assert payload["summary"]["Sample Count"].sum() == len(dataframe)
    assert payload["sample_count"] == 4
    assert payload["cluster_count"] == 2
    assert payload["features"].index.tolist() == dataframe.index.tolist()


def test_result_payload_surfaces_silhouette_unavailable_without_losing_fit():
    dataframe = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]},
        index=[7, 8, 9, 10],
    )

    with patch(
        "user_analytics.ui.pages.customer_segmentation.silhouette_score",
        side_effect=ValueError("Silhouette is undefined for this result."),
    ):
        payload = make_result_payload(dataframe, ["a", "b"], n_clusters=2, random_state=1)

    assert payload["labels"].index.tolist() == dataframe.index.tolist()
    assert payload["inertia"] >= 0
    assert payload["silhouette"] is None
    assert "undefined" in payload["silhouette_note"]


def test_payload_silhouette_uses_fitted_labels_and_preprocessed_features():
    dataframe = pd.DataFrame(
        {"a": [0.0, 0.2, 0.8, 5.0, 5.2, 5.8], "b": [0.0, 0.1, 1.0, 5.0, 5.1, 6.0]},
        index=[10, 20, 30, 40, 50, 60],
    )
    prepared = prepare_clustering_data(dataframe, ["a", "b"])

    with patch("user_analytics.ui.pages.customer_segmentation.evaluate_kmeans") as evaluate:
        payload = make_result_payload(dataframe, ["a", "b"], n_clusters=2, random_state=7)

    expected = silhouette_score(prepared.features.to_numpy(), payload["labels"].to_numpy())
    assert payload["silhouette"] == pytest.approx(expected)
    evaluate.assert_not_called()


def test_payload_preserves_existing_kmeans_labels_centers_and_inertia():
    dataframe = pd.DataFrame(
        {"a": [0.0, 0.2, 0.8, 5.0, 5.2, 5.8], "b": [0.0, 0.1, 1.0, 5.0, 5.1, 6.0]},
        index=[10, 20, 30, 40, 50, 60],
    )
    prepared = prepare_clustering_data(dataframe, ["a", "b"])
    expected_fit = fit_kmeans(prepared, n_clusters=2, random_state=7)

    payload = make_result_payload(dataframe, ["a", "b"], n_clusters=2, random_state=7)

    pd.testing.assert_series_equal(payload["labels"], expected_fit.labels)
    pd.testing.assert_frame_equal(payload["centers"], expected_fit.centers)
    assert payload["inertia"] == pytest.approx(expected_fit.inertia)


def test_current_k_clustering_fits_kmeans_once(monkeypatch):
    dataframe = pd.DataFrame(
        {"a": [0.0, 0.2, 0.8, 5.0, 5.2, 5.8], "b": [0.0, 0.1, 1.0, 5.0, 5.1, 6.0]}
    )

    class CountingKMeans(KMeans):
        fit_calls = 0

        def fit(self, X, y=None, sample_weight=None):
            type(self).fit_calls += 1
            return super().fit(X, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(clustering, "KMeans", CountingKMeans)
    payload = make_result_payload(dataframe, ["a", "b"], n_clusters=2, random_state=7)

    assert CountingKMeans.fit_calls == 1
    assert payload["silhouette"] is not None


def test_collapsed_fitted_clusters_make_silhouette_unavailable_without_scoring():
    dataframe = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]}, index=[4, 5, 6])
    fitted = SimpleNamespace(
        labels=pd.Series([0, 0, 0], index=dataframe.index, name="Cluster"),
        centers=pd.DataFrame(),
        inertia=1.0,
    )

    with patch("user_analytics.ui.pages.customer_segmentation.fit_kmeans", return_value=fitted), patch(
        "user_analytics.ui.pages.customer_segmentation.silhouette_score"
    ) as scorer:
        payload = make_result_payload(dataframe, ["a", "b"], n_clusters=2, random_state=7)

    assert payload["silhouette"] is None
    assert "distinct fitted clusters" in payload["silhouette_note"]
    scorer.assert_not_called()


def test_silhouette_calculation_failure_keeps_fitted_cluster_result():
    dataframe = pd.DataFrame(
        {"a": [0.0, 0.2, 0.8, 5.0, 5.2, 5.8], "b": [0.0, 0.1, 1.0, 5.0, 5.1, 6.0]}
    )

    with patch(
        "user_analytics.ui.pages.customer_segmentation.silhouette_score",
        side_effect=ValueError("Silhouette is undefined for these labels."),
    ):
        payload = make_result_payload(dataframe, ["a", "b"], n_clusters=2, random_state=7)

    assert len(payload["labels"]) == len(dataframe)
    assert payload["inertia"] >= 0
    assert payload["silhouette"] is None
    assert "undefined" in payload["silhouette_note"]



def test_result_match_requires_current_metadata_and_pca_inputs():
    current = {
        "dataset_fingerprint": "abc",
        "selected_columns": ("spend", "visits"),
        "n_clusters": 3,
        "random_state": 42,
        "features": pd.DataFrame({"PC-input-1": [0.0], "PC-input-2": [1.0]}),
        "labels": pd.Series([0]),
    }

    assert result_matches_current(current, "abc", ["spend", "visits"], 3, 42)
    assert not result_matches_current(
        current, "abc", ["spend", "visits"], 2, 42
    )
    old_cached_result = {key: value for key, value in current.items() if key != "features"}
    assert not result_matches_current(old_cached_result, "abc", ["spend", "visits"], 3, 42)


def test_profile_tables_generate_profiles_for_mixed_selected_fields():
    dataframe = pd.DataFrame(
        {"spend": [10.0, 20.0, 30.0], "tier": ["A", "A", "B"]},
        index=[5, 8, 13],
    )
    labels = pd.Series([0, 0, 1], index=dataframe.index)

    profiles = build_cluster_profile_tables(dataframe, labels, ["spend", "tier"])

    assert not profiles.numeric.empty
    assert not profiles.categorical.empty
    assert profiles.numeric.loc[profiles.numeric["Cluster"] == 0, "Mean"].iloc[0] == 15
    assert profiles.categorical["Proportion"].sum() == 2.0


def test_profile_tables_handle_numeric_only_and_categorical_only_selections():
    dataframe = pd.DataFrame(
        {"spend": [10.0, 20.0], "tier": ["A", "B"]}, index=[2, 7]
    )
    labels = pd.Series([0, 1], index=dataframe.index)

    numeric_only = build_cluster_profile_tables(dataframe, labels, ["spend"])
    categorical_only = build_cluster_profile_tables(dataframe, labels, ["tier"])

    assert not numeric_only.numeric.empty
    assert numeric_only.categorical.empty
    assert categorical_only.numeric.empty
    assert not categorical_only.categorical.empty


def test_stale_or_invalid_result_is_not_eligible_for_profile_display():
    stale_result = {
        "dataset_fingerprint": "old-data",
        "selected_columns": ("spend", "tier"),
        "n_clusters": 2,
        "random_state": 42,
        "features": pd.DataFrame({"spend": [0.0], "tier_A": [1.0]}),
        "labels": pd.Series([0], index=[1]),
    }

    assert not result_matches_current(
        stale_result, "current-data", ["spend", "tier"], 2, 42
    )


def test_statistical_analysis_helper_reuses_original_data_and_aligned_labels():
    dataframe = pd.DataFrame(
        {
            "spend": [10.0, 20.0, 30.0, 40.0],
            "tier": ["A", "A", "B", "B"],
        },
        index=[11, 21, 31, 41],
    )
    original = dataframe.copy(deep=True)
    labels = pd.Series([0, 0, 1, 1], index=dataframe.index)

    result = build_cluster_statistics_tables(
        dataframe,
        labels.iloc[::-1],
        numeric_columns=["spend"],
        categorical_columns=["tier"],
    )

    assert set(result.numeric["Variable"]) == {"spend"}
    assert result.numeric.loc[0, "Sample Count"] == 4
    assert result.numeric.loc[0, "ANOVA Status"] == "OK"
    assert set(result.categorical["Variable"]) == {"tier"}
    assert result.categorical.loc[0, "Sample Count"] == 4
    assert result.categorical.loc[0, "Status"] == "OK"
    pd.testing.assert_frame_equal(dataframe, original)


def test_numeric_statistical_tables_split_results_and_preserve_small_p_values_and_reasons():
    results = pd.DataFrame(
        {
            "Variable": ["spend", "visits"],
            "Sample Count": [20, 18],
            "Cluster Count": [2, 2],
            "ANOVA Statistic": [12.34567, None],
            "ANOVA P-value": [1e-320, None],
            "ANOVA Adjusted P-value": [0.0, None],
            "Kruskal Statistic": [10.2, None],
            "Kruskal P-value": [0.0012, None],
            "Kruskal Adjusted P-value": [0.0024, None],
            "Levene Statistic": [1.1, None],
            "Levene P-value": [0.2, None],
            "Variance Homogeneity Status": ["No significant evidence of unequal variances", "Unavailable"],
            "Eta-squared": [0.45678, None],
            "Epsilon-squared": [0.32123, None],
            "Non-finite Count": [1, 0],
            "ANOVA Status": ["OK", "Unavailable"],
            "ANOVA Reason": [None, "Each cluster needs two observations."],
            "Kruskal Status": ["OK", "Unavailable"],
            "Kruskal Reason": [None, "Insufficient observations."],
            "Levene Reason": [None, "At least two clusters are required."],
        }
    )

    main, diagnostics = prepare_numeric_statistical_tables(results)

    assert main.columns.tolist() == [
        "Variable", "Sample Count", "Cluster Count", "ANOVA F", "ANOVA P-value",
        "ANOVA P-value (FDR)", "Kruskal-Wallis H", "Kruskal-Wallis P-value",
        "Kruskal-Wallis P-value (FDR)",
    ]
    assert main.loc[0, "ANOVA P-value"] == "1.000e-320"
    assert main.loc[0, "ANOVA P-value (FDR)"] == "<1e-300"
    assert "ANOVA Status" not in diagnostics
    assert diagnostics.loc[0, "Eta-squared"] == "0.457"
    assert diagnostics.loc[1, "Test Status / Reason"].find("Each cluster needs two observations.") >= 0
    assert diagnostics.loc[1, "Test Status / Reason"].find("Insufficient observations.") >= 0
    assert diagnostics.loc[1, "Test Status / Reason"].find("At least two clusters are required.") >= 0


def test_categorical_statistical_tables_keep_warnings_and_unavailable_reasons():
    results = pd.DataFrame(
        {
            "Variable": ["tier", "region"],
            "Sample Count": [40, 5],
            "Cluster Count": [2, 1],
            "Category Count": [3, 2],
            "Chi-square Statistic": [8.1234, None],
            "Degrees of Freedom": [2, None],
            "P-value": [0.0000123, None],
            "Adjusted P-value": [0.0000246, None],
            "Cramer's V": [0.31876, None],
            "Minimum Expected Frequency": [1.25, None],
            "Expected Cells < 5 Proportion": [0.4, None],
            "Status": ["OK", "Unavailable"],
            "Reason": [None, "At least two clusters must have valid values."],
            "Warning": ["Expected counts are small.", None],
        }
    )

    main, diagnostics = prepare_categorical_statistical_tables(results)

    assert main.columns.tolist() == [
        "Variable", "Sample Count", "Cluster Count", "Category Count",
        "Chi-square Statistic", "Degrees of Freedom", "P-value", "P-value (FDR)",
    ]
    assert main.loc[0, "P-value"] == "1.230e-05"
    assert diagnostics.loc[0, "Cramer's V"] == "0.319"
    assert diagnostics.loc[0, "Expected Cells < 5 Proportion"] == "40.0%"
    assert diagnostics.loc[0, "Expected Frequency Warning"] == "Expected counts are small."
    assert "Unavailable" in diagnostics.loc[1, "Test Status / Reason"]
    assert "At least two clusters" in diagnostics.loc[1, "Test Status / Reason"]
