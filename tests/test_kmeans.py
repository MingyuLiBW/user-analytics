import numpy as np
import pandas as pd
import pytest

from user_analytics.analysis.clustering import (
    evaluate_kmeans,
    fit_kmeans,
    prepare_clustering_data,
    summarize_clusters,
)


def make_features():
    source = pd.DataFrame(
        {"x": [0.0, 0.2, 0.1, 10.0, 10.2, 9.9], "y": [0.0, 0.1, 0.2, 10.0, 9.8, 10.1]},
        index=[11, 12, 13, 21, 22, 23],
    )
    return source, prepare_clustering_data(source, ["x", "y"])


def test_fit_kmeans_returns_model_labels_centers_and_inertia():
    source, prepared = make_features()

    result = fit_kmeans(prepared, n_clusters=2, random_state=7)

    assert len(result.labels) == len(source)
    assert result.labels.index.tolist() == source.index.tolist()
    assert result.centers.shape == (2, 2)
    assert result.inertia >= 0
    assert result.model.n_init == 10


def test_fixed_random_state_produces_reproducible_labels_and_centers():
    _, prepared = make_features()

    first = fit_kmeans(prepared, n_clusters=2, random_state=123)
    second = fit_kmeans(prepared, n_clusters=2, random_state=123)

    np.testing.assert_array_equal(first.labels.to_numpy(), second.labels.to_numpy())
    np.testing.assert_allclose(first.centers.to_numpy(), second.centers.to_numpy())
    assert first.inertia == pytest.approx(second.inertia)


def test_cluster_summary_has_correct_counts_proportions_and_source_means():
    source, prepared = make_features()
    fitted = fit_kmeans(prepared, n_clusters=2, random_state=3)

    summary = summarize_clusters(source, fitted.labels, numeric_columns=["x", "y"])

    assert summary["Sample Count"].sum() == len(source)
    assert summary["Sample Proportion"].sum() == pytest.approx(1.0)
    assert summary["Sample Count"].sort_values().tolist() == [3, 3]
    assert summary.loc[summary["x"].idxmin(), "x"] < 1
    assert summary.loc[summary["x"].idxmax(), "x"] > 9


def test_cluster_summary_accepts_exact_series_index_match():
    source = pd.DataFrame({"value": [1.0, 3.0, 10.0, 12.0]}, index=[10, 20, 30, 40])
    labels = pd.Series([0, 0, 1, 1], index=source.index, name="Cluster")

    summary = summarize_clusters(source, labels, numeric_columns=["value"])

    assert summary["Sample Count"].to_dict() == {0: 2, 1: 2}
    assert summary["Sample Proportion"].to_dict() == {0: 0.5, 1: 0.5}
    assert summary["value"].to_dict() == {0: 2.0, 1: 11.0}


def test_cluster_summary_reorders_series_labels_by_matching_index():
    source = pd.DataFrame({"value": [1.0, 3.0, 10.0, 12.0]}, index=[10, 20, 30, 40])
    labels = pd.Series([1, 0, 1, 0], index=[30, 10, 40, 20], name="Cluster")

    summary = summarize_clusters(source, labels, numeric_columns=["value"])

    assert summary["Sample Count"].to_dict() == {0: 2, 1: 2}
    assert summary["value"].to_dict() == {0: 2.0, 1: 11.0}


@pytest.mark.parametrize(
    "label_index",
    ([10, 20, 30], [10, 20, 30, 40, 50]),
    ids=["missing-index", "extra-index"],
)
def test_cluster_summary_rejects_incomplete_or_extra_series_indices(label_index):
    source = pd.DataFrame({"value": [1.0, 3.0, 10.0, 12.0]}, index=[10, 20, 30, 40])
    labels = pd.Series(range(len(label_index)), index=label_index)

    with pytest.raises(ValueError, match="match.*index exactly"):
        summarize_clusters(source, labels)


def test_cluster_summary_rejects_duplicate_dataframe_index():
    source = pd.DataFrame({"value": [1.0, 3.0, 10.0]}, index=[10, 10, 30])
    labels = pd.Series([0, 0, 1], index=[10, 10, 30])

    with pytest.raises(ValueError, match="DataFrame index must be unique"):
        summarize_clusters(source, labels)


def test_cluster_summary_rejects_duplicate_series_index():
    source = pd.DataFrame({"value": [1.0, 3.0, 10.0]}, index=[10, 20, 30])
    labels = pd.Series([0, 0, 1], index=[10, 10, 30])

    with pytest.raises(ValueError, match="label index must be unique"):
        summarize_clusters(source, labels)


def test_cluster_summary_rejects_array_labels_with_wrong_length():
    source = pd.DataFrame({"value": [1.0, 3.0, 10.0, 12.0]}, index=[10, 20, 30, 40])

    with pytest.raises(ValueError, match="label count must match"):
        summarize_clusters(source, [0, 0, 1])


def test_evaluate_kmeans_returns_elbow_and_silhouette_values():
    _, prepared = make_features()

    result = evaluate_kmeans(prepared.features, [2, 3], random_state=8)

    assert result["K"].tolist() == [2, 3]
    assert (result["Inertia"] >= 0).all()
    assert result["Silhouette Score"].between(-1, 1).all()


def test_invalid_cluster_counts_and_insufficient_unique_points_raise_clear_errors():
    _, prepared = make_features()
    with pytest.raises(ValueError, match="between 2"):
        fit_kmeans(prepared, n_clusters=6)
    with pytest.raises(ValueError, match="integer"):
        fit_kmeans(prepared, n_clusters=2.0)

    repeated = pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 2, 2]})
    repeated_features = prepare_clustering_data(repeated, ["a", "b"])
    with pytest.raises(ValueError, match="unique data points"):
        fit_kmeans(repeated_features, n_clusters=2)
    with pytest.raises(ValueError, match="unique data points"):
        evaluate_kmeans(repeated_features.features, [2])


def test_evaluation_rejects_too_few_samples_and_out_of_range_k():
    with pytest.raises(ValueError, match="at least 3 samples"):
        evaluate_kmeans(pd.DataFrame([[0.0], [1.0]]), [2])
    with pytest.raises(ValueError, match="invalid"):
        evaluate_kmeans(pd.DataFrame([[0.0], [1.0], [2.0]]), [1])
    with pytest.raises(ValueError, match="invalid"):
        evaluate_kmeans(pd.DataFrame([[0.0], [1.0], [2.0]]), [3])


def test_original_data_is_not_modified_by_fitting_evaluation_or_summary():
    source, prepared = make_features()
    original = source.copy(deep=True)

    fitted = fit_kmeans(prepared, n_clusters=2)
    evaluate_kmeans(prepared.features, [2])
    summarize_clusters(source, fitted.labels)

    pd.testing.assert_frame_equal(source, original)
