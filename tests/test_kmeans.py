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
