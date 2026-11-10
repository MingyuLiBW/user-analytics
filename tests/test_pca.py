import numpy as np
import pandas as pd
import pytest

from user_analytics.analysis.clustering import prepare_pca_data


def sample_features():
    return pd.DataFrame(
        {"feature_a": [-2.0, -1.0, 1.0, 2.0], "feature_b": [1.0, -1.0, 1.0, -1.0]},
        index=[10, 20, 40, 80],
    )


def test_prepare_pca_returns_two_components_and_variance_ratios():
    features = sample_features()
    labels = pd.Series([0, 0, 1, 1], index=features.index, name="Cluster")

    result = prepare_pca_data(features, labels)

    assert result.coordinates[["PC1", "PC2"]].shape == (4, 2)
    assert 0 <= result.pc1_explained_variance_ratio <= 1
    assert 0 <= result.pc2_explained_variance_ratio <= 1
    assert result.cumulative_explained_variance_ratio == pytest.approx(
        result.pc1_explained_variance_ratio + result.pc2_explained_variance_ratio
    )
    assert result.cumulative_explained_variance_ratio <= 1 + 1e-12
    assert result.model.n_components_ == 2


def test_pca_preserves_nonconsecutive_indices_and_label_alignment():
    features = sample_features()
    labels = pd.Series([1, 0, 1, 0], index=[80, 10, 40, 20], name="Cluster")

    result = prepare_pca_data(features, labels)

    assert result.sample_indices.tolist() == [10, 20, 40, 80]
    assert result.coordinates.index.tolist() == [10, 20, 40, 80]
    assert result.coordinates["Sample Index"].tolist() == [10, 20, 40, 80]
    assert result.coordinates["Cluster"].tolist() == [0, 0, 1, 1]


def test_pca_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 2 samples"):
        prepare_pca_data(pd.DataFrame({"a": [1], "b": [2]}), [0])


def test_pca_rejects_fewer_than_two_feature_dimensions():
    with pytest.raises(ValueError, match="at least 2 feature dimensions"):
        prepare_pca_data(pd.DataFrame({"a": [1, 2, 3]}), [0, 1, 1])


def test_pca_rejects_label_length_mismatch():
    with pytest.raises(ValueError, match="label count must match"):
        prepare_pca_data(sample_features(), [0, 1])


def test_pca_does_not_modify_feature_matrix():
    features = sample_features()
    original = features.copy(deep=True)
    labels = pd.Series([0, 0, 1, 1], index=features.index)

    prepare_pca_data(features, labels)

    pd.testing.assert_frame_equal(features, original)
