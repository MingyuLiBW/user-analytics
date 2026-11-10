import numpy as np
import pandas as pd
import pytest

from user_analytics.analysis.clustering import prepare_clustering_data


def test_numeric_missing_values_are_filled_with_median_and_scaled():
    dataframe = pd.DataFrame({"income": [1.0, None, 9.0], "age": [20.0, 30.0, 40.0]})

    result = prepare_clustering_data(dataframe, ["income", "age"])

    assert result.features.loc[1, "income"] == pytest.approx(0.0)
    assert result.features["age"].mean() == pytest.approx(0.0)
    assert result.features["age"].std(ddof=0) == pytest.approx(1.0)
    assert result.numeric_columns == ["income", "age"]
    assert result.categorical_columns == []


def test_categorical_missing_values_are_filled_with_mode_and_one_hot_encoded():
    dataframe = pd.DataFrame({"age": [10, 20, 30], "segment": ["A", "A", None]})

    result = prepare_clustering_data(dataframe, ["age", "segment"])

    assert "segment_A" in result.feature_names
    assert result.features.loc[2, "segment_A"] == 1
    assert result.features.loc[2].filter(like="segment_").sum() == 1


def test_all_missing_categorical_field_uses_missing_placeholder_category():
    dataframe = pd.DataFrame({"age": [10, 20], "segment": [None, None]})

    result = prepare_clustering_data(dataframe, ["age", "segment"])

    assert "segment_**MISSING**" in result.feature_names
    assert result.features["segment_**MISSING**"].tolist() == [1.0, 1.0]


def test_unknown_categories_are_ignored_during_transform():
    dataframe = pd.DataFrame({"age": [10, 20, 30], "segment": ["A", "B", "A"]})
    result = prepare_clustering_data(dataframe, ["age", "segment"])
    new_data = pd.DataFrame({"age": [40], "segment": ["unseen"]})

    transformed = result.preprocessor.transform(new_data)

    assert transformed.shape[1] == len(result.feature_names)
    assert np.isfinite(transformed).all()
    assert transformed[0, result.feature_names.index("segment_A")] == 0
    assert transformed[0, result.feature_names.index("segment_B")] == 0


def test_invalid_duplicate_boolean_date_and_insufficient_features_are_rejected():
    dataframe = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [2, 3],
            "flag": [True, False],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )

    with pytest.raises(ValueError, match="not found"):
        prepare_clustering_data(dataframe, ["a", "missing"])
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_clustering_data(dataframe, ["a", "a"])
    with pytest.raises(ValueError, match="Boolean"):
        prepare_clustering_data(dataframe, ["a", "flag"])
    with pytest.raises(ValueError, match="Date"):
        prepare_clustering_data(dataframe, ["a", "when"])
    with pytest.raises(ValueError, match="at least two"):
        prepare_clustering_data(dataframe, ["a"])


def test_output_preserves_original_index_and_input_is_unchanged():
    dataframe = pd.DataFrame(
        {"amount": [1.0, None, 3.0], "kind": ["x", "y", None]},
        index=[10, 20, 30],
    )
    original = dataframe.copy(deep=True)

    result = prepare_clustering_data(dataframe, ["amount", "kind"])

    assert result.features.index.tolist() == [10, 20, 30]
    assert result.sample_indices.tolist() == [10, 20, 30]
    pd.testing.assert_frame_equal(dataframe, original)
