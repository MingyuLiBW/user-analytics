from unittest.mock import patch

import pandas as pd

from user_analytics.ui.pages.customer_segmentation import (
    make_result_payload,
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
        "user_analytics.ui.pages.customer_segmentation.evaluate_kmeans",
        side_effect=ValueError("Silhouette is undefined for this result."),
    ):
        payload = make_result_payload(dataframe, ["a", "b"], n_clusters=2, random_state=1)

    assert payload["labels"].index.tolist() == dataframe.index.tolist()
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
