"""Session-state helpers for the Home page CSV upload lifecycle."""

from user_analytics.data.loader import load_csv


def clear_uploaded_data(session_state):
    """Clear the active dataset and derived segmentation results after a file change."""
    session_state["uploaded_dataframe"] = None
    session_state["uploaded_filename"] = None
    for key in (
        "customer_segmentation_result",
        "customer_segmentation_evaluation",
        "customer_segmentation_pca",
        "customer_segmentation_statistics",
        "customer_segmentation_dataset",
    ):
        session_state.pop(key, None)


def sync_uploaded_data(session_state, uploaded_file, has_header=True):
    """Load a newly selected file, or retain the active dataset when no event exists.

    The uploader's change callback explicitly clears state when the user removes
    its file. A ``None`` value here therefore means no current upload event (for
    example, a page transition unmounted the widget), not an implicit clear.
    """
    if uploaded_file is None:
        return session_state.get("uploaded_dataframe")

    # Invalidate the previous dataset before attempting to parse a replacement.
    clear_uploaded_data(session_state)
    dataframe = load_csv(uploaded_file, has_header=has_header)
    session_state["uploaded_dataframe"] = dataframe
    session_state["uploaded_filename"] = getattr(uploaded_file, "name", None)
    return dataframe
