import pandas as pd
import pytest

from user_analytics.data.loader import MAX_UPLOAD_SIZE_BYTES, CSVLoadError
from user_analytics.ui.upload_state import clear_uploaded_data, sync_uploaded_data


class UploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def test_upload_saves_dataframe_in_session_state():
    state = {"uploaded_dataframe": None, "uploaded_filename": None}

    dataframe = sync_uploaded_data(
        state, UploadedFile("a.csv", b"name,value\nA,1\n")
    )

    assert dataframe.loc[0, "name"] == "A"
    pd.testing.assert_frame_equal(state["uploaded_dataframe"], dataframe)
    assert state["uploaded_filename"] == "a.csv"


def test_explicit_uploader_clear_clears_dataframe_and_metadata():
    state = {
        "uploaded_dataframe": pd.DataFrame({"old": [1]}),
        "uploaded_filename": "a.csv",
        "customer_segmentation_result": {"old": True},
        "customer_segmentation_evaluation": {"old": True},
        "customer_segmentation_pca": {"old": True},
        "customer_segmentation_statistics": {"old": True},
        "customer_segmentation_dataset": "old",
    }

    clear_uploaded_data(state)

    assert state["uploaded_dataframe"] is None
    assert state["uploaded_filename"] is None
    assert not any(key.startswith("customer_segmentation_") for key in state)


def test_uploading_new_file_replaces_old_dataframe():
    state = {
        "uploaded_dataframe": None,
        "uploaded_filename": None,
        "customer_segmentation_result": {"old": True},
    }
    sync_uploaded_data(state, UploadedFile("a.csv", b"name\nA\n"))
    state["customer_segmentation_result"] = {"old": True}

    dataframe = sync_uploaded_data(state, UploadedFile("b.csv", b"name\nB\n"))

    assert dataframe["name"].tolist() == ["B"]
    assert state["uploaded_filename"] == "b.csv"
    assert "customer_segmentation_result" not in state


def test_missing_widget_value_retains_dataset_across_page_transition():
    state = {"uploaded_dataframe": None, "uploaded_filename": None}
    original = sync_uploaded_data(state, UploadedFile("a.csv", b"name\nA\n"))

    returned = sync_uploaded_data(state, uploaded_file=None)

    assert returned is original
    assert state["uploaded_dataframe"] is original


def test_invalid_replacement_does_not_leave_previous_dataframe_active():
    state = {"uploaded_dataframe": None, "uploaded_filename": None}
    sync_uploaded_data(state, UploadedFile("a.csv", b"name\nA\n"))

    with pytest.raises(CSVLoadError):
        sync_uploaded_data(state, UploadedFile("b.csv", b"a,b\n1,2,3\n"))

    assert state["uploaded_dataframe"] is None
    assert state["uploaded_filename"] is None


def test_oversized_replacement_does_not_leave_previous_dataframe_active():
    class OversizedUpload:
        name = "too-large.csv"
        size = MAX_UPLOAD_SIZE_BYTES + 10

        def getvalue(self):
            pytest.fail("Oversized upload should be rejected before reading its contents.")

    state = {"uploaded_dataframe": None, "uploaded_filename": None}
    sync_uploaded_data(state, UploadedFile("a.csv", b"name\nA\n"))

    with pytest.raises(CSVLoadError, match="Maximum allowed size is 50 MB"):
        sync_uploaded_data(state, OversizedUpload())

    assert state["uploaded_dataframe"] is None
    assert state["uploaded_filename"] is None


def test_consecutive_uploads_leave_only_the_latest_data_in_session_state():
    state = {"uploaded_dataframe": None, "uploaded_filename": None}
    for name, value in (("a.csv", "A"), ("b.csv", "B"), ("c.csv", "C")):
        sync_uploaded_data(state, UploadedFile(name, f"name\n{value}\n".encode()))

    assert state["uploaded_dataframe"]["name"].tolist() == ["C"]
    assert state["uploaded_filename"] == "c.csv"
