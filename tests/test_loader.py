import pytest
import pandas as pd

from user_analytics.data.loader import (
    MAX_UPLOAD_SIZE_BYTES,
    CSVLoadError,
    load_csv,
)


class UploadedFile:
    def __init__(self, content, size=None):
        self._content = content
        if size is not None:
            self.size = size

    def getvalue(self):
        return self._content


def test_load_csv_with_header_keeps_column_names():
    uploaded = UploadedFile(b"name,age\nAda,36\n")

    dataframe = load_csv(uploaded, has_header=True)

    assert list(dataframe.columns) == ["name", "age"]
    assert dataframe.iloc[0].to_dict() == {"name": "Ada", "age": 36}
    assert dataframe.index.tolist() == [0]


def test_load_csv_without_header_keeps_first_row_and_generates_names():
    uploaded = UploadedFile(b"Ada,36\nGrace,40\n")

    dataframe = load_csv(uploaded, has_header=False)

    assert list(dataframe.columns) == ["Column_1", "Column_2"]
    assert dataframe.to_dict("records") == [
        {"Column_1": "Ada", "Column_2": 36},
        {"Column_1": "Grace", "Column_2": 40},
    ]


def test_load_csv_with_header_handles_utf8_bom():
    uploaded = UploadedFile("\ufeffname,age\nAda,36\n".encode("utf-8"))

    dataframe = load_csv(uploaded)

    assert list(dataframe.columns) == ["name", "age"]
    assert dataframe.iloc[0].to_dict() == {"name": "Ada", "age": 36}


def test_load_csv_rejects_empty_file():
    import pytest

    with pytest.raises(CSVLoadError, match="empty"):
        load_csv(UploadedFile(b""))


def test_load_csv_rejects_header_without_rows():
    import pytest

    with pytest.raises(CSVLoadError, match="no data rows"):
        load_csv(UploadedFile(b"name,age\n"), has_header=True)


def test_load_csv_rejects_malformed_csv():
    import pytest

    uploaded = UploadedFile(b'"unterminated,value\n1,2\n')

    with pytest.raises(CSVLoadError, match="parse"):
        load_csv(uploaded)


def test_load_csv_rejects_data_row_with_more_fields_than_header():
    import pytest

    uploaded = UploadedFile(b"a,b\n1,2,3\n")

    with pytest.raises(CSVLoadError, match="CSV structure is inconsistent"):
        load_csv(uploaded, has_header=True)


def test_load_csv_rejects_data_row_with_fewer_fields_than_header():
    import pytest

    uploaded = UploadedFile(b"a,b,c\n1,2\n")

    with pytest.raises(CSVLoadError, match="CSV structure is inconsistent"):
        load_csv(uploaded, has_header=True)


@pytest.mark.parametrize(
    "content",
    [
        b"1,2\n3,4,5\n",
        b"1,2\n3\n",
    ],
)
def test_load_csv_without_header_rejects_inconsistent_data_row_width(content):
    import pytest

    with pytest.raises(CSVLoadError, match="CSV structure is inconsistent"):
        load_csv(UploadedFile(content), has_header=False)


def test_load_csv_accepts_regular_rows_with_consistent_field_counts():
    uploaded = UploadedFile(b"name,age\nAda,36\nGrace,40\n")

    dataframe = load_csv(uploaded)

    assert dataframe.to_dict("records") == [
        {"name": "Ada", "age": 36},
        {"name": "Grace", "age": 40},
    ]
    assert dataframe.index.tolist() == [0, 1]


def test_load_csv_accepts_file_smaller_than_upload_limit():
    uploaded = UploadedFile(b"name\nAda\n")

    dataframe = load_csv(uploaded)

    assert dataframe["name"].tolist() == ["Ada"]


def test_file_at_upload_limit_is_allowed_to_reach_pandas_parser(monkeypatch):
    uploaded = UploadedFile(b"name\nAda\n", size=MAX_UPLOAD_SIZE_BYTES)
    called = []
    monkeypatch.setattr(
        "user_analytics.data.loader.pd.read_csv",
        lambda *args, **kwargs: (
            called.append((args, kwargs)) or pd.DataFrame({"name": ["Ada"]})
        ),
    )
    dataframe = load_csv(uploaded)

    assert called
    assert dataframe["name"].tolist() == ["Ada"]


def test_load_csv_rejects_oversized_file_before_reading_or_parsing(monkeypatch):
    class OversizedUpload:
        size = MAX_UPLOAD_SIZE_BYTES + 1

        def getvalue(self):
            pytest.fail("Oversized content should be rejected before getvalue().")

    monkeypatch.setattr(
        "user_analytics.data.loader.pd.read_csv",
        lambda *args, **kwargs: pytest.fail("Pandas parser must not run for oversized files."),
    )
    with pytest.raises(CSVLoadError, match="Maximum allowed size is 50 MB") as error:
        load_csv(OversizedUpload())

    assert f"{MAX_UPLOAD_SIZE_BYTES + 1:,} bytes" in str(error.value)
    assert "50 MB" in str(error.value)
