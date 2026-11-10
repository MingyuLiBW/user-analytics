from user_analytics.data.loader import CSVLoadError, load_csv


class UploadedFile:
    def __init__(self, content):
        self._content = content

    def getvalue(self):
        return self._content


def test_load_csv_with_header_keeps_column_names():
    uploaded = UploadedFile(b"name,age\nAda,36\n")

    dataframe = load_csv(uploaded, has_header=True)

    assert list(dataframe.columns) == ["name", "age"]
    assert dataframe.iloc[0].to_dict() == {"name": "Ada", "age": 36}


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
