"""CSV loading helpers for uploaded files."""

import csv
from io import BytesIO, StringIO

import pandas as pd

from user_analytics.data.validation import validate_dataframe


MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


class CSVLoadError(ValueError):
    """Raised when an uploaded file is not a readable, non-empty CSV."""


def _check_file_size(size_bytes):
    """Raise a user-facing error when a file exceeds the configured upload limit."""
    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        size_mib = size_bytes / (1024 * 1024)
        raise CSVLoadError(
            f"CSV file is too large: {size_bytes:,} bytes ({size_mib:.2f} MiB). "
            f"Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB "
            f"({MAX_UPLOAD_SIZE_BYTES:,} bytes)."
        )


def _validate_csv_field_counts(raw, has_header):
    """Reject nonblank CSV records whose number of fields differs from the schema.

    With a header, the header row defines the width. Without a header, the first
    data record defines it. The standard csv reader respects quoted commas and
    multiline fields, unlike counting comma characters in raw lines.
    """
    text = raw.decode("utf-8-sig")
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    expected_fields = None

    for record in reader:
        # Match pandas' default skip_blank_lines behavior for physically blank lines.
        if not record:
            continue
        if expected_fields is None:
            expected_fields = len(record)
            if has_header:
                continue
        actual_fields = len(record)
        if actual_fields != expected_fields:
            raise CSVLoadError(
                "CSV structure is inconsistent: "
                f"data record ending at line {reader.line_num} has {actual_fields} fields; "
                f"expected {expected_fields}."
            )


def load_csv(uploaded_file, has_header=True):
    """Read an uploaded CSV as a DataFrame, optionally treating its first row as data."""
    if uploaded_file is None:
        raise CSVLoadError("Choose a CSV file to upload.")

    # Streamlit's UploadedFile exposes its byte size, allowing oversized files
    # to be rejected before retrieving their contents into application code.
    declared_size = getattr(uploaded_file, "size", None)
    if declared_size is not None:
        _check_file_size(declared_size)

    try:
        raw = uploaded_file.getvalue()
    except AttributeError as exc:
        raise CSVLoadError("The uploaded file could not be read.") from exc

    if not raw:
        raise CSVLoadError("The uploaded file is empty.")
    _check_file_size(len(raw))

    try:
        _validate_csv_field_counts(raw, has_header=has_header)
    except UnicodeDecodeError as exc:
        raise CSVLoadError(f"Could not parse this file as a UTF-8 CSV: {exc}") from exc
    except csv.Error as exc:
        raise CSVLoadError(f"Could not parse this file as a UTF-8 CSV: malformed CSV structure: {exc}") from exc

    try:
        dataframe = pd.read_csv(
            BytesIO(raw),
            encoding="utf-8-sig",
            header=0 if has_header else None,
            index_col=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise CSVLoadError("The CSV is empty or does not contain a readable row.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise CSVLoadError(f"Could not parse this file as a UTF-8 CSV: {exc}") from exc
    except Exception as exc:
        raise CSVLoadError(f"The uploaded file could not be read: {exc}") from exc

    if not has_header:
        dataframe.columns = [f"Column_{index}" for index in range(1, len(dataframe.columns) + 1)]

    result = validate_dataframe(dataframe)
    if not result.is_valid:
        raise CSVLoadError(result.message)
    return dataframe
