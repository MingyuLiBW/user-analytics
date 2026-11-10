"""CSV loading helpers for uploaded files."""

from io import BytesIO

import pandas as pd

from user_analytics.data.validation import validate_dataframe


class CSVLoadError(ValueError):
    """Raised when an uploaded file is not a readable, non-empty CSV."""


def load_csv(uploaded_file, has_header=True):
    """Read an uploaded CSV as a DataFrame, optionally treating its first row as data."""
    if uploaded_file is None:
        raise CSVLoadError("Choose a CSV file to upload.")

    try:
        raw = uploaded_file.getvalue()
    except AttributeError as exc:
        raise CSVLoadError("The uploaded file could not be read.") from exc

    if not raw:
        raise CSVLoadError("The uploaded file is empty.")

    try:
        dataframe = pd.read_csv(
            BytesIO(raw),
            encoding="utf-8-sig",
            header=0 if has_header else None,
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
