"""CSV upload validation helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating an uploaded CSV's basic shape."""

    is_valid: bool
    message: str = ""


def validate_dataframe(dataframe):
    """Check that a parsed CSV contains at least one column and one row."""
    if dataframe.empty and len(dataframe.columns) == 0:
        return ValidationResult(False, "The uploaded CSV is empty or has no readable header.")
    if len(dataframe.columns) == 0:
        return ValidationResult(False, "The uploaded CSV does not contain any columns.")
    if len(dataframe.index) == 0:
        return ValidationResult(False, "The uploaded CSV contains column headers but no data rows.")
    return ValidationResult(True)
