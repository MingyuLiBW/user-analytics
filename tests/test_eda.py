import pandas as pd
import pytest

from user_analytics.analysis.eda import (
    describe_categorical,
    describe_numeric,
    prepare_category_frequencies,
    prepare_numeric_series,
)


def test_numeric_descriptive_statistics():
    dataframe = pd.DataFrame({"score": [1, 2, 3, 4]})

    result = describe_numeric(dataframe).loc["score"]

    assert result["Count"] == 4
    assert result["Mean"] == 2.5
    assert result["Minimum"] == 1
    assert result["25%"] == 1.75
    assert result["Median"] == 2.5
    assert result["75%"] == 3.25
    assert result["Maximum"] == 4


def test_categorical_summary_counts_non_missing_values():
    dataframe = pd.DataFrame({"group": ["A", "A", "B", None]})

    result = describe_categorical(dataframe).loc["group"]

    assert result["Unique Count"] == 2
    assert result["Most Frequent Value"] == "A"
    assert result["Frequency of Most Frequent Value"] == 2


def test_missing_values_are_excluded_from_summaries_and_chart_data():
    dataframe = pd.DataFrame({"number": [1, None, 3], "group": ["x", None, "y"]})

    assert describe_numeric(dataframe).loc["number", "Count"] == 2
    assert len(prepare_numeric_series(dataframe, "number")) == 2
    assert prepare_category_frequencies(dataframe, "group")["Count"].sum() == 2


def test_empty_columns_are_handled():
    dataframe = pd.DataFrame({"number": [None, None], "group": [None, None]})

    assert describe_numeric(dataframe).loc["number", "Count"] == 0
    assert pd.isna(describe_numeric(dataframe).loc["number", "Mean"])
    assert describe_categorical(dataframe).loc["group", "Unique Count"] == 0
    assert describe_categorical(dataframe).loc["group", "Most Frequent Value"] is None
    assert prepare_category_frequencies(dataframe, "group").empty


def test_category_top_n_combines_remaining_values_as_other():
    dataframe = pd.DataFrame({"group": ["A", "A", "B", "C", "D"]})

    result = prepare_category_frequencies(dataframe, "group", top_n=2).set_index("Category")

    assert result["Count"].to_dict() == {"A": 2, "B": 1, "Other": 2}
    assert result["Percentage"].sum() == pytest.approx(100)


def test_eda_helpers_do_not_modify_input_dataframe():
    dataframe = pd.DataFrame({"number": [1, None, 3], "group": ["A", "B", "A"]})
    original = dataframe.copy(deep=True)

    describe_numeric(dataframe)
    describe_categorical(dataframe)
    prepare_numeric_series(dataframe, "number")
    prepare_category_frequencies(dataframe, "group", top_n=1)

    pd.testing.assert_frame_equal(dataframe, original)
