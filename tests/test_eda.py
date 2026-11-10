from io import StringIO

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


def test_top_n_preserves_real_other_category_when_it_is_in_top_n():
    dataframe = pd.DataFrame(
        {"group": ["Other"] * 10 + ["A"] * 9 + ["B"] * 8 + ["C"] * 7 + [None]}
    )

    result = prepare_category_frequencies(dataframe, "group", top_n=1).set_index("Category")

    assert result["Count"].to_dict() == {"Other": 10, "Other (remaining)": 24}
    assert result.loc["Other", "Percentage"] == pytest.approx(10 / 34 * 100)
    assert result.loc["Other (remaining)", "Percentage"] == pytest.approx(24 / 34 * 100)
    assert result["Count"].sum() == 34
    assert result["Percentage"].sum() == pytest.approx(100)


def test_top_n_uses_distinct_tail_label_when_real_other_is_not_in_top_n():
    dataframe = pd.DataFrame(
        {"group": ["A"] * 10 + ["B"] * 9 + ["Other"] * 8 + ["C"] * 7}
    )

    result = prepare_category_frequencies(dataframe, "group", top_n=2).set_index("Category")

    assert result["Count"].to_dict() == {"A": 10, "B": 9, "Other (remaining)": 15}
    assert result["Count"].sum() == 34
    assert result["Percentage"].tolist() == pytest.approx([10 / 34 * 100, 9 / 34 * 100, 15 / 34 * 100])
    assert result["Percentage"].sum() == pytest.approx(100)


def test_top_n_tail_label_avoids_other_real_category_labels():
    dataframe = pd.DataFrame(
        {"group": ["Other"] * 5 + ["A"] * 4 + ["Other (remaining)"] * 3 + ["B"] * 2}
    )

    result = prepare_category_frequencies(dataframe, "group", top_n=1).set_index("Category")

    assert result["Count"].to_dict() == {"Other": 5, "Other (remaining) 2": 9}
    assert result["Count"].sum() == 14
    assert result["Percentage"].sum() == pytest.approx(100)


def test_eda_helpers_do_not_modify_input_dataframe():
    dataframe = pd.DataFrame({"number": [1, None, 3], "group": ["A", "B", "A"]})
    original = dataframe.copy(deep=True)

    describe_numeric(dataframe)
    describe_categorical(dataframe)
    prepare_numeric_series(dataframe, "number")
    prepare_category_frequencies(dataframe, "group", top_n=1)

    pd.testing.assert_frame_equal(dataframe, original)


def test_numeric_description_and_univariate_data_use_finite_values():
    dataframe = pd.DataFrame({"number": [1.0, 2.0, 3.0, None, float("inf"), -float("inf")]})
    original = dataframe.copy(deep=True)

    summary = describe_numeric(dataframe).loc["number"]
    chart_values = prepare_numeric_series(dataframe, "number")

    assert summary["Count"] == 3
    assert summary["Missing Count"] == 1
    assert summary["Non-finite Count"] == 2
    assert summary["Mean"] == pytest.approx(2)
    assert summary["Standard Deviation"] == pytest.approx(1)
    assert summary["Minimum"] == 1
    assert summary["Median"] == 2
    assert summary["Maximum"] == 3
    assert chart_values.tolist() == [1.0, 2.0, 3.0]
    assert summary["Count"] + summary["Missing Count"] + summary["Non-finite Count"] == len(dataframe)
    pd.testing.assert_frame_equal(dataframe, original)


def test_numeric_description_with_only_non_finite_values_returns_empty_statistics():
    dataframe = pd.DataFrame({"number": [None, float("inf"), -float("inf")]})

    result = describe_numeric(dataframe).loc["number"]

    assert result["Count"] == 0
    assert result["Missing Count"] == 1
    assert result["Non-finite Count"] == 2
    assert pd.isna(result["Mean"])
    assert pd.isna(result["Maximum"])


def test_csv_parsed_nonfinite_fixture_counts_33_finite_values_per_field():
    rows = [str(value) for value in range(33)] + ["NaN", "+Inf", "-Inf"]
    csv_text = "age,income,spending_score\n" + "\n".join(
        f"{value},{value},{value}" for value in rows
    )
    dataframe = pd.read_csv(StringIO(csv_text))
    original = dataframe.copy(deep=True)

    result = describe_numeric(dataframe)

    assert result.index.tolist() == ["age", "income", "spending_score"]
    assert result["Count"].tolist() == [33, 33, 33]
    assert result["Missing Count"].tolist() == [1, 1, 1]
    assert result["Non-finite Count"].tolist() == [2, 2, 2]
    assert result["Mean"].tolist() == pytest.approx([16, 16, 16])
    assert result["Standard Deviation"].tolist() == pytest.approx([9.6695398] * 3)
    assert result["Minimum"].tolist() == [0, 0, 0]
    assert result["25%"].tolist() == [8, 8, 8]
    assert result["Median"].tolist() == [16, 16, 16]
    assert result["75%"].tolist() == [24, 24, 24]
    assert result["Maximum"].tolist() == [32, 32, 32]
    assert result[["Count", "Missing Count", "Non-finite Count"]].sum(axis=1).tolist() == [36] * 3
    pd.testing.assert_frame_equal(dataframe, original)


def test_text_inf_remains_a_regular_category_value():
    dataframe = pd.DataFrame({"group": ["inf", "-inf", "inf"]})

    result = prepare_category_frequencies(dataframe, "group").set_index("Category")

    assert result["Count"].to_dict() == {"inf": 2, "-inf": 1}
