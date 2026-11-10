"""Exploratory data analysis page."""

import plotly.express as px
import streamlit as st

from user_analytics.analysis.eda import (
    describe_categorical,
    describe_numeric,
    get_categorical_columns,
    get_numeric_columns,
    prepare_category_frequencies,
    prepare_numeric_series,
)


def render(dataframe):
    """Render descriptive statistics and univariate analysis controls."""
    st.title("Exploratory Data Analysis")
    if dataframe is None:
        st.info("Upload a CSV file on the Home page to explore its data.")
        return
    if dataframe.empty or len(dataframe.columns) == 0:
        st.info("The uploaded dataset has no rows or columns to analyze.")
        return

    numeric_columns = get_numeric_columns(dataframe)
    empty_columns = [column for column in dataframe.columns if dataframe[column].isna().all()]
    numeric_summary_columns = list(dict.fromkeys(numeric_columns + empty_columns))
    categorical_columns = get_categorical_columns(dataframe)

    st.header("Descriptive Statistics")
    if numeric_summary_columns:
        selected_numeric = st.multiselect(
            "Numeric fields and empty fields", numeric_summary_columns,
            default=numeric_summary_columns, key="eda_numeric_fields"
        )
        numeric_summary = describe_numeric(dataframe, selected_numeric)
        if numeric_summary.empty:
            st.caption("Select at least one numeric field to display its statistics.")
        else:
            st.dataframe(numeric_summary, use_container_width=True)
    else:
        st.info("No numeric fields are available for descriptive statistics.")

    if categorical_columns:
        selected_categorical = st.multiselect(
            "Categorical fields", categorical_columns, default=categorical_columns,
            key="eda_categorical_fields",
        )
        categorical_summary = describe_categorical(dataframe, selected_categorical)
        if categorical_summary.empty:
            st.caption("Select at least one categorical field to display its statistics.")
        else:
            st.dataframe(categorical_summary, use_container_width=True)
    else:
        st.info("No categorical fields are available for category statistics.")

    st.header("Univariate Analysis")
    available_columns = numeric_columns + categorical_columns
    if not available_columns:
        st.info("No supported fields are available for univariate analysis.")
        return

    column = st.selectbox("Field", available_columns, key="eda_univariate_field")
    if column in numeric_columns:
        chart_type = st.selectbox("Chart type", ["Histogram", "Box Plot"], key="eda_numeric_chart")
        values = prepare_numeric_series(dataframe, column)
        if values.empty:
            st.info("This numeric field contains no valid values to chart.")
        elif chart_type == "Histogram":
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=30, key="eda_hist_bins")
            fig = px.histogram(values, x=column, nbins=bins, title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.box(values, y=column, title=f"Box plot of {column}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        top_n = st.number_input("Top N categories (0 = show all)", min_value=0, value=10, step=1)
        frequencies = prepare_category_frequencies(dataframe, column, top_n or None)
        if frequencies.empty:
            st.info("This categorical field contains no valid values to chart.")
        else:
            st.dataframe(frequencies, hide_index=True, use_container_width=True)
            fig = px.bar(
                frequencies, x="Category", y="Count", title=f"Category frequency for {column}",
                hover_data=["Percentage"],
            )
            st.plotly_chart(fig, use_container_width=True)
