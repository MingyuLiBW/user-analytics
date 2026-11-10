"""Exploratory data analysis page."""

import plotly.express as px
import streamlit as st

from user_analytics.analysis.eda import (
    describe_categorical,
    describe_numeric,
    calculate_correlation,
    get_categorical_columns,
    get_numeric_columns,
    prepare_category_frequencies,
    prepare_numeric_series,
    prepare_bivariate_data,
    prepare_categorical_crosstab,
    prepare_categorical_numeric_data,
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


    st.header("Bivariate Analysis")
    bivariate_columns = numeric_columns + categorical_columns
    if len(bivariate_columns) < 2:
        st.info("At least two supported fields are required for bivariate analysis.")
    else:
        x_options = bivariate_columns
        x_column = st.selectbox("X-axis field", x_options, key="eda_bivariate_x")
        y_options = [column for column in bivariate_columns if column != x_column]
        y_column = st.selectbox("Y-axis field", y_options, key="eda_bivariate_y")
        x_is_numeric = x_column in numeric_columns
        y_is_numeric = y_column in numeric_columns

        if x_is_numeric and y_is_numeric:
            pair_data = prepare_bivariate_data(dataframe, x_column, y_column)
            if pair_data.empty:
                st.info("No pairwise valid observations are available for these fields.")
            elif pair_data[x_column].nunique() < 2 or pair_data[y_column].nunique() < 2:
                st.info("Both numeric fields need at least two distinct valid values for a scatter plot.")
            else:
                add_trendline = st.checkbox("Show linear trendline", value=False, key="eda_bivariate_trendline")
                color_options = [None] + [
                    column for column in categorical_columns if column not in {x_column, y_column}
                ]
                color_column = st.selectbox(
                    "Color by category (optional)", color_options, key="eda_bivariate_color"
                )
                color_missing_count = 0
                if color_column:
                    pair_data[color_column] = dataframe.loc[pair_data.index, color_column]
                    color_missing_count = int(pair_data[color_column].isna().sum())
                    pair_data = pair_data.dropna(subset=[color_column])
                if color_missing_count:
                    st.caption(
                        f"{color_missing_count} rows missing the selected color field are omitted from this chart."
                    )
                if pair_data.empty:
                    st.info("No complete observations remain after applying the selected color field.")
                else:
                    fig = px.scatter(
                        pair_data, x=x_column, y=y_column, color=color_column,
                        trendline="ols" if add_trendline else None,
                        title=f"{x_column} vs. {y_column}",
                        labels={x_column: x_column, y_column: y_column},
                        hover_data=list(pair_data.columns),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        elif x_is_numeric != y_is_numeric:
            category_column = y_column if x_is_numeric else x_column
            numeric_column = x_column if x_is_numeric else y_column
            chart_type = st.selectbox("Chart type", ["Box Plot", "Violin Plot"], key="eda_cat_num_chart")
            grouped_data = prepare_categorical_numeric_data(dataframe, category_column, numeric_column)
            if grouped_data.empty:
                st.info("No pairwise valid observations are available for these fields.")
            elif grouped_data[category_column].nunique() < 2:
                st.info("At least two categories with valid numeric values are needed for comparison.")
            else:
                chart_function = px.box if chart_type == "Box Plot" else px.violin
                fig = chart_function(
                    grouped_data, x=category_column, y=numeric_column,
                    title=f"{numeric_column} by {category_column}",
                    labels={category_column: category_column, numeric_column: numeric_column},
                    points="outliers" if chart_type == "Box Plot" else "all",
                    hover_data=[category_column, numeric_column],
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            measure = st.selectbox("Display metric", ["Count", "Row proportion"], key="eda_cat_cat_metric")
            crosstab = prepare_categorical_crosstab(
                dataframe, x_column, y_column, normalize=measure == "Row proportion"
            )
            if crosstab.empty:
                st.info("No pairwise valid observations are available for these fields.")
            else:
                long_data = crosstab.rename_axis(index=x_column, columns=y_column).stack().rename("Value").reset_index()
                if measure == "Row proportion":
                    long_data["Value"] *= 100
                fig = px.bar(
                    long_data, x=x_column, y="Value", color=y_column, barmode="group",
                    title=f"{y_column} by {x_column} ({measure.lower()})",
                    labels={"Value": "Percentage (%)" if measure == "Row proportion" else "Count"},
                    hover_data=[x_column, y_column, "Value"],
                )
                st.plotly_chart(fig, use_container_width=True)

    st.header("Correlation Analysis")
    if len(numeric_columns) < 2:
        st.info("Correlation analysis requires at least two numeric fields.")
    else:
        selected_correlation_columns = st.multiselect(
            "Numeric fields for correlation", numeric_columns,
            default=numeric_columns, key="eda_correlation_fields",
        )
        method_label = st.selectbox(
            "Correlation method", ["Pearson", "Spearman"], key="eda_correlation_method"
        )
        st.caption(
            "Pandas computes each pair using pairwise complete observations; rows with a missing "
            "value in either field are excluded only for that pair."
        )
        if len(selected_correlation_columns) < 2:
            st.info("Select at least two numeric fields to display a correlation matrix.")
        else:
            correlation = calculate_correlation(
                dataframe, selected_correlation_columns, method=method_label.lower()
            )
            st.dataframe(correlation.style.format("{:.2f}", na_rep="NA"), use_container_width=True)
            heatmap_values = correlation.to_numpy(dtype=float)
            fig = px.imshow(
                heatmap_values,
                x=correlation.columns,
                y=correlation.index,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                color_continuous_midpoint=0,
                text_auto=".2f",
                aspect="auto",
                title=f"{method_label} correlation matrix",
                labels={"x": "Field", "y": "Field", "color": "Correlation"},
            )
            fig.update_traces(
                hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
                connectgaps=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "- **Positive correlation:** variables tend to change in the same direction.\n"
            "- **Negative correlation:** variables tend to change in opposite directions.\n"
            "- **Near zero:** linear or monotonic association is weak, depending on the selected method.\n"
            "- Correlation does not establish causation."
        )
