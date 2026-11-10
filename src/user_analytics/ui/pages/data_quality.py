"""Data quality analysis page rendering."""

import plotly.express as px
import streamlit as st

from user_analytics.analysis.outliers import analyze_outliers
from user_analytics.analysis.profiling import (
    analyze_data_types,
    analyze_duplicates,
    analyze_missing_values,
)


def render(dataframe):
    """Render the data quality report for a DataFrame or an upload prompt."""
    st.title("Data Quality Analysis")
    if dataframe is None:
        st.info("Upload a CSV file on the Home page to view its data quality report.")
        return

    missing = analyze_missing_values(dataframe)
    duplicates = analyze_duplicates(dataframe)
    types = analyze_data_types(dataframe)
    outliers = analyze_outliers(dataframe)

    st.subheader("Missing values")
    metric1, metric2 = st.columns(2)
    metric1.metric("Missing cells", f"{missing['total_missing']:,}")
    metric2.metric("Dataset missing rate", f"{missing['missing_percentage']:.2f}%")
    st.dataframe(missing["per_column"], hide_index=True, use_container_width=True)
    if not missing["per_column"].empty:
        fig = px.bar(
            missing["per_column"], x="column", y="missing_percentage",
            labels={"column": "Field", "missing_percentage": "Missing (%)"},
            title="Missing values by field",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Duplicate rows")
    st.write(
        f"Found **{duplicates['duplicate_count']:,}** fully duplicated rows "
        f"({duplicates['duplicate_percentage']:.2f}% of {duplicates['row_count']:,} rows). "
        "Rows are only reported; no records are removed."
    )

    st.subheader("Field types")
    st.dataframe(types, hide_index=True, use_container_width=True)

    st.subheader("Potential outliers (IQR)")
    if outliers.empty:
        st.info("No numeric fields are available for IQR outlier analysis.")
    else:
        st.caption("These are statistical flags only; source rows are not changed or removed.")
        st.caption(
            "IQR bounds and outlier percentages use finite numeric values only. "
            "Missing Count and Non-finite Count report NaN and ±inf separately."
        )
        st.dataframe(outliers, hide_index=True, use_container_width=True)
        chart_data = outliers[outliers["status"] == "Analyzed"]
        if not chart_data.empty:
            fig = px.bar(
                chart_data, x="column", y="outlier_count",
                labels={"column": "Field", "outlier_count": "Potential outliers"},
                title="Potential outliers by numeric field",
            )
            st.plotly_chart(fig, use_container_width=True)
