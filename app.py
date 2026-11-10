"""Application entry point and page navigation."""

import streamlit as st

from user_analytics.config import APP_ICON, APP_TITLE, PAGE_LAYOUT
from user_analytics.data.loader import CSVLoadError, load_csv
from user_analytics.ui.pages import customer_segmentation, data_quality, eda

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)

if "uploaded_dataframe" not in st.session_state:
    st.session_state.uploaded_dataframe = None

page = st.sidebar.radio("Page", ["Home", "Data Quality", "EDA", "Customer Segmentation"], key="dashboard_page")

if page == "Home":
    st.title("📊 User Analytics Dashboard")
    st.subheader("Make your data easier to understand.")
    st.write(
        "An interactive workspace for exploring CSV data, checking data quality, "
        "reviewing descriptive statistics, and discovering groups with K-means."
    )
    st.success("Welcome! Upload a CSV file to begin exploring your data.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_upload")
    has_header = st.checkbox("First row contains column names", value=True, key="csv_has_header")

    if uploaded_file is not None:
        try:
            dataframe = load_csv(uploaded_file, has_header=has_header)
        except CSVLoadError as error:
            st.session_state.uploaded_dataframe = None
            st.error(str(error))
        else:
            st.session_state.uploaded_dataframe = dataframe
            st.subheader("Data preview")
            st.dataframe(dataframe.head(10), use_container_width=True)

            st.subheader("Dataset information")
            row_col1, row_col2 = st.columns(2)
            row_col1.metric("Total rows", f"{len(dataframe):,}")
            row_col2.metric("Total columns", f"{len(dataframe.columns):,}")
            st.markdown("**Fields and data types**")
            st.dataframe(
                dataframe.dtypes.astype(str).rename("Data type").rename_axis("Field").reset_index(),
                hide_index=True,
                use_container_width=True,
            )
elif page == "Data Quality":
    data_quality.render(st.session_state.uploaded_dataframe)
elif page == "EDA":
    eda.render(st.session_state.uploaded_dataframe)
else:
    customer_segmentation.render(st.session_state.uploaded_dataframe)
