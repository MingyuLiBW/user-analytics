"""Application entry point and page navigation."""

import streamlit as st

from user_analytics.config import APP_ICON, APP_TITLE, PAGE_LAYOUT
from user_analytics.data.loader import CSVLoadError
from user_analytics.ui.pages import customer_segmentation, data_quality, eda
from user_analytics.ui.upload_state import clear_uploaded_data, sync_uploaded_data

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)

if "uploaded_dataframe" not in st.session_state:
    st.session_state.uploaded_dataframe = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


def _on_csv_upload_change():
    """Invalidate old analysis data only on an explicit uploader interaction."""
    clear_uploaded_data(st.session_state)

page = st.sidebar.radio("Page", ["Home", "Data Quality", "EDA", "Customer Segmentation"], key="dashboard_page")

if page == "Home":
    st.title("📊 User Analytics Dashboard")
    st.subheader("Make your data easier to understand.")
    st.write(
        "An interactive workspace for exploring CSV data, checking data quality, "
        "reviewing descriptive statistics, and discovering groups with K-means."
    )
    st.success("Welcome! Upload a CSV file to begin exploring your data.")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="csv_upload",
        on_change=_on_csv_upload_change,
    )
    has_header = st.checkbox("First row contains column names", value=True, key="csv_has_header")

    if uploaded_file is not None:
        try:
            dataframe = sync_uploaded_data(st.session_state, uploaded_file, has_header=has_header)
        except CSVLoadError as error:
            st.error(str(error))
    else:
        # The file uploader may be absent/blank after a page transition. Keep
        # displaying and sharing the active session dataset until explicitly cleared.
        dataframe = sync_uploaded_data(st.session_state, uploaded_file, has_header=has_header)
        if dataframe is not None:
            filename = st.session_state.get("uploaded_filename")
            description = f" ({filename})" if filename else ""
            st.info(f"Active dataset retained for this session{description}.")

    dataframe = st.session_state.uploaded_dataframe
    if dataframe is not None:
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
