"""Customer segmentation controls and result presentation."""

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from user_analytics.analysis.clustering import (
    evaluate_kmeans,
    fit_kmeans,
    prepare_clustering_data,
    summarize_clusters,
)


def dataset_fingerprint(dataframe):
    """Build a stable session-scoped identifier for the current uploaded dataset."""
    digest = hashlib.sha256()
    digest.update(repr((tuple(dataframe.columns), tuple(map(str, dataframe.dtypes)))).encode())
    digest.update(pd.util.hash_pandas_object(dataframe, index=True).values.tobytes())
    return digest.hexdigest()


def available_feature_columns(dataframe):
    """Return supported numeric and categorical features, excluding bool/date."""
    numeric_columns = [
        column for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        and not pd.api.types.is_bool_dtype(dataframe[column].dtype)
    ]
    categorical_columns = [
        column for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        and not pd.api.types.is_bool_dtype(dataframe[column].dtype)
        and not pd.api.types.is_datetime64_any_dtype(dataframe[column].dtype)
    ]
    return numeric_columns, categorical_columns


def make_result_payload(source, selected_columns, n_clusters, random_state):
    """Run the existing preprocessing and clustering API and package display data."""
    prepared = prepare_clustering_data(source, selected_columns)
    fitted = fit_kmeans(prepared, n_clusters=n_clusters, random_state=random_state)
    summary = summarize_clusters(
        source,
        fitted.labels,
        numeric_columns=prepared.numeric_columns,
    )
    silhouette = None
    silhouette_note = None
    try:
        evaluation = evaluate_kmeans(prepared, [n_clusters], random_state=random_state)
        silhouette = float(evaluation.loc[0, "Silhouette Score"])
    except (ValueError, TypeError) as error:
        silhouette_note = str(error)
    return {
        "labels": fitted.labels,
        "centers": fitted.centers,
        "inertia": fitted.inertia,
        "summary": summary,
        "silhouette": silhouette,
        "silhouette_note": silhouette_note,
        "sample_count": len(prepared.features),
        "cluster_count": n_clusters,
        "feature_names": prepared.feature_names,
        "sample_indices": prepared.sample_indices,
    }


def render_empty_state():
    """Show a friendly prompt when no dataset is available."""
    st.info("Upload a CSV file on the Home page to configure customer segmentation.")


def render(dataframe):
    """Render feature selection, model configuration, and evaluation controls."""
    st.title("Customer Segmentation")
    if dataframe is None:
        render_empty_state()
        return
    if dataframe.empty or dataframe.shape[1] == 0:
        st.info("The uploaded dataset has no rows or columns to cluster.")
        return

    numeric_columns, categorical_columns = available_feature_columns(dataframe)
    supported_columns = numeric_columns + categorical_columns
    st.caption("Choose features intentionally. Boolean fields and date/time fields are excluded.")
    if not supported_columns:
        st.warning("No supported numeric or categorical fields are available for clustering.")
        return

    selected_columns = st.multiselect(
        "Features for clustering",
        options=supported_columns,
        default=[],
        format_func=lambda column: (
            f"{column} — Numeric" if column in numeric_columns else f"{column} — Categorical"
        ),
        key="customer_segmentation_features",
    )
    current_fingerprint = dataset_fingerprint(dataframe)
    result_state_key = "customer_segmentation_result"
    previous_fingerprint = st.session_state.get("customer_segmentation_dataset")
    if previous_fingerprint != current_fingerprint:
        st.session_state.pop(result_state_key, None)
        st.session_state.pop("customer_segmentation_evaluation", None)
        st.session_state.customer_segmentation_dataset = current_fingerprint

    with st.form("customer_segmentation_run_form"):
        k_value = st.number_input("Number of Clusters (K)", min_value=2, value=3, step=1)
        random_state = st.number_input("Random State", value=42, step=1)
        run_clustering = st.form_submit_button("Run Clustering", type="primary")

    if run_clustering:
        if len(selected_columns) < 2:
            st.session_state.pop(result_state_key, None)
            st.error("Select at least two supported fields before running clustering.")
        else:
            try:
                payload = make_result_payload(
                    dataframe, selected_columns, int(k_value), int(random_state)
                )
            except (ValueError, TypeError, KeyError) as error:
                st.session_state.pop(result_state_key, None)
                st.error(f"Could not run clustering: {error}")
            except Exception as error:
                st.session_state.pop(result_state_key, None)
                st.error(f"Clustering failed unexpectedly: {error}")
            else:
                payload.update(
                    {
                        "dataset_fingerprint": current_fingerprint,
                        "selected_columns": tuple(selected_columns),
                        "n_clusters": int(k_value),
                        "random_state": int(random_state),
                    }
                )
                st.session_state[result_state_key] = payload

    result = st.session_state.get(result_state_key)
    result_matches = bool(
        result
        and result["dataset_fingerprint"] == current_fingerprint
        and result["selected_columns"] == tuple(selected_columns)
        and result["n_clusters"] == int(k_value)
        and result["random_state"] == int(random_state)
    )
    if result and not result_matches:
        st.info("Clustering settings changed. Run clustering again to refresh these results.")

    st.header("Model Evaluation")
    if len(selected_columns) < 2:
        st.info("Select at least two fields to evaluate candidate K values.")
    else:
        max_k = max(2, min(len(dataframe) - 1, 10))
        if len(dataframe) < 3:
            st.info("At least 3 rows are needed for Silhouette Score evaluation.")
        else:
            with st.form("customer_segmentation_evaluation_form"):
                eval_min_k = st.number_input("Minimum K", min_value=2, max_value=max_k, value=2, step=1)
                eval_max_k = st.number_input(
                    "Maximum K", min_value=2, max_value=max_k,
                    value=min(max_k, max(3, int(k_value))), step=1,
                )
                run_evaluation = st.form_submit_button("Evaluate K range")
            if run_evaluation:
                if eval_min_k > eval_max_k:
                    st.session_state.pop("customer_segmentation_evaluation", None)
                    st.error("Minimum K must not exceed Maximum K.")
                else:
                    try:
                        prepared_for_eval = prepare_clustering_data(dataframe, selected_columns)
                        evaluation = evaluate_kmeans(
                            prepared_for_eval,
                            list(range(int(eval_min_k), int(eval_max_k) + 1)),
                            random_state=int(random_state),
                        )
                    except (ValueError, TypeError, KeyError) as error:
                        st.session_state.pop("customer_segmentation_evaluation", None)
                        st.error(f"K evaluation could not be calculated: {error}")
                    else:
                        st.session_state.customer_segmentation_evaluation = {
                            "dataset_fingerprint": current_fingerprint,
                            "selected_columns": tuple(selected_columns),
                            "random_state": int(random_state),
                            "min_k": int(eval_min_k),
                            "max_k": int(eval_max_k),
                            "data": evaluation,
                        }
            evaluation_state = st.session_state.get("customer_segmentation_evaluation")
            if evaluation_state and (
                evaluation_state["dataset_fingerprint"] != current_fingerprint
                or evaluation_state["selected_columns"] != tuple(selected_columns)
                or evaluation_state["random_state"] != int(random_state)
            ):
                st.session_state.pop("customer_segmentation_evaluation", None)
                evaluation_state = None
            if evaluation_state:
                evaluation_data = evaluation_state["data"]
                st.plotly_chart(
                    px.line(evaluation_data, x="K", y="Inertia", markers=True, title="Elbow Method — Inertia by K"),
                    use_container_width=True,
                )
                st.dataframe(evaluation_data, hide_index=True, use_container_width=True)
                st.caption(
                    "Use these metrics as guidance, not as an absolute choice of optimal K. "
                    "Consider the data context and the trade-off between compactness and separation."
                )

    st.header("Clustering Results")
    if not result_matches:
        st.info("Run clustering to view cluster assignments and profiles.")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Samples", f"{result['sample_count']:,}")
    metric_columns[1].metric("Clusters", f"{result['cluster_count']:,}")
    metric_columns[2].metric("Inertia", f"{result['inertia']:.4f}")
    if result["silhouette"] is not None:
        metric_columns[3].metric("Silhouette Score", f"{result['silhouette']:.4f}")
    else:
        metric_columns[3].metric("Silhouette Score", "Unavailable")
        if result["silhouette_note"]:
            st.caption(result["silhouette_note"])

    st.subheader("Cluster sizes and profiles")
    st.dataframe(result["summary"], use_container_width=True)
    st.subheader("Cluster centers (preprocessed feature space)")
    st.dataframe(result["centers"], use_container_width=True)
    st.caption("Cluster IDs are aligned to the original row index. Source records are not changed or removed.")
