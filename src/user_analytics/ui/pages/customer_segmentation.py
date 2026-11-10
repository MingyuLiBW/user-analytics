"""Customer segmentation controls and result presentation."""

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from user_analytics.analysis.clustering import (
    evaluate_kmeans,
    fit_kmeans,
    prepare_clustering_data,
    prepare_pca_data,
    summarize_cluster_profiles,
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


def result_matches_current(result, fingerprint, selected_columns, n_clusters, random_state):
    """Check cached clustering metadata and required PCA inputs against current controls."""
    return bool(
        result
        and "features" in result
        and "labels" in result
        and result.get("dataset_fingerprint") == fingerprint
        and result.get("selected_columns") == tuple(selected_columns)
        and result.get("n_clusters") == int(n_clusters)
        and result.get("random_state") == int(random_state)
    )


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
        "features": prepared.features.copy(),
    }


def build_cluster_profile_tables(dataframe, labels, selected_columns):
    """Call the analysis-layer profiler for selected original-data fields."""
    numeric_columns = [
        column for column in selected_columns
        if column in dataframe.columns
        and pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        and not pd.api.types.is_bool_dtype(dataframe[column].dtype)
    ]
    categorical_columns = [
        column for column in selected_columns
        if column in dataframe.columns
        and not pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        and not pd.api.types.is_datetime64_any_dtype(dataframe[column].dtype)
    ]
    return summarize_cluster_profiles(
        dataframe,
        labels,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _format_numeric_profile(profile):
    """Format numeric profile values for display while preserving missing values."""
    formatted = profile.copy()
    numeric_fields = ["Mean", "Median", "Standard Deviation", "Minimum", "Maximum"]
    for column in numeric_fields:
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:,.3f}" if pd.notna(value) else "—"
            )
    return formatted


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
    result_matches = result_matches_current(
        result, current_fingerprint, selected_columns, k_value, random_state
    )
    if result and not result_matches:
        if "features" not in result or "labels" not in result:
            st.session_state.pop(result_state_key, None)
            result = None
            st.info("Saved clustering results are outdated. Run clustering again to generate PCA inputs.")
        else:
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
        st.subheader("PCA Visualization")
        st.info("PCA will be available after a successful clustering run with at least two transformed features.")
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

    st.subheader("Customer Segment Insights")
    try:
        profiles = build_cluster_profile_tables(
            dataframe, result["labels"], result["selected_columns"]
        )
    except (ValueError, TypeError, KeyError) as error:
        st.warning(f"Cluster profiles could not be calculated: {error}")
    else:
        numeric_tab, categorical_tab = st.tabs(["Numeric Profile", "Categorical Profile"])
        with numeric_tab:
            if profiles.numeric.empty:
                st.info("No numeric fields were selected for cluster profiling.")
            else:
                st.dataframe(
                    _format_numeric_profile(profiles.numeric),
                    hide_index=True,
                    use_container_width=True,
                )
                chart_data = profiles.numeric.dropna(subset=["Mean"])
                if not chart_data.empty:
                    fig = px.bar(
                        chart_data,
                        x="Field",
                        y="Mean",
                        color=chart_data["Cluster"].astype(str),
                        barmode="group",
                        title="Numeric field means by cluster",
                        labels={"Field": "Numeric field", "Mean": "Mean (original units)", "color": "Cluster"},
                        hover_data={
                            "Cluster": True,
                            "Field": True,
                            "Count": True,
                            "Mean": ":.3f",
                            "Median": ":.3f",
                            "Standard Deviation": ":.3f",
                            "Minimum": ":.3f",
                            "Maximum": ":.3f",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Selected numeric fields contain no non-missing values to compare.")
        with categorical_tab:
            if profiles.categorical.empty:
                st.info("No categorical fields were selected for cluster profiling.")
            else:
                categorical_display = profiles.categorical.copy()
                categorical_display["Proportion"] = categorical_display["Proportion"].map(
                    lambda value: f"{value:.1%}" if pd.notna(value) else "—"
                )
                st.dataframe(categorical_display, hide_index=True, use_container_width=True)
                chart_data = profiles.categorical.copy()
                if not chart_data.empty:
                    chart_data["Cluster"] = chart_data["Cluster"].astype(str)
                    chart_data["Proportion (%)"] = chart_data["Proportion"] * 100
                    fig = px.bar(
                        chart_data,
                        x="Cluster",
                        y="Proportion (%)",
                        color="Category",
                        facet_col="Field",
                        barmode="stack",
                        title="Category composition within each cluster",
                        labels={"Cluster": "Cluster", "Category": "Category", "Field": "Categorical field"},
                        hover_data={"Count": True, "Proportion (%)": ":.1f"},
                    )
                    fig.update_yaxes(range=[0, 100], title="Within-cluster proportion (%)")
                    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Visualization")
    pca_state_key = "customer_segmentation_pca"
    pca_signature = (
        current_fingerprint,
        tuple(selected_columns),
        int(k_value),
        int(random_state),
    )
    pca_result = st.session_state.get(pca_state_key)
    if not pca_result or pca_result["signature"] != pca_signature:
        try:
            pca_result = {
                "signature": pca_signature,
                "data": prepare_pca_data(result["features"], result["labels"]),
            }
            st.session_state[pca_state_key] = pca_result
        except (ValueError, TypeError) as error:
            st.session_state.pop(pca_state_key, None)
            st.info(f"PCA visualization is unavailable: {error}")
        except Exception as error:
            st.session_state.pop(pca_state_key, None)
            st.info(f"PCA visualization could not be generated: {error}")
    if pca_result and pca_result["signature"] == pca_signature:
        pca = pca_result["data"]
        coordinates = pca.coordinates
        variance_1 = pca.pc1_explained_variance_ratio
        variance_2 = pca.pc2_explained_variance_ratio
        fig = px.scatter(
            coordinates,
            x="PC1",
            y="PC2",
            color=coordinates["Cluster"].astype(str),
            hover_data={
                "Sample Index": True,
                "Cluster": True,
                "PC1": ":.4f",
                "PC2": ":.4f",
            },
            title="Customer Segments — PCA Projection",
            labels={
                "PC1": f"PC1 ({variance_1:.1%} explained variance)",
                "PC2": f"PC2 ({variance_2:.1%} explained variance)",
                "color": "Cluster",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        variance_columns = st.columns(3)
        variance_columns[0].metric("PC1 explained variance", f"{variance_1:.2%}")
        variance_columns[1].metric("PC2 explained variance", f"{variance_2:.2%}")
        variance_columns[2].metric(
            "Cumulative explained variance",
            f"{pca.cumulative_explained_variance_ratio:.2%}",
        )
        st.caption("PCA is used only to visualize the fitted clusters; it does not change K-means assignments.")
