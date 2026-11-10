"""Preprocessing utilities for customer segmentation."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ClusteringPreprocessingResult:
    """Prepared feature matrix and fitted objects needed for later transforms."""

    features: pd.DataFrame
    preprocessor: ColumnTransformer
    feature_names: list[str]
    sample_indices: pd.Index
    numeric_columns: list[Any]
    categorical_columns: list[Any]



def _impute_categorical_values(values, columns, fill_values):
    """Fill each categorical column with its fitted mode or explicit empty marker."""
    result = pd.DataFrame(values, columns=columns)
    for column in columns:
        result[column] = result[column].fillna(fill_values[column])
    return result

def prepare_clustering_data(dataframe, selected_columns):
    """Validate and preprocess selected numeric/categorical clustering features.

    Missing values are imputed, categorical data is one-hot encoded, and only
    numeric columns are standardized. The returned feature matrix keeps the
    original row index and leaves the input DataFrame untouched.
    """
    selected_columns = list(selected_columns)
    if len(selected_columns) != len(set(selected_columns)):
        raise ValueError("Duplicate feature selections are not allowed.")
    if len(selected_columns) < 2:
        raise ValueError("Select at least two features for clustering.")

    missing_columns = [column for column in selected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Selected field(s) not found: {missing_columns}")

    boolean_columns = [
        column for column in selected_columns
        if pd.api.types.is_bool_dtype(dataframe[column].dtype)
    ]
    if boolean_columns:
        raise ValueError(f"Boolean fields are not supported: {boolean_columns}")

    datetime_columns = [
        column for column in selected_columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column].dtype)
    ]
    if datetime_columns:
        raise ValueError(f"Date fields cannot be used directly for clustering: {datetime_columns}")

    numeric_columns = [
        column for column in selected_columns
        if pd.api.types.is_numeric_dtype(dataframe[column].dtype)
    ]
    categorical_columns = [column for column in selected_columns if column not in numeric_columns]
    if len(numeric_columns) + len(categorical_columns) != len(selected_columns):
        raise ValueError("One or more selected fields have unsupported data types.")
    if dataframe.empty:
        raise ValueError("There are no rows available for clustering.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_imputer = FunctionTransformer(
        _impute_categorical_values,
        validate=False,
        kw_args={
            "columns": categorical_columns,
            "fill_values": {
                column: (
                    "**MISSING**"
                    if dataframe[column].isna().all()
                    else dataframe[column].mode(dropna=True).iloc[0]
                )
                for column in categorical_columns
            },
        },
        feature_names_out="one-to-one",
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", categorical_imputer),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    selected_data = dataframe.loc[:, selected_columns].copy()
    try:
        transformed = preprocessor.fit_transform(selected_data)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Could not preprocess the selected features: {exc}") from exc

    feature_names = preprocessor.get_feature_names_out().tolist()
    features = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=dataframe.index.copy(),
    )
    return ClusteringPreprocessingResult(
        features=features,
        preprocessor=preprocessor,
        feature_names=feature_names,
        sample_indices=dataframe.index.copy(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


@dataclass(frozen=True)
class KMeansResult:
    """Fitted K-means outputs aligned to source sample indices."""

    model: KMeans
    labels: pd.Series
    centers: pd.DataFrame
    inertia: float


def _feature_frame(features):
    """Extract a feature DataFrame and source index from supported inputs."""
    if isinstance(features, ClusteringPreprocessingResult):
        return features.features, features.sample_indices
    if isinstance(features, pd.DataFrame):
        return features, features.index
    raise TypeError("features must be a DataFrame or ClusteringPreprocessingResult.")


def _validate_kmeans_input(features, n_clusters):
    """Validate matrix shape, numeric contents, and requested cluster count."""
    matrix, sample_indices = _feature_frame(features)
    if matrix.empty or matrix.shape[1] == 0:
        raise ValueError("K-means requires at least one sample and one feature.")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise ValueError("K-means features must all be numeric.")
    values = matrix.to_numpy(dtype=float)
    import numpy as np
    if not np.isfinite(values).all():
        raise ValueError("K-means features must not contain missing or infinite values.")
    sample_count = len(matrix)
    unique_count = len(matrix.drop_duplicates())
    if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
        raise ValueError("n_clusters must be an integer.")
    if n_clusters < 2 or n_clusters >= sample_count:
        raise ValueError(f"n_clusters must be between 2 and {sample_count - 1} for {sample_count} samples.")
    if unique_count < n_clusters:
        raise ValueError(
            f"Cannot create {n_clusters} clusters: only {unique_count} unique data points are available."
        )
    return matrix, sample_indices, values


def fit_kmeans(features, n_clusters, random_state=42):
    """Fit deterministic K-means and return labels, centers, and inertia."""
    matrix, sample_indices, values = _validate_kmeans_input(features, n_clusters)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(values)
    center_frame = pd.DataFrame(
        model.cluster_centers_, columns=matrix.columns, index=range(n_clusters)
    )
    center_frame.index.name = "Cluster"
    label_series = pd.Series(labels, index=sample_indices.copy(), name="Cluster")
    return KMeansResult(
        model=model,
        labels=label_series,
        centers=center_frame,
        inertia=float(model.inertia_),
    )


def summarize_clusters(original_dataframe, labels, numeric_columns=None):
    """Summarize cluster sizes, proportions, and means of source numeric fields."""
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels, index=original_dataframe.index, name="Cluster")
    if not labels.index.is_unique or not original_dataframe.index.is_unique:
        raise ValueError("Cluster labels and source rows must have unique indices for alignment.")
    common_index = original_dataframe.index.intersection(labels.index, sort=False)
    if common_index.empty:
        raise ValueError("No cluster labels match the original data index.")
    aligned_labels = labels.loc[common_index]
    numeric_candidates = [
        column for column in original_dataframe.columns
        if pd.api.types.is_numeric_dtype(original_dataframe[column].dtype)
        and not pd.api.types.is_bool_dtype(original_dataframe[column].dtype)
    ]
    selected_numeric = numeric_candidates if numeric_columns is None else list(numeric_columns)
    invalid_numeric = [column for column in selected_numeric if column not in numeric_candidates]
    if invalid_numeric:
        raise ValueError(f"Selected profile field(s) are not numeric: {invalid_numeric}")

    summary = pd.DataFrame({"Cluster": aligned_labels.to_numpy()}, index=common_index)
    summary_counts = summary["Cluster"].value_counts().sort_index()
    proportions = summary_counts / len(summary)
    result = pd.DataFrame(
        {
            "Sample Count": summary_counts,
            "Sample Proportion": proportions,
        }
    )
    if selected_numeric:
        source_values = original_dataframe.loc[common_index, selected_numeric].copy()
        means = source_values.groupby(aligned_labels, sort=True).mean()
        means.index.name = "Cluster"
        result = result.join(means)
    result.index.name = "Cluster"
    return result


def evaluate_kmeans(features, k_values, random_state=42):
    """Calculate inertia and silhouette for valid K values, failing on invalid K."""
    matrix, _ = _feature_frame(features)
    if matrix.empty or matrix.shape[1] == 0:
        raise ValueError("K evaluation requires at least one sample and one feature.")
    import numpy as np

    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("K evaluation features must not contain missing or infinite values.")
    sample_count = len(matrix)
    unique_count = len(matrix.drop_duplicates())
    if sample_count < 3:
        raise ValueError("K evaluation needs at least 3 samples to calculate silhouette scores.")
    rows = []
    for k in k_values:
        if not isinstance(k, int) or isinstance(k, bool) or k < 2 or k >= sample_count:
            raise ValueError(f"K={k} is invalid; choose an integer from 2 to {sample_count - 1}.")
        if unique_count < k:
            raise ValueError(
                f"K={k} exceeds the {unique_count} unique data points available."
            )
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(values)
        observed_clusters = len(set(labels))
        if observed_clusters < 2 or observed_clusters >= sample_count:
            raise ValueError(
                f"K={k} produced {observed_clusters} distinct clusters; silhouette score is undefined."
            )
        rows.append(
            {
                "K": k,
                "Inertia": float(model.inertia_),
                "Silhouette Score": float(silhouette_score(values, labels)),
            }
        )
    return pd.DataFrame(rows, columns=["K", "Inertia", "Silhouette Score"])


@dataclass(frozen=True)
class PCAResult:
    """Two-dimensional PCA coordinates aligned to source samples and labels."""

    coordinates: pd.DataFrame
    sample_indices: pd.Index
    pc1_explained_variance_ratio: float
    pc2_explained_variance_ratio: float
    cumulative_explained_variance_ratio: float
    model: PCA


def prepare_pca_data(features, labels):
    """Project preprocessed clustering features to two PCs, preserving labels/index."""
    matrix, sample_indices = _feature_frame(features)
    if len(matrix) < 2:
        raise ValueError("PCA visualization requires at least 2 samples.")
    if matrix.shape[1] < 2:
        raise ValueError("PCA visualization requires at least 2 feature dimensions.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise ValueError("PCA input features must all be numeric.")

    if isinstance(labels, pd.Series):
        if labels.index.equals(sample_indices):
            aligned_labels = labels.copy()
        elif labels.index.is_unique and sample_indices.is_unique:
            if not sample_indices.isin(labels.index).all():
                raise ValueError("Cluster labels do not match the feature sample indices.")
            aligned_labels = labels.loc[sample_indices].copy()
        else:
            raise ValueError("Cannot align labels to non-unique sample indices.")
        label_values = aligned_labels.to_numpy()
    else:
        label_values = list(labels)
        if len(label_values) != len(matrix):
            raise ValueError("Cluster label count must match the number of feature rows.")

    if len(label_values) != len(matrix):
        raise ValueError("Cluster label count must match the number of feature rows.")

    import numpy as np

    values = matrix.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("PCA input features must not contain missing or infinite values.")
    model = PCA(n_components=2)
    projected = model.fit_transform(values)
    coordinate_frame = pd.DataFrame(
        {
            "PC1": projected[:, 0],
            "PC2": projected[:, 1],
            "Cluster": label_values,
            "Sample Index": sample_indices.to_list(),
        },
        index=sample_indices.copy(),
    )
    coordinate_frame.index.name = "Original Index"
    explained = model.explained_variance_ratio_
    return PCAResult(
        coordinates=coordinate_frame,
        sample_indices=sample_indices.copy(),
        pc1_explained_variance_ratio=float(explained[0]),
        pc2_explained_variance_ratio=float(explained[1]),
        cumulative_explained_variance_ratio=float(explained.sum()),
        model=model,
    )
