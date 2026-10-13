"""Flexible preprocessing transforms for jet constituents and jet-level features."""

import copy
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class LogScaleTransform(BaseEstimator, TransformerMixin):
    """Apply log transform to features, optionally with offset for stability.

    Useful for features with heavy-tailed distributions like pt.
    """

    def __init__(self, offset: float = 1.0):
        """Initialize log transform.

        Args:
            offset: Added before log to avoid log(0). Default 1.0 gives log(1+x).
        """
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X + self.offset)

    def inverse_transform(self, X):
        return np.exp(X) - self.offset


class FeatureWiseTransformer(BaseEstimator, TransformerMixin):
    """Apply transformers to specific feature indices only.

    This allows selective preprocessing like log scaling only pt features.
    """

    def __init__(self, feature_configs: list[dict]):
        """Initialize feature-wise transformer.

        Args:
            feature_configs: List of dicts with keys:
                - 'indices': int or list of ints for feature indices
                - 'transformers': list of transformer instances to apply in sequence

        Example:
            feature_configs = [
                {'indices': [0], 'transformers': [LogScaleTransform()]},
            ]
        """
        self.feature_configs = feature_configs
        self._validate_configs()

    def _validate_configs(self):
        """Ensure indices don't overlap and are properly formatted."""
        all_indices = set()
        for config in self.feature_configs:
            indices = config["indices"]
            if isinstance(indices, int):
                indices = [indices]
            if any(idx in all_indices for idx in indices):
                raise ValueError("Feature indices must not overlap across configs")
            all_indices.update(indices)

    def _get_indices(self, config: dict) -> list[int]:
        """Extract indices as list."""
        indices = config["indices"]
        return [indices] if isinstance(indices, int) else indices

    def fit(self, X, y=None):
        """Fit transformers on their respective features."""
        # Track number of input features (sklearn convention)
        self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
        for config in self.feature_configs:
            indices = self._get_indices(config)
            X_subset = X[:, indices]

            # Apply transformers sequentially
            for transformer in config["transformers"]:
                transformer.fit(X_subset)
                X_subset = transformer.transform(X_subset)

        return self

    def transform(self, X):
        """Transform features using fitted transformers."""
        X_transformed = X.copy()

        for config in self.feature_configs:
            indices = self._get_indices(config)
            X_subset = X_transformed[:, indices]

            # Apply transformers sequentially
            for transformer in config["transformers"]:
                X_subset = transformer.transform(X_subset)

            X_transformed[:, indices] = X_subset

        return X_transformed

    def inverse_transform(self, X):
        """Inverse transform features using fitted transformers."""
        X_inverse = X.copy()

        for config in self.feature_configs:
            indices = self._get_indices(config)
            X_subset = X_inverse[:, indices]

            # Apply inverse transformers in reverse order
            for transformer in reversed(config["transformers"]):
                X_subset = transformer.inverse_transform(X_subset)

            X_inverse[:, indices] = X_subset

        return X_inverse


class CompositeTransformer(BaseEstimator, TransformerMixin):
    """Apply log scaling to specific features, then a final transform to all features."""

    def __init__(self, log_transformer, final_transformer):
        """Initialize composite transformer.

        Args:
            log_transformer: FeatureWiseTransformer for log scaling
            final_transformer: StandardScaler or QuantileTransformer for all features
        """
        self.log_transformer = log_transformer
        self.final_transformer = final_transformer
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """Fit both transformers sequentially."""
        self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
        X_logged = self.log_transformer.fit_transform(X)
        self.final_transformer.fit(X_logged)
        return self

    def transform(self, X):
        """Apply log scaling then final transform."""
        X_logged = self.log_transformer.transform(X)
        return self.final_transformer.transform(X_logged)

    def inverse_transform(self, X):
        """Reverse both transforms."""
        X_unscaled = self.final_transformer.inverse_transform(X)
        return self.log_transformer.inverse_transform(X_unscaled)


def create_preprocessing_transformer(
    mode: Literal["quantile", "log_quantile", "log_standard", "standard"],
    log_feature_indices: list[int] | None = None,
    n_quantiles: int = 500,
    log_offset: float = 1.0,
    n_features: int | None = None,
) -> BaseEstimator:
    """Factory function to create preprocessing transformers with common configurations.

    Args:
        mode: Preprocessing strategy:
            - "quantile": QuantileTransformer on all features
            - "log_quantile": Log scale specified features, then quantile transform all
            - "log_standard": Log scale specified features, then standard scale all
            - "standard": StandardScaler on all features
        log_feature_indices: Which features to log scale (only used for log_* modes)
        n_quantiles: Number of quantiles for QuantileTransformer
        log_offset: Offset for log transform (log(x + offset))
        n_features: Total number of features (required for log_* modes)

    Returns:
        Configured transformer

    Example:
        # Log scale pt (index 0), quantile transform everything
        transformer = create_preprocessing_transformer(
            mode="log_quantile",
            log_feature_indices=[0],
            n_features=4
        )
    """
    if mode == "quantile":
        return QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=None,
        )

    elif mode == "standard":
        return StandardScaler()

    elif mode in ["log_quantile", "log_standard"]:
        if log_feature_indices is None or n_features is None:
            raise ValueError(f"Mode '{mode}' requires log_feature_indices and n_features")

        # Build feature configs for log scaling only
        feature_configs = [
            {
                "indices": log_feature_indices,
                "transformers": [LogScaleTransform(offset=log_offset)],
            }
        ]

        log_transformer = FeatureWiseTransformer(feature_configs)

        # Determine final transformer to apply to ALL features
        if mode == "log_quantile":
            final_transform = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                subsample=None,
            )
        else:  # log_standard
            final_transform = StandardScaler()

        # Return composite: log specific features, then transform all
        return CompositeTransformer(log_transformer, final_transform)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def slice_transformer(transformer: CompositeTransformer, keep: list[int]) -> CompositeTransformer:
    """Restrict a fitted log+standard ``CompositeTransformer`` to the feature columns ``keep``.

    Every column is transformed independently, so the sliced transformer applies exactly
    the same transform to the kept columns as the full one. Used for positions-out, where
    position columns are removed from ``csts`` but the positions-in scaler is reused.
    """
    if not isinstance(transformer, CompositeTransformer) or not isinstance(
        transformer.final_transformer, StandardScaler
    ):
        raise TypeError(
            f"only CompositeTransformer(..., StandardScaler) supported, got {transformer}"
        )
    new_index = {old: new for new, old in enumerate(keep)}

    configs = []
    for cfg in transformer.log_transformer.feature_configs:
        idx = [cfg["indices"]] if isinstance(cfg["indices"], int) else list(cfg["indices"])
        kept = [new_index[i] for i in idx if i in new_index]
        if kept:
            configs.append({"indices": kept, "transformers": copy.deepcopy(cfg["transformers"])})
    log_t = FeatureWiseTransformer(configs)
    log_t.n_features_in_ = len(keep)

    final = copy.deepcopy(transformer.final_transformer)
    for attr in ("mean_", "scale_", "var_"):
        if getattr(final, attr, None) is not None:
            setattr(final, attr, np.asarray(getattr(final, attr))[keep])
    if np.ndim(final.n_samples_seen_) > 0:
        final.n_samples_seen_ = np.asarray(final.n_samples_seen_)[keep]
    final.n_features_in_ = len(keep)

    out = CompositeTransformer(log_t, final)
    out.n_features_in_ = len(keep)
    return out
