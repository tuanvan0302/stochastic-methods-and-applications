"""Gaussian HMM-specific metrics.

Generic, model-agnostic metrics (AIC/BIC formula, transition tables, state
summaries) now live in src/shared/metrics.py so the GMM baseline can reuse
them for a fair comparison.
"""

from __future__ import annotations


def count_hmm_parameters(n_components: int, n_features: int, covariance_type: str) -> int:
    """Count free parameters for a Gaussian HMM.

    Includes the initial state distribution, the transition matrix, the
    per-state means, and the per-state covariances (which depend on
    covariance_type: spherical, diagonal, full, or tied).
    """

    if n_components <= 0 or n_features <= 0:
        raise ValueError("n_components and n_features must be positive.")

    startprob_params = n_components - 1
    transmat_params = n_components * (n_components - 1)
    mean_params = n_components * n_features

    if covariance_type == "spherical":
        covariance_params = n_components
    elif covariance_type == "diag":
        covariance_params = n_components * n_features
    elif covariance_type == "full":
        covariance_params = n_components * n_features * (n_features + 1) // 2
    elif covariance_type == "tied":
        covariance_params = n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return startprob_params + transmat_params + mean_params + covariance_params
