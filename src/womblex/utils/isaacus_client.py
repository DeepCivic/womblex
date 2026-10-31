"""Re-export of :mod:`womblex.providers.isaacus.client`; deprecated, removed in 0.7.0."""

import warnings

from womblex.providers.isaacus.client import (
    API_KEY_ENV,
    ENDPOINTS_ENV,
    PROFILE_ENV,
    REGION_ENV,
    SageMakerEndpoint,
    check_reachable,
    endpoints_from_env,
    is_model_served,
    make_ai_chunking_client,
    make_isaacus_client,
    misconfigured_endpoints_var,
    parse_endpoints,
    sagemaker_configured,
    unserved_models,
    with_retries,
)

__all__ = [
    "API_KEY_ENV",
    "ENDPOINTS_ENV",
    "PROFILE_ENV",
    "REGION_ENV",
    "SageMakerEndpoint",
    "check_reachable",
    "endpoints_from_env",
    "is_model_served",
    "make_ai_chunking_client",
    "make_isaacus_client",
    "misconfigured_endpoints_var",
    "parse_endpoints",
    "sagemaker_configured",
    "unserved_models",
    "with_retries",
]

warnings.warn(
    "womblex.utils.isaacus_client is deprecated and will be removed in 0.7.0; "
    "import from womblex.providers.isaacus.client",
    DeprecationWarning,
    stacklevel=2,
)
