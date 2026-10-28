"""An endpoint's name and region with any AWS account identifier or credential removed.

Shared by the run record's services and the deployments footer key, so neither can emit an
account id or a secret a URL carries.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

#: The variable the Isaacus SDK reads its hosted-API base URL from.
BASE_URL_ENV = "ISAACUS_BASE_URL"

#: An endpoint given as a full ARN carries the AWS account id in its fifth
#: field. The record must not, so the ARN is reduced to the endpoint name it
#: ends with and the region it names — better data as well as safer.
_ARN_PREFIX = "arn:aws:sagemaker:"


def endpoint_identity(name: str, region: str | None) -> tuple[str, str | None]:
    """*name* and *region* with any account identifier removed.

    ``arn:aws:sagemaker:<region>:<account>:endpoint/<name>`` reduces to
    ``<name>`` plus ``<region>``. A string that is not an ARN passes through
    untouched; one that is, but in a shape this does not recognise, still does
    not — it yields ``unnamed-endpoint`` rather than risk emitting the account
    field as a name. The declared region wins over the ARN's, since a deployer
    who spelled one out meant it.
    """
    if not name.startswith(_ARN_PREFIX):
        return name, region
    # Parse strictly by ARN field index rather than by "the last segment":
    # field 4 is the account, and an ARN truncated at it would otherwise be
    # emitted as the endpoint name. Fields are region(3), account(4),
    # resource(5+) — so an ARN with no resource has no name to give.
    parts = name.split(":")
    resource = ":".join(parts[5:]) if len(parts) > 5 else ""
    resolved = resource.rsplit("/", 1)[-1].strip() or "unnamed-endpoint"
    arn_region = parts[3].strip() if len(parts) > 3 else ""
    return resolved, region or arn_region or None


def api_endpoint() -> str:
    """The hosted API's endpoint as ``ISAACUS_BASE_URL`` names it, else ``sdk-default``.

    Scheme, host, port and path only: userinfo, query and fragment can carry a credential,
    so they are dropped. A value that is not a URL with a host yields ``unnamed-endpoint``
    rather than risk emitting it. Unset, the SDK's own default is not spelled out.
    """
    url = (os.environ.get(BASE_URL_ENV) or "").strip()
    if not url:
        return "sdk-default"
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return "unnamed-endpoint"
    if not parts.scheme or not host:
        return "unnamed-endpoint"
    return f"{parts.scheme}://{host}{f':{port}' if port else ''}{parts.path.rstrip('/')}"
