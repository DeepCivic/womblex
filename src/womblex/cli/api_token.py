"""``womblex api-token``: mint a service token and its registry entry."""
from __future__ import annotations

import argparse

from womblex.api.auth import SCOPES, generate_token
from womblex.cli._shared import Command


def _register(p: argparse.ArgumentParser) -> None:
    p.add_argument("--client", required=True, help="Client id the token authenticates as.")
    p.add_argument(
        "--scope", action="append", choices=SCOPES, default=None,
        help="Scope to grant (repeatable). Default: submit and read.",
    )


def cmd_api_token(args: argparse.Namespace) -> int:
    """Print a new token once, with the registry entry that holds only its hash."""
    client = args.client.strip()
    if not client:
        print("error: --client must not be empty")
        return 1
    scopes = args.scope or ["submit", "read"]
    token, digest = generate_token()
    print(f"token (shown once; give it to the client): {token}")
    print()
    print("registry entry ($WOMBLEX_API_CLIENTS file, under `clients:`):")
    print(f"  - client_id: {client}")
    print(f"    token_sha256: {digest}")
    print(f"    scopes: [{', '.join(scopes)}]")
    return 0


COMMANDS = [
    Command("api-token", "Mint a service-API token and its registry entry", _register, cmd_api_token),
]
