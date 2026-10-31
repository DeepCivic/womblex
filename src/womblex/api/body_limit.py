"""A request-body size cap enforced before the framework parses the body.

Starlette's multipart parser spools file parts to temporary disk before a route
runs, so a size check inside the route comes after the disk is spent. This
ASGI middleware answers 413 on a declared ``Content-Length`` over the cap, and
cuts a body without a truthful length off as it arrives.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, MutableMapping

    Scope = MutableMapping[str, Any]
    Message = MutableMapping[str, Any]
    Receive = Callable[[], Awaitable[Message]]
    Send = Callable[[Message], Awaitable[None]]
    ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class _TooLarge(Exception):
    pass


class BodyLimit:
    """Refuse a POST to *path* whose body is over *limit* bytes with 413."""

    def __init__(self, app: ASGIApp, *, path: str, limit: int) -> None:
        self.app = app
        self.path = path
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != self.path:
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.limit:
            await self._refuse(send)
            return

        received = 0
        exceeded = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    exceeded = True
                    raise _TooLarge
            return message

        async def guarded_send(message: Message) -> None:
            # The framework turns a failed body read into its own 400; once the
            # cap is passed, that answer is dropped and the 413 below replaces it.
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _TooLarge:
            pass
        if exceeded:
            await self._refuse(send)

    async def _refuse(self, send: Send) -> None:
        body = json.dumps({"detail": f"request body is over {self.limit} bytes"}).encode()
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
