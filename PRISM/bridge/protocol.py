"""JSON-over-TCP protocol for the Live Bridge API.

Messages are JSON objects terminated by a newline character.
Each message has a type (request/response), an id for correlation,
an action name, and parameters or result data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class BridgeMessage:
    """A single message in the bridge protocol.

    Attributes:
        type: 'request' or 'response'.
        id: Correlation ID (echoed in responses).
        action: Command name (e.g., 'run', 'get_state').
        params: Request parameters dict.
        status: Response status ('ok' or 'error').
        data: Response payload dict.
        error: Error description string.
    """
    type: str = "request"           # "request" | "response"
    id: str = ""                    # correlation id
    action: str = ""                # command name
    params: dict = field(default_factory=dict)
    status: str = ""                # "ok" | "error"
    data: dict = field(default_factory=dict)
    error: str = ""

    def to_json(self) -> str:
        """Serialize to a JSON string (no trailing newline)."""
        return json.dumps(asdict(self), ensure_ascii=False)

    def to_bytes(self) -> bytes:
        """Serialize to bytes with newline terminator for TCP."""
        return (self.to_json() + "\n").encode("utf-8")

    @classmethod
    def from_json(cls, raw: str) -> BridgeMessage:
        """Deserialize from a JSON string."""
        d = json.loads(raw.strip())
        return cls(
            type=d.get("type", "request"),
            id=d.get("id", ""),
            action=d.get("action", ""),
            params=d.get("params", {}),
            status=d.get("status", ""),
            data=d.get("data", {}),
            error=d.get("error", ""),
        )

    @classmethod
    def ok_response(cls, request_id: str, data: dict | None = None) -> BridgeMessage:
        """Create a success response."""
        return cls(
            type="response",
            id=request_id,
            status="ok",
            data=data or {},
        )

    @classmethod
    def error_response(cls, request_id: str, error: str) -> BridgeMessage:
        """Create an error response."""
        return cls(
            type="response",
            id=request_id,
            status="error",
            error=error,
        )
