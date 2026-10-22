from __future__ import annotations

import base64
import os
import secrets
import string
from dataclasses import dataclass, field
from urllib.parse import quote, urlsplit

import requests


class WebCADError(RuntimeError):
    """An HTTP or WebCAD business failure."""


class UncertainCallError(WebCADError):
    """The request may have executed remotely; do not retry automatically."""


@dataclass
class WebCADConfig:
    base_url: str
    access_token: str = field(default="", repr=False)
    username: str = ""
    password: str = field(default="", repr=False)
    session_id: str = ""
    verify_tls: bool | str = True
    timeout: float = 30.0
    # Public settings from the supplied cad-agent-create-node.mjs example.
    client_id: str = "e5cd7e4891bf95d1d19206ce24a7b32e"
    tenant_id: str = "000000"

    @classmethod
    def from_env(cls) -> WebCADConfig:
        verify = os.getenv("WEBCAD_CA_BUNDLE") or os.getenv("WEBCAD_VERIFY_TLS", "true").lower() != "false"
        return cls(
            base_url=os.environ["WEBCAD_BASE_URL"],
            access_token=os.getenv("WEBCAD_ACCESS_TOKEN", ""),
            username=os.getenv("WEBCAD_USERNAME", ""),
            password=os.getenv("WEBCAD_PASSWORD", ""),
            session_id=os.getenv("WEBCAD_SESSION_ID", ""),
            verify_tls=verify,
            client_id=os.getenv("WEBCAD_CLIENT_ID", cls.client_id),
            tenant_id=os.getenv("WEBCAD_TENANT_ID", cls.tenant_id),
        )


# This is a public encryption key, not a credential. The supplied example labels
# an SPKI DER key as RSA PUBLIC KEY; load the DER to avoid that PEM-label mismatch.
LOGIN_PUBLIC_KEY = (
    "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC/XwRpoHnvtFxR5cSzw0m220gpyydQ0fO/u2NqajTsa8A3"
    "+nm6LHAufv5rxLWafb3x7OYUnMv/5Llief/tZHNrf7ElQYRtHLk21C/gIJUkwb1F2cns6E3IKoTJOdBUbPyTy0z"
    "FeexxznCkIqmy53zLw4yJFYGK53h0OZdrUbq/ZwIDAQAB"
)


def encrypt_password(password: str) -> str:
    """Match the supplied frontend AES-ECB + RSA PKCS1v15 login algorithm."""
    from cryptography.hazmat.primitives import padding, serialization
    from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    key = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32)).encode()
    padder = padding.PKCS7(128).padder()
    padded = padder.update(password.encode()) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    encrypted = cipher.update(padded) + cipher.finalize()
    public_key = serialization.load_der_public_key(base64.b64decode(LOGIN_PUBLIC_KEY))
    wrapped = public_key.encrypt(base64.b64encode(key), rsa_padding.PKCS1v15())
    return base64.b64encode(wrapped).decode() + "|" + base64.b64encode(encrypted).decode()


class WebCADClient:
    def __init__(self, config: WebCADConfig):
        parts = urlsplit(config.base_url)
        if parts.scheme != "https" or not parts.netloc or parts.username or parts.password:
            raise ValueError("WebCAD base_url must be an HTTPS URL without embedded credentials")
        self.config = config
        self.http = requests.Session()
        self.http.headers.update({"Content-Type": "application/json"})
        if config.access_token:
            self.http.headers["Authorization"] = f"Bearer {config.access_token}"

    def close(self) -> None:
        self.http.close()

    def _request(self, method: str, path: str, body: dict | None = None):
        try:
            response = self.http.request(
                method, self.config.base_url.rstrip("/") + path, json=body,
                timeout=self.config.timeout, verify=self.config.verify_tls,
                allow_redirects=False,
            )
        except requests.exceptions.SSLError as exc:
            raise WebCADError("TLS verification failed; configure WEBCAD_CA_BUNDLE or explicitly opt in to test-only TLS bypass") from exc
        except requests.RequestException as exc:
            raise UncertainCallError(f"{method} {path}: network failure; verify the remote state before retrying") from exc
        if not 200 <= response.status_code < 300:
            error = UncertainCallError if response.status_code >= 500 else WebCADError
            raise error(f"{method} {path}: HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise UncertainCallError(f"{method} {path}: invalid JSON response; verify the remote state") from exc

    def authenticate(self) -> None:
        if self.http.headers.get("Authorization"):
            return
        if not self.config.username or not self.config.password:
            raise WebCADError("Set WEBCAD_ACCESS_TOKEN or WEBCAD_USERNAME and WEBCAD_PASSWORD")
        result = self._request("POST", "/local-api/auth/login", {
            "tenantId": self.config.tenant_id,
            "username": self.config.username,
            "password": encrypt_password(self.config.password),
            "rememberMe": True,
            "clientId": self.config.client_id,
            "grantType": "password",
        })
        data = result.get("data") or {}
        token = data.get("access_token") or data.get("accessToken")
        if not token:
            raise WebCADError(f"Login failed (code={result.get('code', 'unknown')}); no access_token received; verify username, password, clientId and tenantId")
        self.http.headers["Authorization"] = f"Bearer {token}"

    def sessions(self) -> list[dict]:
        result = self._request("GET", "/api/cad-agent/sessions")
        if not isinstance(result.get("sessions"), list):
            raise WebCADError("Unexpected sessions response; confirm against the service schema")
        return result["sessions"]

    def select_session(self) -> dict:
        sessions = self.sessions()
        selected = [s for s in sessions if s.get("id") == self.config.session_id] if self.config.session_id else sessions
        if len(selected) != 1:
            raise WebCADError("Open one target editor, or set WEBCAD_SESSION_ID to an existing session id")
        self.config.session_id = selected[0]["id"]
        return selected[0]

    def call_tool(self, tool: str, args: dict | None = None):
        if not self.config.session_id:
            raise WebCADError("Select the target session before calling tools")
        result = self._request(
            "POST", f"/api/cad-agent/sessions/{quote(self.config.session_id, safe='')}/calls",
            {"tool": tool, "args": args or {}},
        )
        if result.get("ok") is not True:
            error = result.get("error") or {}
            # Tool payloads contain geometry, not login credentials. Login errors
            # are handled separately without echoing the authentication response.
            raise WebCADError(f"{tool}: {error.get('code', 'unknown')}: {error.get('message', 'business failure')}")
        return result.get("value")

    def describe_type(self, node_type: str) -> dict:
        value = self.call_tool("cad.node.describe_type", {"type": node_type})
        if not isinstance(value, dict) or "createSchema" not in value:
            raise WebCADError("describe_type did not return createSchema; confirm the live response contract")
        return value
