"""Copernicus Data Space OAuth2 token management.

The default analysis pipeline (Earth Search STAC / AWS open data) does not need
credentials. This module is only used when fetching full products directly from
the Copernicus Data Space catalogue.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


class CdseAuth:
    """Fetches and caches a CDSE access token from username/password or M2M creds."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        load_dotenv()
        self.client_id = os.getenv("CDSE_CLIENT_ID", "")
        self.client_secret = os.getenv("CDSE_CLIENT_SECRET", "")
        self.username = os.getenv("CDSE_USERNAME", "")
        self.password = os.getenv("CDSE_PASSWORD", "")
        self.ttl = int(os.getenv("CDSE_TOKEN_TTL", "3600"))
        self.cache_file = (cache_dir or Path("data/cache")) / "cdse_token.json"

    @property
    def configured(self) -> bool:
        # Prefer username/password: the cdse-public password grant issues tokens
        # with the download audience. Sentinel-Hub-dashboard M2M clients currently
        # authenticate but their tokens lack the download scope.
        return bool(self.username and self.password) or bool(self.client_id and self.client_secret)

    def _read_cache(self) -> str | None:
        if not self.cache_file.exists():
            return None
        try:
            data = json.loads(self.cache_file.read_text())
            if data.get("expires_at", 0) > time.time() + 60:
                return data["token"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass
        return None

    def _write_cache(self, token: str, expires_in: int) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(
            json.dumps({"token": token, "expires_at": time.time() + expires_in})
        )

    def get_token(self, force: bool = False) -> str:
        if not force:
            cached = self._read_cache()
            if cached:
                return cached
        if not self.configured:
            raise RuntimeError(
                "CDSE credentials are not configured. Fill CDSE_USERNAME/CDSE_PASSWORD or "
                "CDSE_CLIENT_ID/CDSE_CLIENT_SECRET in .env, or use the Earth Search STAC path."
            )

        if self.username and self.password:
            payload = {
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": "cdse-public",
            }
        elif self.client_id:
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
        else:
            raise RuntimeError("CDSE credentials are not configured.")

        resp = requests.post(TOKEN_URL, data=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        token = data["access_token"]
        self._write_cache(token, int(data.get("expires_in", self.ttl)))
        return token
