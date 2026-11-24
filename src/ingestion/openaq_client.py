from __future__ import annotations

import os
import random
import time
from typing import Any, Iterator

import requests

from src.common.config_loader import setup_logger

logger = setup_logger("openaq.client")


class OpenAQClient:
    """Rate-limit aware client for OpenAQ API v3."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        api_key = os.getenv("OPENAQ_API_KEY", "").strip()
        if not api_key or api_key == "your_openaq_api_key_here":
            raise ValueError(
                "OPENAQ_API_KEY belum diset pada file .env. "
                "Tambahkan API key valid dari https://explore.openaq.org/."
            )

        oaq = cfg["openaq"]
        self.base_url = oaq["base_url"].rstrip("/")
        self.timeout = int(oaq.get("request_timeout_sec", 30))
        self.max_retries = int(oaq.get("max_retries", 6))
        self.backoff_base = float(oaq.get("backoff_base_sec", 2.0))
        self.min_interval = float(oaq.get("min_request_interval_sec", 1.05))

        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-API-Key": api_key,
                "Accept": "application/json",
                "User-Agent": "unesa-bigdata-airquality/1.0",
            }
        )

        self._last_request_ts = 0.0
        self.requests_made = 0
        self.rate_limit_retries = 0

    def close(self) -> None:
        self.session.close()

    def _pace_request(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _respect_rate_headers(self, headers: requests.structures.CaseInsensitiveDict) -> None:
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if remaining is not None:
            try:
                rem_val = int(remaining)
                if rem_val <= 1:
                    wait_sec = 5.0
                    if reset is not None:
                        reset_num = float(reset)
                        # OpenAQ x-ratelimit-reset can be seconds-until-reset or unix epoch
                        if reset_num > 1_000_000_000:
                            wait_sec = max(1.0, reset_num - time.time() + 1.0)
                        else:
                            wait_sec = max(1.0, reset_num + 1.0)
                    wait_sec = min(wait_sec, 65.0)
                    logger.info("Quota rate-limit hampir habis (remaining=%d), jeda %.1fs", rem_val, wait_sec)
                    time.sleep(wait_sec)
            except ValueError:
                pass

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            self._pace_request()
            try:
                self._last_request_ts = time.monotonic()
                resp = self.session.get(url, params=params, timeout=self.timeout)
                self.requests_made += 1
            except requests.RequestException as exc:
                delay = self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning("Network error (%s), retry %d/%d dalam %.1fs", exc, attempt, self.max_retries, delay)
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                self._respect_rate_headers(resp.headers)
                return resp.json()

            if resp.status_code == 429:
                self.rate_limit_retries += 1
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("x-ratelimit-reset")
                delay = self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0.5, 2.0)
                if retry_after:
                    try:
                        ra_val = float(retry_after)
                        if ra_val > 1_000_000_000:
                            ra_val = max(1.0, ra_val - time.time() + 1.0)
                        delay = max(delay, min(ra_val, 90.0))
                    except ValueError:
                        pass
                logger.warning("HTTP 429 rate-limited pada %s, backoff %.1fs (retry %d/%d)", path, delay, attempt, self.max_retries)
                time.sleep(delay)
                continue

            if resp.status_code in {408, 500, 502, 503, 504}:
                delay = self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning("HTTP %d pada %s, retry %d/%d dalam %.1fs", resp.status_code, path, attempt, self.max_retries, delay)
                if attempt == self.max_retries:
                    resp.raise_for_status()
                time.sleep(delay)
                continue

            if resp.status_code == 404:
                return {"meta": {"found": 0}, "results": []}

            resp.raise_for_status()

        raise RuntimeError(f"Gagal memanggil {url} setelah {self.max_retries} percobaan.")

    def iter_locations(self, country_iso: str, limit: int = 100) -> Iterator[dict[str, Any]]:
        page = 1
        while True:
            payload = self._get(
                "/locations",
                params={"iso": country_iso, "limit": limit, "page": page},
            )
            results = payload.get("results", [])
            if not results:
                break
            yield from results
            if len(results) < limit:
                break
            page += 1

    def iter_sensor_hours(
        self,
        sensor_id: int,
        datetime_from: str,
        datetime_to: str,
        limit: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        page = 1
        while True:
            payload = self._get(
                f"/sensors/{sensor_id}/hours",
                params={
                    "datetime_from": datetime_from,
                    "datetime_to": datetime_to,
                    "limit": limit,
                    "page": page,
                },
            )
            results = payload.get("results", [])
            if not results:
                break
            yield from results
            if len(results) < limit:
                break
            page += 1
