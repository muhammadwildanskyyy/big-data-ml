from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from src.common.config_loader import load_config, local_path, save_json, setup_logger
from src.ingestion.openaq_client import OpenAQClient

logger = setup_logger("openaq.ingest")


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_date_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cur = start
    step = timedelta(days=chunk_days)
    while cur < end:
        nxt = min(cur + step, end)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def _extract_license_names(loc: dict[str, Any]) -> str:
    licenses = loc.get("licenses") or []
    names = [lic.get("name") for lic in licenses if isinstance(lic, dict) and lic.get("name")]
    return "; ".join(sorted(set(names))) if names else "Unspecified (see OpenAQ provider terms)"


def _rank_locations(
    locations: list[dict[str, Any]],
    allowed_params: set[str],
    primary_param: str,
    win_start: datetime,
    win_end: datetime,
) -> list[dict[str, Any]]:
    ranked: list[tuple[tuple[int, int, float], dict[str, Any]]] = []
    for loc in locations:
        first_dt = _parse_iso((loc.get("datetimeFirst") or {}).get("utc"))
        last_dt = _parse_iso((loc.get("datetimeLast") or {}).get("utc"))
        if not first_dt or not last_dt:
            continue

        eff_start = max(win_start, first_dt)
        eff_end = min(win_end, last_dt)
        span_days = (eff_end - eff_start).total_seconds() / 86400.0
        if span_days < 3.0:
            continue

        sensors = [
            s
            for s in (loc.get("sensors") or [])
            if ((s.get("parameter") or {}).get("name") or "").lower() in allowed_params
        ]
        if not sensors:
            continue

        param_names = {((s.get("parameter") or {}).get("name") or "").lower() for s in sensors}
        has_primary = 1 if primary_param in param_names else 0
        if not has_primary:
            continue

        loc_copy = dict(loc)
        loc_copy["_filtered_sensors"] = sensors
        loc_copy["_eff_start"] = eff_start
        loc_copy["_eff_end"] = eff_end
        loc_copy["_span_days"] = span_days

        # Prioritaskan lokasi dengan multi-polutan + rentang waktu panjang
        score = (1 if len(param_names) >= 2 else 0, len(param_names), span_days)
        ranked.append((score, loc_copy))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked]


def run_ingestion(target_rows_override: int | None = None) -> dict[str, Any]:
    cfg = load_config()
    oaq = cfg["openaq"]
    country_iso = cfg["project"]["country_iso"]
    target_rows = target_rows_override or int(cfg["project"]["target_min_rows"])

    allowed_params = {p.lower() for p in oaq["parameters"]}
    primary_param = oaq["primary_parameter"].lower()
    win_start = _parse_iso(oaq["date_from"])
    win_end = min(_parse_iso(oaq["date_to"]), datetime.now(timezone.utc))
    assert win_start is not None and win_end is not None

    raw_file = local_path(cfg, cfg["paths"]["local_raw_file"])
    meta_file = local_path(cfg, cfg["paths"]["local_metadata_file"])
    raw_file.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAQClient(cfg)
    try:
        logger.info("Mengambil daftar stasiun OpenAQ v3 untuk country_iso=%s ...", country_iso)
        all_locations = list(
            client.iter_locations(country_iso=country_iso, limit=int(oaq["locations_page_limit"]))
        )
        logger.info("Total lokasi terdaftar di OpenAQ (%s): %d", country_iso, len(all_locations))

        candidates = _rank_locations(all_locations, allowed_params, primary_param, win_start, win_end)
        logger.info(
            "Lokasi kandidat aktif dengan %s pada rentang [%s .. %s]: %d",
            primary_param,
            _iso_utc(win_start),
            _iso_utc(win_end),
            len(candidates),
        )
        if not candidates:
            raise RuntimeError(
                f"Tidak ditemukan stasiun aktif di {country_iso} pada rentang {oaq['date_from']}..{oaq['date_to']}."
            )

        max_sensors = int(oaq.get("max_sensors", 60))
        chunk_days = int(oaq.get("chunk_days", 90))
        hours_limit = int(oaq.get("hours_page_limit", 1000))

        total_rows = 0
        sensors_fetched = 0
        locations_used: list[dict[str, Any]] = []
        param_counts: Counter[str] = Counter()
        providers: set[str] = set()
        licenses: set[str] = set()
        min_ts: str | None = None
        max_ts: str | None = None

        with raw_file.open("w", encoding="utf-8") as fout:
            for loc in candidates:
                if total_rows >= target_rows or sensors_fetched >= max_sensors:
                    break

                loc_id = loc["id"]
                loc_name = loc.get("name") or f"Location-{loc_id}"
                locality = loc.get("locality") or loc.get("timezone") or "Unknown"
                coords = loc.get("coordinates") or {}
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                provider_name = (loc.get("provider") or {}).get("name") or "Unknown"
                license_name = _extract_license_names(loc)

                eff_start: datetime = loc["_eff_start"]
                eff_end: datetime = loc["_eff_end"]
                chunks = _iter_date_chunks(eff_start, eff_end, chunk_days)

                loc_rows = 0
                loc_sensor_count = 0

                for sensor in loc["_filtered_sensors"]:
                    if sensors_fetched >= max_sensors:
                        break
                    sensor_id = sensor["id"]
                    param_obj = sensor.get("parameter") or {}
                    param_name = (param_obj.get("name") or "").lower()
                    unit = param_obj.get("units") or ""

                    sensor_rows = 0
                    for c_start, c_end in chunks:
                        for rec in client.iter_sensor_hours(
                            sensor_id=sensor_id,
                            datetime_from=_iso_utc(c_start),
                            datetime_to=_iso_utc(c_end),
                            limit=hours_limit,
                        ):
                            period = rec.get("period") or {}
                            dt_from = period.get("datetimeFrom") or {}
                            ts_utc = dt_from.get("utc")
                            if not ts_utc:
                                continue

                            summary = rec.get("summary") or {}
                            coverage = rec.get("coverage") or {}

                            row = {
                                "location_id": loc_id,
                                "location_name": loc_name,
                                "locality": locality,
                                "country_iso": country_iso,
                                "latitude": lat,
                                "longitude": lon,
                                "provider_name": provider_name,
                                "license_name": license_name,
                                "sensor_id": sensor_id,
                                "parameter": param_name,
                                "unit": unit,
                                "datetime_utc": ts_utc,
                                "datetime_local": dt_from.get("local"),
                                "value": rec.get("value"),
                                "summary_min": summary.get("min"),
                                "summary_max": summary.get("max"),
                                "summary_sd": summary.get("sd"),
                                "coverage_percent": coverage.get("percent"),
                            }
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                            sensor_rows += 1
                            total_rows += 1
                            param_counts[param_name] += 1

                            if min_ts is None or ts_utc < min_ts:
                                min_ts = ts_utc
                            if max_ts is None or ts_utc > max_ts:
                                max_ts = ts_utc

                    if sensor_rows > 0:
                        sensors_fetched += 1
                        loc_sensor_count += 1
                        loc_rows += sensor_rows
                        logger.info(
                            "  [%s] sensor=%d (%s) -> %d baris (kumulatif=%d)",
                            loc_name,
                            sensor_id,
                            param_name,
                            sensor_rows,
                            total_rows,
                        )

                if loc_rows > 0:
                    providers.add(provider_name)
                    licenses.add(license_name)
                    locations_used.append(
                        {
                            "location_id": loc_id,
                            "location_name": loc_name,
                            "locality": locality,
                            "provider": provider_name,
                            "license": license_name,
                            "sensors_fetched": loc_sensor_count,
                            "rows": loc_rows,
                        }
                    )

        metadata = {
            "fetched_at_utc": _iso_utc(datetime.now(timezone.utc)),
            "source": "OpenAQ API v3 (https://openaq.org/)",
            "api_base_url": oaq["base_url"],
            "endpoints_used": [
                f"/locations?iso={country_iso}",
                "/sensors/{sensor_id}/hours",
            ],
            "country_iso": country_iso,
            "requested_date_range": {
                "from": _iso_utc(win_start),
                "to": _iso_utc(win_end),
            },
            "actual_date_range": {
                "min_datetime_utc": min_ts,
                "max_datetime_utc": max_ts,
            },
            "total_raw_rows": total_rows,
            "target_min_rows": target_rows,
            "locations_fetched_count": len(locations_used),
            "sensors_fetched_count": sensors_fetched,
            "rows_per_parameter": dict(param_counts),
            "providers": sorted(providers),
            "licenses": sorted(licenses),
            "api_requests_made": client.requests_made,
            "rate_limit_retries": client.rate_limit_retries,
            "output_file": cfg["paths"]["local_raw_file"],
            "locations_detail": locations_used,
        }
        save_json(metadata, meta_file)
        logger.info(
            "Selesai ingestion: %d baris mentah dari %d lokasi (%d sensor). Metadata -> %s",
            total_rows,
            len(locations_used),
            sensors_fetched,
            meta_file,
        )
        return metadata
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch hourly air quality data for Indonesia from OpenAQ v3.")
    parser.add_argument("--target-rows", type=int, default=None, help="Override target minimum raw rows.")
    args = parser.parse_args()
    run_ingestion(target_rows_override=args.target_rows)


if __name__ == "__main__":
    main()
