from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    hdfs_host = os.getenv("HDFS_NAMENODE_HOST", "namenode")
    hdfs_port = os.getenv("HDFS_NAMENODE_PORT", "8020")
    hdfs_base_uri = f"hdfs://{hdfs_host}:{hdfs_port}"

    mongo_host = os.getenv("MONGO_HOST", "mongodb")
    mongo_port = os.getenv("MONGO_PORT", "27017")
    mongo_db = os.getenv("MONGO_DB", "air_quality_db")
    mongo_user = os.getenv("MONGO_USER", "bigdata")
    mongo_password = os.getenv("MONGO_PASSWORD", "bigdata_secret")
    mongo_uri = (
        f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}"
        f"/{mongo_db}?authSource=admin"
    )

    cfg["runtime"] = {
        "project_root": str(PROJECT_ROOT),
        "hdfs_default_fs": hdfs_base_uri,
        "spark_master_url": os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077"),
        "mongo_uri": mongo_uri,
        "mongo_db": mongo_db,
        "openaq_api_key_set": bool(os.getenv("OPENAQ_API_KEY")),
    }
    return cfg


def hdfs_uri(cfg: dict[str, Any], path_key: str) -> str:
    base = cfg["runtime"]["hdfs_default_fs"].rstrip("/")
    rel = cfg["paths"][path_key]
    return f"{base}{rel}"


def local_path(cfg: dict[str, Any], *parts: str) -> Path:
    return Path(cfg["runtime"]["project_root"]).joinpath(*parts)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def save_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return path


def save_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
