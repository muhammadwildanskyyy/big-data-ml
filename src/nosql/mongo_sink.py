from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pyspark.sql import SparkSession

from src.common.config_loader import load_config, local_path, save_json, setup_logger
from src.common.spark_session import create_spark_session

logger = setup_logger("nosql.mongodb")


def _load_json(path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_to_mongodb_via_spark(spark: SparkSession, cfg: dict[str, Any]) -> dict[str, Any]:
    db_name = cfg["runtime"]["mongo_db"]
    cols = cfg["mongodb"]["collections"]
    out_root = local_path(cfg, cfg["paths"]["local_outputs_dir"])

    analytics_path = out_root / "analytics" / "analytics_summary.json"
    preds_path = out_root / "ml" / "sample_predictions.json"
    bench_path = out_root / "benchmark" / "scale_results.json"

    # 1. Tulis sampel prediksi model ke MongoDB melalui Spark MongoDB Connector
    preds_data = _load_json(preds_path)
    preds_df = spark.createDataFrame(preds_data)
    (
        preds_df.write.format("mongodb")
        .mode("overwrite")
        .option("database", db_name)
        .option("collection", cols["model_predictions"])
        .save()
    )
    logger.info(
        "[WRITE Spark->MongoDB] %d dokumen prediksi ditulis ke %s.%s",
        len(preds_data),
        db_name,
        cols["model_predictions"],
    )

    # 2. Tulis ringkasan analisis eksploratif sebagai dokumen JSON terstruktur
    analytics_data = _load_json(analytics_path)
    analytics_doc = {
        "document_type": "exploratory_analytics_summary",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_summary": analytics_data["overall_summary"],
        "thresholds": analytics_data["thresholds"],
        "top_locations": analytics_data["location_profile"][:10],
        "weekday_vs_weekend": analytics_data["weekday_vs_weekend"],
        "correlations": analytics_data["correlations"],
    }
    analytics_rdd = spark.sparkContext.parallelize([json.dumps(analytics_doc)])
    analytics_df = spark.read.json(analytics_rdd)
    (
        analytics_df.write.format("mongodb")
        .mode("overwrite")
        .option("database", db_name)
        .option("collection", cols["analytics_summary"])
        .save()
    )
    logger.info(
        "[WRITE Spark->MongoDB] Dokumen ringkasan analisis ditulis ke %s.%s",
        db_name,
        cols["analytics_summary"],
    )

    # 3. Tulis hasil eksperimen skala data ke MongoDB
    bench_count = 0
    if bench_path.exists():
        bench_data = _load_json(bench_path)
        bench_df = spark.createDataFrame(bench_data)
        (
            bench_df.write.format("mongodb")
            .mode("overwrite")
            .option("database", db_name)
            .option("collection", cols["benchmark_results"])
            .save()
        )
        bench_count = len(bench_data)
        logger.info(
            "[WRITE Spark->MongoDB] %d dokumen benchmark ditulis ke %s.%s",
            bench_count,
            db_name,
            cols["benchmark_results"],
        )

    return {
        "database": db_name,
        "written_counts": {
            cols["model_predictions"]: len(preds_data),
            cols["analytics_summary"]: 1,
            cols["benchmark_results"]: bench_count,
        },
    }


def verify_mongodb_read(spark: SparkSession, cfg: dict[str, Any]) -> dict[str, Any]:
    db_name = cfg["runtime"]["mongo_db"]
    cols = cfg["mongodb"]["collections"]

    # Verifikasi baca via Spark MongoDB Connector
    spark_read_df = (
        spark.read.format("mongodb")
        .option("database", db_name)
        .option("collection", cols["model_predictions"])
        .load()
    )
    spark_read_count = spark_read_df.count()
    logger.info("=== [READ MongoDB->Spark] Sampel 5 Dokumen Prediksi ===")
    spark_read_df.select(
        "location_name",
        "target_datetime_utc",
        "pm25_current",
        "pm25_actual_next_hour",
        "predicted_label",
        "kategori_prediksi",
        "model",
    ).show(5, truncate=False)

    # Verifikasi baca & query terfilter via PyMongo (simulasi aplikasi konsumen)
    client = MongoClient(cfg["runtime"]["mongo_uri"], serverSelectionTimeoutMS=5000)
    try:
        db = client[db_name]
        counts = {
            name: db[name].count_documents({})
            for name in cols.values()
        }
        unhealthy_samples = list(
            db[cols["model_predictions"]]
            .find({"predicted_label": 1}, {"_id": 0})
            .sort("target_datetime_utc", -1)
            .limit(3)
        )
        summary_doc = db[cols["analytics_summary"]].find_one({}, {"_id": 0, "overall_summary": 1, "generated_at_utc": 1})
    finally:
        client.close()

    logger.info("[READ PyMongo] Jumlah dokumen per koleksi: %s", counts)
    logger.info("[READ PyMongo] Contoh hasil query prediksi Tidak Sehat (predicted_label=1): %d dokumen", len(unhealthy_samples))
    for doc in unhealthy_samples:
        logger.info("  -> %s", json.dumps(doc, ensure_ascii=False))

    return {
        "spark_connector_read_count": spark_read_count,
        "pymongo_collection_counts": counts,
        "sample_query_unhealthy_predictions": unhealthy_samples,
        "analytics_summary_preview": summary_doc,
    }


def run_mongodb_sync(spark: SparkSession, cfg: dict[str, Any]) -> dict[str, Any]:
    write_res = write_to_mongodb_via_spark(spark, cfg)
    read_res = verify_mongodb_read(spark, cfg)

    audit = {
        "status": "verified",
        "connector": "org.mongodb.spark:mongo-spark-connector_2.12:10.4.1",
        "write": write_res,
        "read": read_res,
    }
    out_path = local_path(cfg, cfg["paths"]["local_outputs_dir"], "nosql", "mongodb_verification.json")
    save_json(audit, out_path)
    logger.info("Bukti verifikasi MongoDB tersimpan di %s", out_path)
    return audit


def main() -> None:
    cfg = load_config()
    spark = create_spark_session(cfg, app_name="AirQualityMongoDBSync")
    try:
        run_mongodb_sync(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
