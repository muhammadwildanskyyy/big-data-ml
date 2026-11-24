from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config_loader import hdfs_uri, load_config, local_path, save_json, setup_logger
from src.common.spark_session import create_spark_session
from src.storage.hdfs_manager import read_raw_from_hdfs

logger = setup_logger("processing.preprocess")


def preprocess_dataframe(
    raw_df: DataFrame,
    cfg: dict[str, Any],
) -> tuple[DataFrame, dict[str, Any]]:
    bounds: dict[str, list[float]] = cfg["preprocessing"]["pollutant_bounds"]
    all_params: list[str] = [p.lower() for p in cfg["openaq"]["parameters"]]
    utc_offset = int(cfg["preprocessing"].get("utc_offset_hours", 7))

    raw_count = raw_df.count()
    schema_info = [{"name": f.name, "type": f.dataType.simpleString()} for f in raw_df.schema.fields]

    # Audit null pada kolom kritis
    key_cols = ["location_id", "sensor_id", "parameter", "datetime_utc", "value"]
    null_row = raw_df.select(
        [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in key_cols]
    ).collect()[0]
    null_counts = {c: int(null_row[c] or 0) for c in key_cols}

    # 1. Hapus baris dengan field kritis kosong atau parameter tidak terdaftar
    non_null_df = (
        raw_df.withColumn("parameter", F.lower(F.trim(F.col("parameter"))))
        .dropna(subset=key_cols)
        .filter(F.col("parameter").isin(all_params))
    )
    after_null_count = non_null_df.count()

    # 2. Hapus duplikasi pengukuran (location_id, sensor_id, datetime_utc)
    dedup_df = non_null_df.dropDuplicates(["location_id", "sensor_id", "datetime_utc"])
    after_dedup_count = dedup_df.count()
    duplicates_removed = after_null_count - after_dedup_count

    # 3. Filter anomali sensor berdasarkan batas fisik per polutan
    valid_expr = F.lit(False)
    for param, (low, high) in bounds.items():
        valid_expr = valid_expr | (
            (F.col("parameter") == param)
            & (F.col("value") >= float(low))
            & (F.col("value") <= float(high))
        )

    anomaly_df = dedup_df.filter(~valid_expr)
    anomalies_by_param = {
        r["parameter"]: int(r["count"])
        for r in anomaly_df.groupBy("parameter").count().collect()
    }
    clean_long_df = dedup_df.filter(valid_expr)
    after_anomaly_count = clean_long_df.count()
    anomalies_removed = after_dedup_count - after_anomaly_count

    # 4. Transformasi Long-to-Wide (pivot per location_id & datetime_utc)
    # Normalisasi metadata per location_id agar tidak memecah baris saat pivot
    loc_meta = clean_long_df.groupBy("location_id").agg(
        F.first("location_name", ignorenulls=True).alias("location_name"),
        F.first("locality", ignorenulls=True).alias("locality"),
        F.first("latitude", ignorenulls=True).alias("latitude"),
        F.first("longitude", ignorenulls=True).alias("longitude"),
        F.first("provider_name", ignorenulls=True).alias("provider_name"),
    )

    pivoted_df = (
        clean_long_df.groupBy("location_id", "datetime_utc")
        .pivot("parameter", all_params)
        .agg(F.avg("value"))
    )

    wide_df = pivoted_df.join(loc_meta, on="location_id", how="left")

    # Pastikan seluruh kolom polutan tersedia secara deterministik
    for param in all_params:
        if param not in wide_df.columns:
            wide_df = wide_df.withColumn(param, F.lit(None).cast("double"))

    # 5. Ekstraksi fitur temporal (WIB / UTC+offset untuk interpretasi lokal Indonesia)
    local_ts = F.expr(f"datetime_utc + INTERVAL {utc_offset} HOURS")
    wide_df = (
        wide_df.withColumn("datetime_local_ts", local_ts)
        .withColumn("hour", F.hour("datetime_local_ts"))
        .withColumn("day_of_week", F.dayofweek("datetime_local_ts"))
        .withColumn("month", F.month("datetime_local_ts"))
        .withColumn("year_month", F.date_format("datetime_local_ts", "yyyy-MM"))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
        .withColumn("day_type", F.when(F.col("is_weekend") == 1, "Weekend").otherwise("Weekday"))
    )

    wide_count = wide_df.count()
    non_null_pollutants = wide_df.select(
        [F.count(F.col(p)).alias(p) for p in all_params]
    ).collect()[0]

    audit = {
        "raw_rows": raw_count,
        "initial_schema": schema_info,
        "null_counts_key_cols": null_counts,
        "rows_after_null_filter": after_null_count,
        "duplicates_removed": duplicates_removed,
        "rows_after_dedup": after_dedup_count,
        "anomalies_removed": anomalies_removed,
        "anomalies_by_parameter": anomalies_by_param,
        "clean_long_rows": after_anomaly_count,
        "wide_rows_after_pivot": wide_count,
        "wide_non_null_per_pollutant": {p: int(non_null_pollutants[p]) for p in all_params},
        "utc_offset_hours": utc_offset,
    }
    return wide_df, audit


def run_preprocessing(spark: SparkSession, cfg: dict[str, Any]) -> dict[str, Any]:
    raw_df = read_raw_from_hdfs(spark, cfg)
    logger.info("=== Skema Awal Data Mentah (HDFS) ===")
    raw_df.printSchema()

    wide_df, audit = preprocess_dataframe(raw_df, cfg)

    out_uri = hdfs_uri(cfg, "hdfs_processed_dir")
    wide_df.write.mode("overwrite").parquet(out_uri)
    audit["hdfs_processed_uri"] = out_uri

    audit_path = local_path(cfg, cfg["paths"]["local_outputs_dir"], "processing", "preprocessing_audit.json")
    save_json(audit, audit_path)

    logger.info("=== Ringkasan Preprocessing Spark ===")
    logger.info("Baris mentah awal      : %d", audit["raw_rows"])
    logger.info("Duplikasi dihapus      : %d", audit["duplicates_removed"])
    logger.info("Anomali sensor dihapus : %d (%s)", audit["anomalies_removed"], audit["anomalies_by_parameter"])
    logger.info("Baris bersih (long)    : %d", audit["clean_long_rows"])
    logger.info("Baris wide (pivoted)   : %d", audit["wide_rows_after_pivot"])
    logger.info("Ketersediaan polutan   : %s", audit["wide_non_null_per_pollutant"])
    logger.info("Output Parquet HDFS    : %s", out_uri)
    logger.info("Laporan audit lokal    : %s", audit_path)
    return audit


def main() -> None:
    cfg = load_config()
    spark = create_spark_session(cfg, app_name="AirQualityPreprocessing")
    try:
        run_preprocessing(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
