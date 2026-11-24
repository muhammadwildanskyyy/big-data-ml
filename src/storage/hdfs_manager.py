from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from src.common.config_loader import hdfs_uri, load_config, local_path, save_json, setup_logger
from src.common.spark_session import create_spark_session

logger = setup_logger("storage.hdfs")

RAW_OPENAQ_SCHEMA = StructType(
    [
        StructField("location_id", LongType(), True),
        StructField("location_name", StringType(), True),
        StructField("locality", StringType(), True),
        StructField("country_iso", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("provider_name", StringType(), True),
        StructField("license_name", StringType(), True),
        StructField("sensor_id", LongType(), True),
        StructField("parameter", StringType(), True),
        StructField("unit", StringType(), True),
        StructField("datetime_utc", StringType(), True),
        StructField("datetime_local", StringType(), True),
        StructField("value", DoubleType(), True),
        StructField("summary_min", DoubleType(), True),
        StructField("summary_max", DoubleType(), True),
        StructField("summary_sd", DoubleType(), True),
        StructField("coverage_percent", DoubleType(), True),
    ]
)


def read_raw_from_hdfs(spark: SparkSession, cfg: dict[str, Any]) -> DataFrame:
    uri = hdfs_uri(cfg, "hdfs_raw_file")
    logger.info("Membaca dataset mentah langsung dari HDFS: %s", uri)
    df = spark.read.schema(RAW_OPENAQ_SCHEMA).json(uri)
    return df.withColumn("datetime_utc", F.to_timestamp("datetime_utc"))


def read_processed_from_hdfs(spark: SparkSession, cfg: dict[str, Any]) -> DataFrame:
    uri = hdfs_uri(cfg, "hdfs_processed_dir")
    logger.info("Membaca dataset terproses dari HDFS: %s", uri)
    return spark.read.parquet(uri)


def verify_hdfs_read(spark: SparkSession, cfg: dict[str, Any]) -> dict[str, Any]:
    uri = hdfs_uri(cfg, "hdfs_raw_file")
    df = read_raw_from_hdfs(spark, cfg)
    df.printSchema()

    row_count = df.count()
    loc_count = df.select("location_id").distinct().count()
    param_dist = [
        {"parameter": r["parameter"], "rows": r["count"]}
        for r in df.groupBy("parameter").count().orderBy(F.desc("count")).collect()
    ]
    time_bounds = df.agg(
        F.min("datetime_utc").alias("min_utc"),
        F.max("datetime_utc").alias("max_utc"),
    ).collect()[0]

    audit = {
        "hdfs_uri": uri,
        "row_count": row_count,
        "distinct_locations": loc_count,
        "min_datetime_utc": str(time_bounds["min_utc"]),
        "max_datetime_utc": str(time_bounds["max_utc"]),
        "parameters": param_dist,
        "schema": [{"name": f.name, "type": f.dataType.simpleString()} for f in df.schema.fields],
    }

    out_path = local_path(cfg, cfg["paths"]["local_outputs_dir"], "hdfs", "spark_hdfs_audit.json")
    save_json(audit, out_path)
    logger.info("Verifikasi baca HDFS berhasil: %d baris, %d lokasi -> %s", row_count, loc_count, out_path)
    return audit


def main() -> None:
    cfg = load_config()
    spark = create_spark_session(cfg, app_name="VerifyHDFSRead")
    try:
        verify_hdfs_read(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
