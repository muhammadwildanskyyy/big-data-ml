from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession


def create_spark_session(
    cfg: dict[str, Any],
    app_name: str,
    shuffle_partitions: int | None = None,
) -> SparkSession:
    runtime = cfg["runtime"]
    partitions = shuffle_partitions or cfg["benchmark"].get("shuffle_partitions", 8)

    spark = (
        SparkSession.builder.appName(app_name)
        .master(runtime["spark_master_url"])
        .config("spark.hadoop.fs.defaultFS", runtime["hdfs_default_fs"])
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.executorEnv.PYTHONPATH", "/opt/app")
        .config("spark.mongodb.write.connection.uri", runtime["mongo_uri"])
        .config("spark.mongodb.read.connection.uri", runtime["mongo_uri"])
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
