from __future__ import annotations

import math
from typing import Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

FEATURE_COLUMNS: list[str] = [
    "pm25_t",
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "pm25_lag24",
    "pm25_diff_1h",
    "pm25_roll_mean_3h",
    "pm25_roll_mean_6h",
    "pm25_roll_std_6h",
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
    "target_month",
    "target_is_weekend",
    "latitude",
    "longitude",
]


def build_ml_dataset(wide_df: DataFrame, cfg: dict[str, Any]) -> tuple[DataFrame, list[str]]:
    """Construct next-hour classification features strictly from timestamps <= t.

    Label is defined at t + horizon_hours (default +1h) using exact timestamp alignment,
    ensuring zero data leakage from target time t+1 into input features.
    """
    ml_cfg = cfg["ml"]
    target_col = ml_cfg["target_pollutant"]
    threshold = float(ml_cfg["threshold"])
    horizon_sec = int(ml_cfg.get("horizon_hours", 1)) * 3600
    utc_offset = int(cfg["preprocessing"].get("utc_offset_hours", 7))

    base_df = (
        wide_df.filter(F.col(target_col).isNotNull())
        .withColumn("ts_epoch", F.unix_timestamp("datetime_utc"))
        .withColumn("pm25_t", F.col(target_col).cast("double"))
    )

    w_ord = Window.partitionBy("location_id").orderBy("ts_epoch")
    w_lag1 = w_ord.rangeBetween(-3600, -3600)
    w_lag2 = w_ord.rangeBetween(-7200, -7200)
    w_lag3 = w_ord.rangeBetween(-10800, -10800)
    w_lag24 = w_ord.rangeBetween(-25 * 3600, -23 * 3600)
    w_roll3 = w_ord.rangeBetween(-3 * 3600, 0)
    w_roll6 = w_ord.rangeBetween(-6 * 3600, 0)

    two_pi = 2.0 * math.pi

    feat_df = (
        base_df.withColumn("target_pm25", F.lead("pm25_t", 1).over(w_ord))
        .withColumn("target_ts_epoch", F.lead("ts_epoch", 1).over(w_ord))
        .withColumn("target_datetime_utc", F.lead("datetime_utc", 1).over(w_ord))
        # Fitur lag strictly pada waktu <= t (berbasis detik aktual agar bebas gap)
        .withColumn("pm25_lag1", F.first("pm25_t").over(w_lag1))
        .withColumn("pm25_lag2", F.first("pm25_t").over(w_lag2))
        .withColumn("pm25_lag3", F.first("pm25_t").over(w_lag3))
        .withColumn("pm25_lag24", F.avg("pm25_t").over(w_lag24))
        .withColumn("pm25_roll_mean_3h", F.avg("pm25_t").over(w_roll3))
        .withColumn("pm25_roll_mean_6h", F.avg("pm25_t").over(w_roll6))
        .withColumn("pm25_roll_std_6h", F.coalesce(F.stddev_samp("pm25_t").over(w_roll6), F.lit(0.0)))
        .withColumn("pm25_diff_1h", F.col("pm25_t") - F.col("pm25_lag1"))
        # Validasi horizon prediksi tepat 1 jam ke depan
        .filter((F.col("target_ts_epoch") - F.col("ts_epoch")) == horizon_sec)
        .filter(F.col("target_pm25").isNotNull())
    )

    # Fitur temporal untuk jam target t+1 (diketahui di muka sebelum t+1 terjadi)
    target_local = F.expr(f"target_datetime_utc + INTERVAL {utc_offset} HOURS")
    feat_df = (
        feat_df.withColumn("_target_local_ts", target_local)
        .withColumn("_target_hour", F.hour("_target_local_ts").cast("double"))
        .withColumn("_target_dow", F.dayofweek("_target_local_ts").cast("double"))
        .withColumn("target_month", F.month("_target_local_ts").cast("double"))
        .withColumn("target_is_weekend", F.when(F.col("_target_dow").isin([1.0, 7.0]), 1.0).otherwise(0.0))
        .withColumn("target_hour_sin", F.sin(F.lit(two_pi) * F.col("_target_hour") / F.lit(24.0)))
        .withColumn("target_hour_cos", F.cos(F.lit(two_pi) * F.col("_target_hour") / F.lit(24.0)))
        .withColumn("target_dow_sin", F.sin(F.lit(two_pi) * F.col("_target_dow") / F.lit(7.0)))
        .withColumn("target_dow_cos", F.cos(F.lit(two_pi) * F.col("_target_dow") / F.lit(7.0)))
        .withColumn("label", F.when(F.col("target_pm25") > threshold, 1.0).otherwise(0.0))
        .withColumn("persistence_pred", F.when(F.col("pm25_t") > threshold, 1.0).otherwise(0.0))
    )

    # Pastikan tidak ada fitur leakage
    assert "target_pm25" not in FEATURE_COLUMNS
    assert "label" not in FEATURE_COLUMNS

    select_cols = [
        "location_id",
        "location_name",
        "locality",
        "datetime_utc",
        "target_datetime_utc",
        "ts_epoch",
        "target_pm25",
        "label",
        "persistence_pred",
        *FEATURE_COLUMNS,
    ]
    clean_ml_df = feat_df.select(*select_cols).dropna(subset=[*FEATURE_COLUMNS, "label"])
    return clean_ml_df, list(FEATURE_COLUMNS)
