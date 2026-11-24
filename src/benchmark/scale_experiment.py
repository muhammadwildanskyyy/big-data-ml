from __future__ import annotations

import time
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.common.config_loader import load_config, local_path, save_csv, save_json, setup_logger
from src.common.spark_session import create_spark_session
from src.ml.features import build_ml_dataset
from src.ml.train_linear_svc import train_and_evaluate
from src.processing.preprocess import preprocess_dataframe
from src.storage.hdfs_manager import read_raw_from_hdfs

logger = setup_logger("benchmark.scale")


def run_scale_benchmark(spark: SparkSession, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    fractions: list[float] = [float(f) for f in cfg["benchmark"]["fractions"]]
    results: list[dict[str, Any]] = []

    # Hitung batas kronologis agar setiap fraksi mempertahankan kontinuitas deret waktu
    full_epochs = read_raw_from_hdfs(spark, cfg).select(F.unix_timestamp("datetime_utc").alias("ts")).dropna()
    quantiles = full_epochs.approxQuantile("ts", fractions, 0.001)

    for frac, max_epoch in zip(fractions, quantiles):
        logger.info("=== Eksperimen Skala Data: %.0f%% ===", frac * 100)
        t_total_start = time.perf_counter()

        # 1. Ukur waktu baca dari HDFS (dengan filter skala kronologis)
        t0 = time.perf_counter()
        raw_df = read_raw_from_hdfs(spark, cfg)
        if frac < 1.0:
            raw_df = raw_df.filter(F.unix_timestamp("datetime_utc") <= int(max_epoch))
        raw_df = raw_df.cache()
        raw_rows = raw_df.count()
        hdfs_read_sec = time.perf_counter() - t0

        # 2. Ukur waktu preprocessing (cleaning + pivot + fitur temporal)
        t1 = time.perf_counter()
        wide_df, pre_audit = preprocess_dataframe(raw_df, cfg)
        wide_df = wide_df.cache()
        wide_rows = wide_df.count()
        preprocessing_sec = time.perf_counter() - t1

        # 3. Ukur waktu feature engineering window (lag/rolling/target t+1)
        t2 = time.perf_counter()
        ml_df, feature_cols = build_ml_dataset(wide_df, cfg)
        ml_df = ml_df.cache()
        ml_rows = ml_df.count()
        feature_eng_sec = time.perf_counter() - t2

        # 4. Ukur waktu pelatihan & evaluasi LinearSVC
        _, preds_df, report = train_and_evaluate(ml_df, feature_cols, cfg)
        preds_df.unpersist()

        ml_train_sec = report["timing"]["train_sec"]
        ml_eval_sec = report["timing"]["eval_sec"]
        total_sec = time.perf_counter() - t_total_start
        m = report["linear_svc_metrics"]

        row = {
            "scale_fraction": frac,
            "scale_label": f"{int(frac * 100)}%",
            "raw_rows": raw_rows,
            "clean_long_rows": pre_audit["clean_long_rows"],
            "wide_rows": wide_rows,
            "ml_rows": ml_rows,
            "train_rows": report["train_rows"],
            "test_rows": report["test_rows"],
            "hdfs_read_sec": round(hdfs_read_sec, 3),
            "preprocessing_sec": round(preprocessing_sec, 3),
            "feature_eng_sec": round(feature_eng_sec, 3),
            "ml_train_sec": round(ml_train_sec, 3),
            "ml_eval_sec": round(ml_eval_sec, 3),
            "total_sec": round(total_sec, 3),
            "accuracy": m["accuracy"],
            "precision_pos": m["precision_pos"],
            "recall_pos": m["recall_pos"],
            "f1_pos": m["f1_pos"],
            "weighted_f1": m["weighted_f1"],
            "auc_roc": m["auc_roc"],
        }
        results.append(row)

        logger.info(
            "Skala %s | raw=%d | ml=%d | HDFS=%.2fs | Preproc=%.2fs | Train=%.2fs | Total=%.2fs | Acc=%.4f | F1=%.4f",
            row["scale_label"],
            row["raw_rows"],
            row["ml_rows"],
            row["hdfs_read_sec"],
            row["preprocessing_sec"],
            row["ml_train_sec"],
            row["total_sec"],
            row["accuracy"],
            row["f1_pos"],
        )

        ml_df.unpersist()
        wide_df.unpersist()
        raw_df.unpersist()

    out_dir = local_path(cfg, cfg["paths"]["local_outputs_dir"], "benchmark")
    save_json(results, out_dir / "scale_results.json")
    save_csv(results, out_dir / "scale_results.csv")
    logger.info("Hasil benchmark skala tersimpan di %s", out_dir)
    return results


def main() -> None:
    cfg = load_config()
    spark = create_spark_session(cfg, app_name="AirQualityScaleBenchmark")
    try:
        run_scale_benchmark(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
