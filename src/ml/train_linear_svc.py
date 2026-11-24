from __future__ import annotations

import time
from typing import Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config_loader import hdfs_uri, load_config, local_path, save_csv, save_json, setup_logger
from src.common.spark_session import create_spark_session
from src.ml.features import build_ml_dataset
from src.storage.hdfs_manager import read_processed_from_hdfs

logger = setup_logger("ml.linear_svc")


def chronological_split(
    ml_df: DataFrame,
    train_ratio: float,
) -> tuple[DataFrame, DataFrame, int]:
    cutoff_epoch = int(ml_df.approxQuantile("ts_epoch", [train_ratio], 0.001)[0])
    train_df = ml_df.filter(F.col("ts_epoch") < cutoff_epoch)
    test_df = ml_df.filter(F.col("ts_epoch") >= cutoff_epoch)
    return train_df, test_df, cutoff_epoch


def compute_classification_metrics(
    pred_df: DataFrame,
    prediction_col: str = "prediction",
    has_raw_prediction: bool = True,
) -> dict[str, Any]:
    counts = {
        (int(r["label"]), int(r[prediction_col])): int(r["count"])
        for r in pred_df.groupBy("label", prediction_col).count().collect()
    }
    tp = counts.get((1, 1), 0)
    tn = counts.get((0, 0), 0)
    fp = counts.get((0, 1), 0)
    fn = counts.get((1, 0), 0)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision_pos = tp / (tp + fp) if (tp + fp) else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = (
        2.0 * precision_pos * recall_pos / (precision_pos + recall_pos)
        if (precision_pos + recall_pos)
        else 0.0
    )

    mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol=prediction_col)
    weighted_precision = float(mc_eval.setMetricName("weightedPrecision").evaluate(pred_df)) if total else 0.0
    weighted_recall = float(mc_eval.setMetricName("weightedRecall").evaluate(pred_df)) if total else 0.0
    weighted_f1 = float(mc_eval.setMetricName("f1").evaluate(pred_df)) if total else 0.0

    auc_roc = None
    auc_pr = None
    if has_raw_prediction and (tp + fn) > 0 and (tn + fp) > 0:
        bin_eval = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
        )
        auc_roc = round(float(bin_eval.setMetricName("areaUnderROC").evaluate(pred_df)), 4)
        auc_pr = round(float(bin_eval.setMetricName("areaUnderPR").evaluate(pred_df)), 4)

    return {
        "total_test_rows": total,
        "accuracy": round(accuracy, 4),
        "precision_pos": round(precision_pos, 4),
        "recall_pos": round(recall_pos, 4),
        "f1_pos": round(f1_pos, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def train_and_evaluate(
    ml_df: DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[PipelineModel, DataFrame, dict[str, Any]]:
    ml_cfg = cfg["ml"]
    svc_cfg = ml_cfg["linear_svc"]
    train_ratio = float(ml_cfg.get("train_ratio", 0.80))

    train_df, test_df, cutoff_epoch = chronological_split(ml_df, train_ratio)
    train_df = train_df.cache()
    test_df = test_df.cache()

    train_count = train_df.count()
    test_count = test_df.count()
    train_pos = train_df.filter(F.col("label") == 1.0).count()
    test_pos = test_df.filter(F.col("label") == 1.0).count()

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    svc = LinearSVC(
        featuresCol="features",
        labelCol="label",
        maxIter=int(svc_cfg.get("max_iter", 100)),
        regParam=float(svc_cfg.get("reg_param", 0.1)),
        threshold=float(svc_cfg.get("threshold", 0.0)),
        standardization=bool(svc_cfg.get("standardization", False)),
    )
    pipeline = Pipeline(stages=[assembler, scaler, svc])

    t0 = time.perf_counter()
    model = pipeline.fit(train_df)
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    predictions_df = model.transform(test_df).cache()
    metrics = compute_classification_metrics(predictions_df, prediction_col="prediction", has_raw_prediction=True)
    eval_sec = time.perf_counter() - t1

    baseline_metrics = compute_classification_metrics(
        predictions_df,
        prediction_col="persistence_pred",
        has_raw_prediction=False,
    )

    svc_stage = model.stages[-1]
    coefficients = {
        col: round(float(coef), 4)
        for col, coef in zip(feature_cols, svc_stage.coefficients.toArray().tolist())
    }

    report = {
        "problem_formulation": "Next-Hour Air Quality Binary Classification (t+1 from features <= t)",
        "algorithm": "Spark MLlib LinearSVC",
        "threshold_pm25": float(ml_cfg["threshold"]),
        "threshold_standard": cfg["thresholds"]["standard_name"],
        "split_method": "Chronological timestamp split (no temporal leakage)",
        "split_cutoff_epoch": cutoff_epoch,
        "train_rows": train_count,
        "test_rows": test_count,
        "train_positive_rate": round(train_pos / train_count, 4) if train_count else 0.0,
        "test_positive_rate": round(test_pos / test_count, 4) if test_count else 0.0,
        "feature_columns": feature_cols,
        "hyperparameters": svc_cfg,
        "timing": {
            "train_sec": round(train_sec, 3),
            "eval_sec": round(eval_sec, 3),
        },
        "linear_svc_metrics": metrics,
        "persistence_baseline_metrics": baseline_metrics,
        "model_intercept": round(float(svc_stage.intercept), 4),
        "feature_coefficients": coefficients,
    }

    train_df.unpersist()
    test_df.unpersist()
    return model, predictions_df, report


def run_ml_pipeline(spark: SparkSession, cfg: dict[str, Any]) -> dict[str, Any]:
    wide_df = read_processed_from_hdfs(spark, cfg)
    ml_df, feature_cols = build_ml_dataset(wide_df, cfg)

    ml_uri = hdfs_uri(cfg, "hdfs_ml_dataset_dir")
    ml_df.write.mode("overwrite").parquet(ml_uri)
    ml_df = spark.read.parquet(ml_uri).cache()

    model, predictions_df, report = train_and_evaluate(ml_df, feature_cols, cfg)

    model_uri = hdfs_uri(cfg, "hdfs_model_dir")
    model.write().overwrite().save(model_uri)
    report["hdfs_ml_dataset_uri"] = ml_uri
    report["hdfs_model_uri"] = model_uri

    out_dir = local_path(cfg, cfg["paths"]["local_outputs_dir"], "ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, out_dir / "linear_svc_evaluation.json")

    cm = report["linear_svc_metrics"]["confusion_matrix"]
    save_csv(
        [
            {"actual": "Aman (0)", "pred_aman_0": cm["tn"], "pred_tidak_sehat_1": cm["fp"]},
            {"actual": "Tidak Sehat (1)", "pred_aman_0": cm["fn"], "pred_tidak_sehat_1": cm["tp"]},
        ],
        out_dir / "confusion_matrix.csv",
    )

    sample_limit = int(cfg["mongodb"].get("sample_predictions_limit", 500))
    sample_df = (
        predictions_df.select(
            "location_id",
            "location_name",
            "locality",
            F.date_format("datetime_utc", "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("feature_datetime_utc"),
            F.date_format("target_datetime_utc", "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("target_datetime_utc"),
            F.round("pm25_t", 2).alias("pm25_current"),
            F.round("target_pm25", 2).alias("pm25_actual_next_hour"),
            F.col("label").cast("int").alias("actual_label"),
            F.col("prediction").cast("int").alias("predicted_label"),
            F.when(F.col("prediction") == 1.0, "Tidak Sehat (>35.4 ug/m3)")
            .otherwise("Aman (<=35.4 ug/m3)")
            .alias("kategori_prediksi"),
            F.lit("LinearSVC").alias("model"),
            F.lit("Spark MLlib").alias("sumber"),
        )
        .orderBy(F.desc("target_datetime_utc"))
        .limit(sample_limit)
    )
    sample_rows = [r.asDict() for r in sample_df.collect()]
    save_json(sample_rows, out_dir / "sample_predictions.json")
    save_csv(sample_rows[:100], out_dir / "sample_predictions.csv")

    m = report["linear_svc_metrics"]
    b = report["persistence_baseline_metrics"]
    logger.info("=== Hasil Evaluasi Spark MLlib LinearSVC (Next-Hour, Bebas Data Leakage) ===")
    logger.info("Train rows: %d (pos=%.2f%%) | Test rows: %d (pos=%.2f%%)",
                report["train_rows"], report["train_positive_rate"] * 100,
                report["test_rows"], report["test_positive_rate"] * 100)
    logger.info(
        "LinearSVC  -> Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f | AUC-ROC=%s",
        m["accuracy"], m["precision_pos"], m["recall_pos"], m["f1_pos"], m["auc_roc"],
    )
    logger.info(
        "Baseline   -> Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f (Persistence t -> t+1)",
        b["accuracy"], b["precision_pos"], b["recall_pos"], b["f1_pos"],
    )
    logger.info(
        "Confusion Matrix -> TP=%d, TN=%d, FP=%d, FN=%d",
        cm["tp"], cm["tn"], cm["fp"], cm["fn"],
    )
    logger.info("Model tersimpan di HDFS: %s", model_uri)

    predictions_df.unpersist()
    ml_df.unpersist()
    return report


def main() -> None:
    cfg = load_config()
    spark = create_spark_session(cfg, app_name="AirQualityLinearSVC")
    try:
        run_ml_pipeline(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
