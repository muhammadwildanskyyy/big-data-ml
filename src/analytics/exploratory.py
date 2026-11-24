from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config_loader import load_config, local_path, save_csv, save_json, setup_logger
from src.common.spark_session import create_spark_session
from src.storage.hdfs_manager import read_processed_from_hdfs

logger = setup_logger("analytics.exploratory")


def _rows_to_dicts(df: DataFrame) -> list[dict[str, Any]]:
    return [r.asDict(recursive=True) for r in df.collect()]


def run_exploratory_analytics(spark: SparkSession, cfg: dict[str, Any]) -> dict[str, Any]:
    df = read_processed_from_hdfs(spark, cfg).cache()
    df.createOrReplaceTempView("air_quality_hourly")

    thr_primary = float(cfg["thresholds"]["pm25_unhealthy"])
    thr_who = float(cfg["thresholds"]["who_24h_guideline_pm25"])
    thr_klhk = float(cfg["thresholds"]["klhk_ispu_unhealthy_pm25"])
    pollutants: list[str] = [p.lower() for p in cfg["openaq"]["parameters"]]

    out_dir = local_path(cfg, cfg["paths"]["local_outputs_dir"], "analytics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Profil polusi antar-lokasi (rata-rata, standar deviasi, median, P95, maksimum)
    loc_profile_df = spark.sql(
        """
        SELECT
            location_id,
            location_name,
            locality,
            COUNT(pm25) AS n_hours_pm25,
            ROUND(AVG(pm25), 2) AS mean_pm25,
            ROUND(STDDEV(pm25), 2) AS std_pm25,
            ROUND(PERCENTILE_APPROX(pm25, 0.50), 2) AS p50_pm25,
            ROUND(PERCENTILE_APPROX(pm25, 0.95), 2) AS p95_pm25,
            ROUND(MAX(pm25), 2) AS max_pm25,
            ROUND(AVG(pm10), 2) AS mean_pm10,
            ROUND(AVG(no2), 2) AS mean_no2,
            ROUND(AVG(o3), 2) AS mean_o3
        FROM air_quality_hourly
        WHERE pm25 IS NOT NULL
        GROUP BY location_id, location_name, locality
        ORDER BY mean_pm25 DESC
        """
    )
    logger.info("=== [Analisis 1] Profil Polusi Antar-Lokasi (Top 15 PM2.5) ===")
    loc_profile_df.show(15, truncate=False)
    loc_profile = _rows_to_dicts(loc_profile_df)
    save_csv(loc_profile, out_dir / "01_location_profile.csv")

    # 2a. Pola temporal polutan per jam (WIB)
    hourly_pattern_df = spark.sql(
        f"""
        SELECT
            hour,
            COUNT(pm25) AS n_obs,
            ROUND(AVG(pm25), 2) AS mean_pm25,
            ROUND(PERCENTILE_APPROX(pm25, 0.50), 2) AS median_pm25,
            ROUND(PERCENTILE_APPROX(pm25, 0.95), 2) AS p95_pm25,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_primary} THEN 1.0 ELSE 0.0 END), 2) AS pct_exceed_epa
        FROM air_quality_hourly
        WHERE pm25 IS NOT NULL
        GROUP BY hour
        ORDER BY hour ASC
        """
    )
    logger.info("=== [Analisis 2a] Pola Temporal PM2.5 Per Jam (WIB) ===")
    hourly_pattern_df.show(24, truncate=False)
    hourly_pattern = _rows_to_dicts(hourly_pattern_df)
    save_csv(hourly_pattern, out_dir / "02a_temporal_hourly.csv")

    # 2b. Perbandingan hari kerja (Weekday) vs akhir pekan (Weekend)
    daytype_df = spark.sql(
        f"""
        SELECT
            day_type,
            COUNT(pm25) AS n_obs,
            ROUND(AVG(pm25), 2) AS mean_pm25,
            ROUND(STDDEV(pm25), 2) AS std_pm25,
            ROUND(PERCENTILE_APPROX(pm25, 0.50), 2) AS median_pm25,
            ROUND(PERCENTILE_APPROX(pm25, 0.95), 2) AS p95_pm25,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_primary} THEN 1.0 ELSE 0.0 END), 2) AS pct_exceed_epa
        FROM air_quality_hourly
        WHERE pm25 IS NOT NULL
        GROUP BY day_type
        ORDER BY day_type
        """
    )
    logger.info("=== [Analisis 2b] Perbandingan Polusi Hari Kerja vs Akhir Pekan ===")
    daytype_df.show(truncate=False)
    daytype_pattern = _rows_to_dicts(daytype_df)
    save_csv(daytype_pattern, out_dir / "02b_weekday_vs_weekend.csv")

    # 3a. Pelanggaran ambang batas aman per lokasi
    exceed_loc_df = spark.sql(
        f"""
        SELECT
            location_id,
            location_name,
            locality,
            COUNT(pm25) AS n_hours,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_who} THEN 1.0 ELSE 0.0 END), 2) AS pct_exceed_who_15,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_primary} THEN 1.0 ELSE 0.0 END), 2) AS pct_exceed_epa_35_4,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_klhk} THEN 1.0 ELSE 0.0 END), 2) AS pct_exceed_klhk_55_4
        FROM air_quality_hourly
        WHERE pm25 IS NOT NULL
        GROUP BY location_id, location_name, locality
        ORDER BY pct_exceed_epa_35_4 DESC
        """
    )
    logger.info("=== [Analisis 3a] Persentase Pelanggaran Ambang Batas PM2.5 per Lokasi ===")
    exceed_loc_df.show(15, truncate=False)
    exceed_loc = _rows_to_dicts(exceed_loc_df)
    save_csv(exceed_loc, out_dir / "03a_exceedance_by_location.csv")

    # 3b. Pelanggaran ambang batas aman per bulan (tren musiman)
    exceed_month_df = spark.sql(
        f"""
        SELECT
            year_month,
            COUNT(pm25) AS n_hours,
            ROUND(AVG(pm25), 2) AS mean_pm25,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_who} THEN 1.0 ELSE 0.0 END), 2) AS pct_exceed_who_15,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_primary} THEN 1.0 ELSE 0.0 END), 2) AS pct_exceed_epa_35_4,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_klhk} THEN 1.0 ELSE 0.0 END), 2) AS pct_exceed_klhk_55_4
        FROM air_quality_hourly
        WHERE pm25 IS NOT NULL
        GROUP BY year_month
        ORDER BY year_month ASC
        """
    )
    logger.info("=== [Analisis 3b] Persentase Pelanggaran Ambang Batas PM2.5 per Bulan ===")
    exceed_month_df.show(36, truncate=False)
    exceed_month = _rows_to_dicts(exceed_month_df)
    save_csv(exceed_month, out_dir / "03b_exceedance_by_month.csv")

    # 4. Korelasi antar-polutan (serta korelasi PM2.5 dengan fitur temporal & spasial)
    corr_rows: list[dict[str, Any]] = []
    available_pollutants = [
        p for p in pollutants if df.filter(F.col(p).isNotNull()).limit(30).count() >= 30
    ]

    for i, p1 in enumerate(available_pollutants):
        for p2 in available_pollutants[i + 1 :]:
            pair_df = df.select(p1, p2).dropna()
            n_pairs = pair_df.count()
            if n_pairs >= 30:
                corr_val = pair_df.stat.corr(p1, p2)
                if corr_val is not None and corr_val == corr_val:  # bukan NaN
                    corr_rows.append(
                        {
                            "pair_type": "pollutant_vs_pollutant",
                            "var_a": p1,
                            "var_b": p2,
                            "n_pairs": n_pairs,
                            "pearson_r": round(float(corr_val), 4),
                        }
                    )

    context_cols = ["hour", "month", "is_weekend", "latitude", "longitude"]
    for col_name in context_cols:
        pair_df = df.select("pm25", F.col(col_name).cast("double").alias(col_name)).dropna()
        n_pairs = pair_df.count()
        if n_pairs >= 30:
            corr_val = pair_df.stat.corr("pm25", col_name)
            if corr_val is not None and corr_val == corr_val:
                corr_rows.append(
                    {
                        "pair_type": "pm25_vs_context",
                        "var_a": "pm25",
                        "var_b": col_name,
                        "n_pairs": n_pairs,
                        "pearson_r": round(float(corr_val), 4),
                    }
                )

    logger.info("=== [Analisis 4] Korelasi Antar-Polutan & Konteks Spatio-Temporal ===")
    for row in corr_rows:
        logger.info(
            "  %-22s | %-6s vs %-10s | n=%6d | r=%+.4f",
            row["pair_type"],
            row["var_a"],
            row["var_b"],
            row["n_pairs"],
            row["pearson_r"],
        )
    save_csv(corr_rows, out_dir / "04_correlations.csv")

    overall_row = spark.sql(
        f"""
        SELECT
            COUNT(pm25) AS total_pm25_hours,
            COUNT(DISTINCT location_id) AS total_locations,
            ROUND(AVG(pm25), 2) AS overall_mean_pm25,
            ROUND(PERCENTILE_APPROX(pm25, 0.95), 2) AS overall_p95_pm25,
            ROUND(100.0 * AVG(CASE WHEN pm25 > {thr_primary} THEN 1.0 ELSE 0.0 END), 2) AS overall_pct_exceed_epa
        FROM air_quality_hourly
        WHERE pm25 IS NOT NULL
        """
    ).collect()[0].asDict()

    summary = {
        "thresholds": cfg["thresholds"],
        "overall_summary": overall_row,
        "location_profile": loc_profile,
        "temporal_hourly": hourly_pattern,
        "weekday_vs_weekend": daytype_pattern,
        "exceedance_by_location": exceed_loc,
        "exceedance_by_month": exceed_month,
        "correlations": corr_rows,
    }
    summary_path = out_dir / "analytics_summary.json"
    save_json(summary, summary_path)
    logger.info("Seluruh ringkasan analisis tersimpan di %s", summary_path)

    df.unpersist()
    return summary


def main() -> None:
    cfg = load_config()
    spark = create_spark_session(cfg, app_name="AirQualityExploratoryAnalytics")
    try:
        run_exploratory_analytics(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
