#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin:${PATH}"

SPARK_SUBMIT="docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077"

echo "============================================================"
echo " PIPELINE BIG DATA KUALITAS UDARA INDONESIA (END-TO-END)"
echo "============================================================"

echo ""
echo "[TAHAP 1/6] Unggah & Verifikasi Dataset Mentah ke HDFS"
bash scripts/setup_hdfs.sh

echo ""
echo "[TAHAP 2/6] Verifikasi Pembacaan HDFS dari Apache Spark"
${SPARK_SUBMIT} /opt/app/src/storage/hdfs_manager.py

echo ""
echo "[TAHAP 3/6] Preprocessing & Pivot Long-to-Wide di Apache Spark"
${SPARK_SUBMIT} /opt/app/src/processing/preprocess.py

echo ""
echo "[TAHAP 4/6] Analisis Eksploratif (4 Analisis Spark SQL / DataFrame)"
${SPARK_SUBMIT} /opt/app/src/analytics/exploratory.py

echo ""
echo "[TAHAP 5/6] Pelatihan & Evaluasi Spark MLlib LinearSVC (Bebas Leakage)"
${SPARK_SUBMIT} /opt/app/src/ml/train_linear_svc.py

echo ""
echo "[TAHAP 6/6] Eksperimen Skala Data (25%, 50%, 100%) & Sinkronisasi MongoDB"
${SPARK_SUBMIT} /opt/app/src/benchmark/scale_experiment.py
${SPARK_SUBMIT} /opt/app/src/nosql/mongo_sink.py

echo ""
echo "============================================================"
echo " SELESAI! Seluruh log dan artefak audit tersimpan di outputs/"
echo "============================================================"
