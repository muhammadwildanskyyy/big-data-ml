#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin:${PATH}"

RAW_LOCAL="data/raw/openaq_indonesia_hourly.jsonl"
META_LOCAL="data/raw/ingestion_metadata.json"
HDFS_BASE="/air_quality"
HDFS_RAW="${HDFS_BASE}/raw"
HDFS_PROC="${HDFS_BASE}/processed"
HDFS_MODELS="${HDFS_BASE}/models"
AUDIT_OUT="outputs/hdfs/hdfs_verification.txt"

if [ ! -f "${RAW_LOCAL}" ]; then
  echo "ERROR: File mentah tidak ditemukan di ${RAW_LOCAL}. Jalankan ingestion terlebih dahulu." >&2
  exit 1
fi

mkdir -p "$(dirname "${AUDIT_OUT}")"

{
  echo "=== [1/4] Menunggu HDFS NameNode siap (keluar dari SafeMode) ==="
  docker exec namenode hdfs dfsadmin -safemode wait

  echo "=== [2/4] Membuat struktur direktori HDFS ==="
  docker exec namenode hdfs dfs -mkdir -p "${HDFS_RAW}" "${HDFS_PROC}" "${HDFS_MODELS}"

  echo "=== [3/4] Mengunggah dataset mentah OpenAQ ke HDFS ==="
  docker exec namenode hdfs dfs -put -f /opt/data/raw/openaq_indonesia_hourly.jsonl "${HDFS_RAW}/"
  if [ -f "${META_LOCAL}" ]; then
    docker exec namenode hdfs dfs -put -f /opt/data/raw/ingestion_metadata.json "${HDFS_RAW}/"
  fi

  echo "=== [4/4] Verifikasi struktur & ukuran direktori HDFS ==="
  echo "--- hdfs dfs -ls -R ${HDFS_BASE} ---"
  docker exec namenode hdfs dfs -ls -R "${HDFS_BASE}"

  echo "--- hdfs dfs -du -h ${HDFS_RAW} ---"
  docker exec namenode hdfs dfs -du -h "${HDFS_RAW}"

  echo "--- hdfs dfs -count -h ${HDFS_RAW} ---"
  docker exec namenode hdfs dfs -count -h "${HDFS_RAW}"
} | tee "${AUDIT_OUT}"

echo "Bukti verifikasi HDFS tersimpan di ${AUDIT_OUT}"
