# Proyek Akhir Big Data — Klasifikasi Kualitas Udara Indonesia (OpenAQ v3 + HDFS + Spark MLlib + MongoDB)

Implementasi end-to-end pemrosesan Big Data untuk analisis dan prediksi klasifikasi kualitas udara jam berikutnya (*Next-Hour Binary Classification*) di Indonesia menggunakan **OpenAQ API v3**, **HDFS**, **Apache Spark (SQL, DataFrame API, MLlib `LinearSVC`)**, dan **MongoDB**.

## Arsitektur Sistem

`OpenAQ API v3 (Indonesia)` → `JSONL Mentah + Metadata` → `HDFS (namenode/datanode)` → `Apache Spark Preprocessing (Long-to-Wide Parquet)` → `4 Analisis Eksploratif (Spark SQL)` → `Spark MLlib LinearSVC (Bebas Data Leakage)` → `Benchmark Skala (25%/50%/100%)` → `MongoDB (Spark Connector + PyMongo)`

## Prasyarat & Konfigurasi

1. Salin `.env.example` menjadi `.env` dan isi `OPENAQ_API_KEY` valid dari OpenAQ Explorer:
   ```bash
   cp .env.example .env
   ```
2. Seluruh parameter pipeline (rentang tanggal, batas fisik sensor, ambang batas PM2.5 `35.4 µg/m³`, fitur lag, split kronologis 80/20, dan skala benchmark) terpusat pada `config/settings.yaml`.

## Cara Menjalankan

### 1. Jalankan Cluster Docker (HDFS + Spark + MongoDB)
```bash
docker compose up -d --build
```
Antarmuka Web UI yang tersedia untuk dokumentasi/screenshot:
- **HDFS NameNode UI:** `http://localhost:9870`
- **Spark Master UI:** `http://localhost:8080`
- **Spark Worker UI:** `http://localhost:8081`

### 2. Tarik Data Asli Indonesia dari OpenAQ API v3 (>50.000 baris mentah)
Jalankan dari dalam container `spark-master` (agar tidak perlu instalasi dependensi Python di host):
```bash
docker exec spark-master python3 -m src.ingestion.fetch_indonesia
```
Output mentah tersimpan di `data/raw/openaq_indonesia_hourly.jsonl` beserta dokumentasi sumber & lisensi di `data/raw/ingestion_metadata.json`.

### 3. Jalankan Seluruh Pipeline End-to-End (HDFS → Spark → MLlib → Benchmark → MongoDB)
```bash
bash scripts/run_pipeline.sh
```

Atau jalankan per tahap jika ingin mengambil tangkapan layar terpisah untuk artikel ilmiah:
```bash
bash scripts/setup_hdfs.sh
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/app/src/storage/hdfs_manager.py
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/app/src/processing/preprocess.py
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/app/src/analytics/exploratory.py
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/app/src/ml/train_linear_svc.py
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/app/src/benchmark/scale_experiment.py
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/app/src/nosql/mongo_sink.py
```

## Artefak Audit untuk Artikel Ilmiah (`outputs/`)
- `data/raw/ingestion_metadata.json` — Endpoint, rentang tanggal, jumlah baris, provider, dan lisensi OpenAQ.
- `outputs/hdfs/` — Output `hdfs dfs -ls/-du/-count` & audit skema baca Spark dari `hdfs://`.
- `outputs/processing/` — Statistik baris awal, null, duplikasi, anomali sensor, dan hasil pivot.
- `outputs/analytics/` — CSV & JSON dari 4 analisis eksploratif (profil lokasi, pola jam/akhir pekan, pelanggaran ambang batas WHO/EPA/KLHK, korelasi).
- `outputs/ml/` — Metrik `LinearSVC` vs baseline persistensi, *confusion matrix*, koefisien fitur, dan sampel prediksi.
- `outputs/benchmark/` — Tabel waktu eksekusi & skor evaluasi pada skala 25%, 50%, dan 100%.
- `outputs/nosql/` — Bukti operasi tulis (Spark MongoDB Connector) & baca/query (Spark & PyMongo).
