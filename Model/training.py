"""
Training K-Means VERSI ALTERNATIF (v2)
======================================
Tujuan: membuat cluster yang ditentukan oleh ketiga fitur secara seimbang, bukan
hampir seluruhnya oleh tanggal order terakhir seperti pada model RapidMiner (v1).

Perbedaan dari training.py (v1):
  1. Recency  : "hari sejak order terakhir" terhadap tanggal acuan (SNAPSHOT_DATE,
                sama dengan RapidMiner), bukan hari dalam bulan (1-31).
  1b. Jendela : RFM dihitung dari order dalam WINDOW_DAYS (30) hari terakhir saja, jadi
                Frequency/Monetary = aktivitas bulanan dan Recency bernilai 1..30.
  2. Skewness : sum_order_total dan count_order_total di-log (log1p) sebelum normalisasi.
  3. Outlier  : dideteksi setelah transformasi dan normalisasi sementara, sehingga ketiga
                fitur berbobot sama (v1: pada skala asli, didominasi total belanja).
  4. Label    : nama segmen (Loyal/Reguler/Pasif) ditentukan otomatis dari centroid
                dan disimpan di dalam .pkl, jadi tidak perlu mapping manual.

Output (tidak menimpa model v1):
  kmeans_model_v2.pkl, scaler_v2.pkl
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
ID_COL = "customer_id"
DATE_COL = "order_date"
VALUE_COL = "order_total"

RANDOM_SEED = 2001
N_CLUSTERS = 3
N_INIT = 20
MAX_ITER = 100

REMOVE_OUTLIERS = True      # False -> lewati deteksi outlier
OUTLIER_NEIGHBORS = 20
OUTLIER_COUNT = 100

# Tanggal acuan recency. Sama dengan RapidMiner: date_parse_custom("2025-10-30", "yyyy-MM-dd").
# Isi None untuk memakai tanggal order terakhir di dataset.
SNAPSHOT_DATE = "2025-10-30"

# Jendela RFM bulanan: hanya order dalam WINDOW_DAYS hari terakhir (termasuk hari SNAPSHOT_DATE) yang dipakai.
#   Frequency = jumlah order di jendela, Monetary = total belanja di jendela,
#   Recency   = 1..WINDOW_DAYS (1 = order terakhir pada hari acuan, WINDOW_DAYS = hari pertama jendela).
# Isi None untuk memakai seluruh riwayat order (perilaku lama).
WINDOW_DAYS = 30

FEATURES = ["sum_order_total", "count_order_total", "recency_days"]
LOG_FEATURES = ["sum_order_total", "count_order_total"]   # di-log1p sebelum normalisasi

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "Data" / "food_delivery_orders_dataset-selected-columns.csv"
MODEL_PATH = Path(__file__).resolve().parent / "kmeans_model_v2.pkl"
SCALER_PATH = Path(__file__).resolve().parent / "scaler_v2.pkl"


# ----------------------------------------------------------------------------
# Fungsi tahapan
# ----------------------------------------------------------------------------
def build_customer_features(df: pd.DataFrame):
    """Filter jendela + aggregate per customer + recency. Mengembalikan (fitur, snapshot, window_start)."""
    df = df[[ID_COL, DATE_COL, VALUE_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if SNAPSHOT_DATE is not None:
        snapshot = pd.Timestamp(SNAPSHOT_DATE)
    else:
        snapshot = df[DATE_COL].max().normalize()
    end_excl = snapshot + pd.Timedelta(days=1)          # batas akhir (tidak termasuk)

    window_start = None
    if WINDOW_DAYS:
        window_start = snapshot - pd.Timedelta(days=WINDOW_DAYS - 1)
        n_all = len(df)
        df = df[(df[DATE_COL] >= window_start) & (df[DATE_COL] < end_excl)]
        print(f"Jendela RFM {WINDOW_DAYS} hari: {window_start.date()} s/d {snapshot.date()} "
              f"-> {len(df)} dari {n_all} order dipakai")
        if df.empty:
            raise ValueError("Tidak ada order dalam jendela. Periksa SNAPSHOT_DATE dan WINDOW_DAYS.")

    agg = df.groupby(ID_COL).agg(
        sum_order_total=(VALUE_COL, "sum"),
        count_order_total=(VALUE_COL, "count"),
        last_order=(DATE_COL, "max"),
    )
    if WINDOW_DAYS:
        # 1 = order terakhir di hari acuan ... WINDOW_DAYS = hari pertama jendela (bulat ke atas)
        agg["recency_days"] = np.ceil((end_excl - agg["last_order"]).dt.total_seconds() / 86400.0)
    else:
        agg["recency_days"] = (snapshot - agg["last_order"]).dt.total_seconds() / 86400.0
    return agg[FEATURES], snapshot, window_start


def apply_log(feats: pd.DataFrame) -> pd.DataFrame:
    out = feats.copy()
    for f in LOG_FEATURES:
        out[f] = np.log1p(out[f])
    return out


def detect_outliers_distance(X: np.ndarray, k: int, n_outliers: int) -> np.ndarray:
    """Skor = jarak Euclidean ke tetangga ke-k; n_outliers skor terbesar = outlier."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    distances, _ = nn.kneighbors(X)
    score = distances[:, k]
    mask = np.zeros(len(X), dtype=bool)
    mask[np.argsort(score)[-n_outliers:]] = True
    return mask


def avg_within_centroid_distance(X, labels, centers) -> dict:
    per_cluster = {}
    for c in range(len(centers)):
        pts = X[labels == c]
        per_cluster[c] = float(np.linalg.norm(pts - centers[c], axis=1).mean()) if len(pts) else np.nan
    overall = float(np.linalg.norm(X - centers[labels], axis=1).mean())
    return {"per_cluster": per_cluster, "overall": overall}


def assign_labels(centers_scaled: np.ndarray) -> dict:
    """
    Skor nilai pelanggan = belanja + frekuensi - recency (recency besar = lama tidak order).
    Skor tertinggi = Loyal, tengah = Reguler, terendah = Pasif.
    """
    i_sum, i_cnt, i_rec = (FEATURES.index(f) for f in ("sum_order_total", "count_order_total", "recency_days"))
    score = centers_scaled[:, i_sum] + centers_scaled[:, i_cnt] - centers_scaled[:, i_rec]
    order = np.argsort(score)                          # rendah -> tinggi
    names = ["Pelanggan Pasif", "Pelanggan Reguler", "Pelanggan Loyal"]
    return {int(c): names[rank] for rank, c in enumerate(order)}


def centers_original_units(scaler: MinMaxScaler, centers_scaled: np.ndarray) -> pd.DataFrame:
    """Centroid dikembalikan ke satuan asli (kebalikan normalisasi dan log)."""
    df = pd.DataFrame(scaler.inverse_transform(centers_scaled), columns=FEATURES)
    for f in LOG_FEATURES:
        df[f] = np.expm1(df[f])
    return df


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(input_path: Path, model_path: Path):
    raw = pd.read_csv(input_path)
    print(f"Data awal: {raw.shape[0]} baris")

    customers, snapshot, window_start = build_customer_features(raw)
    print(f"Jumlah customer: {len(customers)} | tanggal acuan recency (snapshot): {snapshot.date()}")

    if (customers["recency_days"] < 0).any():
        print(f"PERINGATAN: {(customers['recency_days'] < 0).sum()} customer punya order SETELAH snapshot "
              f"({snapshot.date()}), recency bernilai negatif. Periksa SNAPSHOT_DATE.")
    print(f"Rentang recency_days: {customers['recency_days'].min():.2f} s/d {customers['recency_days'].max():.2f}")

    X_log = apply_log(customers)

    # Outlier (pada data ter-transformasi + normalisasi sementara)
    outliers_removed = REMOVE_OUTLIERS
    if REMOVE_OUTLIERS and len(customers) <= max(OUTLIER_NEIGHBORS + 1, OUTLIER_COUNT):
        print(f"PERINGATAN: customer ({len(customers)}) terlalu sedikit untuk deteksi outlier "
              f"(k={OUTLIER_NEIGHBORS}, outlier={OUTLIER_COUNT}). Deteksi outlier dilewati; "
              "kecilkan OUTLIER_COUNT jika ingin tetap membuang outlier.")
        outliers_removed = False
    if outliers_removed:
        tmp_scaled = MinMaxScaler().fit_transform(X_log)
        outlier_mask = detect_outliers_distance(tmp_scaled, OUTLIER_NEIGHBORS, OUTLIER_COUNT)
    else:
        outlier_mask = np.zeros(len(customers), dtype=bool)
    print(f"Outlier dibuang: {outlier_mask.sum()} | customer dianalisis: {(~outlier_mask).sum()}")
    clean = X_log[~outlier_mask]

    # Normalize (dilatih dari data bersih)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train = scaler.fit_transform(clean)

    # K-Means
    kmeans = KMeans(
        n_clusters=N_CLUSTERS, init="k-means++", n_init=N_INIT,
        max_iter=MAX_ITER, random_state=RANDOM_SEED,
    )
    labels = kmeans.fit_predict(X_train)
    centers = kmeans.cluster_centers_
    cluster_labels = assign_labels(centers)

    # Performance
    perf = avg_within_centroid_distance(X_train, labels, centers)
    sil = float(silhouette_score(X_train, labels))
    dbi = float(davies_bouldin_score(X_train, labels))
    print("\n=== Performance ===")
    print(f"Avg. within centroid distance : {perf['overall']:.4f}")
    print(f"Silhouette                    : {sil:.4f}")
    print(f"Davies-Bouldin                : {dbi:.4f}")
    print(f"Inertia (SSE)                 : {kmeans.inertia_:.4f}")

    # Diagnosis pengaruh fitur
    spread = centers.max(axis=0) - centers.min(axis=0)
    share = spread / spread.sum() * 100
    print("\n=== Pengaruh tiap fitur terhadap pembagian cluster ===")
    for f, sp, sh in zip(FEATURES, spread, share):
        print(f"  {f:<20} selisih centroid {sp:.3f} | pengaruh {sh:5.1f}%")
    if share.max() > 70:
        print("  PERINGATAN: satu fitur masih mendominasi (>70%).")

    # Profil cluster
    prof = centers_original_units(scaler, centers)
    prof.insert(0, "segmen", [cluster_labels[i] for i in range(len(prof))])
    prof["jumlah"] = np.bincount(labels, minlength=N_CLUSTERS)
    print("\n=== Centroid (satuan asli) ===")
    print(prof.round(2).to_string())

    # Simpan model
    artifact = {
        "model": kmeans,
        "scaler": scaler,
        "features": FEATURES,
        "cluster_labels": cluster_labels,
        "use_rapidminer_day_of_month": False,
        "preprocess": {
            "version": 2,
            "log_features": LOG_FEATURES,
            "recency": "days_in_window" if WINDOW_DAYS else "days_since_snapshot",
            "snapshot_date": snapshot,
            "window_days": WINDOW_DAYS,
            "window_start": window_start,
            "recency_offset": 1 if WINDOW_DAYS else 0,
        },
        "config": {
            "n_clusters": N_CLUSTERS,
            "n_init": N_INIT,
            "max_iter": MAX_ITER,
            "random_seed": RANDOM_SEED,
            "remove_outliers": outliers_removed,
            "outlier_neighbors": OUTLIER_NEIGHBORS,
            "outlier_count": OUTLIER_COUNT,
        },
        "metrics": {
            "avg_within_centroid_distance": perf,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "inertia": float(kmeans.inertia_),
            "feature_influence_pct": dict(zip(FEATURES, map(float, share))),
        },
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModel disimpan ke : {model_path}")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler disimpan ke: {SCALER_PATH}")


if __name__ == "__main__":
    main(INPUT_PATH, MODEL_PATH)