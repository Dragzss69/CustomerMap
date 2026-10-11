import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Model hasil training_v2.py (dicari di CustomerMap/Model/, Menu/, App/, CustomerMap/)
MODEL_FILENAME = "kmeans_model_v2.pkl"

# Satuan kolom order_total pada dataset training.
#   1.0    -> order_total dalam Rupiah
#   1000.0 -> order_total dalam ribuan Rupiah
# Input Monetary (Rp) di aplikasi dibagi nilai ini sebelum masuk ke model,
# sehingga skalanya sama persis dengan data saat training.
ORDER_TOTAL_UNIT = 1000.0

CLUSTER_ICONS = {
    "Pelanggan Loyal": "🏆",
    "Pelanggan Reguler": "🙂",
    "Pelanggan Pasif": "😴",
}

CLUSTER_DESC = {
    "Pelanggan Loyal": "Sering memesan dengan total belanja tinggi dan masih aktif. "
                       "Pertahankan dengan program loyalitas, poin reward, atau akses promo eksklusif.",
    "Pelanggan Reguler": "Memesan dengan frekuensi dan nilai belanja sedang. "
                         "Dorong naik menjadi loyal lewat voucher bundling atau diskon pembelian berikutnya.",
    "Pelanggan Pasif": "Jarang memesan atau sudah lama tidak aktif. "
                       "Lakukan reaktivasi dengan voucher comeback, notifikasi personal, atau survei alasan berhenti.",
}

FEATURE_NAMES = {
    "sum_order_total": "Total Belanja",
    "count_order_total": "Jumlah Order",
    "recency_days": "Recency (hari sejak order terakhir)",
}

SEGMENT_ORDER = ("Pelanggan Reguler", "Pelanggan Pasif", "Pelanggan Loyal")


# ----------------------------------------------------------------------------
# Memuat model
# ----------------------------------------------------------------------------
def model_folders() -> list[Path]:
    here = Path(__file__).resolve().parent
    root = here.parent.parent                      # CustomerMap/
    return [root / "Model", here, here.parent, root]


def find_model_path(filename: str = MODEL_FILENAME):
    for folder in model_folders():
        path = folder / filename
        if path.exists():
            return path
    return None


@st.cache_resource
def _read_artifact(path_str: str, mtime: float) -> dict:
    with open(path_str, "rb") as f:
        return pickle.load(f)


def validate_artifact(art: dict) -> None:
    """Pastikan .pkl adalah hasil training_v2.py."""
    kurang = [k for k in ("model", "scaler", "features", "preprocess", "cluster_labels") if k not in art]
    if kurang:
        raise ValueError(
            "File model bukan hasil training_v2.py (key tidak ada: " + ", ".join(kurang) + "). "
            "Jalankan training_v2.py lalu muat ulang halaman."
        )


def load_model_artifact(filename: str = MODEL_FILENAME) -> dict:
    """Memuat .pkl (model, scaler, fitur, label, metrik). Dimuat ulang otomatis jika file diganti."""
    path = find_model_path(filename)
    if path is None:
        dicari = "\n".join(f"- {f}" for f in model_folders())
        raise FileNotFoundError(
            f"{filename} tidak ditemukan. Jalankan training_v2.py, lalu pastikan file ada di salah satu folder:\n{dicari}"
        )
    art = _read_artifact(str(path), path.stat().st_mtime)
    validate_artifact(art)
    return art


# ----------------------------------------------------------------------------
# Preprocessing (sama dengan training_v2.py)
# ----------------------------------------------------------------------------
def to_model_space(art: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Nilai asli -> nilai yang masuk ke scaler (log1p untuk fitur ber-log)."""
    out = df.copy()
    for f in art["preprocess"]["log_features"]:
        out[f] = np.log1p(out[f])
    return out


def from_model_space(art: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Kebalikan to_model_space."""
    out = df.copy()
    for f in art["preprocess"]["log_features"]:
        out[f] = np.expm1(out[f])
    return out


def feature_ranges(art: dict):
    """Rentang data training (min, max) per fitur dalam satuan asli."""
    sc, feats = art["scaler"], art["features"]
    lo = from_model_space(art, pd.DataFrame([sc.data_min_], columns=feats)).iloc[0].to_dict()
    hi = from_model_space(art, pd.DataFrame([sc.data_max_], columns=feats)).iloc[0].to_dict()
    return lo, hi


def centers_original(art: dict) -> pd.DataFrame:
    """Centroid dalam satuan asli."""
    c = pd.DataFrame(art["scaler"].inverse_transform(art["model"].cluster_centers_), columns=art["features"])
    return from_model_space(art, c)


def predict(art: dict, total: float, count: int, last_date: date, days_since=None):
    """
    Alur sama seperti training_v2.py: fitur (satuan dataset) -> log1p -> MinMaxScaler -> K-Means.
      total      : total belanja dalam Rupiah (dikonversi ke satuan dataset)
      count      : jumlah order
      last_date  : tanggal order terakhir
      days_since : jumlah hari sejak order terakhir; jika None dihitung dari last_date
                   terhadap tanggal acuan training (snapshot_date)
    """
    if days_since is None:
        snap = pd.Timestamp(art["preprocess"]["snapshot_date"])
        days_since = (snap - pd.Timestamp(last_date)).days

    values = {
        "sum_order_total": float(total) / ORDER_TOTAL_UNIT,
        "count_order_total": float(count),
        "recency_days": float(days_since),
    }
    x_raw = pd.DataFrame([[values[f] for f in art["features"]]], columns=art["features"])
    x_scaled = art["scaler"].transform(to_model_space(art, x_raw))
    km = art["model"]

    cluster = int(km.predict(x_scaled)[0])
    dists = np.linalg.norm(km.cluster_centers_ - x_scaled, axis=1)
    return cluster, dists, x_raw, x_scaled


# ----------------------------------------------------------------------------
# Diagnosis
# ----------------------------------------------------------------------------
def feature_influence(art: dict) -> pd.DataFrame:
    """Selisih centroid tertinggi-terendah per fitur (skala 0-1): fitur berselisih besar paling menentukan."""
    centers = art["model"].cluster_centers_
    spread = centers.max(axis=0) - centers.min(axis=0)
    share = spread / spread.sum() * 100 if spread.sum() else spread
    return pd.DataFrame(
        {"Selisih antar centroid (0-1)": spread, "Pengaruh (%)": share},
        index=[FEATURE_NAMES.get(f, f) for f in art["features"]],
    )


def feature_sensitivity(art: dict, x_raw: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Ubah SATU fitur ke batas min/maks data training (fitur lain tetap), lihat apakah cluster berubah."""
    scaler, km = art["scaler"], art["model"]
    lo, hi = feature_ranges(art)
    base = int(km.predict(scaler.transform(to_model_space(art, x_raw)))[0])
    rows = []
    for f in art["features"]:
        out = {"Fitur diubah": FEATURE_NAMES.get(f, f)}
        for nama, nilai in (("ke minimum", lo[f]), ("ke maksimum", hi[f])):
            x = x_raw.copy()
            x[f] = nilai
            c = int(km.predict(scaler.transform(to_model_space(art, x)))[0])
            out[nama] = labels.get(c, f"Cluster {c}") + ("" if c == base else "  ← berubah")
        rows.append(out)
    return pd.DataFrame(rows).set_index("Fitur diubah")


# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
def page_simulator(art: dict = None):
    st.title("Simulator Prediksi Segmen Pelanggan")

    try:
        if art is not None:
            try:
                validate_artifact(art)
            except ValueError:
                art = None               # art kiriman app.py bukan model v2 -> abaikan, muat v2 sendiri
        if art is None:
            art = load_model_artifact()
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()
        return

    m = art.get("metrics", {})
    info = "Model ini digunakan untuk memprediksi dan mengelompokkan segmen perilaku pelanggan berdasarkan analisis pola transaksi RFM (Recency, Frequency, Monetary)."
    st.caption(info)

    feats = art["features"]
    cluster_labels = art["cluster_labels"]
    snap = pd.Timestamp(art["preprocess"]["snapshot_date"])
    dmin, dmax = feature_ranges(art)

    col_in, col_out = st.columns([1, 1.2], gap="large")

    with col_in:
        st.subheader("Input Data Pelanggan")

        days_since = None
        with st.form("sim_form"):
            count = st.number_input(
                "Frekuensi: berapa kali pelanggan ini memesan sejauh ini?",
                min_value=1,
                value=None,
                placeholder="Masukkan jumlah order (misal: 5)",
                step=1,
                help=f"Rentang data training: {dmin['count_order_total']:.0f} – {dmax['count_order_total']:.0f}",
            )
            total = st.number_input(
                "Monetary: total akumulasi belanja pelanggan ini (Rp)",
                min_value=0.0,
                value=None,
                placeholder="Masukkan nominal belanja (misal: 150000)",
                step=10000.0,
                format="%.0f",
                help=(
                    "Rentang data training: Rp "
                    f"{dmin['sum_order_total'] * ORDER_TOTAL_UNIT:,.0f} – "
                    f"Rp {dmax['sum_order_total'] * ORDER_TOTAL_UNIT:,.0f}"
                ),
            )
            last_date = st.date_input(
                "Recency: tanggal order terakhir",
                value=snap.date())
                
            submitted = st.form_submit_button("Prediksi Segmen", type="primary", width="stretch")

    with col_out:
        st.subheader("Hasil Prediksi")
        if not submitted:
            st.info("Isi data di sebelah kiri lalu klik **Prediksi Segmen**.")
            return

        cluster, dists, x_raw, x_scaled = predict(art, total, count, last_date, days_since)

        label = cluster_labels.get(cluster, f"Cluster {cluster}")
        st.success(f"{CLUSTER_ICONS.get(label, '')} **{label}**  (cluster {cluster})")
        st.write(CLUSTER_DESC.get(label, ""))

        inv = 1.0 / (dists + 1e-9)
        closeness = inv / inv.sum()
        clusters_by_label = {lbl: c_id for c_id, lbl in cluster_labels.items()}

        cols = st.columns(len(SEGMENT_ORDER))
        for col, lbl in zip(cols, SEGMENT_ORDER):
            c = clusters_by_label.get(lbl)
            if c is None:
                col.metric(lbl, "-")
                continue
            col.metric(lbl, f"{closeness[c] * 100:.1f}%",
                       help=f"Jarak ke centroid cluster {c}: {dists[c]:.4f}")

        chart_data = [
            {"Segmen": lbl, "Kemiripan (%)": closeness[clusters_by_label[lbl]] * 100.0}
            for lbl in SEGMENT_ORDER if lbl in clusters_by_label
        ]
        st.bar_chart(pd.DataFrame(chart_data).set_index("Segmen"))

        eps = 1e-9
        out_of_range = [
            FEATURE_NAMES.get(f, f)
            for f, v in zip(feats, x_raw.iloc[0])
            if v < dmin[f] - eps or v > dmax[f] + eps
        ]
        if out_of_range:
            st.warning("Nilai berikut berada di luar rentang data training, hasil prediksi "
                       "kurang dapat diandalkan: " + ", ".join(out_of_range))

if __name__ == "__main__":               # dijalankan langsung: streamlit run prediksi.py
    st.set_page_config(page_title="Simulator Prediksi", page_icon="🔮", layout="wide")
    page_simulator()