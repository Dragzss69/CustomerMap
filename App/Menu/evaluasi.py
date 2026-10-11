import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------------
# Konstanta
# ----------------------------------------------------------------------------
# Model hasil training_v2.py (dicari di CustomerMap/Model/, Menu/, App/, CustomerMap/)
MODEL_FILENAME = "kmeans_model_v2.pkl"
REQUIRED_COLS = ["customer_id", "order_date", "order_total"]

FEATURE_NAMES = {
    "sum_order_total": "Total Belanja",
    "count_order_total": "Jumlah Order",
    "recency_days": "Recency (hari sejak order terakhir)",
}
SEGMEN_COLORS = {
    "Pelanggan Loyal": "#3b82f6",
    "Pelanggan Reguler": "#2e9e5b",
    "Pelanggan Pasif": "#f59e0b",
}


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
def build_features(raw: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Aggregate per customer + recency terhadap tanggal acuan training."""
    df = raw[REQUIRED_COLS].copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_total"] = pd.to_numeric(df["order_total"], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLS)

    snapshot = pd.Timestamp(art["preprocess"]["snapshot_date"])
    agg = df.groupby("customer_id").agg(
        sum_order_total=("order_total", "sum"),
        count_order_total=("order_total", "count"),
        last_order=("order_date", "max"),
    )
    agg["recency_days"] = (snapshot - agg["last_order"]).dt.total_seconds() / 86400.0
    return agg[art["features"]]


def to_model_space(art: dict, df: pd.DataFrame) -> pd.DataFrame:
    """log1p untuk fitur ber-log (sebelum scaler)."""
    out = df.copy()
    for f in art["preprocess"]["log_features"]:
        if (out[f] < 0).any():
            raise ValueError(f"Fitur {FEATURE_NAMES.get(f, f)} tidak boleh bernilai negatif.")
        out[f] = np.log1p(out[f])
    return out


def centers_original(art: dict) -> pd.DataFrame:
    """Centroid dalam satuan asli (kebalikan normalisasi dan log)."""
    c = pd.DataFrame(art["scaler"].inverse_transform(art["model"].cluster_centers_), columns=art["features"])
    for f in art["preprocess"]["log_features"]:
        c[f] = np.expm1(c[f])
    return c


def detect_outliers(X: np.ndarray, k: int, n_outliers: int) -> np.ndarray:
    """Skor = jarak Euclidean ke tetangga ke-k; n_outliers skor terbesar = outlier."""
    distances, _ = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X).kneighbors(X)
    score = distances[:, k]
    mask = np.zeros(len(X), dtype=bool)
    mask[np.argsort(score)[-n_outliers:]] = True
    return mask


# ----------------------------------------------------------------------------
# Evaluasi
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Memproses data dengan model v2...")
def run_evaluation(file_bytes: bytes, remove_outlier: bool, model_sig: int, _art: dict) -> dict:
    """
    Alur sama seperti training_v2.py:
      agregasi -> log1p -> (outlier pada normalisasi sementara) -> scaler MODEL -> K-Means.model.predict
    model_sig hanya untuk membedakan cache antar model.
    """
    art = _art
    raw = pd.read_csv(io.BytesIO(file_bytes))
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing))

    features, km, scaler = art["features"], art["model"], art["scaler"]
    cluster_labels = art["cluster_labels"]
    cfg = art.get("config", {})

    feats = build_features(raw, art)
    if feats.empty:
        raise ValueError("Tidak ada data valid setelah pembersihan (cek format tanggal dan order_total).")
    X_log = to_model_space(art, feats)

    # Outlier: seperti training, dideteksi pada data ter-log yang dinormalisasi sementara
    k, n_out = cfg.get("outlier_neighbors", 20), cfg.get("outlier_count", 100)
    mask = np.zeros(len(feats), dtype=bool)
    outlier_skipped = False
    if remove_outlier:
        if len(feats) <= max(k + 1, n_out):
            outlier_skipped = True          # data terlalu sedikit untuk parameter outlier training
        else:
            mask = detect_outliers(MinMaxScaler().fit_transform(X_log), k, n_out)
    keep = ~mask
    if not keep.any():
        raise ValueError("Seluruh data terdeteksi sebagai outlier.")

    Xk = scaler.transform(X_log[keep])      # scaler hasil training (bukan fit ulang)
    labels = km.predict(Xk)
    centers = km.cluster_centers_
    dist = np.linalg.norm(Xk - centers[labels], axis=1)

    result = feats[keep].copy()
    result["cluster"] = labels
    result["Segmen"] = [cluster_labels.get(int(c), f"Cluster {int(c)}") for c in labels]
    result["jarak_ke_centroid"] = dist
    for i, f in enumerate(features):
        result[f"norm_{f}"] = Xk[:, i]

    n_found = len(np.unique(labels))
    metrics = {
        "n_rows": int(len(raw)),
        "n_customers": int(len(feats)),
        "n_outliers": int(mask.sum()),
        "n_analyzed": int(keep.sum()),
        "outlier_skipped": outlier_skipped,
        "avg_within": float(dist.mean()),
        "inertia": float((dist ** 2).sum()),
        "silhouette": None,
        "davies_bouldin": None,
        # persentase data di luar rentang 0-1 (di luar rentang data training) per fitur
        "out_of_range_pct": {
            f: float(((Xk[:, i] < 0) | (Xk[:, i] > 1)).mean() * 100) for i, f in enumerate(features)
        },
    }
    if n_found >= 2 and len(Xk) > n_found:
        sil = silhouette_samples(Xk, labels)
        result["silhouette"] = sil
        metrics["silhouette"] = float(sil.mean())
        metrics["davies_bouldin"] = float(davies_bouldin_score(Xk, labels))

    return {
        "raw_preview": raw.head(20),
        "result": result,
        "outliers": feats[mask].copy(),
        "metrics": metrics,
    }


def feature_influence(art: dict) -> pd.DataFrame:
    """Selisih centroid tertinggi-terendah per fitur (skala 0-1)."""
    centers = art["model"].cluster_centers_
    spread = centers.max(axis=0) - centers.min(axis=0)
    share = spread / spread.sum() * 100 if spread.sum() else spread
    return pd.DataFrame(
        {"Selisih antar centroid (0-1)": spread, "Pengaruh (%)": share},
        index=[FEATURE_NAMES.get(f, f) for f in art["features"]],
    )


# ----------------------------------------------------------------------------
# Tampilan halaman
# ----------------------------------------------------------------------------
def page_evaluasi(art: dict = None):
    st.title("Evaluasi Model K-Means")

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

    features = art["features"]
    km = art["model"]
    cluster_labels = art["cluster_labels"]
    cfg = art.get("config", {})
    tm = art.get("metrics", {})
    model_sig = hash(km.cluster_centers_.tobytes())

    # Inisialisasi session state untuk menyimpan hasil pemrosesan dataset
    if "eval_out" not in st.session_state:
        st.session_state.eval_out = None

    remove_outlier = cfg.get("remove_outliers", True)

    # 1. Tampilkan form unggah HANYA jika data belum diproses
    if st.session_state.eval_out is None:
        st.caption("Unggah dataset order, lalu data diproses dengan model dan scaler dari kmeans_model_v2.pkl.")
        up = st.file_uploader("Unggah dataset (CSV)", type=["csv"])

        if up is None:
            st.info("Unggah dataset untuk memulai evaluasi.")
            return

        try:
            # Proses data dan simpan ke session state
            out = run_evaluation(up.getvalue(), remove_outlier, model_sig, art)
            st.session_state.eval_out = out
            st.rerun()  # Rerun agar container unggah dataset langsung hilang
        except Exception as e:
            st.error(f"Gagal memproses dataset: {e}")
            return

    # 2. Ambil hasil evaluasi yang sudah tersimpan
    out = st.session_state.eval_out

    # (Opsional) Tombol untuk mengganti / unggah ulang dataset jika dibutuhkan
    col_info, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("🔄 Unggah Ulang", use_container_width=True):
            st.session_state.eval_out = None
            st.rerun()

    result, m = out["result"], out["metrics"]

    # 3. Tampilkan tab hasil dataset dan metrik evaluasi
    tab_data, tab_metric = st.tabs(["📄 Hasil Dataset", "📈 Metrik Evaluasi"])

    # ---------------- Tab 1: hasil dataset ----------------
    with tab_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baris order", f"{m['n_rows']:,}")
        c2.metric("Jumlah customer", f"{m['n_customers']:,}")
        c3.metric("Outlier dibuang", f"{m['n_outliers']:,}")
        c4.metric("Customer dianalisis", f"{m['n_analyzed']:,}")

        st.subheader("Hasil clustering per customer")
        show_cols = features + ["cluster", "Segmen", "jarak_ke_centroid"]
        view = result[show_cols].rename(columns=FEATURE_NAMES)
        st.dataframe(view.style.format({
            FEATURE_NAMES["sum_order_total"]: "{:,.0f}",
            FEATURE_NAMES["count_order_total"]: "{:,.0f}",
            FEATURE_NAMES["recency_days"]: "{:,.1f}",
            "jarak_ke_centroid": "{:.4f}",
        }), width="stretch")
        st.download_button(
            "⬇️ Unduh hasil (CSV)",
            result[show_cols].to_csv().encode("utf-8"),
            file_name="hasil_clustering_v2.csv",
            mime="text/csv",
        )

        names = [FEATURE_NAMES[f] for f in features]

        # Gunakan data ternormalisasi (0–1) dan centroid ternormalisasi.
        pts = result[[f"norm_{f}" for f in features]].values
        ctr = km.cluster_centers_
        plot_df = pd.DataFrame(pts, columns=names, index=result.index)
        plot_df["Segmen"] = result["Segmen"]
        plot_df["customer_id"] = result.index.astype(str)
        ctr_labels = [cluster_labels.get(i, f"Cluster {i}") for i in range(len(ctr))]

        st.subheader("Scatter plot clustering")
        c1, c2 = st.columns(2)
        with c1:
            x_name = st.selectbox("Sumbu X", names, index=1)
        with c2:
            y_name = st.selectbox("Sumbu Y", names, index=0)

        fig = px.scatter(
            plot_df,
            x=x_name,
            y=y_name,
            color="Segmen",
            color_discrete_map=SEGMEN_COLORS,
            hover_name="customer_id",
            opacity=0.75,
        )
        xi, yi = names.index(x_name), names.index(y_name)
        fig.add_trace(go.Scatter(
            x=ctr[:, xi],
            y=ctr[:, yi],
            mode="markers+text",
            text=ctr_labels,
            textposition="top center",
            name="Centroid",
            marker=dict(symbol="x", size=16, color="black", line=dict(width=2)),
        ))

        fig.update_layout(height=560, margin=dict(t=20, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tanda ✕ hitam adalah centroid tiap cluster (dalam skala ternormalisasi 0–1).")

    # ---------------- Tab 2: metrik ----------------
    with tab_metric:
        ts = tm.get("silhouette")
        td = tm.get("davies_bouldin")
        ta = tm.get("avg_within_centroid_distance", {}).get("overall")

        if m["silhouette"] is None:
            st.warning("Silhouette dan Davies-Bouldin tidak dapat dihitung karena data hanya "
                       "masuk ke satu cluster atau terlalu sedikit.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Silhouette Score",
                  "-" if m["silhouette"] is None else f"{m['silhouette']:.4f}")
        c2.metric("Davies-Bouldin Index (DBI)",
                  "-" if m["davies_bouldin"] is None else f"{m['davies_bouldin']:.4f}")
        c3.metric("Avg. within centroid distance", f"{m['avg_within']:.4f}")
        c4.metric("Inertia (SSE)", f"{m['inertia']:.4f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Distribusi customer per segmen**")
            dist_df = result["Segmen"].value_counts().rename_axis("Segmen").reset_index(name="Jumlah")
            fig = px.bar(dist_df, x="Segmen", y="Jumlah", color="Segmen",
                         color_discrete_map=SEGMEN_COLORS, text="Jumlah")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        with right:
            if "silhouette" in result.columns:
                st.markdown("**Rata-rata silhouette & jarak ke centroid per segmen**")
                per = result.groupby("Segmen").agg(
                    Silhouette=("silhouette", "mean"),
                    Jarak_ke_centroid=("jarak_ke_centroid", "mean"),
                    Jumlah=("cluster", "size"),
                )
                st.dataframe(per.style.format({"Silhouette": "{:.4f}", "Jarak_ke_centroid": "{:.4f}"}),
                             width="stretch")

        st.markdown("**Pengaruh tiap fitur terhadap pembagian cluster (dari centroid model)**")
        infl = feature_influence(art)
        st.dataframe(infl.style.format({"Selisih antar centroid (0-1)": "{:.3f}",
                                        "Pengaruh (%)": "{:.1f}"}), width="stretch")
        st.caption("Pengaruh yang seimbang berarti ketiga fitur sama-sama menentukan segmen. "
                   "Jika satu fitur >70%, cluster praktis hanya dibedakan oleh fitur itu.")


if __name__ == "__main__":               # dijalankan langsung: streamlit run evaluasi.py
    st.set_page_config(page_title="Evaluasi Model", page_icon="📊", layout="wide")
    page_evaluasi()