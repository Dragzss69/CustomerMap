import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Model hasil training_v2.py (dicari di CustomerMap/Model/, Menu/, App/, CustomerMap/)
MODEL_FILENAME = "kmeans_model_v2.pkl"

# Satuan kolom order_total pada dataset training.
#   1.0    -> order_total dalam Rupiah
#   1000.0 -> order_total dalam ribuan Rupiah
# Input Monetary (Rp) di aplikasi dibagi nilai ini sebelum masuk ke model,
# sehingga skalanya sama persis dengan data saat training.
ORDER_TOTAL_UNIT = 1000.0

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

SEGMENT_ORDER = ("Pelanggan Loyal", "Pelanggan Reguler", "Pelanggan Pasif")

SEGMENT_COLORS = {
    "Pelanggan Loyal": "#3b82f6",
    "Pelanggan Reguler": "#2e9e5b",
    "Pelanggan Pasif": "#f59e0b",
}
CUSTOMER_COLOR = "#8b5cf6"

SEGMENT_ACTIONS = {
    "Pelanggan Loyal": ["Beri poin reward atau program loyalitas", "Tawarkan akses promo eksklusif",
                        "Ajak memberi ulasan atau referral"],
    "Pelanggan Reguler": ["Kirim voucher bundling", "Beri diskon untuk pembelian berikutnya",
                          "Dorong target order bulanan agar naik jadi loyal"],
    "Pelanggan Pasif": ["Kirim voucher comeback", "Kirim notifikasi personal (push atau WhatsApp)",
                        "Lakukan survei alasan berhenti"],
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


def check_window(art: dict) -> None:
    """Input recency 1..N hanya cocok dengan model yang dilatih pada jendela RFM N hari."""
    if not art["preprocess"].get("window_days"):
        raise ValueError(
            "Model ini dilatih pada seluruh riwayat order, bukan jendela RFM bulanan, sehingga tidak cocok "
            "dengan input recency 1–30. Jalankan training_v2.py versi terbaru (WINDOW_DAYS = 30)."
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
    check_window(art)
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
        pre = art["preprocess"]
        snap = pd.Timestamp(pre["snapshot_date"]).normalize()
        days_since = (snap - pd.Timestamp(last_date).normalize()).days + pre.get("recency_offset", 0)

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
# Tampilan (gaya dashboard)
# ----------------------------------------------------------------------------
CSS = """
<style>
.cm-badge{border-radius:14px;padding:16px 14px;text-align:center;color:#fff;margin:6px 0 4px 0}
.cm-badge .big{font-size:1.5rem;font-weight:800;line-height:1.25}
.cm-badge .sub{font-size:.8rem;opacity:.92;margin-top:2px}
.cm-h{font-weight:700;font-size:1.02rem;margin:2px 0 8px 0}
.cm-label{font-size:.85rem;opacity:.7}
.cm-value{font-size:1.85rem;font-weight:800;line-height:1.2;margin:2px 0 8px 0}
.cm-bar{height:8px;border-radius:6px;background:rgba(128,128,128,.25);overflow:hidden}
.cm-bar>div{height:100%;border-radius:6px}
.cm-note{font-size:.78rem;opacity:.7;margin-top:6px}
.cm-row{display:flex;align-items:center;gap:10px;padding:8px 4px;border-bottom:1px solid rgba(128,128,128,.2);font-size:.9rem}
.cm-row:last-child{border-bottom:none}
.cm-ico{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:800;color:#fff;flex:none}
.cm-seg{font-size:1.05rem;font-weight:800;margin-bottom:4px}
.cm-li{font-size:.88rem;margin:3px 0}
</style>
"""


def rupiah(v: float) -> str:
    return "Rp " + f"{v:,.0f}".replace(",", ".")


def kpi_card(label: str, value: str, pos: float = None, color: str = "", note: str = "") -> None:
    st.markdown(
        f'<div class="cm-label">{label}</div>'
        f'<div class="cm-value">{value}</div>',
        unsafe_allow_html=True,
    )


def decision_factors(art: dict, x_scaled: np.ndarray, cluster: int, dists: np.ndarray):
    """
    Fitur mana yang membuat pelanggan lebih dekat ke segmen terpilih dibanding segmen terdekat kedua.
    Kontribusi > 0 = mendukung segmen terpilih, < 0 = condong ke segmen lain.
    """
    centers = art["model"].cluster_centers_
    order = np.argsort(dists)
    runner = int(order[1]) if int(order[0]) == cluster else int(order[0])
    x = x_scaled[0]
    contrib = (x - centers[runner]) ** 2 - (x - centers[cluster]) ** 2
    df = pd.DataFrame({
        "Fitur": [FEATURE_NAMES.get(f, f) for f in art["features"]],
        "Kontribusi": contrib,
    }).sort_values("Kontribusi", ascending=False)
    return df, runner


def donut_chart(names, values, colors, center_text) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=names, values=values, hole=0.66, sort=False, direction="clockwise",
        marker=dict(colors=colors), textinfo="none",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
        annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=22, showarrow=False)],
    )
    return fig


def compare_chart(art: dict, x_scaled: np.ndarray, clusters_by_label: dict) -> go.Figure:
    centers = art["model"].cluster_centers_
    rows = []
    for i, f in enumerate(art["features"]):
        nama = FEATURE_NAMES.get(f, f)
        rows.append({"Fitur": nama, "Seri": "Pelanggan ini", "Nilai": float(np.clip(x_scaled[0][i], 0, 1))})
        for lbl in SEGMENT_ORDER:
            c = clusters_by_label.get(lbl)
            if c is not None:
                rows.append({"Fitur": nama, "Seri": lbl, "Nilai": float(centers[c][i])})
    seri = ["Pelanggan ini", *[s for s in SEGMENT_ORDER if s in clusters_by_label]]
    fig = px.bar(
        pd.DataFrame(rows), x="Nilai", y="Fitur", color="Seri", orientation="h", barmode="group",
        color_discrete_map={**SEGMENT_COLORS, "Pelanggan ini": CUSTOMER_COLOR},
        category_orders={"Seri": seri},
    )
    fig.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10), legend_title_text="",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        xaxis=dict(range=[0, 1.05]), yaxis_title=None,
    )
    return fig


def centroid_table(art: dict, cluster_labels: dict) -> pd.DataFrame:
    """Profil rata-rata tiap segmen dalam satuan yang mudah dibaca."""
    c = centers_original(art)
    out = pd.DataFrame({
        "Frekuensi (order)": c["count_order_total"],
        "Monetary (Rp)": c["sum_order_total"] * ORDER_TOTAL_UNIT,
        "Recency (hari)": c["recency_days"],
    })
    out.index = [cluster_labels.get(i, f"Cluster {i}") for i in range(len(out))]
    order = [s for s in SEGMENT_ORDER if s in out.index] + [i for i in out.index if i not in SEGMENT_ORDER]
    return out.loc[order]


def render_overview(art: dict, cluster_labels: dict, incomplete: bool = False) -> None:
    """Tampilan awal (sebelum prediksi): ringkasan tiap segmen."""
    prof = centroid_table(art, cluster_labels)
    st.markdown('<div class="cm-h">Ringkasan segmen pelanggan</div>', unsafe_allow_html=True)
    cols = st.columns(len(SEGMENT_ORDER))
    for col, lbl in zip(cols, SEGMENT_ORDER):
        with col, st.container(border=True):
            color = SEGMENT_COLORS[lbl]
            st.markdown(f'<div class="cm-seg" style="color:{color}">{lbl}</div>',
                        unsafe_allow_html=True)
            st.caption(CLUSTER_DESC[lbl])
    if incomplete:
        st.warning("Lengkapi dulu Frekuensi, Monetary, dan Recency, lalu klik **Prediksi Segmen**.")
    else:
        st.info("Isi data pelanggan di panel kiri lalu klik **Prediksi Segmen**.")


# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
def page_simulator(art: dict = None):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Simulator Prediksi Segmen Pelanggan")

    try:
        if art is not None:
            try:
                validate_artifact(art)
                check_window(art)
            except ValueError:
                art = None               # art kiriman app.py bukan model v2 -> abaikan, muat v2 sendiri
        if art is None:
            art = load_model_artifact()
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()
        return

    st.caption("Model ini digunakan untuk memprediksi dan mengelompokkan segmen perilaku pelanggan "
               "berdasarkan analisis pola transaksi RFM (Recency, Frequency, Monetary).")

    feats = art["features"]
    cluster_labels = art["cluster_labels"]
    clusters_by_label = {lbl: c_id for c_id, lbl in cluster_labels.items()}
    snap = pd.Timestamp(art["preprocess"]["snapshot_date"])
    W = int(art["preprocess"]["window_days"])
    wstart = art["preprocess"].get("window_start")
    wstart = pd.Timestamp(wstart) if wstart is not None else snap - pd.Timedelta(days=W - 1)
    dmin, dmax = feature_ranges(art)
    sig = hash(art["model"].cluster_centers_.tobytes())

    col_left, col_main = st.columns([1, 2.6], gap="large")

    # ---------------- Panel kiri: input ----------------
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="cm-h">Input Data Pelanggan</div>', unsafe_allow_html=True)
            with st.form("sim_form", border=False):
                count = st.number_input(
                    f"Frekuensi: berapa kali memesan dalam {W} hari terakhir?",
                    min_value=1,
                    value=None,
                    placeholder="5",
                    step=1,
                    help=f"Rentang data training: {dmin['count_order_total']:.0f} – {dmax['count_order_total']:.0f}",
                )
                total = st.number_input(
                    f"Monetary: total belanja dalam {W} hari terakhir (Rp)",
                    min_value=0.0,
                    value=None,
                    placeholder="150000",
                    step=10000.0,
                    format="%.0f",
                    help=("Rentang data training: "
                          f"{rupiah(dmin['sum_order_total'] * ORDER_TOTAL_UNIT)} – "
                          f"{rupiah(dmax['sum_order_total'] * ORDER_TOTAL_UNIT)}"),
                )
                recency = st.number_input(
                    f"Recency: berapa hari sejak order terakhir? (1–{W})",
                    min_value=1,
                    max_value=W,
                    value=None,
                    placeholder=f"1 - 30",
                    step=1,
                    help=f"1 = order terakhir pada hari acuan, {W} = order terakhir di hari pertama jendela.",
                )
                st.caption(f"Semua isian dihitung dalam jendela {W} hari terakhir")
                submitted = st.form_submit_button("Prediksi Segmen", type="primary", width="stretch")

    incomplete = False
    if submitted:
        if count is None or total is None or recency is None:
            incomplete = True
            st.session_state.pop("sim_result", None)
        else:
            cluster, dists, x_raw, x_scaled = predict(art, total, count, None, int(recency))
            st.session_state["sim_result"] = {
                "sig": sig, "cluster": cluster, "dists": dists, "x_raw": x_raw, "x_scaled": x_scaled,
            }
    res = st.session_state.get("sim_result")
    if res is not None and res["sig"] != sig:
        res = None                        # model diganti -> hasil lama tidak berlaku

    # ---------------- Panel kiri: hasil ringkas ----------------
    if res is not None:
        cluster, dists, x_raw, x_scaled = res["cluster"], res["dists"], res["x_raw"], res["x_scaled"]
        label = cluster_labels.get(cluster, f"Cluster {cluster}")
        color = SEGMENT_COLORS.get(label, "#64748b")
        inv = 1.0 / (dists + 1e-9)
        closeness = inv / inv.sum()
        runner = decision_factors(art, x_scaled, cluster, dists)

        with col_left:
            with st.container(border=True):
                st.markdown('<div class="cm-h">Hasil Prediksi</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="cm-badge" style="background:{color}">'
                    f'<div class="big">{label}</div>'
                    f'<div class="sub">kemiripan {closeness[cluster] * 100:.1f}% · cluster {cluster}</div></div>',
                    unsafe_allow_html=True,
                )

    # ---------------- Panel utama ----------------
    with col_main:
        if res is None:
            render_overview(art, cluster_labels, incomplete)
        else:
            eps = 1e-9
            out_of_range = [
                FEATURE_NAMES.get(f, f)
                for f, v in zip(feats, x_raw.iloc[0])
                if v < dmin[f] - eps or v > dmax[f] + eps
            ]
            if out_of_range:
                st.warning("Nilai berikut berada di luar rentang data training, hasil prediksi "
                           "kurang dapat diandalkan: " + ", ".join(out_of_range))

            prof = centroid_table(art, cluster_labels)
            seg = prof.loc[label] if label in prof.index else None
            xs = dict(zip(feats, x_scaled[0]))

            k1, k2, k3 = st.columns(3)
            with k1, st.container(border=True):
                kpi_card("Frekuensi", f"{x_raw['count_order_total'].iloc[0]:,.0f} order",
                         None, color,"")
            with k2, st.container(border=True):
                kpi_card("Monetary", rupiah(x_raw["sum_order_total"].iloc[0] * ORDER_TOTAL_UNIT),
                         None, color, "")
            with k3, st.container(border=True):
                kpi_card("Recency", f"{x_raw['recency_days'].iloc[0]:,.0f} hari",
                         None, color, "")

            g1, g2 = st.columns([1, 1.25])
            with g1, st.container(border=True):
                st.markdown('<div class="cm-h">Kedekatan ke tiap segmen</div>', unsafe_allow_html=True)
                names = [s for s in SEGMENT_ORDER if s in clusters_by_label]
                vals = [closeness[clusters_by_label[s]] * 100 for s in names]
                st.plotly_chart(
                    donut_chart(names, vals, [SEGMENT_COLORS[s] for s in names],
                                f"{closeness[cluster] * 100:.0f}%"),
                    width="stretch",
                )
            with g2, st.container(border=True):
                st.markdown('<div class="cm-h">Posisi pelanggan dibanding rata-rata segmen</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(compare_chart(art, x_scaled, clusters_by_label), width="stretch")

            with st.container(border=True):
                st.markdown('<div class="cm-h">Rekomendasi tindakan</div>', unsafe_allow_html=True)
                st.write(CLUSTER_DESC.get(label, ""))
                st.markdown("".join(f'<div class="cm-li">• {a}</div>' for a in SEGMENT_ACTIONS.get(label, [])),
                            unsafe_allow_html=True)


if __name__ == "__main__":               # dijalankan langsung: streamlit run prediksi.py
    st.set_page_config(page_title="Simulator Prediksi", page_icon="🔮", layout="wide")
    page_simulator()