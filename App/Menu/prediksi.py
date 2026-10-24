import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # supaya cm_style ketemu saat dijalankan langsung
from cm_style import (CUSTOMER_COLOR, SEGMENT_COLORS, action_list, hero, inject, kpi, result_card,  # noqa: E402
                      section, segment_cards, steps, style_fig)

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

SHORT_NAMES = {
    "sum_order_total": "Belanja",
    "count_order_total": "Order",
    "recency_days": "Recency",
}

SEGMENT_ORDER = ("Pelanggan Loyal", "Pelanggan Reguler", "Pelanggan Pasif")

SEGMENT_ACTIONS = {
    "Pelanggan Loyal": ["Beri poin reward atau program loyalitas", "Tawarkan akses promo eksklusif",
                        "Ajak memberi ulasan atau referral"],
    "Pelanggan Reguler": ["Kirim voucher bundling", "Beri diskon untuk pembelian berikutnya",
                          "Dorong target order bulanan agar naik jadi loyal"],
    "Pelanggan Pasif": ["Kirim voucher comeback", "Kirim notifikasi personal (push atau WhatsApp)",
                        "Lakukan survei alasan berhenti"],
}

CHART_CONFIG = {"displayModeBar": False}


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
            out[nama] = labels.get(c, f"Cluster {c}") + ("" if c == base else "  (berubah)")
        rows.append(out)
    return pd.DataFrame(rows).set_index("Fitur diubah")


# ----------------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------------
def rupiah(v: float) -> str:
    return "Rp " + f"{v:,.0f}".replace(",", ".")


def ringkas_rp(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f} jt"
    if v >= 1_000:
        return f"{v / 1_000:.0f} rb"
    return f"{v:.0f}"


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
        labels=names, values=values, hole=0.7, sort=False, direction="clockwise",
        marker=dict(colors=colors, line=dict(color="#0b1226", width=3)), textinfo="none",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
        annotations=[dict(text=f"<b>{center_text}</b>", x=0.5, y=0.5, font=dict(size=30, color="#ffffff"), showarrow=False)],
    )
    return style_fig(fig, 310)


def compare_chart(art: dict, x_scaled: np.ndarray, clusters_by_label: dict) -> go.Figure:
    centers = art["model"].cluster_centers_
    rows = []
    for i, f in enumerate(art["features"]):
        nama = SHORT_NAMES.get(f, f)
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
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        showlegend=False,                       # legenda diganti versi HTML (lihat compare_legend)
        margin=dict(t=8, b=8, l=8, r=16), bargap=0.30, bargroupgap=0.06,
        xaxis=dict(range=[0, 1.05], title=None, tickvals=[0, 0.25, 0.5, 0.75, 1]),
        yaxis=dict(title=None, automargin=True),
    )
    return style_fig(fig, 300)


def compare_legend(clusters_by_label: dict) -> None:
    """Legenda di atas grafik (HTML), supaya tidak pernah menimpa judul sumbu."""
    items = [("Pelanggan ini", CUSTOMER_COLOR)] + [
        (s, SEGMENT_COLORS[s]) for s in SEGMENT_ORDER if s in clusters_by_label
    ]
    html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;'
        f'font-size:.8rem;color:#cbd5e1">'
        f'<i style="width:10px;height:10px;border-radius:3px;background:{c};display:inline-block"></i>{n}</span>'
        for n, c in items
    )
    st.markdown(f'<div style="margin:0 0 6px 2px">{html}</div>', unsafe_allow_html=True)


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
    """Tampilan awal (sebelum prediksi): ringkasan tiap segmen + langkah pakai."""
    prof = centroid_table(art, cluster_labels)
    section("Ringkasan segmen pelanggan", "profil rata-rata tiap segmen dari model")
    items = []
    for lbl in SEGMENT_ORDER:
        tiles = None
        if lbl in prof.index:
            r = prof.loc[lbl]
            tiles = [(f"{r['Frekuensi (order)']:.1f}x", "Frekuensi"),
                     (ringkas_rp(r["Monetary (Rp)"]), "Belanja"),
                     (f"{r['Recency (hari)']:.1f} hr", "Recency")]
        items.append({"name": lbl, "color": SEGMENT_COLORS[lbl], "desc": CLUSTER_DESC[lbl], "tiles": tiles})
    segment_cards(items)
    if incomplete:
        st.warning("Lengkapi dulu Frekuensi, Monetary, dan Recency, lalu klik **Prediksi Segmen**.")
    steps([("Isi data", "Masukkan frekuensi, monetary, dan recency di panel kiri."),
           ("Klik Prediksi Segmen", "Model menghitung kedekatan pelanggan ke tiap segmen."),
           ("Lihat hasil", "Segmen, grafik perbandingan, dan rekomendasi tindakan.")])


# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
def page_simulator(art: dict = None):
    inject()
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

    hero("Simulator Prediksi Segmen Pelanggan",
         "Prediksi dan kelompokkan segmen perilaku pelanggan berdasarkan pola transaksi RFM "
         "(Recency, Frequency, Monetary).",
         ["Model K-Means", f"{len(art['cluster_labels'])} segmen",
          f"Jendela {int(art['preprocess']['window_days'])} hari"])

    feats = art["features"]
    cluster_labels = art["cluster_labels"]
    clusters_by_label = {lbl: c_id for c_id, lbl in cluster_labels.items()}
    W = int(art["preprocess"]["window_days"])
    dmin, dmax = feature_ranges(art)
    sig = hash(art["model"].cluster_centers_.tobytes())

    col_left, col_main = st.columns([1, 2.6], gap="large")

    # ---------------- Panel kiri: input ----------------
    with col_left:
        with st.container(border=True, key="cmcard_input"):
            section("Input Data Pelanggan")
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
                    placeholder=f"1 - {W}",
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
        color = SEGMENT_COLORS.get(label, "#94a3b8")
        inv = 1.0 / (dists + 1e-9)
        closeness = inv / inv.sum()

        with col_left:
            with st.container(border=True, key="cmcard_hasil"):
                section("Hasil Prediksi")
                result_card(label, color, float(closeness[cluster] * 100), cluster)

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

            xs = dict(zip(feats, x_scaled[0]))
            k1, k2, k3 = st.columns(3)
            with k1:
                kpi("Frekuensi", f"{x_raw['count_order_total'].iloc[0]:,.0f} order", "F", color,
                    pos=xs["count_order_total"],
                    note=f"Rentang training: {dmin['count_order_total']:.0f} – {dmax['count_order_total']:.0f} order")
            with k2:
                kpi("Monetary", rupiah(x_raw["sum_order_total"].iloc[0] * ORDER_TOTAL_UNIT), "M", color,
                    pos=xs["sum_order_total"],
                    note=(f"Rentang training: {ringkas_rp(dmin['sum_order_total'] * ORDER_TOTAL_UNIT)} – "
                          f"{ringkas_rp(dmax['sum_order_total'] * ORDER_TOTAL_UNIT)}"))
            with k3:
                kpi("Recency", f"{x_raw['recency_days'].iloc[0]:,.0f} hari", "R", color,
                    pos=xs["recency_days"],
                    note=f"Rentang training: {dmin['recency_days']:.0f} – {dmax['recency_days']:.0f} hari")

            g1, g2 = st.columns([1, 1.25])
            with g1, st.container(border=True, key="cmcard_g1"):
                section("Kedekatan ke tiap segmen")
                names = [s for s in SEGMENT_ORDER if s in clusters_by_label]
                vals = [closeness[clusters_by_label[s]] * 100 for s in names]
                st.plotly_chart(
                    donut_chart(names, vals, [SEGMENT_COLORS[s] for s in names],
                                f"{closeness[cluster] * 100:.0f}%"),
                    width="stretch",
                    config=CHART_CONFIG,
                )
            with g2, st.container(border=True, key="cmcard_g2"):
                section("Posisi pelanggan dibanding rata-rata segmen")
                compare_legend(clusters_by_label)
                st.plotly_chart(compare_chart(art, x_scaled, clusters_by_label),
                                width="stretch", config=CHART_CONFIG)
                st.caption("Skala 0–1: semakin panjang batang, semakin tinggi nilainya dibanding data training.")

            with st.container(border=True, key="cmcard_rekom"):
                section("Rekomendasi tindakan", label)
                st.markdown(f'<div class="cx-rec-desc">{CLUSTER_DESC.get(label, "")}</div>', unsafe_allow_html=True)
                action_list(SEGMENT_ACTIONS.get(label, []), color)


if __name__ == "__main__":               # dijalankan langsung: streamlit run prediksi.py
    st.set_page_config(page_title="Simulator Prediksi", layout="wide")
    page_simulator()