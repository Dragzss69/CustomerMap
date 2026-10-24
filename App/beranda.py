import base64

import streamlit as st

# Halaman beranda (landing page) portal CustomerMap. Hanya tampil saat aplikasi pertama kali dibuka.
# Tombol memindahkan pengguna ke halaman simulator lewat st.session_state["page"] (dibaca app.py).
PAGE_START = "simulator"

APP_NAME = "CustomerMap"
APP_TAG = "SEGMENTASI &amp; PREDIKSI"
BADGE = "Segmentasi Pelanggan Tepat Sasaran"
HEADLINE_1 = "Kenali Pelanggan Anda"
HEADLINE_2 = "Berbasis Data Akurat"
INTRO = ("CustomerMap mengelompokkan pelanggan layanan pesan-antar makanan ke dalam segmen Loyal, Reguler, "
         "dan Pasif berdasarkan analisis RFM (Recency, Frequency, Monetary) dengan algoritma K-Means, "
         "sehingga strategi promosi dan retensi dapat diarahkan secara objektif dan terukur.")
BUTTON_TEXT = "Coba Sekarang"
FEATURES_TITLE = "Fitur Unggulan Platform"
FEATURES_SUB = "Segmentasi pelanggan yang transparan dan siap ditindaklanjuti"
FOOTER_TEXT = "© 2026 CustomerMap. Hak Cipta Dilindungi."
FOOTER_TAG = "SEGMENTASI K-MEANS"

# ---------------------------------------------------------------------------
# Palet warna: sama dengan halaman Simulator Prediksi, Evaluasi Model, dan sidebar
# (aksen biru, warna segmen biru/hijau/kuning, teks #1f2937, latar #f4f6fb)
# ---------------------------------------------------------------------------
ACCENT = "#3b82f6"          # biru utama (tombol, judul beraksen)
ACCENT_DARK = "#1d4ed8"     # biru gelap (hover, teks pada lencana)
ACCENT_SOFT = "#e0ecff"     # biru muda (latar lencana, sama dengan menu aktif sidebar)
ACCENT_SOFT_BORDER = "#bfdbfe"
SEG_BLUE, SEG_GREEN, SEG_AMBER = "#3b82f6", "#2e9e5b", "#f59e0b"   # warna segmen
BG = "#f4f6fb"              # latar halaman (sama dengan config.toml)
TEXT = "#1f2937"
TEXT_BODY = "#4b5563"
MUTED = "#6b7280"
BORDER = "#e5e7eb"
CARD_DARK_1, CARD_DARK_2, CARD_MUTED = "#1f2937", "#111827", "#9ca3af"

# Ikon (gaya garis): (kode SVG, warna garis (terang untuk latar gelap), warna latar kotak ikon)
_ICONS = {
    "calc": ('<rect width="16" height="20" x="4" y="2" rx="2"/><path d="M8 6h8M16 14v4M8 10h.01M12 10h.01M16 10h.01'
             'M8 14h.01M12 14h.01M8 18h.01M12 18h.01"/>', "#93c5fd", "rgba(59,130,246,.22)"),
    "brief": ('<path d="M16 20V4a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/><rect width="20" height="14" x="2" y="6" rx="2"/>',
              "#86efac", "rgba(46,158,91,.25)"),
    "bell": ('<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
             "#fcd34d", "rgba(245,158,11,.22)"),
}

FEATURES = [
    ("calc", "Segmentasi RFM K-Means",
     "Pelanggan dikelompokkan otomatis dari pola recency, frekuensi, dan nilai belanja dalam jendela 30 hari."),
    ("brief", "Simulator Prediksi Segmen",
     "Masukkan data satu pelanggan, dapatkan segmen, tingkat kemiripan, dan rekomendasi tindakan."),
    ("bell", "Evaluasi Model Transparan",
     "Unggah dataset dan lihat Silhouette, Davies-Bouldin, serta sebaran segmen secara langsung."),
]


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _icon_uri(name: str) -> str:
    body, color, _ = _ICONS[name]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
    return _data_uri(svg)


def _logo_uri() -> str:
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><defs>'
           f'<linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#60a5fa"/>'
           f'<stop offset="1" stop-color="{ACCENT_DARK}"/></linearGradient></defs>'
           '<rect width="40" height="40" rx="10" fill="url(#g)"/>'
           '<g stroke="#fff" stroke-width="2" stroke-linecap="round" fill="none">'
           '<path d="M14.5 15.2l11 1.4M14 16.5l4.3 8.2M26 18l-4.4 6.6"/>'
           '<circle cx="12.5" cy="14.5" r="3.3" fill="#fff"/><circle cx="27.5" cy="16.8" r="3.3" fill="#fff"/>'
           '<circle cx="20" cy="27" r="3.3" fill="#fff"/></g></svg>')
    return _data_uri(svg)


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ---------- Kerangka halaman (tanpa sidebar & header Streamlit) ---------- */
.stApp{{background:{BG}}}
header[data-testid="stHeader"],section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],[data-testid="stExpandSidebarButton"]{{display:none !important}}
html,body,.stApp,[data-testid="stAppViewContainer"]{{overflow-x:hidden}}
.block-container{{max-width:1240px;padding:0 2rem 0 2rem !important}}
.lp, .lp *{{font-family:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif}}

/* ---------- Navbar ---------- */
.lp-nav{{width:100vw;margin-left:calc(50% - 50vw);background:#fff;border-bottom:1px solid {BORDER}}}
.lp-nav-in{{max-width:1240px;margin:0 auto;padding:.7rem 2rem;display:flex;align-items:center;gap:12px}}
.lp-logo{{width:40px;height:40px}}
.lp-brand{{font-weight:800;font-size:1.15rem;line-height:1.15;color:{TEXT}}}
.lp-brand-sub{{font-weight:600;font-size:.62rem;letter-spacing:.14em;color:{MUTED};margin-top:1px}}

/* ---------- Hero ---------- */
.lp-pill{{display:inline-block;background:{ACCENT_SOFT};border:1px solid {ACCENT_SOFT_BORDER};color:{ACCENT_DARK};
         font-weight:600;font-size:.8rem;padding:.38rem .9rem;border-radius:999px;margin-bottom:1.5rem}}
.lp-h1{{font-weight:800;font-size:3.1rem;line-height:1.12;letter-spacing:-.02em;color:{TEXT};margin:0 0 1.3rem 0}}
.lp-h1 span{{color:{ACCENT}}}
.lp-p{{color:{TEXT_BODY};font-size:1.05rem;line-height:1.75;max-width:640px;margin:0 0 1.9rem 0}}

/* ---------- Tombol hero ---------- */
.st-key-lp_start button{{background:{ACCENT} !important;border:none !important;color:#fff !important;
    border-radius:10px !important;padding:.9rem 2.2rem !important;min-height:0;transition:.15s;
    box-shadow:0 8px 20px rgba(59,130,246,.30)}}
.st-key-lp_start button p{{font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1rem;color:#fff !important}}
.st-key-lp_start button:hover{{background:{ACCENT_DARK} !important;transform:translateY(-1px);
    box-shadow:0 10px 24px rgba(29,78,216,.35)}}

/* ---------- Kartu fitur ---------- */
.lp-card{{background:linear-gradient(135deg,{CARD_DARK_1} 0%,{CARD_DARK_2} 100%);border-radius:28px;
         padding:2.4rem 2.5rem;box-shadow:0 24px 50px rgba(17,24,39,.22);position:relative;overflow:hidden}}
.lp-card:before{{content:"";position:absolute;top:-80px;right:-80px;width:260px;height:260px;border-radius:50%;
                background:radial-gradient(circle,rgba(59,130,246,.38),transparent 70%)}}
.lp-card-t{{position:relative;color:#fff;font-weight:700;font-size:1.45rem;margin-bottom:.4rem}}
.lp-card-s{{position:relative;color:{CARD_MUTED};font-size:.85rem;margin-bottom:1.8rem}}
.lp-feat{{position:relative;display:flex;gap:14px;margin-bottom:1.35rem}}
.lp-feat:last-child{{margin-bottom:0}}
.lp-ico{{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;flex:none}}
.lp-ico img{{width:18px;height:18px}}
.lp-ft{{color:#fff;font-weight:700;font-size:.95rem;margin-bottom:2px}}
.lp-fd{{color:{CARD_MUTED};font-size:.84rem;line-height:1.55}}

/* ---------- Footer ---------- */
.lp-foot{{width:100vw;margin-left:calc(50% - 50vw);border-top:1px solid {BORDER};background:#fff;margin-top:8vh}}
.lp-foot-in{{max-width:1240px;margin:0 auto;padding:1.3rem 2rem;display:flex;justify-content:space-between;
            align-items:center;color:{MUTED};font-size:.82rem}}
.lp-tag{{background:{ACCENT_SOFT};border:1px solid {ACCENT_SOFT_BORDER};color:{ACCENT_DARK};font-weight:700;
        font-size:.68rem;letter-spacing:.04em;padding:.35rem .8rem;border-radius:8px}}
</style>
"""


def nav_html() -> str:
    return (f'<div class="lp"><div class="lp-nav"><div class="lp-nav-in"><img class="lp-logo" src="{_logo_uri()}" alt="logo">'
            f'<div><div class="lp-brand">{APP_NAME}</div><div class="lp-brand-sub">{APP_TAG}</div></div></div></div></div>')


def hero_html() -> str:
    return (f'<div class="lp"><div class="lp-pill">{BADGE}</div>'
            f'<div class="lp-h1">{HEADLINE_1}<br><span>{HEADLINE_2}</span></div>'
            f'<div class="lp-p">{INTRO}</div></div>')


def card_html() -> str:
    items = ""
    for key, title, desc in FEATURES:
        bg = _ICONS[key][2]
        items += (f'<div class="lp-feat"><div class="lp-ico" style="background:{bg}"><img src="{_icon_uri(key)}" alt=""></div>'
                  f'<div><div class="lp-ft">{title}</div><div class="lp-fd">{desc}</div></div></div>')
    return (f'<div class="lp"><div class="lp-card"><div class="lp-card-t">{FEATURES_TITLE}</div>'
            f'<div class="lp-card-s">{FEATURES_SUB}</div>{items}</div></div>')


def footer_html() -> str:
    return (f'<div class="lp"><div class="lp-foot"><div class="lp-foot-in"><span>{FOOTER_TEXT}</span>'
            f'<span class="lp-tag">{FOOTER_TAG}</span></div></div></div>')


def _start() -> None:
    st.session_state["page"] = PAGE_START


def page_beranda() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(nav_html(), unsafe_allow_html=True)
    st.markdown('<div style="height:9vh"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1], gap="large", vertical_alignment="center")
    with left:
        st.markdown(hero_html(), unsafe_allow_html=True)
        st.button(BUTTON_TEXT, key="lp_start", on_click=_start)
    with right:
        st.markdown(card_html(), unsafe_allow_html=True)

    st.markdown(footer_html(), unsafe_allow_html=True)


if __name__ == "__main__":               # dijalankan langsung: streamlit run beranda.py
    st.set_page_config(page_title="CustomerMap", layout="wide")
    page_beranda()