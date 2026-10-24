import base64
from pathlib import Path

import streamlit as st

from beranda import page_beranda
from Menu.evaluasi import page_evaluasi
from Menu.prediksi import page_simulator

st.set_page_config(page_title="CustomerMap", layout="wide")

# ----------------------------------------------------------------------------
# Konfigurasi sidebar
# ----------------------------------------------------------------------------
APP_TITLE = "CustomerMap"
APP_SUBTITLE = "Segmentasi Pelanggan"

# Logo opsional: letakkan file di App/assets/logo.png (atau .svg/.jpg). Jika tidak ada, dipakai logo bawaan.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Daftar menu: satu seksi "MENU UTAMA" berisi semua opsi. Tambah menu baru cukup menambah baris di sini.
# Ikon memakai Material Symbols (https://fonts.google.com/icons), tulis sebagai :material/nama_ikon:
MENU = [
    {"section": "MENU UTAMA", "items": [
        {"id": "simulator", "label": "Simulator Prediksi", "icon": ":material/insights:", "page": page_simulator},
        {"id": "evaluasi", "label": "Evaluasi Model", "icon": ":material/analytics:", "page": page_evaluasi},
    ]},
]
PAGES = {item["id"]: item for sec in MENU for item in sec["items"]}
PAGES["beranda"] = {"id": "beranda", "page": page_beranda}   # hanya tampil saat aplikasi pertama dibuka
DEFAULT_PAGE = "beranda"        # halaman pembuka portal

# Warna (sidebar putih)
SB_BG = "#ffffff"        # latar sidebar
SB_BORDER = "#e5e7eb"
SB_ACTIVE = "#e0ecff"    # latar menu aktif
SB_ACTIVE_TEXT = "#1d4ed8"
SB_HOVER = "#f1f5f9"
SB_TEXT = "#1f2937"
SB_MUTED = "#6b7280"
SB_WIDTH = "290px"

DEFAULT_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<circle cx="32" cy="32" r="28" fill="none" stroke="#1f2937" stroke-width="4"/>'
    '<path d="M22 24 L42 26 L32 43 Z" fill="none" stroke="#1f2937" stroke-width="2" opacity=".45"/>'
    '<circle cx="22" cy="24" r="6.5" fill="#3b82f6"/><circle cx="42" cy="26" r="6.5" fill="#2e9e5b"/>'
    '<circle cx="32" cy="43" r="6.5" fill="#f59e0b"/></svg>'
)

CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

/* ---------- Kerangka sidebar (putih, lebar tetap) ---------- */
section[data-testid="stSidebar"]{
    background:__BG__ !important;border-right:1px solid __BORDER__;
    width:__W__ !important;min-width:__W__ !important;max-width:__W__ !important}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarResizeHandle"]{display:none !important}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:1.3rem 1rem 1rem 1rem}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.6rem}

/* ---------- Header: logo + judul ---------- */
.cm-brand{display:flex;align-items:center;gap:12px}
.cm-brand img{width:46px;height:46px;object-fit:contain;flex:none}
.cm-title{font-family:'Poppins',sans-serif;font-weight:700;font-size:1.3rem;line-height:1.15;color:__TEXT__}
.cm-sub{font-family:'Poppins',sans-serif;font-size:.82rem;color:__MUTED__;margin-top:2px}
.cm-line{border:none;border-top:1px solid __BORDER__;margin:.9rem -1rem .2rem -1rem}

/* ---------- Label seksi (jarak ke menu diperlebar) ---------- */
.cm-sec{font-family:'Poppins',sans-serif;font-size:.78rem;font-weight:700;letter-spacing:.09em;
        color:__MUTED__;margin:1.4rem .5rem 1.1rem .5rem}

/* ---------- Tombol menu ---------- */
section[data-testid="stSidebar"] [class*="st-key-nav_"] button{
    background:transparent !important;border:none !important;box-shadow:none !important;transform:none !important;
    color:__TEXT__ !important;border-radius:14px !important;width:100%;
    justify-content:flex-start;padding:.85rem 1rem !important;transition:background .15s}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button p{
    font-family:'Poppins',sans-serif;font-weight:600;font-size:1.02rem;color:inherit !important}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover{background:__HOVER__ !important}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button:focus{box-shadow:none !important;outline:none}
section[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stIconMaterial"]{
    font-size:1.5rem !important;color:inherit !important}
__ACTIVE__
</style>
"""


def build_css(active: str) -> str:
    sel = f'section[data-testid="stSidebar"] .st-key-nav_{active} button'
    active_css = (f"{sel},\n{sel}:hover{{background:{SB_ACTIVE} !important;color:{SB_ACTIVE_TEXT} !important}}")
    css = CSS_TEMPLATE
    for token, value in {
        "__BG__": SB_BG, "__BORDER__": SB_BORDER, "__TEXT__": SB_TEXT, "__MUTED__": SB_MUTED,
        "__HOVER__": SB_HOVER, "__W__": SB_WIDTH, "__ACTIVE__": active_css,
    }.items():
        css = css.replace(token, value)
    return css


def logo_uri() -> str:
    """Data URI logo: file di App/assets/ jika ada, selain itu logo bawaan."""
    mimes = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
    for name in ("logo.png", "logo.svg", "logo.jpg", "logo.jpeg"):
        path = ASSETS_DIR / name
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            return f"data:{mimes[path.suffix.lower()]};base64,{data}"
    return "data:image/svg+xml;base64," + base64.b64encode(DEFAULT_LOGO_SVG.encode()).decode()


# ----------------------------------------------------------------------------
# Navigasi
# ----------------------------------------------------------------------------
def go_to(page_id: str) -> None:
    st.session_state["page"] = page_id


def render_sidebar() -> None:
    active = st.session_state["page"]

    with st.sidebar:
        st.markdown(build_css(active), unsafe_allow_html=True)
        st.markdown(
            f'<div class="cm-brand"><img src="{logo_uri()}" alt="logo"><div>'
            f'<div class="cm-title">{APP_TITLE}</div><div class="cm-sub">{APP_SUBTITLE}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="cm-line">', unsafe_allow_html=True)

        for sec in MENU:
            st.markdown(f'<div class="cm-sec">{sec["section"]}</div>', unsafe_allow_html=True)
            for item in sec["items"]:
                st.button(item["label"], key=f"nav_{item['id']}", icon=item["icon"],
                          on_click=go_to, args=(item["id"],), width="stretch")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    st.session_state.setdefault("page", DEFAULT_PAGE)
    if st.session_state["page"] not in PAGES:
        st.session_state["page"] = DEFAULT_PAGE

    page = st.session_state["page"]
    if page != "beranda":                          # beranda tampil penuh tanpa sidebar
        render_sidebar()
    PAGES[page]["page"]()                          # halaman memuat model v2-nya sendiri


if __name__ == "__main__":
    main()