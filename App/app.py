import pickle
from pathlib import Path

import streamlit as st

from cm_style import inject, brand, model_card
from Menu.prediksi import page_simulator
from Menu.evaluasi import page_evaluasi

MODEL_PATH = Path(__file__).parent.parent / "Model" / "kmeans_model_v2.pkl"
SCALER_PATH = Path(__file__).parent.parent / "Model" / "scaler_v2.pkl"

NAV_SIMULATOR = ":material/query_stats: Simulator Prediksi"
NAV_EVALUASI = ":material/fact_check: Evaluasi Model"

st.set_page_config(page_title="CustomerMap", page_icon=":material/hub:", layout="wide")
inject()


# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifact(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if not MODEL_PATH.exists():
        st.error(f"File model tidak ditemukan: `{MODEL_PATH.name}`. "
                 "Jalankan `training.py` terlebih dahulu.")
        st.stop()
    if not SCALER_PATH.exists():
        st.error(f"File scaler tidak ditemukan: `{SCALER_PATH.name}`. "
                 "Jalankan ulang `training.py` untuk membuat model dan scaler.")
        st.stop()

    art = load_artifact(MODEL_PATH)
    art["scaler"] = load_artifact(SCALER_PATH)

    brand()
    menu = st.sidebar.radio("Menu", [NAV_SIMULATOR, NAV_EVALUASI], label_visibility="collapsed")
    st.sidebar.divider()
    model_card(art)

    if menu == NAV_SIMULATOR:
        page_simulator(art)
    else:
        page_evaluasi(art)


if __name__ == "__main__":
    main()