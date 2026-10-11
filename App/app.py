import pickle
from pathlib import Path

import streamlit as st

from Menu.prediksi import page_simulator
from Menu.evaluasi import page_evaluasi

MODEL_PATH = Path(__file__).parent.parent / "Model" / "kmeans_model.pkl"
SCALER_PATH = Path(__file__).parent.parent / "Model" / "scaler.pkl"

st.set_page_config(page_title="CustomerMap", layout="wide")


# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifact(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Menu 2
# ----------------------------------------------------------------------------
def page_menu_dua(art: dict):
    page_evaluasi(art)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if not MODEL_PATH.exists():
        st.error(f"File model tidak ditemukan: `{MODEL_PATH.name}`. "
                 "Jalankan `Training.py` terlebih dahulu.")
        st.stop()
    if not SCALER_PATH.exists():
        st.error(f"File scaler tidak ditemukan: `{SCALER_PATH.name}`. "
                 "Jalankan ulang `Training.py` untuk membuat model dan scaler.")
        st.stop()

    art = load_artifact(MODEL_PATH)
    art["scaler"] = load_artifact(SCALER_PATH)

    st.sidebar.title("CustomerMap")
    menu = st.sidebar.radio("Menu", ["Simulator Prediksi", "Evaluasi Model"])
    st.sidebar.divider()

    if menu == "Simulator Prediksi":
        page_simulator(art)
    else:
        page_menu_dua(art)


if __name__ == "__main__":
    main()