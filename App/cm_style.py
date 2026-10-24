"""Tema & komponen tampilan CustomerMap (dark, aurora + glass).

Semua gaya ada di sini supaya app.py / Menu/*.py tinggal memanggil fungsi kecil:
inject(), brand(), model_card(), hero(), section(), kpi(), segment_cards(), steps(),
result_card(), action_list(), empty_state(), style_fig().
"""
from __future__ import annotations

import streamlit as st

# ----------------------------------------------------------------------------
# Palet
# ----------------------------------------------------------------------------
SEGMENT_COLORS = {
    "Pelanggan Loyal": "#10b981",
    "Pelanggan Reguler": "#3b82f6",
    "Pelanggan Pasif": "#f43f5e",
}
CUSTOMER_COLOR = "#f59e0b"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --cx-text: #e8edff; --cx-muted: #94a3b8; --cx-faint: #64748b;
  --cx-line: rgba(255,255,255,.09); --cx-glass: rgba(255,255,255,.045);
  --cx-cyan: #22d3ee; --cx-violet: #8b5cf6; --cx-rose: #f43f5e;
}

/* ---------- Dasar ---------- */
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"] {font-family: 'Inter', system-ui, sans-serif;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {font-family: 'Space Grotesk', 'Inter', sans-serif; color: #fff; letter-spacing: -.01em;}
.stApp {
  color: var(--cx-text);
  background:
    radial-gradient(900px 520px at 8% -10%, rgba(99,102,241,.36), transparent 60%),
    radial-gradient(760px 460px at 100% 0%, rgba(34,211,238,.20), transparent 60%),
    radial-gradient(760px 520px at 60% 115%, rgba(244,63,94,.16), transparent 60%),
    #060a16;
  background-attachment: fixed;
}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1380px;}
[data-testid="stCaptionContainer"], .stApp small {color: var(--cx-muted);}
::-webkit-scrollbar {width: 8px; height: 8px;}
::-webkit-scrollbar-thumb {background: rgba(255,255,255,.14); border-radius: 8px;}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {background: linear-gradient(180deg, rgba(13,19,42,.97), rgba(6,10,22,.99)); border-right: 1px solid var(--cx-line);}
[data-testid="stSidebar"] hr {border-color: var(--cx-line);}
[data-testid="stSidebar"] [role="radiogroup"] {gap: 6px;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .7rem .95rem; border-radius: 14px; width: 100%; border: 1px solid transparent; transition: all .18s ease; cursor: pointer;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:not([data-testid]) {display: none;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(255,255,255,.06);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(34,211,238,.20), rgba(139,92,246,.16));
  border-color: rgba(34,211,238,.38); box-shadow: 0 0 24px rgba(34,211,238,.12);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color: #fff; font-weight: 600;}

.cx-brand {display: flex; align-items: center; gap: 12px; padding: .3rem 0 1.4rem 0;}
.cx-logo {width: 46px; height: 46px; border-radius: 15px; display: flex; align-items: center; justify-content: center; flex: none;
  background: linear-gradient(135deg, #22d3ee, #8b5cf6 60%, #f43f5e); color: #fff; font-family: 'Space Grotesk', sans-serif;
  font-weight: 700; letter-spacing: .5px; box-shadow: 0 0 28px rgba(139,92,246,.55);}
.cx-brand-name {color: #fff; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.28rem; line-height: 1.1;}
.cx-brand-sub {color: var(--cx-muted); font-size: .78rem; margin-top: 2px;}
.cx-nav-label {color: var(--cx-faint); font-size: .68rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; margin: .2rem 0 .55rem .25rem;}

.cx-mcard {background: var(--cx-glass); border: 1px solid var(--cx-line); border-radius: 16px; padding: 14px 15px; margin-top: .4rem;}
.cx-mcard-t {display: flex; align-items: center; gap: 8px; font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--cx-muted); margin-bottom: 10px;}
.cx-mrow {display: flex; justify-content: space-between; font-size: .82rem; color: #b4bfd8; padding: 4px 0;}
.cx-mrow b {color: #fff; font-family: 'Space Grotesk', sans-serif; font-weight: 600;}
.cx-sidenote {color: var(--cx-faint); font-size: .76rem; line-height: 1.55; margin-top: 14px;}

/* ---------- Banner atas ---------- */
.cx-hero {position: relative; overflow: hidden; border-radius: 26px; padding: 32px 36px; margin-bottom: 26px;
  display: flex; align-items: center; justify-content: space-between; gap: 26px; flex-wrap: wrap;
  background:
    linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px) 0 0 / 34px 34px,
    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px) 0 0 / 34px 34px,
    linear-gradient(135deg, rgba(79,70,229,.55) 0%, rgba(14,165,233,.28) 55%, rgba(244,63,94,.24) 100%);
  border: 1px solid rgba(255,255,255,.14); box-shadow: 0 24px 60px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.12);}
.cx-hero::before {content: ""; position: absolute; width: 340px; height: 340px; right: -90px; top: -130px; border-radius: 50%;
  background: radial-gradient(circle, rgba(34,211,238,.55), transparent 65%); filter: blur(10px); animation: cxdrift 9s ease-in-out infinite alternate;}
@keyframes cxdrift {from {transform: translate(0,0) scale(1);} to {transform: translate(-40px,30px) scale(1.15);}}
.cx-hero-main {position: relative; z-index: 1; flex: 1 1 460px;}
.cx-title {font-family: 'Space Grotesk', sans-serif; font-size: 2.35rem; font-weight: 700; line-height: 1.1; letter-spacing: -.02em;
  background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 55%, #67e8f9 100%); -webkit-background-clip: text; background-clip: text; color: transparent;}
.cx-sub {margin-top: 12px; max-width: 640px; font-size: .96rem; line-height: 1.65; color: rgba(226,232,255,.84);}
.cx-chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px;}
.cx-chip {display: inline-flex; align-items: center; gap: 7px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
  padding: 5px 13px; border-radius: 999px; font-size: .76rem; font-weight: 600; color: #e8edff; backdrop-filter: blur(6px);}
.cx-dot {width: 8px; height: 8px; border-radius: 50%; background: #34d399; animation: cxpulse 1.8s infinite;}
@keyframes cxpulse {0% {box-shadow: 0 0 0 0 rgba(52,211,153,.6);} 100% {box-shadow: 0 0 0 10px rgba(52,211,153,0);}}
.cx-rfm {position: relative; z-index: 1; display: flex; gap: 12px;}
.cx-rfm-t {width: 88px; height: 88px; border-radius: 24px; display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.18); backdrop-filter: blur(10px); box-shadow: 0 10px 28px rgba(0,0,0,.3);}
.cx-rfm-t:nth-child(2) {transform: translateY(12px);}
.cx-rfm-t b {font-family: 'Space Grotesk', sans-serif; font-size: 1.95rem; line-height: 1; color: #fff;}
.cx-rfm-t i {font-style: normal; font-size: .6rem; letter-spacing: .09em; text-transform: uppercase; color: rgba(226,232,255,.72); margin-top: 5px;}

/* ---------- Kartu kaca (container ber-key cmcard_*) ---------- */
div[class*="st-key-cmcard"] {background: var(--cx-glass); border: 1px solid var(--cx-line) !important; border-radius: 22px;
  backdrop-filter: blur(14px); box-shadow: 0 18px 40px rgba(0,0,0,.35);}
.cx-sec {display: flex; align-items: center; gap: 10px; margin: 2px 0 12px 0;}
.cx-sec::before {content: ""; width: 4px; height: 1.15em; border-radius: 3px; background: linear-gradient(#22d3ee, #8b5cf6); flex: none;}
.cx-sec-t {font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.08rem; color: #fff;}
.cx-sec-s {font-size: .8rem; color: var(--cx-muted); margin-left: 4px;}

/* ---------- Kartu angka (KPI) ---------- */
.cx-kpi {position: relative; overflow: hidden; background: var(--cx-glass); border: 1px solid var(--cx-line); border-radius: 20px; padding: 18px 20px 16px 20px;
  box-shadow: 0 18px 40px rgba(0,0,0,.30); backdrop-filter: blur(14px);}
.cx-kpi::after {content: ""; position: absolute; right: -40px; top: -40px; width: 130px; height: 130px; border-radius: 50%;
  background: radial-gradient(circle, var(--c), transparent 70%); opacity: .20;}
.cx-kpi-top {position: relative; z-index: 1; display: flex; align-items: center; gap: 10px;}
.cx-kpi-ico {width: 32px; height: 32px; border-radius: 11px; display: flex; align-items: center; justify-content: center; flex: none;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: .95rem; color: #fff;
  background: linear-gradient(135deg, var(--c), rgba(139,92,246,.85)); box-shadow: 0 0 16px var(--c);}
.cx-kpi-label {font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: var(--cx-muted);}
.cx-kpi-val {position: relative; z-index: 1; font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #fff; margin: 12px 0 12px 0; line-height: 1.1;}
.cx-bar {position: relative; z-index: 1; height: 7px; border-radius: 6px; background: rgba(255,255,255,.10); overflow: hidden;}
.cx-bar > i {display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg, rgba(255,255,255,.25), var(--c));}
.cx-kpi-note {position: relative; z-index: 1; font-size: .74rem; color: var(--cx-faint); margin-top: 8px;}

/* ---------- Hasil prediksi ---------- */
.cx-result {position: relative; overflow: hidden; display: flex; align-items: center; gap: 18px; padding: 20px 18px; border-radius: 20px;
  background: linear-gradient(135deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); border: 1px solid var(--c);
  box-shadow: 0 0 34px color-mix(in srgb, var(--c) 35%, transparent), 0 14px 30px rgba(0,0,0,.4);}
.cx-result::after {content: ""; position: absolute; right: -60px; top: -60px; width: 190px; height: 190px; border-radius: 50%;
  background: radial-gradient(circle, var(--c), transparent 70%); opacity: .22;}
.cx-ring {position: relative; z-index: 1; width: 82px; height: 82px; border-radius: 50%; flex: none; display: flex; align-items: center; justify-content: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.12) 0);}
.cx-ring::before {content: ""; position: absolute; inset: 7px; border-radius: 50%; background: #0b1226;}
.cx-ring b {position: relative; font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; color: #fff;}
.cx-res-body {position: relative; z-index: 1; min-width: 0;}
.cx-res-k {font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--cx-muted);}
.cx-res-name {font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 700; color: #fff; line-height: 1.2; margin: 3px 0 4px 0;}
.cx-res-sub {font-size: .78rem; color: #b4bfd8;}

/* ---------- Kartu segmen ---------- */
.cx-seg-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin-bottom: 22px;}
.cx-seg {position: relative; overflow: hidden; background: var(--cx-glass); border: 1px solid var(--cx-line); border-radius: 22px;
  padding: 22px 20px 20px 20px; backdrop-filter: blur(14px); box-shadow: 0 18px 40px rgba(0,0,0,.35);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;}
.cx-seg::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; background: linear-gradient(90deg, var(--c), transparent);}
.cx-seg::after {content: ""; position: absolute; right: -50px; top: -50px; width: 180px; height: 180px; border-radius: 50%;
  background: radial-gradient(circle, var(--c), transparent 70%); opacity: .20;}
.cx-seg:hover {transform: translateY(-5px); border-color: var(--c); box-shadow: 0 0 28px rgba(0,0,0,.4), 0 22px 44px rgba(0,0,0,.45);}
.cx-seg-name {position: relative; z-index: 1; display: flex; align-items: center; gap: 10px; font-family: 'Space Grotesk', sans-serif;
  font-weight: 700; font-size: 1.16rem; color: #fff; margin-bottom: 8px;}
.cx-seg-name::before {content: ""; width: 10px; height: 10px; border-radius: 50%; background: var(--c); box-shadow: 0 0 14px var(--c); flex: none;}
.cx-seg-desc {position: relative; z-index: 1; color: #b4bfd8; font-size: .84rem; line-height: 1.6; margin: 0 0 16px 0;}
.cx-avg {position: relative; z-index: 1; font-size: .66rem; color: var(--cx-faint); letter-spacing: .1em; text-transform: uppercase; margin: 0 0 7px 2px;}
.cx-tiles {position: relative; z-index: 1; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;}
.cx-tile {background: rgba(255,255,255,.055); border: 1px solid var(--cx-line); border-radius: 13px; padding: 10px 4px; text-align: center;}
.cx-tile b {display: block; font-family: 'Space Grotesk', sans-serif; font-size: 1rem; color: #fff; white-space: nowrap;}
.cx-tile span {display: block; font-size: .6rem; color: var(--cx-muted); text-transform: uppercase; letter-spacing: .07em; margin-top: 3px;}

/* ---------- Langkah ---------- */
.cx-steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;}
.cx-step {display: flex; gap: 14px; align-items: flex-start; background: var(--cx-glass); border: 1px solid var(--cx-line); border-radius: 18px; padding: 16px 18px;}
.cx-num {flex: none; width: 34px; height: 34px; border-radius: 12px; font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #fff;
  background: linear-gradient(135deg, #22d3ee, #8b5cf6); display: flex; align-items: center; justify-content: center; box-shadow: 0 0 18px rgba(34,211,238,.35);}
.cx-step b {display: block; color: #fff; font-size: .92rem; margin-bottom: 3px;}
.cx-step span {color: #a3b0cc; font-size: .8rem; line-height: 1.5;}

/* ---------- Rekomendasi ---------- */
.cx-rec-desc {color: #cbd5e1; font-size: .92rem; line-height: 1.65; margin-bottom: 14px;}
.cx-act {display: flex; align-items: center; gap: 12px; padding: 11px 14px; margin-bottom: 8px; border-radius: 14px; font-size: .9rem; color: #e2e8f0;
  background: rgba(255,255,255,.04); border: 1px solid var(--cx-line); border-left: 3px solid var(--c);}
.cx-act i {font-style: normal; flex: none; width: 24px; height: 24px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
  font-size: .75rem; font-weight: 700; color: #06101f; background: var(--c);}

/* ---------- Kosong / info ---------- */
.cx-empty {text-align: center; padding: 8px 12px 4px 12px;}
.cx-empty-ico {width: 54px; height: 54px; margin: 0 auto 12px auto; border-radius: 18px; display: flex; align-items: center; justify-content: center;
  font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: #fff;
  background: linear-gradient(135deg, #22d3ee, #8b5cf6); box-shadow: 0 0 26px rgba(34,211,238,.35);}
.cx-empty-t {font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: #fff;}
.cx-empty-d {color: var(--cx-muted); font-size: .86rem; line-height: 1.6; margin: 6px auto 12px auto; max-width: 520px;}
.cx-cols {display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;}
.cx-cols code {background: rgba(34,211,238,.10); color: #67e8f9; border: 1px solid rgba(34,211,238,.28); padding: 3px 11px; border-radius: 999px; font-size: .78rem;}
.cx-status {display: inline-flex; align-items: center; gap: 8px; background: rgba(52,211,153,.10); border: 1px solid rgba(52,211,153,.30);
  color: #6ee7b7; padding: 7px 14px; border-radius: 999px; font-size: .82rem; font-weight: 600;}

/* ---------- Input, tombol, tab, tabel ---------- */
[data-testid="stNumberInput"] [data-baseweb="input"] {background: rgba(255,255,255,.06); border: 1px solid var(--cx-line); border-radius: 12px;}
[data-testid="stNumberInput"] input {color: var(--cx-text);}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {border-color: #22d3ee; box-shadow: 0 0 0 3px rgba(34,211,238,.18);}
[data-baseweb="select"] > div {background: rgba(255,255,255,.06); border-color: var(--cx-line); border-radius: 12px;}
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #06b6d4, #6366f1 55%, #d946ef); border: none; border-radius: 14px; font-weight: 700; padding: .7rem 1rem; color: #fff;
  box-shadow: 0 10px 26px rgba(99,102,241,.45); transition: transform .15s, box-shadow .15s, filter .15s;}
[data-testid="stBaseButton-primaryFormSubmit"]:hover, [data-testid="stBaseButton-primary"]:hover {transform: translateY(-2px); filter: brightness(1.1); box-shadow: 0 16px 34px rgba(99,102,241,.55);}
[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {border-radius: 12px; font-weight: 600; background: rgba(255,255,255,.06); border: 1px solid var(--cx-line); color: var(--cx-text);}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover {border-color: #22d3ee; color: #67e8f9;}
[data-testid="stFileUploaderDropzone"] {border: 2px dashed rgba(34,211,238,.45); background: rgba(34,211,238,.05); border-radius: 20px; padding: 2rem;}
[data-testid="stFileUploaderDropzone"]:hover {background: rgba(34,211,238,.09); border-color: #22d3ee;}
[data-testid="stAlert"] {border-radius: 16px; border: 1px solid rgba(56,189,248,.28); background: rgba(56,189,248,.09); backdrop-filter: blur(8px);}
[data-testid="stDataFrame"] {border-radius: 16px; overflow: hidden; border: 1px solid var(--cx-line);}
[data-baseweb="tab-list"] {gap: 6px; background: rgba(255,255,255,.05); border: 1px solid var(--cx-line); padding: 5px; border-radius: 16px; width: fit-content; margin-bottom: 14px;}
[data-baseweb="tab"] {border-radius: 11px; padding: .5rem 1.25rem; height: auto;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, rgba(34,211,238,.25), rgba(139,92,246,.25)); color: #fff;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display: none;}
[data-testid="stMetric"] {background: var(--cx-glass); border: 1px solid var(--cx-line); border-radius: 18px; padding: 16px 20px; box-shadow: 0 14px 30px rgba(0,0,0,.28);}
[data-testid="stMetricValue"] {font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #fff;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {color: var(--cx-muted);}

/* ---------- Grafik ---------- */
[data-testid="stPlotlyChart"] {border-radius: 14px; overflow: hidden;}

@media (max-width: 760px) {
  .cx-hero {padding: 24px 22px;} .cx-title {font-size: 1.7rem;} .cx-rfm {display: none;}
}

/* ================= TEMA TERANG (override) ================= */
:root {
  --cx-text: #0f172a; --cx-muted: #64748b; --cx-faint: #94a3b8;
  --cx-line: rgba(15,23,42,.09); --cx-glass: #ffffff;
}
.stApp {
  color: var(--cx-text);
  background:
    radial-gradient(900px 520px at 8% -10%, rgba(99,102,241,.10), transparent 60%),
    radial-gradient(760px 460px at 100% 0%, rgba(34,211,238,.10), transparent 60%),
    #f6f8fc;
  background-attachment: fixed;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {color: #0f172a;}
::-webkit-scrollbar-thumb {background: rgba(15,23,42,.18);}

/* Hero: tetap berwarna agar jadi aksen utama */
.cx-hero {
  background:
    linear-gradient(rgba(255,255,255,.06) 1px, transparent 1px) 0 0 / 34px 34px,
    linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px) 0 0 / 34px 34px,
    linear-gradient(135deg, #4f46e5 0%, #0ea5e9 60%, #8b5cf6 100%);
  border: none; box-shadow: 0 18px 44px rgba(79,70,229,.28);}
.cx-title {background: none; -webkit-background-clip: initial; background-clip: initial; color: #fff;}
.cx-sub {color: rgba(255,255,255,.88);}

/* Kartu */
div[class*="st-key-cmcard"] {background: #fff; border: 1px solid var(--cx-line) !important;
  box-shadow: 0 10px 30px rgba(15,23,42,.06); backdrop-filter: none;}
.cx-sec-t {color: #0f172a;}

/* KPI */
.cx-kpi {background: #fff; box-shadow: 0 10px 30px rgba(15,23,42,.06);}
.cx-kpi-val {color: #0f172a;}
.cx-kpi-ico {box-shadow: 0 4px 12px color-mix(in srgb, var(--c) 45%, transparent);}
.cx-bar {background: rgba(15,23,42,.08);}
.cx-bar > i {background: linear-gradient(90deg, color-mix(in srgb, var(--c) 45%, #fff), var(--c));}

/* Hasil prediksi */
.cx-result {background: #fff; box-shadow: 0 10px 30px color-mix(in srgb, var(--c) 22%, transparent);}
.cx-ring {background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(15,23,42,.10) 0);}
.cx-ring::before {background: #fff;}
.cx-ring b, .cx-res-name {color: #0f172a;}
.cx-res-sub {color: #64748b;}

/* Kartu segmen */
.cx-seg {background: #fff; box-shadow: 0 10px 30px rgba(15,23,42,.06); backdrop-filter: none;}
.cx-seg::after {opacity: .10;}
.cx-seg:hover {box-shadow: 0 16px 36px rgba(15,23,42,.12);}
.cx-seg-name {color: #0f172a;}
.cx-seg-name::before {box-shadow: 0 0 10px color-mix(in srgb, var(--c) 60%, transparent);}
.cx-seg-desc {color: #475569;}
.cx-tile {background: #f1f5f9;}
.cx-tile b {color: #0f172a;}

/* Langkah */
.cx-step {background: #fff; box-shadow: 0 6px 18px rgba(15,23,42,.05);}
.cx-step b {color: #0f172a;}
.cx-step span {color: #64748b;}
.cx-num {box-shadow: 0 4px 12px rgba(99,102,241,.30);}

/* Rekomendasi */
.cx-rec-desc {color: #334155;}
.cx-act {color: #1e293b; background: #f8fafc;}
.cx-act i {color: #fff;}

/* Input, tombol, tab, tabel */
[data-testid="stNumberInput"] [data-baseweb="input"] {background: #f8fafc; border: 1px solid rgba(15,23,42,.14);}
[data-testid="stNumberInput"] input {color: #0f172a;}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,.18);}
[data-baseweb="select"] > div {background: #f8fafc; border-color: rgba(15,23,42,.14);}
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #4f46e5, #0ea5e9); box-shadow: 0 8px 20px rgba(79,70,229,.30);}
[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {background: #fff; color: #0f172a; border: 1px solid rgba(15,23,42,.14);}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover {border-color: #6366f1; color: #4f46e5;}
[data-testid="stFileUploaderDropzone"] {background: rgba(99,102,241,.04); border-color: rgba(99,102,241,.45);}
[data-testid="stAlert"] {background: #eff6ff; border-color: rgba(59,130,246,.25);}
[data-testid="stDataFrame"] {background: #fff;}
[data-baseweb="tab-list"] {background: #fff;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(14,165,233,.14)); color: #4f46e5;}
[data-testid="stMetric"] {background: #fff; box-shadow: 0 8px 22px rgba(15,23,42,.05);}
[data-testid="stMetricValue"] {color: #0f172a;}
</style>
"""


# ----------------------------------------------------------------------------
# Komponen
# ----------------------------------------------------------------------------
def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand() -> None:
    st.sidebar.markdown(
        '<div class="cx-brand"><div class="cx-logo">CM</div><div>'
        '<div class="cx-brand-name">CustomerMap</div>'
        '<div class="cx-brand-sub">Segmentasi pelanggan</div></div></div>'
        '<div class="cx-nav-label">Navigasi</div>',
        unsafe_allow_html=True,
    )


def model_card(art: dict) -> None:
    """Ringkasan model di sidebar (metrik dari file .pkl)."""
    m = art.get("metrics", {}) or {}
    cfg = art.get("config", {}) or {}
    rows = [("Algoritma", "K-Means"), ("Jumlah segmen", str(cfg.get("n_clusters", len(art["cluster_labels"]))))]
    if m.get("silhouette") is not None:
        rows.append(("Silhouette", f"{m['silhouette']:.3f}"))
    if m.get("davies_bouldin") is not None:
        rows.append(("Davies-Bouldin", f"{m['davies_bouldin']:.3f}"))
    body = "".join(f'<div class="cx-mrow"><span>{k}</span><b>{v}</b></div>' for k, v in rows)
    st.sidebar.markdown(
        f'<div class="cx-mcard"><div class="cx-mcard-t"><i class="cx-dot"></i>Model aktif</div>{body}</div>'
        '<div class="cx-sidenote">Segmentasi berbasis RFM (Recency, Frequency, Monetary).</div>',
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, chips=()) -> None:
    chip_html = '<span class="cx-chip"><i class="cx-dot"></i>Model aktif</span>' + "".join(
        f'<span class="cx-chip">{c}</span>' for c in chips)
    rfm = "".join(f'<div class="cx-rfm-t"><b>{a}</b><i>{b}</i></div>'
                  for a, b in (("R", "Recency"), ("F", "Frequency"), ("M", "Monetary")))
    st.markdown(
        f'<div class="cx-hero"><div class="cx-hero-main"><div class="cx-title">{title}</div>'
        f'<div class="cx-sub">{subtitle}</div><div class="cx-chips">{chip_html}</div></div>'
        f'<div class="cx-rfm">{rfm}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "") -> None:
    s = f'<span class="cx-sec-s">{sub}</span>' if sub else ""
    st.markdown(f'<div class="cx-sec"><span class="cx-sec-t">{title}</span>{s}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, icon: str, color: str, pos: float | None = None, note: str = "") -> None:
    bar = f'<div class="cx-bar"><i style="width:{max(0.0, min(1.0, pos)) * 100:.0f}%"></i></div>' if pos is not None else ""
    note_html = f'<div class="cx-kpi-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="cx-kpi" style="--c:{color}"><div class="cx-kpi-top"><span class="cx-kpi-ico">{icon}</span>'
        f'<span class="cx-kpi-label">{label}</span></div><div class="cx-kpi-val">{value}</div>{bar}{note_html}</div>',
        unsafe_allow_html=True,
    )


def segment_cards(items: list[dict]) -> None:
    """items: [{name, color, desc, tiles: [(nilai, label), ...] | None}]"""
    cards = []
    for it in items:
        tiles = ""
        if it.get("tiles"):
            tiles = ('<div class="cx-avg">Rata-rata segmen</div><div class="cx-tiles">'
                     + "".join(f'<div class="cx-tile"><b>{v}</b><span>{l}</span></div>' for v, l in it["tiles"])
                     + "</div>")
        cards.append(f'<div class="cx-seg" style="--c:{it["color"]}"><div class="cx-seg-name">{it["name"]}</div>'
                     f'<div class="cx-seg-desc">{it["desc"]}</div>{tiles}</div>')
    st.markdown('<div class="cx-seg-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    html = "".join(f'<div class="cx-step"><div class="cx-num">{i}</div><div><b>{t}</b><span>{d}</span></div></div>'
                   for i, (t, d) in enumerate(items, 1))
    st.markdown(f'<div class="cx-steps">{html}</div>', unsafe_allow_html=True)


def result_card(label: str, color: str, pct: float, cluster: int) -> None:
    st.markdown(
        f'<div class="cx-result" style="--c:{color};--p:{pct:.1f}"><div class="cx-ring"><b>{pct:.0f}%</b></div>'
        f'<div class="cx-res-body"><div class="cx-res-k">Segmen pelanggan</div><div class="cx-res-name">{label}</div>'
        f'<div class="cx-res-sub">kedekatan ke segmen · cluster {cluster}</div></div></div>',
        unsafe_allow_html=True,
    )


def action_list(actions: list[str], color: str) -> None:
    html = "".join(f'<div class="cx-act" style="--c:{color}"><i>{n}</i><span>{a}</span></div>'
                   for n, a in enumerate(actions, 1))
    st.markdown(html, unsafe_allow_html=True)


def empty_state(icon: str, title: str, text: str, chips=()) -> None:
    chip_html = f'<div class="cx-cols">{"".join(f"<code>{c}</code>" for c in chips)}</div>' if chips else ""
    st.markdown(
        f'<div class="cx-empty"><div class="cx-empty-ico">{icon}</div><div class="cx-empty-t">{title}</div>'
        f'<div class="cx-empty-d">{text}</div>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def status_chip(text: str) -> None:
    st.markdown(f'<span class="cx-status"><i class="cx-dot"></i>{text}</span>', unsafe_allow_html=True)


def style_fig(fig, height: int | None = None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        legend=dict(font=dict(color="#334155")),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(15,23,42,.15)", font=dict(color="#0f172a")),
    )
    fig.update_xaxes(gridcolor="rgba(15,23,42,.07)", zerolinecolor="rgba(15,23,42,.12)", linecolor="rgba(15,23,42,.12)")
    fig.update_yaxes(gridcolor="rgba(15,23,42,.07)", zerolinecolor="rgba(15,23,42,.12)", linecolor="rgba(15,23,42,.12)")
    if height:
        fig.update_layout(height=height)
    return fig