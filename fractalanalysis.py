import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Market Fractal Analyzer", layout="wide")

CSV_PATH = "data/Market_Data.csv"

# ---------------------------
# Load Symbol Master
# ---------------------------
@st.cache_data
def load_symbol_master():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df

symbols_df = load_symbol_master()

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("🔎 Instrument Selection")

exchange = st.sidebar.selectbox(
    "Select Exchange",
    sorted(symbols_df["Exchange"].unique())
)

company_input = st.sidebar.text_input(
    "Type Company Name",
    placeholder="e.g. Apple, Reliance, Tesla"
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

rolling_window = st.sidebar.slider("Rolling Hurst Window", 100, 500, 252)
max_scale = st.sidebar.slider("MF-DFA Max Scale", 100, 500, 250)

run_button = st.sidebar.button("▶ Run Analysis")

# ---------------------------
# Helper Functions
# ---------------------------
def hurst(ts):
    lags = range(2, 100)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def mfdfa(signal, scales, q_vals):
    signal = np.array(signal)
    signal -= np.mean(signal)
    profile = np.cumsum(signal)

    Fq = np.zeros((len(q_vals), len(scales)))

    for i, scale in enumerate(scales):
        segments = len(profile) // scale
        rms = []

        for j in range(segments):
            seg = profile[j*scale:(j+1)*scale]
            x = np.arange(scale)
            trend = np.polyval(np.polyfit(x, seg, 1), x)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))

        rms = np.array(rms)

        for qi, q in enumerate(q_vals):
            if q == 0:
                Fq[qi, i] = np.exp(0.5 * np.mean(np.log(rms**2)))
            else:
                Fq[qi, i] = (np.mean(rms**q))**(1/q)

    return Fq

# ---------------------------
# Resolve Company → Ticker
# ---------------------------
if company_input:
    filtered = symbols_df[
        (symbols_df["Exchange"] == exchange) &
        (symbols_df["Name"].str.contains(company_input, case=False, na=False))
    ]

    if not filtered.empty:
        selected_row = st.sidebar.selectbox(
            "Matching Companies",
            filtered.itertuples(),
            format_func=lambda x: f"{x.Name} ({x.Ticker})"
        )
        selected_ticker = selected_row.Ticker
    else:
        selected_ticker = None
else:
    selected_ticker = None

# ---------------------------
# Main Analysis
# ---------------------------
st.title("📈 Fractal & Multifractal Market Analysis")

if run_button:

    if not selected_ticker:
        st.error("❌ No matching company found. Please refine your search.")
        st.stop()

    st.success(f"✅ Selected Ticker: {selected_ticker}")

    data = yf.download(selected_ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()

    prices = data["Adj Close"].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()

    col1, col2 = st.columns(2)

    # Price
    with col1:
        fig, ax = plt.subplots()
        ax.plot(prices)
        ax.set_title("Adjusted Close Price")
        st.pyplot(fig)

    # Returns
    with col2:
        fig, ax = plt.subplots()
        ax.plot(returns)
        ax.set_title("Log Returns")
        st.pyplot(fig)

    # QQ Plot
    fig, ax = plt.subplots()
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title("Return Distribution (QQ Plot)")
    st.pyplot(fig)

    # Hurst
    H = hurst(returns.values)
    st.metric("Global Hurst Exponent", round(H, 3))

    rolling_H = returns.rolling(rolling_window).apply(
        lambda x: hurst(x.values), raw=False
    )

    fig, ax = plt.subplots()
    ax.plot(rolling_H)
    ax.axhline(0.5, color="red", linestyle="--")
    ax.set_title("Rolling Hurst Exponent")
    st.pyplot(fig)

    # MF-DFA
    scales = np.arange(10, max_scale, 10)
    q_vals = np.arange(-5, 6)

    Fq = mfdfa(returns.values, scales, q_vals)
    Hq = [np.polyfit(np.log(scales), np.log(Fq[i]), 1)[0] for i in range(len(q_vals))]

    fig, ax = plt.subplots()
    ax.plot(q_vals, Hq, marker="o")
    ax.set_title("Generalized Hurst Exponents H(q)")
    st.pyplot(fig)

    # Spectrum
    tau_q = q_vals * np.array(Hq) - 1
    alpha = np.gradient(tau_q, q_vals)
    f_alpha = q_vals * alpha - tau_q

    fig, ax = plt.subplots()
    ax.plot(alpha, f_alpha, marker="o")
    ax.set_title("Multifractal Spectrum")
    st.pyplot(fig)

    st.success("🎯 Analysis Complete")



