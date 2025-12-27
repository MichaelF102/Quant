import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Market Fractal Analyzer",
    layout="wide"
)

st.title("📈 Market Fractal & Multifractal Analyzer")
st.markdown(
    """
    This application performs **Fractal (Hurst)** and  
    **Multifractal (MF-DFA)** analysis on financial markets.

    ✔ Exchange-aware instrument selection  
    ✔ Company name → ticker resolution  
    ✔ Research-grade diagnostics  
    """
)

# =====================================================
# LOAD SYMBOL MASTER (ROBUST PATH HANDLING)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "Market_Data.csv")

if not os.path.exists(CSV_PATH):
    st.error(f"❌ CSV file not found at:\n{CSV_PATH}")
    st.stop()

@st.cache_data
def load_symbol_master(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required_cols = {"Ticker", "Name", "Exchange"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain Ticker, Name, Exchange columns")
    return df

symbols_df = load_symbol_master(CSV_PATH)

# =====================================================
# SIDEBAR — USER INPUTS
# =====================================================
st.sidebar.header("🔎 Instrument Selection")

exchange = st.sidebar.selectbox(
    "Select Exchange",
    sorted(symbols_df["Exchange"].unique())
)

company_name = st.sidebar.text_input(
    "Company Name",
    placeholder="e.g. Apple, Reliance, Tesla"
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

rolling_window = st.sidebar.slider("Rolling Hurst Window", 100, 500, 252)
max_scale = st.sidebar.slider("MF-DFA Max Scale", 100, 500, 250)

run_analysis = st.sidebar.button("▶ Run Analysis")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def hurst_exponent(series):
    lags = range(2, 100)
    tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    return slope

def mfdfa(signal, scales, q_values):
    signal = np.array(signal)
    signal -= np.mean(signal)
    profile = np.cumsum(signal)

    Fq = np.zeros((len(q_values), len(scales)))

    for i, scale in enumerate(scales):
        segments = len(profile) // scale
        rms = []

        for j in range(segments):
            segment = profile[j * scale:(j + 1) * scale]
            x = np.arange(scale)
            trend = np.polyval(np.polyfit(x, segment, 1), x)
            rms.append(np.sqrt(np.mean((segment - trend) ** 2)))

        rms = np.array(rms)

        for qi, q in enumerate(q_values):
            if q == 0:
                Fq[qi, i] = np.exp(0.5 * np.mean(np.log(rms ** 2)))
            else:
                Fq[qi, i] = (np.mean(rms ** q)) ** (1 / q)

    return Fq

# =====================================================
# COMPANY → TICKER RESOLUTION
# =====================================================
selected_ticker = None

if company_name:
    matches = symbols_df[
        (symbols_df["Exchange"] == exchange) &
        (symbols_df["Name"].str.contains(company_name, case=False, na=False))
    ]

    if not matches.empty:
        choice = st.sidebar.selectbox(
            "Matching Companies",
            matches.itertuples(),
            format_func=lambda x: f"{x.Name} ({x.Ticker})"
        )
        selected_ticker = choice.Ticker
    else:
        st.sidebar.warning("No matching company found.")

# =====================================================
# MAIN ANALYSIS
# =====================================================
if run_analysis:

    if selected_ticker is None:
        st.error("❌ Please select a valid company before running analysis.")
        st.stop()

    st.success(f"✅ Selected Ticker: {selected_ticker}")

    data = yf.download(selected_ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("❌ No data returned from Yahoo Finance.")
        st.stop()

    prices = data["Adj Close"].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()

    # ---------------- PRICE & RETURNS ----------------
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(prices)
        ax.set_title("Adjusted Close Price")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(returns)
        ax.set_title("Log Returns")
        st.pyplot(fig)

    # ---------------- DISTRIBUTION ----------------
    st.subheader("📊 Return Distribution")
    fig, ax = plt.subplots()
    stats.probplot(returns, dist="norm", plot=ax)
    st.pyplot(fig)

    # ---------------- HURST ----------------
    st.subheader("📐 Fractal Analysis")

    H = hurst_exponent(returns.values)
    st.metric("Global Hurst Exponent", round(H, 3))

    rolling_H = returns.rolling(rolling_window).apply(
        lambda x: hurst_exponent(x.values), raw=False
    )

    fig, ax = plt.subplots()
    ax.plot(rolling_H)
    ax.axhline(0.5, linestyle="--", color="red")
    ax.set_title("Rolling Hurst Exponent")
    st.pyplot(fig)

    # ---------------- MF-DFA ----------------
    st.subheader("🌈 Multifractal Analysis (MF-DFA)")

    scales = np.arange(10, max_scale, 10)
    q_vals = np.arange(-5, 6)

    Fq = mfdfa(returns.values, scales, q_vals)
    Hq = [np.polyfit(np.log(scales), np.log(Fq[i]), 1)[0] for i in range(len(q_vals))]

    fig, ax = plt.subplots()
    ax.plot(q_vals, Hq, marker="o")
    ax.set_title("Generalized Hurst Exponents H(q)")
    st.pyplot(fig)

    # ---------------- SPECTRUM ----------------
    tau_q = q_vals * np.array(Hq) - 1
    alpha = np.gradient(tau_q, q_vals)
    f_alpha = q_vals * alpha - tau_q

    fig, ax = plt.subplots()
    ax.plot(alpha, f_alpha, marker="o")
    ax.set_title("Multifractal Spectrum")
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    st.pyplot(fig)

    st.success("🎯 Analysis Complete")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "**Fractal • Multifractal • Market Regimes**  \n"
    "Built for Quantitative Research & Financial Analysis"
)
