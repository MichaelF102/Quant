import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Fractal & Multifractal Market Analysis",
    layout="wide"
)

st.title("📈 Fractal & Multifractal Analysis of Financial Markets")
st.markdown(
    """
    **Fractals reveal long memory.  
    Multifractals reveal market complexity.**

    This app analyzes real financial data using:
    - Hurst Exponent
    - Rolling Fractal Regimes
    - Multifractal Detrended Fluctuation Analysis (MF-DFA)
    """
)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.text_input("Ticker", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

window = st.sidebar.slider("Rolling Hurst Window", 100, 500, 252)
max_scale = st.sidebar.slider("Max MF-DFA Scale", 100, 500, 250)

run = st.sidebar.button("▶ Run Analysis")

# ---------------------------
# Utility Functions
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
            segment = profile[j*scale:(j+1)*scale]
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms.append(np.sqrt(np.mean((segment - trend)**2)))

        rms = np.array(rms)

        for q_i, q in enumerate(q_vals):
            if q == 0:
                Fq[q_i, i] = np.exp(0.5 * np.mean(np.log(rms**2)))
            else:
                Fq[q_i, i] = (np.mean(rms**q))**(1/q)

    return Fq

# ---------------------------
# Main Logic
# ---------------------------
if run:
    st.subheader("📥 Downloading Data")
    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        st.error("No data found. Check ticker or date range.")
        st.stop()

    prices = data["Adj Close"].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()

    col1, col2 = st.columns(2)

    # ---------------------------
    # Price Plot
    # ---------------------------
    with col1:
        st.markdown("### 💲 Price Series")
        fig, ax = plt.subplots()
        ax.plot(prices)
        ax.set_title("Adjusted Close Price")
        st.pyplot(fig)

    # ---------------------------
    # Returns Plot
    # ---------------------------
    with col2:
        st.markdown("### 🔄 Log Returns")
        fig, ax = plt.subplots()
        ax.plot(returns)
        ax.set_title("Log Returns")
        st.pyplot(fig)

    # ---------------------------
    # Distribution
    # ---------------------------
    st.markdown("### 📊 Return Distribution")
    fig, ax = plt.subplots()
    stats.probplot(returns, dist="norm", plot=ax)
    st.pyplot(fig)

    # ---------------------------
    # Hurst Exponent
    # ---------------------------
    st.markdown("## 📐 Fractal Analysis")

    H = hurst(returns.values)
    st.metric("Global Hurst Exponent", round(H, 3))

    rolling_H = returns.rolling(window).apply(
        lambda x: hurst(x.values), raw=False
    )

    fig, ax = plt.subplots()
    ax.plot(rolling_H)
    ax.axhline(0.5, color="red", linestyle="--")
    ax.set_title("Rolling Hurst Exponent")
    st.pyplot(fig)

    # ---------------------------
    # MF-DFA
    # ---------------------------
    st.markdown("## 🌈 Multifractal Analysis (MF-DFA)")

    scales = np.arange(10, max_scale, 10)
    q_vals = np.arange(-5, 6)

    Fq = mfdfa(returns.values, scales, q_vals)

    Hq = []
    for i in range(len(q_vals)):
        coeffs = np.polyfit(np.log(scales), np.log(Fq[i]), 1)
        Hq.append(coeffs[0])

    Hq = np.array(Hq)

    fig, ax = plt.subplots()
    ax.plot(q_vals, Hq, marker="o")
    ax.set_xlabel("q")
    ax.set_ylabel("H(q)")
    ax.set_title("Generalized Hurst Exponents")
    st.pyplot(fig)

    # ---------------------------
    # Multifractal Spectrum
    # ---------------------------
    tau_q = q_vals * Hq - 1
    alpha = np.gradient(tau_q, q_vals)
    f_alpha = q_vals * alpha - tau_q

    fig, ax = plt.subplots()
    ax.plot(alpha, f_alpha, marker="o")
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    ax.set_title("Multifractal Spectrum")
    st.pyplot(fig)

    st.success("✅ Analysis Complete")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    """
    **Built for Quant Research & Market Microstructure Analysis**  
    Fractal • Multifractal • Regime Detection
    """
)
