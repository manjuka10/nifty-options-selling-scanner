import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Nifty 100 Options Selling Scanner",
    page_icon="📊",
    layout="wide"
)

# =========================
# SETTINGS
# =========================
EMA_FAST = 21
EMA_MID = 50
EMA_SLOW = 200

MIN_EXTENSION_PCT = 5.0
MIN_ATR_PCT = 1.0
MAX_ATR_PCT = 8.0
MIN_REALIZED_VOL_PCT = 15.0
MAX_REALIZED_VOL_PCT = 80.0

# Core Nifty 100 symbols. This list can be expanded/updated without
# changing the scanner logic.
NIFTY100 = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "BEL.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
    "COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","ETERNAL.NS","GRASIM.NS",
    "HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS",
    "HINDUNILVR.NS","ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS","INDUSINDBK.NS",
    "INFY.NS","ITC.NS","JINDALSTEL.NS","JSWSTEEL.NS","KOTAKBANK.NS",
    "LT.NS","LICI.NS","LTIM.NS","M&M.NS","MARUTI.NS","MAXHEALTH.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","PFC.NS","PIDILITIND.NS",
    "POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SHRIRAMFIN.NS",
    "SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TCS.NS",
    "TECHM.NS","TITAN.NS","TORNTPHARM.NS","TRENT.NS","TVSMOTOR.NS",
    "ULTRACEMCO.NS","WIPRO.NS","ZYDUSLIFE.NS",
]

# =========================
# DATA
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(symbols):
    rows = []
    for symbol in symbols:
        try:
            df = yf.download(
                symbol,
                period="2y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in required):
                continue

            df = df.dropna(subset=["Close"])
            if len(df) < 210:
                continue

            close = pd.to_numeric(df["Close"], errors="coerce")
            high = pd.to_numeric(df["High"], errors="coerce")
            low = pd.to_numeric(df["Low"], errors="coerce")

            ema21 = close.ewm(span=EMA_FAST, adjust=False).mean()
            ema50 = close.ewm(span=EMA_MID, adjust=False).mean()
            ema200 = close.ewm(span=EMA_SLOW, adjust=False).mean()

            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)

            atr14 = tr.rolling(14).mean()
            atr_pct = atr14 / close * 100

            returns = close.pct_change()
            realized_vol = returns.rolling(20).std() * np.sqrt(252) * 100

            price = float(close.iloc[-1])
            e21 = float(ema21.iloc[-1])
            e50 = float(ema50.iloc[-1])
            e200 = float(ema200.iloc[-1])
            extension = (price - e21) / e21 * 100
            atrp = float(atr_pct.iloc[-1])
            rvol = float(realized_vol.iloc[-1])

            bullish = price > e21 > e50 > e200
            bearish = price < e21 < e50 < e200
            volatility_ok = (
                MIN_ATR_PCT <= atrp <= MAX_ATR_PCT
                and MIN_REALIZED_VOL_PCT <= rvol <= MAX_REALIZED_VOL_PCT
            )

            call_setup = bullish and extension >= MIN_EXTENSION_PCT and volatility_ok
            put_setup = bearish and extension <= -MIN_EXTENSION_PCT and volatility_ok

            if call_setup:
                setup = "CALL"
            elif put_setup:
                setup = "PUT"
            else:
                setup = ""

            trend = "Bullish" if bullish else "Bearish" if bearish else "Neutral"

            rows.append({
                "Stock": symbol.replace(".NS", ""),
                "Price": price,
                "Trend": trend,
                "Extension": extension,
                "Setup": setup,
                # Background fields — not shown in main table
                "EMA21": e21,
                "EMA50": e50,
                "EMA200": e200,
                "ATR%": atrp,
                "RealizedVol%": rvol,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def run_scan():
    return get_market_data(tuple(NIFTY100))


# =========================
# UI
# =========================
st.title("Nifty 100 Options Selling Scanner")
st.caption("Background indicators are calculated automatically. Only qualifying CALL/PUT setups are displayed.")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("↻ Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col2:
    mode = st.radio("Filter", ["CALL", "PUT"], horizontal=True, label_visibility="collapsed")

if "last_updated" not in st.session_state:
    st.session_state.last_updated = None

if st.session_state.last_updated is None:
    st.session_state.last_updated = datetime.now().astimezone()

with col3:
    st.markdown(
        f"**Last updated:** {st.session_state.last_updated.strftime('%d-%b-%Y %I:%M:%S %p %Z')}"
    )

with st.spinner("Scanning Nifty 100 stocks..."):
    results = run_scan()

if results.empty:
    st.warning("No market data was returned. Click Refresh and try again.")
else:
    filtered = results[results["Setup"] == mode].copy()

    if filtered.empty:
        st.info(f"No qualifying {mode} setups at the latest refresh.")
    else:
        filtered["Price"] = filtered["Price"].map(lambda x: f"₹{x:,.2f}")
        filtered["Extension"] = filtered["Extension"].map(lambda x: f"{x:+.1f}%")

        st.dataframe(
            filtered[["Stock", "Price", "Trend", "Extension", "Setup"]],
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    "Prototype: stock-side screening only. Live option-chain IV, IV Rank, Delta, OI, "
    "OI change, volume and bid/ask should be connected through a broker/exchange data source "
    "before treating a row as a complete options-selling setup."
)
