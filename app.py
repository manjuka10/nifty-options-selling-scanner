import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# NIFTY 100 OPTIONS SELLING SCANNER - VERSION 2
# ============================================================
# Two-file project:
#   app.py
#   requirements.txt
#
# Data source in this prototype:
#   Yahoo Finance / yfinance
#
# Option-side checks now include:
#   IV, estimated Delta, OI, OI change (when available),
#   volume, bid/ask spread, liquidity, IV vs realized volatility.
#
# True IV Rank/Percentile requires a historical IV series.
# Yahoo Finance does not reliably provide that history, so V2
# does NOT pretend that current IV is IV Rank.
# ============================================================

st.set_page_config(
    page_title="Nifty 100 Options Selling Scanner",
    page_icon="📊",
    layout="wide",
)

IST = ZoneInfo("Asia/Kolkata")

# -----------------------------
# Strategy settings
# -----------------------------
EMA21 = 21
EMA50 = 50
EMA200 = 200

MIN_EXTENSION = 5.0

MIN_ATR_PCT = 1.0
MAX_ATR_PCT = 8.0

MIN_RVOL = 15.0
MAX_RVOL = 80.0

# Short-option target delta.
MIN_DELTA = 0.15
MAX_DELTA = 0.30

# Option liquidity.
MIN_OI = 500
MIN_VOLUME = 100
MAX_SPREAD_PCT = 3.0

# Require IV to be meaningfully above realized volatility.
MIN_IV_RVOL_SPREAD = 0.0

# Prefer expiries with at least this many days remaining.
MIN_DTE = 7
MAX_DTE = 60

# Only scan option chains for stocks which already pass
# the stock-side setup. This keeps the app reasonably fast.
MAX_OPTION_STOCKS_PER_REFRESH = 25


# -----------------------------
# Nifty 100 universe
# -----------------------------
NIFTY100 = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "BEL.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
    "COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","ETERNAL.NS","GRASIM.NS",
    "HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS",
    "HINDUNILVR.NS","ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS",
    "INDUSINDBK.NS","INFY.NS","ITC.NS","JINDALSTEL.NS","JSWSTEEL.NS",
    "KOTAKBANK.NS","LT.NS","LICI.NS","LTIM.NS","M&M.NS","MARUTI.NS",
    "MAXHEALTH.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS","PFC.NS",
    "PIDILITIND.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS",
    "SHRIRAMFIN.NS","SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS",
    "TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","TORNTPHARM.NS",
    "TRENT.NS","TVSMOTOR.NS","ULTRACEMCO.NS","WIPRO.NS","ZYDUSLIFE.NS",
]


# -----------------------------
# Helpers
# -----------------------------
def now_ist():
    return datetime.now(IST)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(spot, strike, t_years, iv, option_type, rate=0.06):
    """Black-Scholes delta estimate."""
    if any(pd.isna(v) for v in [spot, strike, t_years, iv]):
        return np.nan
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return np.nan

    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * iv * iv) * t_years
    ) / (iv * math.sqrt(t_years))

    if option_type == "CE":
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0


def clean_symbol(symbol):
    return symbol.replace(".NS", "")


# -----------------------------
# Stock-side calculations
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_candidates(symbols):
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

            needed = ["High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in needed):
                continue

            df = df.dropna(subset=["Close"])
            if len(df) < 210:
                continue

            close = pd.to_numeric(df["Close"], errors="coerce")
            high = pd.to_numeric(df["High"], errors="coerce")
            low = pd.to_numeric(df["Low"], errors="coerce")

            ema21 = close.ewm(span=EMA21, adjust=False).mean()
            ema50 = close.ewm(span=EMA50, adjust=False).mean()
            ema200 = close.ewm(span=EMA200, adjust=False).mean()

            prev = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev).abs(),
                (low - prev).abs(),
            ], axis=1).max(axis=1)

            atr = tr.rolling(14).mean()
            atr_pct = atr / close * 100

            daily_returns = close.pct_change()
            realized_vol = (
                daily_returns.rolling(20).std()
                * np.sqrt(252)
                * 100
            )

            price = float(close.iloc[-1])
            e21 = float(ema21.iloc[-1])
            e50 = float(ema50.iloc[-1])
            e200 = float(ema200.iloc[-1])
            atrp = float(atr_pct.iloc[-1])
            rvol = float(realized_vol.iloc[-1])

            extension = (price - e21) / e21 * 100

            bullish = price > e21 > e50 > e200
            bearish = price < e21 < e50 < e200

            volatility_ok = (
                MIN_ATR_PCT <= atrp <= MAX_ATR_PCT
                and MIN_RVOL <= rvol <= MAX_RVOL
            )

            call_core = (
                bullish
                and extension >= MIN_EXTENSION
                and volatility_ok
            )

            put_core = (
                bearish
                and extension <= -MIN_EXTENSION
                and volatility_ok
            )

            rows.append({
                "Symbol": symbol,
                "Stock": clean_symbol(symbol),
                "Price": price,
                "Trend": (
                    "Bullish" if bullish
                    else "Bearish" if bearish
                    else "Neutral"
                ),
                "Extension": extension,
                "ATR%": atrp,
                "RealizedVol%": rvol,
                "CallCore": call_core,
                "PutCore": put_core,
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


# -----------------------------
# Option-chain functions
# -----------------------------
def choose_expiry(expiries):
    """Choose the first expiry between MIN_DTE and MAX_DTE."""
    today = date.today()

    valid = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            dte = (d - today).days
            if MIN_DTE <= dte <= MAX_DTE:
                valid.append((dte, e))
        except Exception:
            pass

    if not valid:
        return None

    valid.sort()
    return valid[0][1]


def normalize_option_frame(frame, option_type, symbol, spot, expiry):
    if frame is None or frame.empty:
        return pd.DataFrame()

    x = frame.copy()

    rename = {
        "contractSymbol": "Contract",
        "strike": "Strike",
        "lastPrice": "Premium",
        "impliedVolatility": "IV",
        "openInterest": "OI",
        "volume": "Volume",
        "bid": "Bid",
        "ask": "Ask",
    }

    for old, new in rename.items():
        if old in x.columns:
            x[new] = x[old]

    required = ["Strike", "Premium", "IV", "OI", "Volume", "Bid", "Ask"]
    for c in required:
        if c not in x.columns:
            x[c] = np.nan

    x["Strike"] = pd.to_numeric(x["Strike"], errors="coerce")
    x["Premium"] = pd.to_numeric(x["Premium"], errors="coerce")
    x["IV"] = pd.to_numeric(x["IV"], errors="coerce")
    x["OI"] = pd.to_numeric(x["OI"], errors="coerce")
    x["Volume"] = pd.to_numeric(x["Volume"], errors="coerce")
    x["Bid"] = pd.to_numeric(x["Bid"], errors="coerce")
    x["Ask"] = pd.to_numeric(x["Ask"], errors="coerce")

    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    dte = max((expiry_date - date.today()).days, 1)
    t = dte / 365.0

    x["OptionType"] = option_type
    x["Symbol"] = clean_symbol(symbol)
    x["Expiry"] = expiry
    x["DTE"] = dte

    x["Delta"] = x.apply(
        lambda r: bs_delta(
            spot,
            float(r["Strike"]) if not pd.isna(r["Strike"]) else np.nan,
            t,
            float(r["IV"]) if not pd.isna(r["IV"]) else np.nan,
            option_type,
        ),
        axis=1,
    )

    x["AbsDelta"] = x["Delta"].abs()

    x["SpreadPct"] = np.where(
        x["Premium"] > 0,
        (x["Ask"] - x["Bid"]) / x["Premium"] * 100,
        np.nan,
    )

    # Current IV is a decimal in Yahoo's option data.
    x["IVPct"] = x["IV"] * 100

    return x


def get_option_candidate(stock_row, option_type):
    symbol = stock_row["Symbol"]
    spot = float(stock_row["Price"])
    rvol = float(stock_row["RealizedVol%"])

    try:
        ticker = yf.Ticker(symbol)
        expiries = ticker.options
        if not expiries:
            return None

        expiry = choose_expiry(expiries)
        if not expiry:
            return None

        chain = ticker.option_chain(expiry)

        if option_type == "CE":
            frame = chain.calls
        else:
            frame = chain.puts

        x = normalize_option_frame(
            frame,
            option_type,
            symbol,
            spot,
            expiry,
        )

        if x.empty:
            return None

        # Directional strike side:
        # CALL selling -> OTM calls only
        # PUT selling  -> OTM puts only
        if option_type == "CE":
            x = x[x["Strike"] > spot]
        else:
            x = x[x["Strike"] < spot]

        # Core option-selling checks.
        x = x[
            x["AbsDelta"].between(MIN_DELTA, MAX_DELTA)
            & (x["OI"] >= MIN_OI)
            & (x["Volume"].fillna(0) >= MIN_VOLUME)
            & (x["Bid"].fillna(0) > 0)
            & (x["Ask"].fillna(0) >= x["Bid"].fillna(0))
            & (x["SpreadPct"] <= MAX_SPREAD_PCT)
            & (x["IVPct"] >= rvol + MIN_IV_RVOL_SPREAD)
        ].copy()

        if x.empty:
            return None

        # Select the strike closest to the middle of our delta zone.
        target_delta = (MIN_DELTA + MAX_DELTA) / 2
        x["DeltaDistance"] = (x["AbsDelta"] - target_delta).abs()

        x = x.sort_values(
            ["DeltaDistance", "OI", "Volume"],
            ascending=[True, False, False],
        )

        row = x.iloc[0].to_dict()

        # OI change is not consistently exposed by Yahoo.
        row["OIChange"] = np.nan

        return row

    except Exception:
        return None


# -----------------------------
# Full scanner
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def run_full_scan():
    stocks = load_stock_candidates(tuple(NIFTY100))

    if stocks.empty:
        return pd.DataFrame()

    # Scan only stocks that have passed the stock-side setup.
    candidates = stocks[
        stocks["CallCore"] | stocks["PutCore"]
    ].copy()

    # Avoid excessive API calls.
    candidates = candidates.sort_values(
        "Extension",
        key=lambda s: s.abs(),
        ascending=False,
    ).head(MAX_OPTION_STOCKS_PER_REFRESH)

    results = []

    for _, stock in candidates.iterrows():

        if bool(stock["CallCore"]):
            option = get_option_candidate(stock, "CE")
            if option:
                results.append({
                    "Stock": stock["Stock"],
                    "Price": stock["Price"],
                    "Trend": stock["Trend"],
                    "Extension": stock["Extension"],
                    "Setup": "CALL",
                    "Strike": option["Strike"],
                    "Premium": option["Premium"],
                    "Delta": option["Delta"],
                    "IV": option["IVPct"],
                    "OI": option["OI"],
                    "OIChange": option["OIChange"],
                    "Volume": option["Volume"],
                    "SpreadPct": option["SpreadPct"],
                    "Expiry": option["Expiry"],
                    "DTE": option["DTE"],
                    "ATR%": stock["ATR%"],
                    "RealizedVol%": stock["RealizedVol%"],
                })

        if bool(stock["PutCore"]):
            option = get_option_candidate(stock, "PE")
            if option:
                results.append({
                    "Stock": stock["Stock"],
                    "Price": stock["Price"],
                    "Trend": stock["Trend"],
                    "Extension": stock["Extension"],
                    "Setup": "PUT",
                    "Strike": option["Strike"],
                    "Premium": option["Premium"],
                    "Delta": option["Delta"],
                    "IV": option["IVPct"],
                    "OI": option["OI"],
                    "OIChange": option["OIChange"],
                    "Volume": option["Volume"],
                    "SpreadPct": option["SpreadPct"],
                    "Expiry": option["Expiry"],
                    "DTE": option["DTE"],
                    "ATR%": stock["ATR%"],
                    "RealizedVol%": stock["RealizedVol%"],
                })

    return pd.DataFrame(results)


# -----------------------------
# UI
# -----------------------------
st.title("Options Selling Scanner")
st.caption(
    "Nifty 100 • Stock trend + extension + volatility + live option-chain checks"
)

top1, top2, top3 = st.columns([1.0, 1.0, 2.4])

with top1:
    if st.button("↻ Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.session_state["last_updated"] = now_ist()
        st.rerun()

with top2:
    selected = st.radio(
        "Filter",
        ["CALL", "PUT"],
        horizontal=True,
        label_visibility="collapsed",
    )

if "last_updated" not in st.session_state:
    st.session_state["last_updated"] = now_ist()

with top3:
    st.markdown(
        "**Last updated:** "
        + st.session_state["last_updated"].strftime(
            "%d-%b-%Y %I:%M:%S %p IST"
        )
    )

with st.spinner("Scanning stock conditions and live option chains..."):
    results = run_full_scan()

if results.empty:
    st.info(
        f"No qualifying {selected} setups at the latest refresh. "
        "This means either the stock conditions or option conditions did not pass."
    )
else:
    visible = results[results["Setup"] == selected].copy()

    if visible.empty:
        st.info(
            f"No qualifying {selected} setups at the latest refresh."
        )
    else:
        visible["Price"] = visible["Price"].map(
            lambda x: f"₹{x:,.2f}"
        )
        visible["Extension"] = visible["Extension"].map(
            lambda x: f"{x:+.1f}%"
        )

        # EXACTLY the requested visible columns.
        table = visible[
            ["Stock", "Price", "Trend", "Extension", "Setup"]
        ].copy()

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        # Background details are available without adding columns
        # to the main screen.
        with st.expander("Selected setup details"):
            details = visible.copy()
            details["Premium"] = details["Premium"].map(
                lambda x: f"₹{x:,.2f}"
            )
            details["Delta"] = details["Delta"].map(
                lambda x: f"{x:+.2f}"
            )
            details["IV"] = details["IV"].map(
                lambda x: f"{x:.1f}%"
            )
            details["SpreadPct"] = details["SpreadPct"].map(
                lambda x: f"{x:.2f}%"
            )
            details["RealizedVol%"] = details["RealizedVol%"].map(
                lambda x: f"{x:.1f}%"
            )
            details["ATR%"] = details["ATR%"].map(
                lambda x: f"{x:.1f}%"
            )

            st.dataframe(
                details[
                    [
                        "Stock", "Setup", "Expiry", "DTE",
                        "Strike", "Premium", "Delta", "IV",
                        "OI", "OIChange", "Volume",
                        "SpreadPct", "ATR%", "RealizedVol%",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

st.divider()

st.caption(
    "V2 uses current option IV, estimated Black-Scholes Delta, OI, volume "
    "and bid/ask spread. True IV Rank/Percentile requires historical IV data "
    "and is not fabricated when the data source does not provide it. "
    "No orders are placed by this app."
)
