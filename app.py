import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from kiteconnect import KiteConnect

# ============================================================
# NIFTY 100 OPTIONS SELLING SCANNER - V3
# ============================================================
# Files required:
#   app.py
#   requirements.txt
#
# Live option data:
#   Zerodha Kite Connect
#
# Add these to Streamlit Cloud -> Settings -> Secrets:
#
# KITE_API_KEY = "your_api_key"
# KITE_ACCESS_TOKEN = "your_daily_access_token"
#
# The access token is normally refreshed for the trading day.
#
# Main screen:
#   Stock | Price | Trend | Extension | Setup
#
# Background checks:
#   21/50/200 EMA
#   Distance from 21 EMA
#   ATR
#   Realized volatility
#   Current IV
#   IV premium vs realized volatility
#   Black-Scholes Delta
#   OI
#   OI change (from the previous scan in this running app)
#   Volume
#   Bid/ask spread
#   Expiry/DTE
#
# True IV Rank/Percentile needs a stored historical IV series.
# V3 deliberately does not pretend that current IV is IV Rank.
# ============================================================

st.set_page_config(
    page_title="Nifty 100 Options Selling Scanner",
    page_icon="📊",
    layout="wide",
)

IST = ZoneInfo("Asia/Kolkata")

# -----------------------------
# Strategy parameters
# -----------------------------
EMA21 = 21
EMA50 = 50
EMA200 = 200

MIN_EXTENSION = 5.0

MIN_ATR_PCT = 1.0
MAX_ATR_PCT = 8.0

MIN_REALIZED_VOL = 15.0
MAX_REALIZED_VOL = 80.0

MIN_DELTA = 0.15
MAX_DELTA = 0.30
TARGET_DELTA = 0.22

MIN_OI = 500
MIN_VOLUME = 100
MAX_SPREAD_PCT = 3.0

# Require option IV to be at least this many percentage points
# above 20-day annualized realized volatility.
MIN_IV_PREMIUM = 0.0

MIN_DTE = 7
MAX_DTE = 60

RISK_FREE_RATE = 0.06

# Only inspect option chains for stocks that pass stock-side
# conditions. This controls API calls.
MAX_OPTION_STOCKS = 30

# Nifty 100 universe used by this app.
# You can replace/update this list later without changing the logic.
NIFTY100 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT",
    "AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
    "BEL","BHARTIARTL","BPCL","BRITANNIA","CIPLA",
    "COALINDIA","DRREDDY","EICHERMOT","ETERNAL","GRASIM",
    "HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO",
    "HINDUNILVR","ICICIBANK","ICICIGI","ICICIPRULI",
    "INDUSINDBK","INFY","ITC","JINDALSTEL","JSWSTEEL",
    "KOTAKBANK","LT","LICI","LTIM","M&M","MARUTI",
    "MAXHEALTH","NESTLEIND","NTPC","ONGC","PFC",
    "PIDILITIND","POWERGRID","RELIANCE","SBILIFE","SBIN",
    "SHRIRAMFIN","SUNPHARMA","TATACONSUM","TATAMOTORS",
    "TATASTEEL","TCS","TECHM","TITAN","TORNTPHARM","TRENT",
    "TVSMOTOR","ULTRACEMCO","WIPRO","ZYDUSLIFE",
]


# ============================================================
# KITE CONNECTION
# ============================================================
@st.cache_resource
def get_kite():
    api_key = st.secrets.get("KITE_API_KEY", "")
    access_token = st.secrets.get("KITE_ACCESS_TOKEN", "")

    if not api_key or not access_token:
        return None, (
            "Kite credentials are missing. Add KITE_API_KEY and "
            "KITE_ACCESS_TOKEN in Streamlit Secrets."
        )

    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        # Small authenticated test.
        kite.profile()
        return kite, None
    except Exception as e:
        return None, f"Kite connection failed: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_instruments(_kite):
    return pd.DataFrame(_kite.instruments("NSE"))


@st.cache_data(ttl=3600, show_spinner=False)
def load_nfo_instruments(_kite):
    return pd.DataFrame(_kite.instruments("NFO"))


# ============================================================
# MATH / OPTIONS
# ============================================================
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_d1(spot, strike, t, sigma, rate):
    return (
        math.log(spot / strike)
        + (rate + 0.5 * sigma * sigma) * t
    ) / (sigma * math.sqrt(t))


def bs_delta(spot, strike, t, sigma, option_type, rate=RISK_FREE_RATE):
    if (
        spot <= 0 or strike <= 0 or t <= 0
        or sigma <= 0 or pd.isna(sigma)
    ):
        return np.nan

    d1 = bs_d1(spot, strike, t, sigma, rate)

    if option_type == "CE":
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0


def bs_price(spot, strike, t, sigma, option_type, rate=RISK_FREE_RATE):
    if (
        spot <= 0 or strike <= 0 or t <= 0
        or sigma <= 0 or pd.isna(sigma)
    ):
        return np.nan

    d1 = bs_d1(spot, strike, t, sigma, rate)
    d2 = d1 - sigma * math.sqrt(t)

    if option_type == "CE":
        return (
            spot * norm_cdf(d1)
            - strike * math.exp(-rate * t) * norm_cdf(d2)
        )

    return (
        strike * math.exp(-rate * t) * norm_cdf(-d2)
        - spot * norm_cdf(-d1)
    )


def implied_volatility(market_price, spot, strike, t, option_type):
    """Simple bisection IV solver."""
    if (
        pd.isna(market_price)
        or pd.isna(spot)
        or pd.isna(strike)
        or market_price <= 0
        or spot <= 0
        or strike <= 0
        or t <= 0
    ):
        return np.nan

    intrinsic = max(
        0.0,
        spot - strike if option_type == "CE" else strike - spot
    )

    if market_price <= intrinsic:
        return np.nan

    lo = 0.01
    hi = 5.0

    try:
        for _ in range(80):
            mid = (lo + hi) / 2.0
            p = bs_price(spot, strike, t, mid, option_type)

            if pd.isna(p):
                return np.nan

            if p > market_price:
                hi = mid
            else:
                lo = mid

        return (lo + hi) / 2.0
    except Exception:
        return np.nan


# ============================================================
# STOCK SIDE
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def stock_screen():
    rows = []

    for symbol in NIFTY100:
        try:
            ticker = f"{symbol}.NS"
            df = yf.download(
                ticker,
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

            needed = ["High", "Low", "Close"]
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
            tr = pd.concat(
                [
                    high - low,
                    (high - prev).abs(),
                    (low - prev).abs(),
                ],
                axis=1,
            ).max(axis=1)

            atr = tr.rolling(14).mean()
            atr_pct = atr / close * 100

            returns = close.pct_change()
            rvol = returns.rolling(20).std() * np.sqrt(252) * 100

            price = float(close.iloc[-1])
            e21 = float(ema21.iloc[-1])
            e50 = float(ema50.iloc[-1])
            e200 = float(ema200.iloc[-1])
            atrp = float(atr_pct.iloc[-1])
            realized_vol = float(rvol.iloc[-1])

            extension = (price - e21) / e21 * 100

            bullish = price > e21 > e50 > e200
            bearish = price < e21 < e50 < e200

            vol_ok = (
                MIN_ATR_PCT <= atrp <= MAX_ATR_PCT
                and MIN_REALIZED_VOL <= realized_vol <= MAX_REALIZED_VOL
            )

            call_core = (
                bullish
                and extension >= MIN_EXTENSION
                and vol_ok
            )

            put_core = (
                bearish
                and extension <= -MIN_EXTENSION
                and vol_ok
            )

            rows.append(
                {
                    "Symbol": symbol,
                    "Price": price,
                    "Trend": (
                        "Bullish"
                        if bullish
                        else "Bearish"
                        if bearish
                        else "Neutral"
                    ),
                    "Extension": extension,
                    "ATR%": atrp,
                    "RealizedVol%": realized_vol,
                    "CallCore": call_core,
                    "PutCore": put_core,
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)


# ============================================================
# OPTION CHAIN
# ============================================================
def option_instruments_for_symbol(nfo, symbol):
    x = nfo[
        (nfo["name"] == symbol)
        & (nfo["instrument_type"].isin(["CE", "PE"]))
    ].copy()

    if x.empty:
        return x

    x["expiry"] = pd.to_datetime(x["expiry"]).dt.date
    today = date.today()

    x["DTE"] = x["expiry"].apply(lambda d: (d - today).days)

    return x[
        x["DTE"].between(MIN_DTE, MAX_DTE)
    ].copy()


def get_quote_map(kite, instruments):
    """
    Fetch live quotes in chunks. Kite's quote endpoint is used for
    current LTP, OI, volume, bid/ask and OHLC-related fields.
    """
    result = {}

    names = []
    for _, r in instruments.iterrows():
        names.append(
            f"NFO:{r['tradingsymbol']}"
        )

    # Keep requests manageable.
    for i in range(0, len(names), 200):
        chunk = names[i:i + 200]
        try:
            result.update(kite.quote(chunk))
        except Exception:
            continue

    return result


def build_option_candidates(kite, nfo, stock_row, option_type):
    symbol = stock_row["Symbol"]
    spot = float(stock_row["Price"])
    realized_vol = float(stock_row["RealizedVol%"])

    inst = option_instruments_for_symbol(nfo, symbol)

    if inst.empty:
        return pd.DataFrame()

    inst = inst[
        inst["instrument_type"] == option_type
    ].copy()

    # OTM only.
    if option_type == "CE":
        inst = inst[inst["strike"] > spot]
    else:
        inst = inst[inst["strike"] < spot]

    if inst.empty:
        return pd.DataFrame()

    # To keep the live scan light, focus around spot.
    inst["distance"] = (inst["strike"] - spot).abs()
    inst = (
        inst.sort_values("distance")
        .groupby("expiry", as_index=False, group_keys=False)
        .head(25)
    )

    quotes = get_quote_map(kite, inst)

    rows = []

    for _, ins in inst.iterrows():
        key = f"NFO:{ins['tradingsymbol']}"
        q = quotes.get(key)

        if not q:
            continue

        depth = q.get("depth", {})
        buy = depth.get("buy", []) or []
        sell = depth.get("sell", []) or []

        bid = float(buy[0]["price"]) if buy else 0.0
        ask = float(sell[0]["price"]) if sell else 0.0

        ltp = float(q.get("last_price") or 0.0)
        oi = float(q.get("oi") or 0.0)
        volume = float(q.get("volume") or 0.0)

        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        else:
            mid = ltp

        if mid <= 0:
            continue

        expiry = ins["expiry"]
        dte = max((expiry - date.today()).days, 1)
        t = dte / 365.0

        iv = implied_volatility(
            mid,
            spot,
            float(ins["strike"]),
            t,
            option_type,
        )

        if pd.isna(iv):
            continue

        delta = bs_delta(
            spot,
            float(ins["strike"]),
            t,
            iv,
            option_type,
        )

        iv_pct = iv * 100
        abs_delta = abs(delta)

        spread_pct = (
            (ask - bid) / mid * 100
            if mid > 0
            else np.nan
        )

        rows.append(
            {
                "Symbol": symbol,
                "Tradingsymbol": ins["tradingsymbol"],
                "Expiry": expiry,
                "DTE": dte,
                "OptionType": option_type,
                "Strike": float(ins["strike"]),
                "Premium": mid,
                "Bid": bid,
                "Ask": ask,
                "SpreadPct": spread_pct,
                "IV": iv_pct,
                "Delta": delta,
                "AbsDelta": abs_delta,
                "OI": oi,
                "Volume": volume,
                "RealizedVol": realized_vol,
                "IVPremium": iv_pct - realized_vol,
                "InstrumentToken": int(ins["instrument_token"]),
            }
        )

    x = pd.DataFrame(rows)

    if x.empty:
        return x

    # Core option conditions.
    x["LiquidityOK"] = (
        (x["OI"] >= MIN_OI)
        & (x["Volume"] >= MIN_VOLUME)
        & (x["Bid"] > 0)
        & (x["Ask"] >= x["Bid"])
        & (x["SpreadPct"] <= MAX_SPREAD_PCT)
    )

    x["DeltaOK"] = x["AbsDelta"].between(
        MIN_DELTA, MAX_DELTA
    )

    x["IVOK"] = (
        x["IVPremium"] >= MIN_IV_PREMIUM
    )

    x["OptionOK"] = (
        x["LiquidityOK"]
        & x["DeltaOK"]
        & x["IVOK"]
    )

    x = x[x["OptionOK"]].copy()

    if x.empty:
        return x

    x["DeltaDistance"] = (
        x["AbsDelta"] - TARGET_DELTA
    ).abs()

    # Prefer nearest target delta, then higher OI,
    # then tighter spread.
    x = x.sort_values(
        ["DeltaDistance", "OI", "SpreadPct"],
        ascending=[True, False, True],
    )

    return x


# ============================================================
# SCAN
# ============================================================
def full_scan(kite):
    stocks = stock_screen()

    if stocks.empty:
        return pd.DataFrame()

    candidates = stocks[
        stocks["CallCore"] | stocks["PutCore"]
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Strongest extensions first.
    candidates["abs_ext"] = candidates["Extension"].abs()
    candidates = candidates.sort_values(
        "abs_ext",
        ascending=False,
    ).head(MAX_OPTION_STOCKS)

    nfo = load_nfo_instruments(kite)

    if nfo.empty:
        return pd.DataFrame()

    results = []

    for _, stock in candidates.iterrows():

        if bool(stock["CallCore"]):
            options = build_option_candidates(
                kite,
                nfo,
                stock,
                "CE",
            )

            if not options.empty:
                o = options.iloc[0]

                results.append(
                    {
                        "Stock": stock["Symbol"],
                        "Price": stock["Price"],
                        "Trend": stock["Trend"],
                        "Extension": stock["Extension"],
                        "Setup": "CALL",
                        "Expiry": o["Expiry"],
                        "DTE": o["DTE"],
                        "Strike": o["Strike"],
                        "Premium": o["Premium"],
                        "Delta": o["Delta"],
                        "IV": o["IV"],
                        "OI": o["OI"],
                        "Volume": o["Volume"],
                        "SpreadPct": o["SpreadPct"],
                        "IVPremium": o["IVPremium"],
                        "ATR%": stock["ATR%"],
                        "RealizedVol%": stock["RealizedVol%"],
                    }
                )

        if bool(stock["PutCore"]):
            options = build_option_candidates(
                kite,
                nfo,
                stock,
                "PE",
            )

            if not options.empty:
                o = options.iloc[0]

                results.append(
                    {
                        "Stock": stock["Symbol"],
                        "Price": stock["Price"],
                        "Trend": stock["Trend"],
                        "Extension": stock["Extension"],
                        "Setup": "PUT",
                        "Expiry": o["Expiry"],
                        "DTE": o["DTE"],
                        "Strike": o["Strike"],
                        "Premium": o["Premium"],
                        "Delta": o["Delta"],
                        "IV": o["IV"],
                        "OI": o["OI"],
                        "Volume": o["Volume"],
                        "SpreadPct": o["SpreadPct"],
                        "IVPremium": o["IVPremium"],
                        "ATR%": stock["ATR%"],
                        "RealizedVol%": stock["RealizedVol%"],
                    }
                )

    return pd.DataFrame(results)


# ============================================================
# UI
# ============================================================
st.title("Options Selling Scanner")
st.caption(
    "Nifty 100 • Stock trend + extension + volatility + live Kite option-chain checks"
)

kite, kite_error = get_kite()

if kite_error:
    st.error(kite_error)
    st.info(
        "Add KITE_API_KEY and KITE_ACCESS_TOKEN under "
        "Streamlit Cloud → Settings → Secrets, then reboot the app."
    )
    st.stop()

top1, top2, top3 = st.columns([1.0, 1.0, 2.5])

with top1:
    if st.button(
        "↻ Refresh",
        type="primary",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now(IST)
        st.rerun()

with top2:
    selected = st.radio(
        "Filter",
        ["CALL", "PUT"],
        horizontal=True,
        label_visibility="collapsed",
    )

if "last_updated" not in st.session_state:
    st.session_state["last_updated"] = datetime.now(IST)

with top3:
    st.markdown(
        "**Last updated:** "
        + st.session_state["last_updated"].strftime(
            "%d-%b-%Y %I:%M:%S %p IST"
        )
    )

with st.spinner(
    "Scanning Nifty 100 and live option chains..."
):
    try:
        results = full_scan(kite)
    except Exception as e:
        results = pd.DataFrame()
        st.error(f"Scan failed: {e}")

if results.empty:
    st.info(
        f"No qualifying {selected} setups at the latest refresh. "
        "This can mean the stock-side conditions or live option conditions "
        "did not pass."
    )
else:
    visible = results[
        results["Setup"] == selected
    ].copy()

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

        st.dataframe(
            visible[
                [
                    "Stock",
                    "Price",
                    "Trend",
                    "Extension",
                    "Setup",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Selected setup details"
        ):
            details = visible.copy()

            details["Expiry"] = details[
                "Expiry"
            ].astype(str)

            details["Strike"] = details[
                "Strike"
            ].map(lambda x: f"{x:,.0f}")

            details["Premium"] = details[
                "Premium"
            ].map(lambda x: f"₹{x:,.2f}")

            details["Delta"] = details[
                "Delta"
            ].map(lambda x: f"{x:+.2f}")

            details["IV"] = details[
                "IV"
            ].map(lambda x: f"{x:.1f}%")

            details["IVPremium"] = details[
                "IVPremium"
            ].map(lambda x: f"{x:+.1f}%")

            details["SpreadPct"] = details[
                "SpreadPct"
            ].map(lambda x: f"{x:.2f}%")

            details["RealizedVol%"] = details[
                "RealizedVol%"
            ].map(lambda x: f"{x:.1f}%")

            details["ATR%"] = details[
                "ATR%"
            ].map(lambda x: f"{x:.1f}%")

            st.dataframe(
                details[
                    [
                        "Stock",
                        "Setup",
                        "Expiry",
                        "DTE",
                        "Strike",
                        "Premium",
                        "Delta",
                        "IV",
                        "IVPremium",
                        "OI",
                        "Volume",
                        "SpreadPct",
                        "ATR%",
                        "RealizedVol%",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

st.divider()

st.caption(
    "V3 uses Kite Connect live option quotes and builds the option chain "
    "from the NFO instrument dump. IV and Delta are calculated locally. "
    "Current OI, volume and bid/ask come from live quotes. OI change is "
    "not treated as available unless a previous live snapshot is stored. "
    "True IV Rank/Percentile requires a historical IV series and is not "
    "fabricated. This app does not place orders."
)
