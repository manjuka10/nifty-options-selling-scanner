import pandas as pd
from indicators.technical import add_technical_indicators
from config.settings import (
    MIN_EXTENSION_PCT, MIN_ATR_PCT, MAX_ATR_PCT,
    MIN_REALIZED_VOL_PCT, MAX_REALIZED_VOL_PCT,
    MIN_SHORT_DELTA, MAX_SHORT_DELTA,
    MIN_OPTION_VOLUME, MIN_OPTION_OI,
    REQUIRE_IV_RANK, MIN_IV_RANK,
)

def _latest_technical(market_df):
    rows = []
    for symbol, group in market_df.groupby("Symbol"):
        group = group.sort_index()
        tech = add_technical_indicators(group).dropna()
        if tech.empty:
            continue
        row = tech.iloc[-1].copy()
        row["Symbol"] = symbol
        rows.append(row)
    return pd.DataFrame(rows)

def _option_quality(options_df):
    if options_df.empty:
        return options_df.copy()

    x = options_df.copy()
    x["AbsDelta"] = x["Delta"].abs()
    x["SpreadPct"] = (x["Ask"] - x["Bid"]) / x["Premium"].replace(0, pd.NA) * 100

    x["Liquid"] = (
        (x["Volume"] >= MIN_OPTION_VOLUME) &
        (x["OI"] >= MIN_OPTION_OI) &
        (x["Bid"] > 0) &
        (x["Ask"] >= x["Bid"])
    )

    x["DeltaOK"] = x["AbsDelta"].between(MIN_SHORT_DELTA, MAX_SHORT_DELTA)

    if REQUIRE_IV_RANK:
        x["IVRankOK"] = x["IVRank"] >= MIN_IV_RANK
    else:
        x["IVRankOK"] = True

    x["OptionOK"] = x["Liquid"] & x["DeltaOK"] & x["IVRankOK"]
    return x

def scan_universe(market_df, options_df):
    tech = _latest_technical(market_df)
    if tech.empty:
        return pd.DataFrame(columns=["Symbol", "Price", "Trend", "Extension", "Setup"])

    tech["Price"] = tech["Close"]
    tech["Extension"] = tech["Distance21Pct"]

    tech["VolOK"] = (
        tech["ATRPct"].between(MIN_ATR_PCT, MAX_ATR_PCT) &
        tech["RealizedVolPct"].between(MIN_REALIZED_VOL_PCT, MAX_REALIZED_VOL_PCT)
    )

    tech["CallCore"] = (
        tech["BullTrend"] &
        (tech["Distance21Pct"] >= MIN_EXTENSION_PCT) &
        tech["VolOK"]
    )

    tech["PutCore"] = (
        tech["BearTrend"] &
        (tech["Distance21Pct"] <= -MIN_EXTENSION_PCT) &
        tech["VolOK"]
    )

    opts = _option_quality(options_df)

    # If live option data is not connected yet, preserve the stock-side
    # candidates so the prototype remains usable. Once options are connected,
    # only candidates with matching OptionOK records are returned.
    if opts.empty:
        tech["Setup"] = ""
        tech.loc[tech["CallCore"], "Setup"] = "CALL"
        tech.loc[tech["PutCore"], "Setup"] = "PUT"
    else:
        call_symbols = set(opts.loc[
            (opts["OptionType"].str.upper() == "CE") & opts["OptionOK"], "Symbol"
        ])
        put_symbols = set(opts.loc[
            (opts["OptionType"].str.upper() == "PE") & opts["OptionOK"], "Symbol"
        ])
        tech["Setup"] = ""
        tech.loc[tech["CallCore"] & tech["Symbol"].isin(call_symbols), "Setup"] = "CALL"
        tech.loc[tech["PutCore"] & tech["Symbol"].isin(put_symbols), "Setup"] = "PUT"

    tech["Trend"] = "Neutral"
    tech.loc[tech["BullTrend"], "Trend"] = "Bullish"
    tech.loc[tech["BearTrend"], "Trend"] = "Bearish"

    result = tech.loc[tech["Setup"].isin(["CALL", "PUT"]),
                       ["Symbol", "Price", "Trend", "Extension", "Setup"]].copy()
    return result.sort_values(["Setup", "Extension"], ascending=[True, False])
