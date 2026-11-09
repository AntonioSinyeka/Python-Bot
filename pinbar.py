"""
Pin Bar / Doji / Engulfing + Trend Alert Bot (Deriv) - GitHub Actions version
==========================================================

Runs ONCE per invocation (GitHub Actions starts it every hour). For each
symbol it checks the last closed H1 candle for a pin bar that agrees with
the H1 RSI bias, then sends a Telegram text + voice alert.

Rules:
  - BOOM indices  -> only BEARISH pin bars (SELL)
  - CRASH indices -> only BULLISH pin bars (BUY)
  - Gold / forex / Volatility -> both directions, as long as the pin bar
                     agrees with the RSI bias (and the 200 EMA, if enabled)

Signal bot only -- it does not place trades.
"""
import asyncio
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import websockets
from gtts import gTTS

try:  # optional: lets you test locally with a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------- CONFIG ---------------------------------

# Public market-data gateway (no login or token needed for candles)
DERIV_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BOOM_SYMBOLS = ["BOOM50", "BOOM100", "BOOM150N", "BOOM200", "BOOM300N", "BOOM500", "BOOM600", "BOOM900", "BOOM1000"]
CRASH_SYMBOLS = ["CRASH50", "CRASH100", "CRASH150N", "CRASH200", "CRASH300N", "CRASH500", "CRASH600", "CRASH900", "CRASH1000"]
GOLD_AND_FOREX = [
    "frxXAUUSD",   # gold
    "frxEURUSD",
    "frxGBPUSD",
    "frxUSDJPY",
    "frxEURGBP",
    "frxEURJPY",
    "frxGBPJPY",
]
# Volatility indices: regular ones, plus the "1s" (one-second tick) versions
VOLATILITY_SYMBOLS = [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
]
# Step indices (Step Index plus the 200 / 300 / 400 / 500 versions): both directions, 24/7
STEP_SYMBOLS = ["stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5"]
# Jump indices (Jump 10 / 25 / 50 / 75 / 100): both directions, 24/7
JUMP_SYMBOLS = ["JD10", "JD25", "JD50", "JD75", "JD100"]
SYMBOLS = BOOM_SYMBOLS + CRASH_SYMBOLS + VOLATILITY_SYMBOLS + STEP_SYMBOLS + JUMP_SYMBOLS + GOLD_AND_FOREX

RSI_PERIOD = 7
H1_GRANULARITY = 3600        # candles used for pin bar detection
BIAS_GRANULARITY = 3600      # candles used for the RSI trend bias (3600 = H1, 14400 = H4)
BIAS_LABEL = "H1" if BIAS_GRANULARITY == 3600 else "H4"
CANDLE_COUNT = 1000          # history per symbol (the 200 EMA needs a long warm-up)

# 200 EMA trend filter on H1: BUY only when the pin bar closes ABOVE the EMA,
# SELL only when it closes BELOW it.
# Applied to Boom/Crash too: Boom SELLs only below the EMA, Crash BUYs only above it.
# Set APPLY_EMA_TO_BOOM_CRASH = False to turn the EMA rule off for those indices.
EMA_PERIOD = 200
APPLY_EMA_TO_BOOM_CRASH = True
APPLY_EMA_TO_VOLATILITY = True   # set False to skip the EMA rule on Volatility indices

# Which patterns must pass the 200 EMA rule (BUY above it, SELL below it)?
# True = the rule is applied to that pattern, False = that pattern ignores the EMA.
# (The per-asset switches above still decide which symbols use the EMA rule at all.)
EMA_FILTER_BY_PATTERN = {
    "Pin Bar": True,
    "Shooting Star": False,
    "Hanging Man": False,
    "Engulfing": False,
    "Doji": False,
    "Trend + RSI": False,   # this signal already contains its own EMA rule
}

PIN_WICK_RATIO = 0.55          # long wick: was 0.66 (66% of the candle), now 55%
PIN_BODY_RATIO = 0.40          # body: was 0.33 (33%), now up to 40%
PIN_OPPOSITE_WICK_RATIO = 0.25 # opposite wick: was 0.15 (15%), now up to 25%

# Only alert if the pin bar candle closed within this many seconds.
# The workflow runs hourly, so each candle is seen by one run; this is what
# stops duplicate alerts (nothing is remembered between runs). Keep < 3600.
ALERT_WINDOW_SECONDS = 55 * 60

# Doji at a key level (support / resistance) -- a second pattern besides pin bars.
ENABLE_DOJI = True
DOJI_BODY_RATIO = 0.10          # body must be <= 10% of the candle's range
DOJI_MIN_RANGE_ATR = 0.5        # candle range must be >= 0.5 x ATR (ignores tiny, meaningless candles)
ATR_PERIOD = 14
KEY_LEVEL_LOOKBACK = 150        # H1 candles searched for swing highs / lows (about 6 days)
PIVOT_STRENGTH = 3              # a swing point beats the 3 candles on each side of it
KEY_LEVEL_MIN_TOUCHES = 2       # a level counts only if price turned there at least twice before
KEY_LEVEL_TOLERANCE_ATR = 0.3   # "at the level" = doji high/low within 0.3 x ATR of it
DOJI_REQUIRE_BIAS = True        # False = skip the RSI bias rule for doji signals

# Engulfing candle at a key level (uses the same key level settings as the doji).
# Bullish engulfing at support = BUY, bearish engulfing at resistance = SELL.
# Shooting star: a looser "rejection at the top" candle, only checked when the candle is NOT a pin bar.
# Long upper wick, small body, little lower wick, after a rise. Always a SELL signal.
ENABLE_SHOOTING_STAR = True
SHOOTING_STAR_BODY_RATIO = 0.35      # body <= 35% of the candle's range
SHOOTING_STAR_UPPER_RATIO = 0.50     # upper wick >= 50% of the range (and >= 2 x the body)
SHOOTING_STAR_LOWER_RATIO = 0.25     # lower wick <= 25% of the range
SHOOTING_STAR_MIN_RANGE_ATR = 0.5    # range >= 0.5 x ATR (ignores tiny candles)
SHOOTING_STAR_REQUIRE_BIAS = False   # True = also require the H1 RSI bias to agree (a rise usually means RSI > 50, which blocks sells)

# Hanging man: the shooting star's cousin at the top of a rise. Small body near the top, long
# LOWER wick, little upper wick, after a rise. Only checked when the candle is NOT a pin bar.
# Always a SELL signal (it looks like a hammer, but after a rise it warns of a reversal down).
ENABLE_HANGING_MAN = True
HANGING_MAN_BODY_RATIO = 0.35        # body <= 35% of the candle's range
HANGING_MAN_LOWER_RATIO = 0.50       # lower wick >= 50% of the range (and >= 2 x the body)
HANGING_MAN_UPPER_RATIO = 0.25       # upper wick <= 25% of the range
HANGING_MAN_MIN_RANGE_ATR = 0.5      # range >= 0.5 x ATR (ignores tiny candles)
HANGING_MAN_NEAR_HIGH_ATR = 0.5      # its high must be within 0.5 x ATR of the recent highs
HANGING_MAN_REQUIRE_BIAS = False     # True = also require the H1 RSI bias to agree (a rise usually means RSI > 50, which blocks sells)

# Trend + RSI signal (no candle pattern needed), on the H1 chart:
#   price BELOW the 200 EMA and RSI under 50  -> SELL
#   price ABOVE the 200 EMA and RSI over 50   -> BUY
# "on_change"    = alert on the first closed candle where the condition becomes true (no repeats while it holds)
# "every_candle" = alert on EVERY closed candle where it is true (can mean an alert every hour per symbol)
ENABLE_TREND_RSI_SIGNAL = True
TREND_RSI_ALERT_MODE = "on_change"

ENABLE_ENGULFING = True
ENGULF_MIN_BODY_ATR = 0.5       # the engulfing candle's body must be >= 0.5 x ATR (ignores tiny candles)
ENGULFING_REQUIRE_BIAS = False  # True = also require the H1 RSI bias to agree (blocks most buys at support)

VOICE_TEXT = "Takondwa, check your trades"
VOICE_FILE = "alert.mp3"

# ------------------------------------------------------------------------


STATS = Counter()   # why candles were skipped, printed as a summary at the end of every run


# Symbols that get a short "check report" on Telegram after EVERY run (arrives silently).
# It shows exactly what the bot saw on the last closed candle and why it did or did not alert,
# so you can compare it with your chart. Set to [] to turn the reports off.
DEBUG_REPORT_SYMBOLS = ["BOOM200", "frxXAUUSD"]
REPORTS = {}


def rep(symbol: str, text: str) -> None:
    REPORTS.setdefault(symbol, []).append(text)


def note(symbol: str, key: str, text: str) -> None:
    STATS[key] += 1
    print(f"[{symbol}] {text}")
    rep(symbol, text)


def send_debug_reports() -> None:
    for sym in DEBUG_REPORT_SYMBOLS:
        lines = REPORTS.get(sym)
        if lines:
            body = "\n".join(lines)
        elif sym in SYMBOLS:
            body = "No data was checked for this symbol in this run."
        else:
            body = "This symbol is not in the SYMBOLS list."
        send_telegram_alert(f"🔎 Check report - {sym}\n{body}", silent=True)


def print_summary() -> None:
    order = ["candles checked", "alerts sent", "no pattern", "wrong direction for index",
             "RSI bias disagreed", "blocked by 200 EMA", "outside alert window",
             "not enough candles", "fetch errors", "symbol codes not found"]
    parts = [f"{k}: {STATS[k]}" for k in order if STATS[k]]
    print("SUMMARY | " + (" | ".join(parts) if parts else "nothing checked"))


def allowed_direction(symbol: str):
    """Boom -> bearish only, Crash -> bullish only, anything else -> both."""
    if symbol.startswith("BOOM"):
        return "bearish"
    if symbol.startswith("CRASH"):
        return "bullish"
    return None


def ema_filter_applies(symbol: str) -> bool:
    if symbol.startswith(("BOOM", "CRASH")):
        return APPLY_EMA_TO_BOOM_CRASH
    if symbol in VOLATILITY_SYMBOLS:
        return APPLY_EMA_TO_VOLATILITY
    return True


def telegram_ready() -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
        return False
    return True


def send_telegram_alert(message: str, silent: bool = False) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    if silent:
        data["disable_notification"] = True   # arrives without buzzing your phone
    try:
        resp = requests.post(url, data=data, timeout=15)
        if resp.status_code != 200:
            print(f"[telegram] text failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        print(f"[telegram] text error: {exc}")


def send_voice_alert() -> None:
    """Generate the spoken clip and send it silently (the text alert is the one that notifies)."""
    try:
        gTTS(VOICE_TEXT).save(VOICE_FILE)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAudio"
        with open(VOICE_FILE, "rb") as f:
            resp = requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "title": VOICE_TEXT,
                    "disable_notification": True,
                },
                files={"audio": f},
                timeout=30,
            )
        if resp.status_code != 200:
            print(f"[telegram] voice failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        print(f"[telegram] voice error: {exc}")


async def fetch_candles(ws, symbol: str, granularity: int, count: int) -> pd.DataFrame:
    request = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "granularity": granularity,
        "style": "candles",
    }
    await ws.send(json.dumps(request))
    response = json.loads(await ws.recv())

    if "error" in response:
        raise RuntimeError(f"Deriv API error for {symbol}: {response['error']['message']}")

    df = pd.DataFrame(response.get("candles", []))
    if df.empty:
        raise RuntimeError(f"No candles returned for {symbol}")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def get_bias(df_bias: pd.DataFrame) -> str:
    closed = df_bias.iloc[:-1]  # drop the still-forming candle
    rsi = compute_rsi(closed["close"], RSI_PERIOD)
    return "bullish" if rsi.iloc[-1] > 50 else "bearish"


def detect_pin_bar(candle: pd.Series):
    """Returns 'bullish', 'bearish', or None."""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    rng = h - l
    if rng <= 0:
        return None

    body_ratio = abs(c - o) / rng
    upper_ratio = (h - max(o, c)) / rng
    lower_ratio = (min(o, c) - l) / rng

    if body_ratio > PIN_BODY_RATIO:
        return None
    if lower_ratio >= PIN_WICK_RATIO and upper_ratio <= PIN_OPPOSITE_WICK_RATIO:
        return "bullish"
    if upper_ratio >= PIN_WICK_RATIO and lower_ratio <= PIN_OPPOSITE_WICK_RATIO:
        return "bearish"
    return None


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR; returns the value at the last candle of df."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1])


def is_doji(candle: pd.Series, atr: float) -> bool:
    rng = candle["high"] - candle["low"]
    if rng <= 0 or atr is None or np.isnan(atr) or atr <= 0:
        return False
    if rng < DOJI_MIN_RANGE_ATR * atr:
        return False
    return abs(candle["close"] - candle["open"]) / rng <= DOJI_BODY_RATIO


def doji_at_key_level(df: pd.DataFrame, doji: pd.Series, atr: float):
    """Returns ('support' | 'resistance', level_price) or None.

    Key levels are swing lows / highs found in the candles BEFORE the doji.
    The doji is "at" a level when its low (support) or high (resistance) is
    within KEY_LEVEL_TOLERANCE_ATR x ATR of at least KEY_LEVEL_MIN_TOUCHES
    earlier swing points.
    """
    sub = df.iloc[:-2].tail(KEY_LEVEL_LOOKBACK).reset_index(drop=True)  # excludes doji and forming candle
    n = PIVOT_STRENGTH
    if len(sub) < 2 * n + 1:
        return None
    tol = KEY_LEVEL_TOLERANCE_ATR * atr
    highs, lows = sub["high"].to_numpy(), sub["low"].to_numpy()

    swing_highs, swing_lows = [], []
    for i in range(n, len(sub) - n):
        if highs[i] == highs[i - n:i + n + 1].max():
            swing_highs.append(highs[i])
        if lows[i] == lows[i - n:i + n + 1].min():
            swing_lows.append(lows[i])

    near_res = [p for p in swing_highs if abs(doji["high"] - p) <= tol]
    near_sup = [p for p in swing_lows if abs(doji["low"] - p) <= tol]
    at_res = len(near_res) >= KEY_LEVEL_MIN_TOUCHES
    at_sup = len(near_sup) >= KEY_LEVEL_MIN_TOUCHES

    if at_res and at_sup:
        return None  # squeezed between both: unclear, skip
    if at_res:
        return ("resistance", float(np.mean(near_res)))
    if at_sup:
        return ("support", float(np.mean(near_sup)))
    return None


def swing_points(sub: pd.DataFrame):
    """Swing highs and lows: a point that beats PIVOT_STRENGTH candles on each side."""
    n = PIVOT_STRENGTH
    highs, lows = sub["high"].to_numpy(), sub["low"].to_numpy()
    swing_highs, swing_lows = [], []
    for i in range(n, len(sub) - n):
        if highs[i] == highs[i - n:i + n + 1].max():
            swing_highs.append(highs[i])
        if lows[i] == lows[i - n:i + n + 1].min():
            swing_lows.append(lows[i])
    return swing_highs, swing_lows


def trend_rsi_state(close: float, ema: float, rsi: float):
    """'bearish' = below the EMA with RSI under 50, 'bullish' = above it with RSI over 50, else None."""
    if np.isnan(ema) or np.isnan(rsi):
        return None
    if close < ema and rsi < 50:
        return "bearish"
    if close > ema and rsi > 50:
        return "bullish"
    return None


def is_shooting_star(df: pd.DataFrame, atr: float) -> bool:
    """Last closed candle (-2): small body, long upper wick, tiny lower wick,
    poking above the recent highs after a rise."""
    if len(df) < 9 or atr is None or np.isnan(atr) or atr <= 0:
        return False
    c = df.iloc[-2]
    rng = c["high"] - c["low"]
    if rng <= 0 or rng < SHOOTING_STAR_MIN_RANGE_ATR * atr:
        return False
    body = abs(c["close"] - c["open"])
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    if body / rng > SHOOTING_STAR_BODY_RATIO:
        return False
    if upper / rng < SHOOTING_STAR_UPPER_RATIO or upper < 2 * body:
        return False
    if lower / rng > SHOOTING_STAR_LOWER_RATIO:
        return False
    rising = df["close"].iloc[-3] > df["close"].iloc[-6]             # price was rising into it
    new_high = c["high"] >= df["high"].iloc[-8:-2].max()             # and it pushed above recent highs
    return bool(rising and new_high)


def is_hanging_man(df: pd.DataFrame, atr: float) -> bool:
    """Last closed candle (-2): small body near the top, long lower wick, tiny upper
    wick, appearing after a rise and near the recent highs."""
    if len(df) < 9 or atr is None or np.isnan(atr) or atr <= 0:
        return False
    c = df.iloc[-2]
    rng = c["high"] - c["low"]
    if rng <= 0 or rng < HANGING_MAN_MIN_RANGE_ATR * atr:
        return False
    body = abs(c["close"] - c["open"])
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    if body / rng > HANGING_MAN_BODY_RATIO:
        return False
    if lower / rng < HANGING_MAN_LOWER_RATIO or lower < 2 * body:
        return False
    if upper / rng > HANGING_MAN_UPPER_RATIO:
        return False
    rising = df["close"].iloc[-3] > df["close"].iloc[-6]             # price was rising into it
    prior_high = df["high"].iloc[-8:-2].max()
    near_high = c["high"] >= prior_high - HANGING_MAN_NEAR_HIGH_ATR * atr
    return bool(rising and near_high)


def engulfing_at_key_level(df: pd.DataFrame, atr: float):
    """Returns ('bullish' | 'bearish', 'support' | 'resistance', level_price) or None.

    The last closed candle (-2) must engulf the body of the one before it (-3),
    and the pattern's low (bullish) or high (bearish) must sit at a key level
    built from the candles BEFORE the pattern.
    """
    if len(df) < 2 * PIVOT_STRENGTH + 4 or atr is None or np.isnan(atr) or atr <= 0:
        return None
    cur, prev = df.iloc[-2], df.iloc[-3]
    if abs(cur["close"] - cur["open"]) < ENGULF_MIN_BODY_ATR * atr:
        return None

    bullish = (prev["close"] < prev["open"] and cur["close"] > cur["open"]
               and cur["open"] <= prev["close"] and cur["close"] >= prev["open"])
    bearish = (prev["close"] > prev["open"] and cur["close"] < cur["open"]
               and cur["open"] >= prev["close"] and cur["close"] <= prev["open"])
    if not (bullish or bearish):
        return None

    sub = df.iloc[:-3].tail(KEY_LEVEL_LOOKBACK).reset_index(drop=True)  # before the pattern
    if len(sub) < 2 * PIVOT_STRENGTH + 1:
        return None
    swing_highs, swing_lows = swing_points(sub)
    tol = KEY_LEVEL_TOLERANCE_ATR * atr

    if bullish:
        pattern_low = min(cur["low"], prev["low"])
        near = [p for p in swing_lows if abs(pattern_low - p) <= tol]
        if len(near) >= KEY_LEVEL_MIN_TOUCHES:
            return ("bullish", "support", float(np.mean(near)))
    else:
        pattern_high = max(cur["high"], prev["high"])
        near = [p for p in swing_highs if abs(pattern_high - p) <= tol]
        if len(near) >= KEY_LEVEL_MIN_TOUCHES:
            return ("bearish", "resistance", float(np.mean(near)))
    return None


async def fetch_active_symbols(ws) -> set:
    """Asks Deriv which symbol codes exist (public request, no token needed)."""
    await ws.send(json.dumps({"active_symbols": "brief"}))
    response = json.loads(await ws.recv())
    items = response.get("active_symbols", [])
    return {i.get("underlying_symbol") or i.get("symbol") for i in items} - {None}


def report_missing_symbols(active: set) -> None:
    """Prints (and silently messages you) any configured symbol code Deriv does not list."""
    if not active:
        print("[symbols] Deriv returned no symbol list, skipping the symbol check")
        return
    missing = [s for s in SYMBOLS if s not in active]
    if not missing:
        print("[symbols] every configured symbol code exists on Deriv")
        return
    STATS["symbol codes not found"] += len(missing)
    hints = sorted(a for a in active if any(k in a.upper() for k in ("BOOM", "CRASH", "XAUUSD", "GOLD", "STPRNG", "STEP", "JD", "JUMP")))
    text = "🧭 Symbol check\nCodes Deriv does not list (they will fail to load): " + ", ".join(missing)
    if hints:
        text += "\nSimilar codes that DO exist: " + ", ".join(hints)
    print(text)
    send_telegram_alert(text, silent=True)


async def check_symbol(ws, symbol: str) -> bool:
    """Returns True if an alert was sent."""
    try:
        df_h1 = await fetch_candles(ws, symbol, H1_GRANULARITY, CANDLE_COUNT)
        if BIAS_GRANULARITY == H1_GRANULARITY:
            df_bias = df_h1
        else:
            df_bias = await fetch_candles(ws, symbol, BIAS_GRANULARITY, CANDLE_COUNT)
    except Exception as exc:
        note(symbol, "fetch errors", f"fetch error: {exc}")
        return False

    if len(df_h1) < 2 or len(df_bias) < RSI_PERIOD + 2:
        note(symbol, "not enough candles", "not enough candles")
        return False

    last_closed_h1 = df_h1.iloc[-2]  # -1 is still forming
    epoch = int(last_closed_h1["epoch"])
    age = time.time() - (epoch + H1_GRANULARITY)  # seconds since that candle closed

    rep(symbol, f"Last closed H1 candle: {datetime.fromtimestamp(epoch, tz=timezone.utc).strftime('%d %b %H:%M UTC')} (closed {age / 60:.0f} min ago)")
    rep(symbol, f"O {last_closed_h1['open']} H {last_closed_h1['high']} L {last_closed_h1['low']} C {last_closed_h1['close']}")

    if age < 0 or age > ALERT_WINDOW_SECONDS:
        note(symbol, "outside alert window", "last closed H1 is outside the alert window (market closed, or the run started late), skipping")
        return False

    STATS["candles checked"] += 1
    rng = last_closed_h1["high"] - last_closed_h1["low"]
    if rng > 0:
        body_pct = abs(last_closed_h1["close"] - last_closed_h1["open"]) / rng * 100
        up_pct = (last_closed_h1["high"] - max(last_closed_h1["open"], last_closed_h1["close"])) / rng * 100
        lo_pct = (min(last_closed_h1["open"], last_closed_h1["close"]) - last_closed_h1["low"]) / rng * 100
        print(f"[{symbol}] candle: body {body_pct:.0f}% | upper wick {up_pct:.0f}% | lower wick {lo_pct:.0f}%")
        rep(symbol, f"Shape: body {body_pct:.0f}% | upper wick {up_pct:.0f}% | lower wick {lo_pct:.0f}%")

    # --- 1. collect EVERY pattern this candle matches; each is judged on its own below ---
    main = []   # pin bar, engulfing, doji
    alt = []    # shooting star, hanging man: used when there is no pin bar, or when the pin bar is rejected by a filter
    atr = compute_atr(df_h1.iloc[:-1], ATR_PERIOD)  # closed candles only

    pin = detect_pin_bar(last_closed_h1)
    if pin is not None:
        main.append(dict(pattern="Pin Bar", key="Pin Bar", signal=pin, require_bias=True, level_note=""))
    if ENABLE_SHOOTING_STAR and is_shooting_star(df_h1, atr):
        alt.append(dict(pattern="Shooting Star", key="Shooting Star", signal="bearish",
                        require_bias=SHOOTING_STAR_REQUIRE_BIAS, level_note=""))
    if ENABLE_HANGING_MAN and is_hanging_man(df_h1, atr):
        alt.append(dict(pattern="Hanging Man", key="Hanging Man", signal="bearish",
                        require_bias=HANGING_MAN_REQUIRE_BIAS, level_note=""))
    if ENABLE_ENGULFING:
        hit = engulfing_at_key_level(df_h1, atr)
        if hit is not None:
            sig, kind, level = hit
            main.append(dict(pattern=f"{sig.capitalize()} Engulfing at {kind}", key="Engulfing", signal=sig,
                             require_bias=ENGULFING_REQUIRE_BIAS,
                             level_note=f"Key level ({kind}): {level:.5g}\n"))
    if ENABLE_DOJI and is_doji(last_closed_h1, atr):
        hit = doji_at_key_level(df_h1, last_closed_h1, atr)
        if hit is not None:
            kind, level = hit
            main.append(dict(pattern=f"Doji at {kind}", key="Doji",
                             signal="bullish" if kind == "support" else "bearish",
                             require_bias=DOJI_REQUIRE_BIAS,
                             level_note=f"Key level ({kind}): {level:.5g}\n"))
    if ENABLE_TREND_RSI_SIGNAL and len(df_h1) >= EMA_PERIOD + 3:
        closes_all = df_h1["close"]
        ema_all = closes_all.ewm(span=EMA_PERIOD, adjust=False).mean()
        rsi_all = compute_rsi(closes_all.iloc[:-1], RSI_PERIOD)   # closed candles only
        i = len(df_h1) - 2                                         # the last closed candle
        now_state = trend_rsi_state(closes_all.iloc[i], ema_all.iloc[i], rsi_all.iloc[i])
        prev_state = trend_rsi_state(closes_all.iloc[i - 1], ema_all.iloc[i - 1], rsi_all.iloc[i - 1])
        fresh = now_state is not None and now_state != prev_state
        rep(symbol, f"Trend + RSI: {now_state or 'no signal'}" + (" (new this candle)" if fresh else (" (already running)" if now_state else "")))
        if now_state is not None and (TREND_RSI_ALERT_MODE == "every_candle" or fresh):
            below = now_state == "bearish"
            main.append(dict(pattern="Trend + RSI", key="Trend + RSI", signal=now_state, require_bias=False,
                             level_note=f"Price {'below' if below else 'above'} the {EMA_PERIOD} EMA and RSI({RSI_PERIOD}) {rsi_all.iloc[i]:.1f} ({'under' if below else 'over'} 50)\n"))
    rep(symbol, "Patterns matched: " + (", ".join(c["pattern"] for c in main + alt) or "none"))
    try:
        rsi_val = compute_rsi(df_bias.iloc[:-1]["close"], RSI_PERIOD).iloc[-1]
        rep(symbol, f"{BIAS_LABEL} RSI({RSI_PERIOD}): {rsi_val:.1f} -> {'bullish' if rsi_val > 50 else 'bearish'} bias")
        if len(df_h1) >= EMA_PERIOD:
            ema_now = df_h1["close"].ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-2]
            rep(symbol, f"{EMA_PERIOD} EMA: {ema_now:.5g} (price is {'above' if last_closed_h1['close'] > ema_now else 'below'} it)")
    except Exception:
        pass
    if not main and not alt:
        note(symbol, "no pattern", "no pin bar, shooting star, hanging man, engulfing or doji at a key level")
        return False

    # --- 2. test each pattern against the filters separately: one failing pattern never hides another ---
    allowed = allowed_direction(symbol)
    bias = get_bias(df_bias)
    cache = {}

    def evaluate(cands):
        ok = []
        for c in cands:
            sig, pat = c["signal"], c["pattern"]
            if allowed is not None and sig != allowed:
                note(symbol, "wrong direction for index", f"{pat} ({sig}) ignored")
                continue
            if c["require_bias"] and sig != bias:
                note(symbol, "RSI bias disagreed", f"{pat} ({sig}) disagrees with {BIAS_LABEL} bias ({bias})")
                continue
            c["ema_note"] = ""
            if EMA_FILTER_BY_PATTERN.get(c["key"], True) and ema_filter_applies(symbol):
                if len(df_h1) < EMA_PERIOD:
                    note(symbol, "not enough candles", f"not enough candles for the {EMA_PERIOD} EMA, skipping")
                    continue
                if "ema" not in cache:
                    cache["ema"] = df_h1["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
                trend = "bullish" if last_closed_h1["close"] > cache["ema"].iloc[-2] else "bearish"
                if sig != trend:
                    where = "below" if trend == "bearish" else "above"
                    note(symbol, "blocked by 200 EMA", f"{pat} ({sig}) blocked: price is {where} the {EMA_PERIOD} EMA")
                    continue
                c["ema_note"] = f"Price {'above' if trend == 'bullish' else 'below'} {EMA_PERIOD} EMA: yes\n"
            ok.append(c)
        return ok

    passed = evaluate(main)
    if alt and not any(c["pattern"] == "Pin Bar" for c in passed):
        passed += evaluate(alt)   # shooting star / hanging man, when no pin bar passed

    if not passed:
        return False

    # --- 3. one alert per direction, listing every pattern that passed ---
    ts = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for direction in ("bullish", "bearish"):
        group = [c for c in passed if c["signal"] == direction]
        if not group:
            continue
        names = " + ".join(c["pattern"] for c in group)
        action = "SELL" if direction == "bearish" else "BUY"
        level_notes = "".join(dict.fromkeys(c["level_note"] for c in group))
        ema_notes = "".join(dict.fromkeys(c["ema_note"] for c in group))
        message = (
            f"🚨 Takondwa, check your trades!\n"
            f"{names} Alert - {symbol}\n"
            f"Signal: {action} ({names}, {BIAS_LABEL} RSI bias: {bias})\n"
            f"H1 candle close: {last_closed_h1['close']}\n"
            f"{level_notes}"
            f"{ema_notes}"
            f"Candle time: {ts}"
        )
        print(message)
        STATS["alerts sent"] += 1
        rep(symbol, f"ALERT SENT: {action} - {names}")
        send_telegram_alert(message)
        send_voice_alert()
    return True


async def run_checks() -> None:
    # Deriv/Cloudflare sometimes rejects the handshake (HTTP 520), so retry.
    last_exc = None
    for attempt in range(1, 4):
        try:
            async with websockets.connect(DERIV_WS_URL, open_timeout=20) as ws:
                try:
                    report_missing_symbols(await fetch_active_symbols(ws))
                except Exception as exc:
                    print(f"[symbols] symbol check failed: {exc}")
                for symbol in SYMBOLS:
                    await check_symbol(ws, symbol)
                    await asyncio.sleep(0.3)  # stay well under Deriv's rate limit
            print_summary()
            send_debug_reports()
            return
        except Exception as exc:
            last_exc = exc
            print(f"[deriv] connection attempt {attempt} failed: {exc}")
            await asyncio.sleep(5 * attempt)
    print(f"[deriv] giving up after 3 attempts: {last_exc}")
    sys.exit(1)


def main() -> None:
    if not telegram_ready():
        sys.exit(1)

    if os.getenv("TEST_ALERT") == "1":
        send_telegram_alert("🚨 Takondwa, check your trades! (test alert)")
        send_voice_alert()
        print("Test alert sent.")
        return

    asyncio.run(run_checks())


if __name__ == "__main__":
    main()
