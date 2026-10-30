"""
Pin Bar / Doji + Trend Alert Bot (Deriv) - GitHub Actions version
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

BOOM_SYMBOLS = ["BOOM100", "BOOM300N", "BOOM500", "BOOM600", "BOOM900", "BOOM1000"]
CRASH_SYMBOLS = ["CRASH100", "CRASH300N", "CRASH500", "CRASH600", "CRASH900", "CRASH1000"]
GOLD_AND_FOREX = [
    "frxXAUUSD",   # gold
    "frxEURUSD",
    "frxGBPUSD",
    "frxUSDJPY",
    "frxUSDCHF",
    "frxAUDUSD",
    "frxUSDCAD",
    "frxNZDUSD",
]
# Volatility indices: regular ones, plus the "1s" (one-second tick) versions
VOLATILITY_SYMBOLS = [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
]
SYMBOLS = BOOM_SYMBOLS + CRASH_SYMBOLS + VOLATILITY_SYMBOLS + GOLD_AND_FOREX

RSI_PERIOD = 10
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

PIN_WICK_RATIO = 0.66
PIN_BODY_RATIO = 0.33
PIN_OPPOSITE_WICK_RATIO = 0.15

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

VOICE_TEXT = "Takondwa, check your trades"
VOICE_FILE = "alert.mp3"

# ------------------------------------------------------------------------


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


def send_telegram_alert(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
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


async def check_symbol(ws, symbol: str) -> bool:
    """Returns True if an alert was sent."""
    try:
        df_h1 = await fetch_candles(ws, symbol, H1_GRANULARITY, CANDLE_COUNT)
        if BIAS_GRANULARITY == H1_GRANULARITY:
            df_bias = df_h1
        else:
            df_bias = await fetch_candles(ws, symbol, BIAS_GRANULARITY, CANDLE_COUNT)
    except Exception as exc:
        print(f"[{symbol}] fetch error: {exc}")
        return False

    if len(df_h1) < 2 or len(df_bias) < RSI_PERIOD + 2:
        print(f"[{symbol}] not enough candles")
        return False

    last_closed_h1 = df_h1.iloc[-2]  # -1 is still forming
    epoch = int(last_closed_h1["epoch"])
    age = time.time() - (epoch + H1_GRANULARITY)  # seconds since that candle closed

    if age < 0 or age > ALERT_WINDOW_SECONDS:
        print(f"[{symbol}] last closed H1 is outside the alert window (market closed?), skipping")
        return False

    # --- find a pattern: a pin bar first, otherwise a doji at a key level ---
    pattern = None
    level_note = ""
    signal = detect_pin_bar(last_closed_h1)
    if signal is not None:
        pattern = "Pin Bar"
    elif ENABLE_DOJI:
        atr = compute_atr(df_h1.iloc[:-1], ATR_PERIOD)  # closed candles only
        if is_doji(last_closed_h1, atr):
            hit = doji_at_key_level(df_h1, last_closed_h1, atr)
            if hit is not None:
                kind, level = hit
                signal = "bullish" if kind == "support" else "bearish"
                pattern = f"Doji at {kind}"
                level_note = f"Key level ({kind}): {level:.5g}\n"
    if signal is None:
        print(f"[{symbol}] no pin bar or doji at a key level")
        return False

    allowed = allowed_direction(symbol)
    if allowed is not None and signal != allowed:
        print(f"[{symbol}] {pattern} ({signal}) ignored")
        return False

    bias = get_bias(df_bias)
    needs_bias = pattern == "Pin Bar" or DOJI_REQUIRE_BIAS
    if needs_bias and signal != bias:
        print(f"[{symbol}] {pattern} ({signal}) disagrees with {BIAS_LABEL} bias ({bias})")
        return False

    ema_note = ""
    if ema_filter_applies(symbol):
        if len(df_h1) < EMA_PERIOD:
            print(f"[{symbol}] not enough candles for the {EMA_PERIOD} EMA, skipping")
            return False
        ema = df_h1["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        ema_at_signal = ema.iloc[-2]  # EMA value at the signal candle
        trend = "bullish" if last_closed_h1["close"] > ema_at_signal else "bearish"
        if signal != trend:
            where = "below" if trend == "bearish" else "above"
            print(f"[{symbol}] {pattern} ({signal}) blocked: price is {where} the {EMA_PERIOD} EMA")
            return False
        ema_note = f"Price {'above' if trend == 'bullish' else 'below'} {EMA_PERIOD} EMA: yes\n"

    ts = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    action = "SELL" if signal == "bearish" else "BUY"
    message = (
        f"🚨 Takondwa, check your trades!\n"
        f"{pattern} Alert - {symbol}\n"
        f"Signal: {action} ({pattern}, {BIAS_LABEL} RSI bias: {bias})\n"
        f"H1 candle close: {last_closed_h1['close']}\n"
        f"{level_note}"
        f"{ema_note}"
        f"Candle time: {ts}"
    )
    print(message)
    send_telegram_alert(message)
    send_voice_alert()
    return True


async def run_checks() -> None:
    # Deriv/Cloudflare sometimes rejects the handshake (HTTP 520), so retry.
    last_exc = None
    for attempt in range(1, 4):
        try:
            async with websockets.connect(DERIV_WS_URL, open_timeout=20) as ws:
                for symbol in SYMBOLS:
                    await check_symbol(ws, symbol)
                    await asyncio.sleep(0.3)  # stay well under Deriv's rate limit
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
