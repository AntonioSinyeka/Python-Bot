"""
Pin Bar + Trend Alert Bot (Deriv Boom/Crash) - GitHub Actions version
=====================================================================

Runs ONCE per invocation (GitHub Actions starts it every hour), checks the
last closed H1 candle for a pin bar that agrees with the H4 RSI bias, and
sends a Telegram text + voice alert: "Takondwa, check your trades".

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

DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOLS = ["BOOM900", "CRASH900"]

# Boom only alerts on bearish pin bars (sell), Crash only on bullish (buy).
SYMBOL_DIRECTION = {
    "BOOM900": "bearish",
    "CRASH900": "bullish",
}

RSI_PERIOD = 10
H1_GRANULARITY = 3600
H4_GRANULARITY = 14400
CANDLE_COUNT = 60

PIN_WICK_RATIO = 0.66
PIN_BODY_RATIO = 0.33
PIN_OPPOSITE_WICK_RATIO = 0.15

# Only alert if the pin bar candle closed within this many seconds.
# The workflow runs hourly, so each candle is seen by exactly one run
# (this is what prevents duplicate alerts, since nothing is remembered
# between runs). Keep this below 3600.
ALERT_WINDOW_SECONDS = 55 * 60

VOICE_TEXT = "Takondwa, check your trades"
VOICE_FILE = "alert.mp3"

# ------------------------------------------------------------------------


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
    """Generate the spoken clip and send it as an audio message."""
    try:
        gTTS(VOICE_TEXT).save(VOICE_FILE)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAudio"
        with open(VOICE_FILE, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "title": VOICE_TEXT},
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


def get_h4_bias(df_h4: pd.DataFrame) -> str:
    closed = df_h4.iloc[:-1]  # drop the still-forming H4 candle
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


async def check_symbol(ws, symbol: str) -> bool:
    """Returns True if an alert was sent."""
    try:
        df_h1 = await fetch_candles(ws, symbol, H1_GRANULARITY, CANDLE_COUNT)
        df_h4 = await fetch_candles(ws, symbol, H4_GRANULARITY, CANDLE_COUNT)
    except Exception as exc:
        print(f"[{symbol}] fetch error: {exc}")
        return False

    if len(df_h1) < 2 or len(df_h4) < RSI_PERIOD + 2:
        print(f"[{symbol}] not enough candles")
        return False

    last_closed_h1 = df_h1.iloc[-2]  # -1 is still forming
    epoch = int(last_closed_h1["epoch"])
    age = time.time() - (epoch + H1_GRANULARITY)  # seconds since that candle closed

    if age < 0 or age > ALERT_WINDOW_SECONDS:
        print(f"[{symbol}] last closed H1 is outside the alert window, skipping")
        return False

    pin_direction = detect_pin_bar(last_closed_h1)
    if pin_direction is None:
        print(f"[{symbol}] no pin bar")
        return False

    allowed = SYMBOL_DIRECTION.get(symbol)
    if allowed and pin_direction != allowed:
        print(f"[{symbol}] {pin_direction} pin bar ignored (only {allowed} allowed)")
        return False

    bias = get_h4_bias(df_h4)
    if pin_direction != bias:
        print(f"[{symbol}] {pin_direction} pin bar disagrees with H4 bias ({bias})")
        return False

    ts = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    action = "SELL" if pin_direction == "bearish" else "BUY"
    message = (
        f"🚨 Takondwa, check your trades!\n"
        f"Pin Bar Alert - {symbol}\n"
        f"Signal: {action} ({pin_direction.upper()} pin bar, H4 RSI bias: {bias})\n"
        f"H1 candle close: {last_closed_h1['close']}\n"
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
