"""
CONFIG — Forensic copy of extract_trades_4configs_bidask.py CONFIG dict
========================================================================
Every value here is locked to the backtest that produced:
  london_selective_v1: 15,068 trades, +51,304 pips, +3.405 p/trade, 53% WR

DO NOT CHANGE THESE VALUES unless you re-run the backtest with new values
and verify the results are still profitable.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OANDA CONNECTION ──
OANDA_ENVIRONMENT = os.getenv('OANDA_ENVIRONMENT', 'practice')
OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', '')
OANDA_API_TOKEN = os.getenv('OANDA_API_TOKEN', '')

if OANDA_ENVIRONMENT == 'live':
    OANDA_BASE_URL = 'https://api-fxtrade.oanda.com'
    OANDA_STREAM_URL = 'https://stream-fxtrade.oanda.com'
else:
    OANDA_BASE_URL = 'https://api-fxpractice.oanda.com'
    OANDA_STREAM_URL = 'https://stream-fxpractice.oanda.com'

# ── POSITION SIZING ──
UNITS_PER_TRADE = int(os.getenv('UNITS_PER_TRADE', '1000'))

# ── PAIRS (from backtest) ──
_pairs_env = os.getenv('PAIRS', '')
if _pairs_env:
    PAIRS = [p.strip() for p in _pairs_env.split(',') if p.strip()]
else:
    PAIRS = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD',
        'AUD_USD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'EUR_AUD',
        'GBP_JPY', 'GBP_AUD', 'AUD_JPY', 'AUD_CAD', 'AUD_NZD',
        'CAD_JPY', 'CHF_JPY', 'NZD_JPY', 'NZD_CAD',
    ]

# ── LOGGING ──
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = os.getenv('LOG_DIR', '/var/log/forex-bot')

# ── SCHEDULE ──
# Reprofile windows weekly on Sunday at 22:00 UTC
# Markets have reopened (Wellington) but liquidity is paper-thin,
# no signal would pass the spread filter. Fresh profile ready for Monday London.
REPROFILE_DAY_UTC = int(os.getenv('REPROFILE_DAY_UTC', '6'))    # 0=Mon, 6=Sun
REPROFILE_HOUR_UTC = int(os.getenv('REPROFILE_HOUR_UTC', '22'))

# Fetch maximum available history for expanding training window.
# Backtest used ANCHORED EXPANDING folds: by the final fold, training was
# ~6 years (2019-2025). Live bot must match this — use ALL available data.
# OANDA provides ~10 years of H1 data. 50,000 H1 bars ≈ 5.7 years.
# M30 is needed for profiling, but OANDA caps at ~10yr too.
H1_WARMUP_CANDLES = int(os.getenv('H1_WARMUP_CANDLES', '50000'))
M30_WARMUP_CANDLES = int(os.getenv('M30_WARMUP_CANDLES', '100000'))

# ====================================================================
# BACKTEST PARAMETERS — LOCKED — DO NOT MODIFY
# Source: extract_trades_4configs_bidask.py lines 51-94
# ====================================================================

# ── EMA PERIODS ──
# Source: lines 56-57
EMA_FAST = 9
EMA_SLOW = 21

# ── MTF WEIGHTS ──
# Source: line 55
# Each timeframe's EMA signal is weighted and summed to create MTF bias
MTF_WEIGHTS = {'M5': 0.20, 'M15': 0.30, 'H1': 0.25, 'H4': 0.20}

# ── HURST EXPONENT ──
# Source: lines 58-59
# Computed on H1 close returns, rolling window
HURST_WINDOW = 504        # H1 bars (504 hours = 21 days)
HURST_STEP = 24           # Recompute every 24 H1 bars (1 day)
HURST_MIN_WINDOW = 10     # Smallest R/S window size
HURST_MAX_WINDOW = 200    # Largest R/S window size
HURST_NUM_WINDOWS = 20    # Number of log-spaced window sizes

# ── VOLATILITY OF VOLATILITY ──
# Source: lines 60-61
VOV_ATR_PERIOD = 14       # ATR computed on H1 OHLC
VOV_ROLLING_WINDOW = 168  # H1 bars (168 hours = 1 week)

# ── V1 REGIME THRESHOLDS — LOCKED ──
# Source: lines 62-64
# These are the ENTRY FILTER thresholds that produced +3.405 p/trade
V1_MTF_THRESH = 0.8       # abs(mtf_bias) must be >= this
V1_HURST_THRESH = 0.55    # hurst must be > this (trending regime)
V1_VOV_PCT = 90           # VoV must be < 90th percentile of training window

# ── LONDON WINDOWS ──
# Source: lines 64-65
# M30 window IDs: hour * 2 + (minute >= 30)
# London hours 8-12 GMT → window IDs 16-25
LONDON_WINDOWS = set(range(16, 26))

# ── WINDOW SELECTION THRESHOLDS ──
# Source: lines 72-78
# London windows within LONDON_WINDOWS must meet these on training data:
LONDON_MIN_SHARPE = 0.2
LONDON_MIN_MEAN_PNL = 0.5   # pips per trade
LONDON_MIN_TRADES = 15       # minimum trades in training period

# Addon windows (outside London) must meet these stricter thresholds:
ADDON_MIN_SHARPE = 0.7
ADDON_MIN_MEAN_PNL = 2.5
ADDON_MIN_TRADES = 40

# ── HOLD PERIOD ──
# Source: line 68
# 48 M5 bars = 4 hours maximum hold time
HOLD_BARS_M5 = 48
BARS_PER_DAY_M5 = 288     # 24h * 12 bars/h

# ── H1 → M5 SHIFT ──
# Source: line 69
# H1 indicators are shifted forward by 55 minutes when projected to M5
# This prevents look-ahead: H1 bar closing at 10:00 shouldn't affect
# M5 bars between 10:00-10:55 (the H1 bar isn't complete yet)
H1_SHIFT_MINUTES_M5 = 55

# ── COSTS ──
# Source: lines 83-90
# Spread is organic in bid/ask prices — NOT deducted separately
SLIPPAGE_PIPS = 0.2          # Execution slippage on all trades
SL_SLIPPAGE_PIPS = 0.5       # Additional slippage on SL fills
SWAP_PER_DAY_PIPS = 0.3      # Daily swap cost
COST_SWAP_PER_BAR = SWAP_PER_DAY_PIPS / BARS_PER_DAY_M5

# ── STOP LOSS ──
# Using 5x ATR (from sl_5.0 variant: 15,068 trades, +51,304 pips)
# SL hit rate was 0.9% — catastrophe guard only
SL_ATR_MULT = 5.0

# ── SPREAD FILTER ──
# Source: line 90
# Skip entry if bid/ask spread exceeds this
MAX_SPREAD_PIPS = 5.0

# ── REGIME EXIT ──
# Source: line 91
# OFF for london_selective_v1 (the strategy we're deploying)
REGIME_EXIT = False

# ── WALK-FORWARD ──
# Source: lines 81
# Training window for window profiling
WF_TRAIN_YEARS = 2

# ── PROFILING RESOLUTION ──
# Source: lines 1019-1025
# Window profiling is done on M30 with V1 params
# But live trading is on M5
PROFILE_RESOLUTION = 'M30'
TRADE_RESOLUTION = 'M5'

# ── OANDA CANDLE GRANULARITY MAPPING ──
OANDA_GRANULARITY = {
    'M5': 'M5',
    'M15': 'M15',
    'M30': 'M30',
    'H1': 'H1',
    'H4': 'H4',
}


def pip_multiplier(instrument):
    """Exact copy from backtest line 128-129."""
    s = instrument.upper().replace('/', '_').replace(' ', '_')
    return 100.0 if 'JPY' in s else 10000.0
