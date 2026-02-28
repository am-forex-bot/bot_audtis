"""
INDICATOR ENGINE — FORENSIC COPY OF BACKTEST COMPUTATIONS
===========================================================
Every function here is a direct copy of the logic from
extract_trades_4configs_bidask.py with source line references.

CRITICAL: Do not modify these functions. They produce the exact
indicator values that generated +51,304 pips in the backtest.
If you change ANYTHING here, you are no longer trading the same
strategy that was tested.
"""

import logging
import numpy as np
import pandas as pd

from config import (
    EMA_FAST, EMA_SLOW,
    MTF_WEIGHTS,
    HURST_WINDOW, HURST_STEP, HURST_MIN_WINDOW, HURST_MAX_WINDOW,
    HURST_NUM_WINDOWS,
    VOV_ATR_PERIOD, VOV_ROLLING_WINDOW,
    V1_MTF_THRESH, V1_HURST_THRESH, V1_VOV_PCT,
    H1_SHIFT_MINUTES_M5,
    SL_ATR_MULT, MAX_SPREAD_PIPS,
    pip_multiplier,
)

log = logging.getLogger('bot.indicators')


# ====================================================================
# HURST EXPONENT — R/S ANALYSIS
# Source: lines 183-289 (Numba + pure-Python implementations)
# Using pure-Python version for portability. If you want Numba,
# install numba and swap in the @njit versions from the backtest.
# ====================================================================

def _rs_single_window(series, w):
    """Rescaled range for a single window size.
    Source: lines 183-202 (_rs_single_window Numba kernel)"""
    n_seg = len(series) // w
    if n_seg < 1:
        return []
    rs_values = []
    for i in range(n_seg):
        seg = series[i * w:(i + 1) * w]
        cumdev = np.cumsum(seg - np.mean(seg))
        R = np.max(cumdev) - np.min(cumdev)
        S = np.std(seg)
        if S > 1e-12:
            rs_values.append(R / S)
    return rs_values


def hurst_rs(series, window_sizes):
    """
    Compute Hurst exponent via R/S analysis.
    Source: lines 204-252 (_hurst_rs)

    Returns: float (Hurst exponent) or NaN
    """
    log_ws, log_rs = [], []
    for w in window_sizes:
        rs_vals = _rs_single_window(series, w)
        if rs_vals:
            log_ws.append(np.log(w))
            log_rs.append(np.log(np.mean(rs_vals)))

    if len(log_ws) < 3:
        return np.nan

    x = np.array(log_ws)
    y = np.array(log_rs)
    mx, my = np.mean(x), np.mean(y)
    num = np.sum((x - mx) * (y - my))
    den = np.sum((x - mx) ** 2)
    return num / den if den > 1e-12 else np.nan


def compute_rolling_hurst(h1_close):
    """
    Rolling Hurst exponent on H1 close returns.
    Source: lines 403-434 (DataEngine._compute_rolling_hurst)

    Args:
        h1_close: pd.Series of H1 close prices with DatetimeIndex

    Returns:
        pd.Series of Hurst values, forward-filled, aligned to h1_close index
    """
    returns = h1_close.astype(np.float64).pct_change().dropna()
    returns_arr = returns.values
    window = HURST_WINDOW
    step = HURST_STEP

    if len(returns_arr) < window:
        return pd.Series(np.nan, index=h1_close.index, dtype=np.float32)

    # Window sizes: log-spaced from HURST_MIN_WINDOW to min(HURST_MAX_WINDOW, window//2)
    # Source: lines 410-416
    min_w = HURST_MIN_WINDOW
    max_w = min(HURST_MAX_WINDOW, window // 2)
    if max_w < min_w * 2:
        return pd.Series(np.nan, index=h1_close.index, dtype=np.float32)

    window_sizes = np.unique(np.logspace(
        np.log10(min_w), np.log10(max_w), HURST_NUM_WINDOWS
    ).astype(np.int64))
    window_sizes = window_sizes[window_sizes >= min_w]

    # Rolling computation
    # Source: lines 424-431 (pure Python path)
    n_calcs = (len(returns_arr) - window) // step
    hurst_values = np.empty(n_calcs)
    hurst_times = []
    for k in range(n_calcs):
        i = window + k * step
        hurst_values[k] = hurst_rs(returns_arr[i - window:i], window_sizes)
        hurst_times.append(returns.index[i])

    hurst_sparse = pd.Series(hurst_values, index=pd.DatetimeIndex(hurst_times),
                             dtype=np.float64)
    return hurst_sparse.reindex(h1_close.index, method='ffill').astype(np.float32)


# ====================================================================
# EMA + ATR INDICATORS
# Source: lines 377-384 (DataEngine._compute_indicators)
# ====================================================================

def compute_ema_atr(df):
    """
    Compute EMA(9), EMA(21), ATR(14) on a DataFrame with OHLC columns.
    Modifies df in-place.
    Source: lines 377-384
    """
    close = df['close'].astype(np.float64)
    high = df['high'].astype(np.float64)
    low = df['low'].astype(np.float64)

    df['ema_9'] = close.ewm(span=EMA_FAST, adjust=False).mean().astype(np.float32)
    df['ema_21'] = close.ewm(span=EMA_SLOW, adjust=False).mean().astype(np.float32)

    # True Range: max(high-low, |high-prev_close|, |low-prev_close|)
    # Source: line 383
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(VOV_ATR_PERIOD).mean().astype(np.float32)

    return df


# ====================================================================
# TF SIGNAL: +1, -1, or 0
# Source: lines 392-394 (DataEngine._compute_mtf_bias_h1)
# ====================================================================

def compute_tf_signal(df):
    """
    Compute directional signal for a single timeframe.
    Source: lines 392-394

    Signal logic:
      +1 if EMA_9 > EMA_21 AND close > EMA_9  (bullish)
      -1 if EMA_9 < EMA_21 AND close < EMA_9  (bearish)
       0 otherwise (neutral)
    """
    signal = pd.Series(0.0, index=df.index, dtype=np.float32)
    signal[(df['ema_9'] > df['ema_21']) & (df['close'] > df['ema_9'])] = 1.0
    signal[(df['ema_9'] < df['ema_21']) & (df['close'] < df['ema_9'])] = -1.0
    return signal


# ====================================================================
# MTF BIAS — COMPUTED ON H1 INDEX
# Source: lines 386-401 (DataEngine._compute_mtf_bias_h1)
# ====================================================================

def compute_mtf_bias_on_h1(data):
    """
    Multi-timeframe bias computed at H1 resolution.

    All TF signals are forward-filled onto the H1 index, weighted, and summed.
    Source: lines 386-401

    Args:
        data: dict of {'M5': df, 'M15': df, 'H1': df, 'H4': df}
              Each df must have 'ema_9', 'ema_21', 'close' columns.

    Returns:
        pd.Series of MTF bias values at H1 resolution, clipped to [-1, 1]
    """
    h1 = data['H1']
    h1_index = h1.index

    tf_signals = {}
    for tf_name in ['M5', 'M15', 'H1', 'H4']:
        tf_df = data[tf_name]
        signal = compute_tf_signal(tf_df)
        tf_signals[tf_name] = signal.reindex(h1_index, method='ffill')

    total_weight = sum(MTF_WEIGHTS.values())
    mtf_bias = pd.Series(0.0, index=h1_index, dtype=np.float64)
    for tf_name, weight in MTF_WEIGHTS.items():
        mtf_bias += tf_signals[tf_name].fillna(0.0) * weight

    return (mtf_bias / total_weight).clip(-1.0, 1.0).astype(np.float32)


# ====================================================================
# VOLATILITY OF VOLATILITY
# Source: lines 436-441 (DataEngine._compute_rolling_vov)
# ====================================================================

def compute_vov(h1_atr):
    """
    VoV = rolling_std(ATR) / rolling_mean(ATR) over VOV_ROLLING_WINDOW bars.
    Source: lines 436-441

    Args:
        h1_atr: pd.Series of H1 ATR values

    Returns:
        pd.Series of VoV values
    """
    atr = h1_atr.astype(np.float64)
    window = VOV_ROLLING_WINDOW
    if len(atr.dropna()) < window:
        return pd.Series(np.nan, index=h1_atr.index, dtype=np.float32)
    return (
        atr.rolling(window).std() /
        atr.rolling(window).mean().replace(0, np.nan)
    ).astype(np.float32)


# ====================================================================
# INDICATOR PROJECTION — H1 → M5
# Source: lines 447-505 (compute_indicators_at_resolution)
# ====================================================================

def project_indicators_to_m5(data, h1_df):
    """
    Project MTF bias, Hurst, VoV, ATR from H1 onto M5 grid.

    Key detail: H1 indicators are SHIFTED FORWARD by 55 minutes before
    projection. This prevents look-ahead — an H1 bar closing at 10:00
    doesn't affect M5 bars until 10:55.

    Source: lines 447-505

    Args:
        data: dict of {'M5': df, 'M15': df, 'H1': df, 'H4': df}
        h1_df: H1 DataFrame with computed hurst, vov, atr, mtf_bias columns

    Returns:
        dict with numpy arrays: 'mtf_bias', 'hurst', 'vov', 'atr'
        All aligned to data['M5'].index
    """
    target_df = data['M5']
    target_idx = target_df.index
    h1_shift = pd.Timedelta(minutes=H1_SHIFT_MINUTES_M5)

    # ── MTF BIAS at M5 resolution ──
    # Source: lines 462-484
    # Recompute signals at M5 resolution (not just forward-fill from H1)
    weights = MTF_WEIGHTS
    total_weight = sum(weights.values())

    signals = {}
    for tf_name in ['M5', 'M15']:
        tf_df = data[tf_name]
        sig = compute_tf_signal(tf_df)
        signals[tf_name] = sig.reindex(target_idx, method='ffill').fillna(0.0)

    # H1 signal: shift forward by 55 min, then forward-fill to M5
    # Source: lines 470-475
    sig_h1 = compute_tf_signal(data['H1'])
    sig_h1_shifted = sig_h1.copy()
    sig_h1_shifted.index = sig_h1_shifted.index + h1_shift
    signals['H1'] = sig_h1_shifted.reindex(target_idx, method='ffill').fillna(0.0)

    # H4 signal: forward-fill to M5 (no shift — H4 is already stale enough)
    # Source: lines 477-481
    sig_h4 = compute_tf_signal(data['H4'])
    signals['H4'] = sig_h4.reindex(target_idx, method='ffill').fillna(0.0)

    mtf_bias = sum(signals[tf] * weights[tf] for tf in weights) / total_weight
    mtf_bias = mtf_bias.clip(-1.0, 1.0).astype(np.float32)

    # ── H1 INDICATORS shifted to M5 ──
    # Source: lines 486-495
    def shift_and_fill(series):
        s = series.copy()
        s.index = s.index + h1_shift
        return s.reindex(target_idx, method='ffill')

    return {
        'mtf_bias': mtf_bias.values.astype(np.float64),
        'hurst': shift_and_fill(h1_df['hurst']).values.astype(np.float64),
        'vov': shift_and_fill(h1_df['vov']).values.astype(np.float64),
        'atr': shift_and_fill(h1_df['atr']).values.astype(np.float64),
    }


# ====================================================================
# REGIME MASK — V1 PARAMS
# Source: lines 774-775
# ====================================================================

def compute_regime_mask(indicators, vov_thresh):
    """
    Compute boolean regime mask using V1 parameters.
    Source: lines 774-775

    Regime is ON when ALL of:
      - abs(MTF bias) >= 0.8
      - Hurst > 0.55
      - VoV < vov_thresh (p90 of training data)
      - VoV is not NaN
      - Hurst is not NaN
    """
    mtf_abs = np.abs(indicators['mtf_bias'])
    hurst = indicators['hurst']
    vov = indicators['vov']

    regime_mask = (
        (mtf_abs >= V1_MTF_THRESH) &
        (hurst > V1_HURST_THRESH) &
        (vov < vov_thresh) &
        (~np.isnan(vov)) &
        (~np.isnan(hurst))
    )
    return regime_mask


# ====================================================================
# M30 WINDOW ID COMPUTATION
# Source: line 781
# ====================================================================

def compute_window_id(timestamp):
    """
    Compute M30 window ID from a timestamp.
    Source: line 781: window_ids = hours * 2 + (minutes >= 30)

    Window 0 = 00:00-00:29, Window 1 = 00:30-00:59, ...
    Window 16 = 08:00-08:29 (London open), Window 25 = 12:30-12:59
    """
    return timestamp.hour * 2 + (1 if timestamp.minute >= 30 else 0)


def compute_window_ids_array(index):
    """Compute window IDs for an entire DatetimeIndex. Source: line 781."""
    hours = index.hour
    minutes = index.minute
    return (hours * 2 + (minutes >= 30).astype(int)).values.astype(np.int64)
