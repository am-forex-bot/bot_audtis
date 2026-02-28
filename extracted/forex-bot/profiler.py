"""
WINDOW PROFILER — Walk-Forward Window Selection
=================================================
Implements the exact window profiling logic from the backtest.

In the backtest, each fold had:
  1. 2-year training period
  2. Simulate ALL windows on training data with V1 params
  3. Group trades by M30 window ID (0-47)
  4. Select London windows meeting thresholds + addon windows meeting stricter thresholds
  5. Apply those windows to OOS data

For live trading, we replicate this with a rolling 2-year window:
  - Fetch 2 years of M30 candle data
  - Compute indicators
  - Simulate trades (all windows, V1 params)
  - Profile each window
  - Select windows meeting the london_selective thresholds

This reprofiles daily at REPROFILE_HOUR_UTC (default 22:00 = after NY close).
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from config import (
    V1_MTF_THRESH, V1_HURST_THRESH, V1_VOV_PCT,
    LONDON_WINDOWS,
    LONDON_MIN_SHARPE, LONDON_MIN_MEAN_PNL, LONDON_MIN_TRADES,
    ADDON_MIN_SHARPE, ADDON_MIN_MEAN_PNL, ADDON_MIN_TRADES,
    SL_ATR_MULT, MAX_SPREAD_PIPS, SLIPPAGE_PIPS, SL_SLIPPAGE_PIPS,
    SWAP_PER_DAY_PIPS, REGIME_EXIT,
    pip_multiplier,
)
from indicators import (
    compute_ema_atr, compute_mtf_bias_on_h1, compute_rolling_hurst,
    compute_vov, compute_tf_signal, compute_regime_mask,
    compute_window_ids_array,
)

log = logging.getLogger('bot.profiler')


def simulate_training_trades(m30_df, instrument, indicators, vov_thresh):
    """
    Simulate trades on M30 training data using V1 params.
    ALL windows are active (no window filter) — we profile the results afterwards.

    This is a simplified version of the backtest kernel operating on M30 bars.
    Source: simulate_trades_m5 (lines 737-854) adapted for M30 resolution.

    Args:
        m30_df: M30 DataFrame with OHLC + bid/ask columns
        instrument: e.g. 'EUR_USD'
        indicators: dict from compute_indicators_at_resolution
        vov_thresh: absolute VoV threshold (p90 of training data)

    Returns:
        list of trade dicts with 'pnl_pips' and 'm30_window'
    """
    pip_mult = pip_multiplier(instrument)
    hold_bars = 8  # M30 hold: 8 bars = 4 hours (same as M5 × 48 = 4 hours)
    bars_per_day = 48  # 24h × 2 bars/h
    cost_swap_per_bar = SWAP_PER_DAY_PIPS / bars_per_day

    mtf_bias = indicators['mtf_bias']
    mtf_abs = np.abs(mtf_bias)
    hurst = indicators['hurst']
    vov = indicators['vov']
    atr = indicators['atr']

    # Regime mask: V1 params
    regime_mask = (
        (mtf_abs >= V1_MTF_THRESH) &
        (hurst > V1_HURST_THRESH) &
        (vov < vov_thresh) &
        (~np.isnan(vov)) &
        (~np.isnan(hurst))
    )

    n_bars = len(m30_df)
    index = m30_df.index
    window_ids = compute_window_ids_array(index)

    # Use mid-price if bid/ask not available
    has_bidask = all(c in m30_df.columns for c in
                     ['bid_open', 'bid_close', 'bid_low',
                      'ask_open', 'ask_close', 'ask_high'])

    if has_bidask:
        bid_open = m30_df['bid_open'].values.astype(np.float64)
        bid_close = m30_df['bid_close'].values.astype(np.float64)
        bid_low = m30_df['bid_low'].values.astype(np.float64)
        ask_open = m30_df['ask_open'].values.astype(np.float64)
        ask_close = m30_df['ask_close'].values.astype(np.float64)
        ask_high = m30_df['ask_high'].values.astype(np.float64)
    else:
        close = m30_df['close'].values.astype(np.float64)
        bid_open = close.copy()
        bid_close = close.copy()
        bid_low = m30_df['low'].values.astype(np.float64)
        ask_open = close.copy()
        ask_close = close.copy()
        ask_high = m30_df['high'].values.astype(np.float64)

    # Active mask = regime only (ALL windows open for profiling)
    active_mask = regime_mask

    # ── SIMULATION LOOP — Exact copy of _simulate_trades_kernel ──
    # Source: lines 539-654
    trades = []
    i = 0
    just_exited = False

    while i < n_bars:
        if not active_mask[i]:
            i += 1
            just_exited = False
            continue

        # Signal transition: first bar of regime, or just re-entered
        if i > 0 and active_mask[i - 1] and not just_exited:
            i += 1
            continue

        just_exited = False
        entry_bar = i + 1
        if entry_bar >= n_bars:
            i += 1
            continue

        # Max spread filter
        entry_spread = (ask_open[entry_bar] - bid_open[entry_bar]) * pip_mult
        if entry_spread > MAX_SPREAD_PIPS:
            i += 1
            continue

        direction = 1.0 if mtf_bias[i] > 0 else -1.0

        if direction > 0:
            entry_price = ask_open[entry_bar]
        else:
            entry_price = bid_open[entry_bar]

        entry_atr = atr[i] if not np.isnan(atr[i]) else 0.0
        sl_pips = entry_atr * pip_mult * SL_ATR_MULT if entry_atr > 0 else 999.0

        max_exit_bar = min(entry_bar + hold_bars, n_bars - 1)
        hit_sl = False
        actual_exit_bar = max_exit_bar

        for j in range(entry_bar + 1, max_exit_bar + 1):
            if j >= n_bars:
                break
            # SL check
            if direction > 0:
                adverse = (entry_price - bid_low[j]) * pip_mult
            else:
                adverse = (ask_high[j] - entry_price) * pip_mult
            if adverse >= sl_pips:
                hit_sl = True
                actual_exit_bar = j
                break

        # Exit pricing
        if hit_sl:
            pnl_gross = -(sl_pips + SL_SLIPPAGE_PIPS)
        else:
            if direction > 0:
                exit_price = bid_close[actual_exit_bar]
            else:
                exit_price = ask_close[actual_exit_bar]
            pnl_gross = (exit_price - entry_price) * direction * pip_mult

        bars_held = actual_exit_bar - entry_bar
        swap_cost = cost_swap_per_bar * bars_held
        pnl_net = pnl_gross - SLIPPAGE_PIPS - swap_cost

        trades.append({
            'pnl_pips': round(float(pnl_net), 2),
            'm30_window': int(window_ids[i]),
        })

        next_i = actual_exit_bar + 1
        if next_i <= i:
            next_i = i + 1
        i = next_i
        just_exited = True

    return trades


def compute_window_metrics(trades):
    """
    Compute metrics for a set of trades from one M30 window.
    Source: lines 894-916 (compute_window_metrics)

    Returns dict with: n_trades, total_pips, sharpe, win_rate, mean_pnl
    """
    if len(trades) < 2:
        return {
            'n_trades': len(trades), 'total_pips': 0,
            'sharpe': -999, 'win_rate': 0, 'mean_pnl': 0,
        }

    pnls = np.array([t['pnl_pips'] for t in trades])
    n = len(pnls)
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls, ddof=1)) if n > 1 else 1.0

    # Annualised Sharpe approximation
    # Source: lines 905-909
    tpy = n  # trades in this window over the training period
    sharpe = (mean_pnl / std_pnl * np.sqrt(tpy)) if std_pnl > 1e-10 else 0

    return {
        'n_trades': n,
        'total_pips': round(float(np.sum(pnls)), 2),
        'sharpe': round(float(sharpe), 4),
        'win_rate': round(float(np.mean(pnls > 0)) * 100, 2),
        'mean_pnl': round(mean_pnl, 2),
    }


def select_windows_london_selective(window_profile):
    """
    Select windows using london_plus_selective strategy.
    Source: lines 860-880 (select_windows)

    London windows (IDs 16-25): must meet LONDON thresholds
    Addon windows (all others): must meet stricter ADDON thresholds
    """
    selected = set()

    # London windows
    for wid in LONDON_WINDOWS:
        if wid in window_profile:
            m = window_profile[wid]
            if (m['sharpe'] >= LONDON_MIN_SHARPE and
                    m['mean_pnl'] >= LONDON_MIN_MEAN_PNL and
                    m['n_trades'] >= LONDON_MIN_TRADES):
                selected.add(wid)

    # Addon windows (outside London)
    for wid, m in window_profile.items():
        if wid not in LONDON_WINDOWS:
            if (m['sharpe'] >= ADDON_MIN_SHARPE and
                    m['mean_pnl'] >= ADDON_MIN_MEAN_PNL and
                    m['n_trades'] >= ADDON_MIN_TRADES):
                selected.add(wid)

    return selected


def compute_indicators_for_profiling(data, h1_df, vov_abs):
    """
    Compute indicators at M30 resolution for window profiling.
    Source: lines 447-505 (compute_indicators_at_resolution) with resolution='M30'

    The key difference from M5 projection: H1 shift is 30 minutes for M30.
    Source: line 69: 'h1_shift_minutes': {'M30': 30, 'M15': 45, 'M5': 55}
    """
    target_df = data['M30']
    target_idx = target_df.index
    h1_shift = pd.Timedelta(minutes=30)  # M30 uses 30-min shift

    weights = MTF_WEIGHTS
    total_weight = sum(weights.values())

    signals = {}
    for tf_name in ['M5', 'M15']:
        if tf_name in data:
            sig = compute_tf_signal(data[tf_name])
            signals[tf_name] = sig.reindex(target_idx, method='ffill').fillna(0.0)
        else:
            signals[tf_name] = pd.Series(0.0, index=target_idx, dtype=np.float32)

    sig_h1 = compute_tf_signal(data['H1'])
    sig_h1_shifted = sig_h1.copy()
    sig_h1_shifted.index = sig_h1_shifted.index + h1_shift
    signals['H1'] = sig_h1_shifted.reindex(target_idx, method='ffill').fillna(0.0)

    sig_h4 = compute_tf_signal(data['H4'])
    signals['H4'] = sig_h4.reindex(target_idx, method='ffill').fillna(0.0)

    mtf_bias = sum(signals[tf] * weights[tf] for tf in weights) / total_weight
    mtf_bias = mtf_bias.clip(-1.0, 1.0).astype(np.float32)

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


def profile_windows_for_pair(data, h1_df, instrument):
    """
    Full window profiling pipeline for a single pair.

    1. Compute VoV p90 threshold from H1 training data
    2. Compute M30 indicators
    3. Simulate ALL windows with V1 params
    4. Profile each window
    5. Select windows meeting london_selective thresholds

    Args:
        data: dict of DataFrames {'M5': df, 'M15': df, 'M30': df, 'H1': df, 'H4': df}
        h1_df: H1 DataFrame with hurst, vov, atr columns already computed
        instrument: e.g. 'EUR_USD'

    Returns:
        set of active M30 window IDs, or empty set if insufficient data
    """
    # VoV threshold from training data
    # Source: lines 1008-1014
    vov_vals = h1_df['vov'].values.astype(np.float64)
    vov_valid = vov_vals[~np.isnan(vov_vals)]
    if len(vov_valid) < 50:
        log.warning(f'{instrument}: insufficient VoV data ({len(vov_valid)} valid), skipping')
        return set(), 999.0

    vov_abs = float(np.percentile(vov_valid, V1_VOV_PCT))

    # M30 indicators
    indicators = compute_indicators_for_profiling(data, h1_df, vov_abs)

    # Simulate training trades (all windows open)
    trades = simulate_training_trades(data['M30'], instrument, indicators, vov_abs)

    if len(trades) < 10:
        log.warning(f'{instrument}: only {len(trades)} training trades, skipping')
        return set(), vov_abs

    # Profile each window
    # Source: lines 1027-1031
    window_profile = {}
    for wid in range(48):
        w_trades = [t for t in trades if t['m30_window'] == wid]
        if len(w_trades) >= 2:
            window_profile[wid] = compute_window_metrics(w_trades)

    # Select windows
    selected = select_windows_london_selective(window_profile)

    log.info(f'{instrument}: {len(trades)} training trades, '
             f'{len(selected)} windows selected: {sorted(selected)}')

    return selected, vov_abs
