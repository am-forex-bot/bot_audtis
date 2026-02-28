"""
DATA MANAGER — Fetches and maintains candle data from OANDA
=============================================================
Converts OANDA candle JSON into the exact DataFrame format
that the backtest used, so indicators compute identically.

OANDA S5 → resample to M5, M15, M30, H1, H4 (backtest did this)
OR: Fetch each granularity directly from OANDA (more efficient for live)

We use the direct approach: fetch M5, M15, M30, H1, H4 candles separately.
This is more API-efficient than fetching S5 and resampling.

CRITICAL DIFFERENCE vs backtest:
  The backtest resampled bid_close/ask_close from S5 to get:
    bid_open  = first(bid_close)  — i.e. the S5 bid_close at bar open
    bid_low   = min(bid_close)    — lowest bid during the bar
    ask_open  = first(ask_close)  — i.e. the S5 ask_close at bar open
    ask_high  = max(ask_close)    — highest ask during the bar

  OANDA's native candle API with price='BA' gives:
    bid: o, h, l, c  — OHLC of bid prices
    ask: o, h, l, c  — OHLC of ask prices

  These are FUNCTIONALLY EQUIVALENT:
    bid_open  ≈ bid.o  (opening bid price)
    bid_low   ≈ bid.l  (lowest bid during bar)
    bid_close ≈ bid.c  (closing bid price)
    ask_open  ≈ ask.o  (opening ask price)
    ask_high  ≈ ask.h  (highest ask during bar)
    ask_close ≈ ask.c  (closing ask price)

  The slight difference: backtest derived everything from S5 bid_close/ask_close,
  while OANDA gives true OHLC. OANDA's version is actually MORE accurate for
  live trading. The backtest's resampling was an approximation necessitated
  by the data format available.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from config import (
    OANDA_GRANULARITY, H1_WARMUP_CANDLES, M30_WARMUP_CANDLES,
    VOV_ATR_PERIOD, VOV_ROLLING_WINDOW,
    HURST_WINDOW, HURST_STEP,
)
from indicators import compute_ema_atr, compute_rolling_hurst, compute_vov

log = logging.getLogger('bot.data')


def candles_to_dataframe(candles, include_bidask=True):
    """
    Convert OANDA candle JSON list to a pandas DataFrame matching
    the backtest's format.

    Args:
        candles: list of OANDA candle dicts (with 'mid', 'bid', 'ask' fields)
        include_bidask: if True, include bid/ask OHLC columns

    Returns:
        pd.DataFrame with DatetimeIndex (UTC) and columns:
          open, high, low, close, volume,
          bid_open, bid_close, bid_low, ask_open, ask_close, ask_high
    """
    if not candles:
        return pd.DataFrame()

    records = []
    for c in candles:
        if not c.get('complete', True) and len(candles) > 1:
            continue  # Skip incomplete candles in historical data

        mid = c.get('mid', {})
        record = {
            'time': pd.Timestamp(c['time']).tz_convert('UTC'),
            'open': float(mid.get('o', 0)),
            'high': float(mid.get('h', 0)),
            'low': float(mid.get('l', 0)),
            'close': float(mid.get('c', 0)),
            'volume': int(c.get('volume', 0)),
        }

        if include_bidask:
            bid = c.get('bid', {})
            ask = c.get('ask', {})
            if bid and ask:
                record['bid_open'] = float(bid.get('o', 0))
                record['bid_close'] = float(bid.get('c', 0))
                record['bid_low'] = float(bid.get('l', 0))
                record['ask_open'] = float(ask.get('o', 0))
                record['ask_close'] = float(ask.get('c', 0))
                record['ask_high'] = float(ask.get('h', 0))

        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.set_index('time').sort_index()

    # Cast to float32 to match backtest precision
    float_cols = df.select_dtypes(include=[np.float64]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    return df


class DataManager:
    """
    Manages candle data for all timeframes and pairs.

    Stores DataFrames in memory, appends new candles as they arrive,
    and provides the dict-of-DataFrames format that indicators expect.
    """

    def __init__(self, oanda_client):
        self.client = oanda_client
        # data[instrument][timeframe] = DataFrame
        self.data = {}

    def warm_up(self, instrument, h1_count=None, m30_count=None):
        """
        Fetch historical candles for all timeframes to warm up indicators.

        CRITICAL: Uses EXPANDING training window to match the backtest.
        The backtest's final fold used ~6 years of training data for profiling.
        We fetch as much H1 and M30 history as OANDA provides.

        M5/M15/H4 only need recent data for live signal computation.
        H1 needs full history for Hurst, VoV, and MTF bias.
        M30 needs full history for window profiling simulation.

        Args:
            instrument: e.g. 'EUR_USD'
            h1_count: number of H1 candles to fetch (default: H1_WARMUP_CANDLES)
            m30_count: number of M30 candles to fetch (default: M30_WARMUP_CANDLES)
        """
        if h1_count is None:
            h1_count = H1_WARMUP_CANDLES
        if m30_count is None:
            m30_count = M30_WARMUP_CANDLES

        self.data[instrument] = {}

        # Fetch each timeframe
        # H1 + M30: need maximum history for expanding window profiling
        # H4: proportional to H1 (needed for MTF bias computation)
        # M5/M15: only recent data needed for live signals
        tf_counts = {
            'H4': max(h1_count // 4, 2000),
            'H1': h1_count,
            'M30': m30_count,
            'M15': 5000,   # ~17 days — enough for live signal + EMA warm-up
            'M5': 5000,    # ~17 days — enough for live signal + EMA warm-up
        }

        for tf, count in tf_counts.items():
            gran = OANDA_GRANULARITY[tf]
            log.info(f'  {instrument} {tf}: fetching {count} candles...')

            if count > 5000:
                candles = self.client.fetch_candles_bulk(
                    instrument, gran, count, price='MBA')
            else:
                candles = self.client.fetch_candles(
                    instrument, gran, count=count, price='MBA')

            df = candles_to_dataframe(candles)
            if len(df) == 0:
                log.error(f'  {instrument} {tf}: NO DATA returned')
                continue

            # Compute EMA and ATR
            compute_ema_atr(df)
            self.data[instrument][tf] = df
            log.info(f'  {instrument} {tf}: {len(df)} candles '
                     f'({df.index[0].date()} → {df.index[-1].date()})')

    def compute_h1_features(self, instrument):
        """
        Compute H1-level features: MTF bias, Hurst, VoV.
        Must be called after warm_up and after all TF DataFrames have EMA/ATR.

        Source: lines 366-371 (DataEngine.load_and_compute — H1 feature block)
        """
        if instrument not in self.data or 'H1' not in self.data[instrument]:
            log.error(f'{instrument}: no H1 data for feature computation')
            return

        h1 = self.data[instrument]['H1']
        data = self.data[instrument]

        # MTF bias at H1 resolution
        # Source: line 367
        from indicators import compute_mtf_bias_on_h1
        h1['mtf_bias'] = compute_mtf_bias_on_h1(data)

        # Hurst exponent
        # Source: line 368
        h1['hurst'] = compute_rolling_hurst(h1['close'])

        # VoV
        # Source: line 369
        h1['vov'] = compute_vov(h1['atr'])

        self.data[instrument]['H1'] = h1

    def update_candle(self, instrument, timeframe):
        """
        Fetch the latest candle(s) and append to stored data.
        Called periodically to keep data current.

        Returns: number of new candles added
        """
        gran = OANDA_GRANULARITY[timeframe]
        existing = self.data.get(instrument, {}).get(timeframe)

        if existing is not None and len(existing) > 0:
            # Fetch candles since last known time
            last_time = existing.index[-1].isoformat()
            candles = self.client.fetch_candles(
                instrument, gran, from_time=last_time, price='MBA')
        else:
            # No existing data, fetch last few candles
            candles = self.client.fetch_candles(
                instrument, gran, count=10, price='MBA')

        if not candles:
            return 0

        new_df = candles_to_dataframe(candles)
        if len(new_df) == 0:
            return 0

        compute_ema_atr(new_df)

        if existing is not None and len(existing) > 0:
            # Merge: update last candle (may have been incomplete) + append new ones
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()

            # Recompute EMA/ATR on the tail to ensure continuity
            # (EMA is path-dependent, so we recompute from scratch on the full series)
            # But for efficiency, only if we have new data
            if len(combined) > len(existing):
                compute_ema_atr(combined)

            self.data[instrument][timeframe] = combined
            new_count = len(combined) - len(existing)
        else:
            if instrument not in self.data:
                self.data[instrument] = {}
            self.data[instrument][timeframe] = new_df
            new_count = len(new_df)

        return max(new_count, 0)

    def get_data(self, instrument):
        """Get all timeframe DataFrames for an instrument."""
        return self.data.get(instrument, {})

    def get_h1(self, instrument):
        """Get H1 DataFrame with features computed."""
        return self.data.get(instrument, {}).get('H1')

    def get_m5(self, instrument):
        """Get M5 DataFrame."""
        return self.data.get(instrument, {}).get('M5')

    def get_latest_m5_time(self, instrument):
        """Get the timestamp of the latest M5 candle."""
        m5 = self.get_m5(instrument)
        if m5 is not None and len(m5) > 0:
            return m5.index[-1]
        return None

    def trim_old_data(self, instrument, max_h1_bars=55000):
        """Trim old data to prevent unbounded memory growth.
        H1 and M30 are kept large for expanding window profiling.
        M5/M15 are trimmed more aggressively — only needed for live signals."""
        if instrument not in self.data:
            return
        for tf in self.data[instrument]:
            df = self.data[instrument][tf]
            if tf == 'H1' and len(df) > max_h1_bars:
                self.data[instrument][tf] = df.iloc[-max_h1_bars:]
            elif tf == 'M30' and len(df) > max_h1_bars * 2:
                self.data[instrument][tf] = df.iloc[-(max_h1_bars * 2):]
            elif tf == 'M5' and len(df) > 10000:
                self.data[instrument][tf] = df.iloc[-10000:]
            elif tf == 'M15' and len(df) > 10000:
                self.data[instrument][tf] = df.iloc[-10000:]
            elif tf == 'H4' and len(df) > max_h1_bars // 4:
                self.data[instrument][tf] = df.iloc[-(max_h1_bars // 4):]
