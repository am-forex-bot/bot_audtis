"""
TRADER — Signal Detection & Order Management
===============================================
Replicates the backtest's entry/exit logic using OANDA live orders.

BACKTEST → LIVE MAPPING:
==========================
  Backtest: Signal at bar i → entry at bar i+1 open
  Live:     Signal when M5 bar closes → place market order (fills at next tick ≈ next bar open)

  Backtest: SL checked intrabar on bid_low (long) / ask_high (short)
  Live:     SL order attached to trade → OANDA monitors and fills

  Backtest: Timed exit at bar entry_bar + 48 at bid_close / ask_close
  Live:     Track open time, close when 48 M5 bars elapsed (4 hours)

  Backtest: Max spread filter → skip if spread > 5.0 pips
  Live:     Check current spread before placing order → skip if > 5.0 pips

  Backtest: just_exited flag → can re-enter on next bar after exit
  Live:     Track per-pair cooldown after exit

  Backtest: regime_exit = False → hold for full duration regardless
  Live:     Same — hold for 48 bars, only exit on SL hit or time

ENTRY CONDITION (Source: lines 542-564):
  1. active_mask[i] is True (regime ON + window active)
  2. Transition: bar i-1 was NOT active, OR just_exited from previous trade
  3. entry_bar = i + 1 (next bar)
  4. spread at entry_bar < MAX_SPREAD_PIPS
  5. direction = sign(mtf_bias[i])
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from config import (
    V1_MTF_THRESH, V1_HURST_THRESH,
    SL_ATR_MULT, MAX_SPREAD_PIPS,
    HOLD_BARS_M5, UNITS_PER_TRADE,
    LONDON_WINDOWS,
    pip_multiplier,
)
from indicators import (
    project_indicators_to_m5, compute_regime_mask,
    compute_window_id, compute_window_ids_array,
)

log = logging.getLogger('bot.trader')


class OpenPosition:
    """Tracks a live position opened by the bot."""

    def __init__(self, trade_id, instrument, direction, entry_time,
                 entry_price, sl_pips, m5_bar_count=0):
        self.trade_id = trade_id
        self.instrument = instrument
        self.direction = direction  # 'long' or 'short'
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.sl_pips = sl_pips
        self.m5_bars_held = m5_bar_count

    def __repr__(self):
        return (f'Position({self.instrument} {self.direction} '
                f'@ {self.entry_price} | {self.m5_bars_held}/{HOLD_BARS_M5} bars '
                f'| SL={self.sl_pips:.1f} pips | id={self.trade_id})')


class Trader:
    """
    Manages signal detection and trade execution.

    State per pair:
      - active_windows: set of M30 window IDs (from profiler)
      - vov_thresh: absolute VoV threshold (from profiler)
      - position: OpenPosition or None
      - prev_active: whether previous M5 bar had active_mask=True
      - just_exited: whether we just closed a position (allows re-entry)
    """

    def __init__(self, oanda_client, data_manager):
        self.client = oanda_client
        self.dm = data_manager

        # Per-pair state
        self.active_windows = {}   # instrument → set of M30 window IDs
        self.vov_thresholds = {}   # instrument → float
        self.positions = {}        # instrument → OpenPosition
        self.prev_active = {}      # instrument → bool
        self.just_exited = {}      # instrument → bool

        # Tracking
        self.trades_today = 0
        self.total_trades = 0

    def set_pair_config(self, instrument, windows, vov_thresh):
        """Set active windows and VoV threshold from profiler.
        Seeds prev_active from the second-to-last M5 bar's active_mask
        to prevent spurious signals after startup or reprofile."""
        self.active_windows[instrument] = windows
        self.vov_thresholds[instrument] = vov_thresh

        # Seed prev_active from historical data to avoid spurious first signal
        prev_was_active = self._compute_prev_active(instrument, windows, vov_thresh)
        self.prev_active[instrument] = prev_was_active

        if instrument not in self.just_exited:
            self.just_exited[instrument] = False

    def _compute_prev_active(self, instrument, windows, vov_thresh):
        """Compute whether the second-to-last M5 bar had active_mask=True.
        This prevents spurious entry signals on startup when the regime
        is already active (the bot would incorrectly see a 'transition')."""
        data = self.dm.get_data(instrument)
        h1 = self.dm.get_h1(instrument)
        m5 = self.dm.get_m5(instrument)

        if not data or h1 is None or m5 is None or len(m5) < 3:
            return False

        try:
            indicators = project_indicators_to_m5(data, h1)
            regime_mask = compute_regime_mask(indicators, vov_thresh)

            if not windows:
                return False

            window_ids = compute_window_ids_array(m5.index)
            wf_arr = np.array(sorted(windows), dtype=np.int64)
            sess_mask = np.isin(window_ids, wf_arr)
            active_mask = regime_mask & sess_mask

            # Check second-to-last bar (the bar before the current latest)
            return bool(active_mask[-2])
        except Exception:
            return False

    def check_signal(self, instrument, last_processed_time=None):
        """
        Check all new completed M5 bars for an entry signal.

        Evaluates every bar since last_processed_time (not just the last bar)
        to prevent silently losing signals when a cycle is missed.

        This replicates the backtest's inner loop logic:
          Source: lines 542-564

        Returns:
          dict with signal details, or None if no signal

        The signal dict contains:
          'direction': 'long' or 'short'
          'mtf_bias': float (the bias value)
          'atr': float (H1 ATR at signal bar)
          'window_id': int (M30 window ID)
          'hurst': float
          'vov': float
          'signal_bar_time': pd.Timestamp
        """
        data = self.dm.get_data(instrument)
        h1 = self.dm.get_h1(instrument)
        m5 = self.dm.get_m5(instrument)

        if not data or h1 is None or m5 is None:
            return None
        if len(m5) < 2 or len(h1) < HOLD_BARS_M5:
            return None

        windows = self.active_windows.get(instrument, set())
        vov_thresh = self.vov_thresholds.get(instrument, 999.0)

        if not windows:
            return None

        # Already have a position for this pair
        if instrument in self.positions:
            return None

        # Project indicators to M5
        indicators = project_indicators_to_m5(data, h1)

        # Regime mask
        regime_mask = compute_regime_mask(indicators, vov_thresh)

        # Window filter
        window_ids = compute_window_ids_array(m5.index)
        wf_arr = np.array(sorted(windows), dtype=np.int64)
        sess_mask = np.isin(window_ids, wf_arr)

        # Active mask = regime AND session
        active_mask = regime_mask & sess_mask

        # Determine range of bars to evaluate: all bars since last_processed_time
        if last_processed_time is not None:
            start_idx = int((m5.index > last_processed_time).argmax())
            if start_idx == 0 and m5.index[0] <= last_processed_time:
                start_idx = len(m5) - 1  # No new bars, just check latest
        else:
            start_idx = len(m5) - 1  # First call, check latest only

        # Evaluate each new bar sequentially (replicates backtest's per-bar loop)
        for i in range(start_idx, len(m5)):
            if not active_mask[i]:
                self.prev_active[instrument] = False
                self.just_exited[instrument] = False
                continue

            # Transition check: must be first bar of regime, or just re-entered
            # Source: lines 548-549
            prev_was_active = self.prev_active.get(instrument, False)
            just_exited = self.just_exited.get(instrument, False)

            if prev_was_active and not just_exited:
                # Regime was already active last bar and we didn't just exit
                # → not a new signal, just continuation
                self.prev_active[instrument] = True
                self.just_exited[instrument] = False
                continue

            # We have a valid signal!
            self.prev_active[instrument] = True
            self.just_exited[instrument] = False

            # Direction: sign of MTF bias
            # Source: line 564
            mtf_val = float(indicators['mtf_bias'][i])
            direction = 'long' if mtf_val > 0 else 'short'

            # ATR for SL computation
            atr_val = float(indicators['atr'][i])
            if np.isnan(atr_val):
                atr_val = 0.0

            return {
                'direction': direction,
                'mtf_bias': mtf_val,
                'atr': atr_val,
                'window_id': int(window_ids[i]),
                'hurst': float(indicators['hurst'][i]),
                'vov': float(indicators['vov'][i]),
                'signal_bar_time': m5.index[i],
            }

        return None

    def on_entry_failed(self, instrument):
        """Reset prev_active so the signal can re-fire on the next bar.
        Without this, a spread rejection permanently consumes the signal
        transition and the regime must turn OFF and ON again."""
        self.prev_active[instrument] = False

    def execute_entry(self, instrument, signal):
        """
        Execute an entry order based on a signal.

        Replicates backtest logic:
          Source: lines 558-574

        1. Check current spread
        2. Compute SL price from ATR
        3. Place market order with SL
        """
        pip_mult = pip_multiplier(instrument)

        # Get current price to check spread
        price_info = self.client.get_current_price(instrument)
        if not price_info:
            log.warning(f'{instrument}: cannot get price, skipping entry')
            return False

        if not price_info.get('tradeable', True):
            log.info(f'{instrument}: market not tradeable, skipping')
            return False

        bid = price_info['bid']
        ask = price_info['ask']
        spread_pips = (ask - bid) * pip_mult

        # Max spread filter
        # Source: lines 558-561
        if spread_pips > MAX_SPREAD_PIPS:
            log.info(f'{instrument}: spread {spread_pips:.1f} > {MAX_SPREAD_PIPS} max, skipping')
            return False

        direction = signal['direction']
        atr = signal['atr']

        # SL computation
        # Source: line 574
        if atr > 0:
            sl_pips = atr * pip_mult * SL_ATR_MULT
        else:
            sl_pips = 999.0  # Effectively no SL

        # Compute SL price
        if direction == 'long':
            entry_price = ask  # Long enters at ask
            sl_price = entry_price - (sl_pips / pip_mult)
            units = UNITS_PER_TRADE
        else:
            entry_price = bid  # Short enters at bid
            sl_price = entry_price + (sl_pips / pip_mult)
            units = -UNITS_PER_TRADE

        # Place order
        comment = (f'V1 {direction} w{signal["window_id"]} '
                   f'mtf={signal["mtf_bias"]:.2f} h={signal["hurst"]:.3f}')

        fill = self.client.place_market_order(
            instrument, units, sl_price=sl_price, comment=comment)

        if fill:
            trade_id = fill.get('tradeOpened', {}).get('tradeID') or fill.get('id', 'unknown')
            fill_price = float(fill.get('price', entry_price))

            # Use the signal bar's M5 timestamp for entry_time so that
            # bar counting in check_timed_exits matches the backtest's
            # 48-bar hold period exactly (not ~49 bars from wall clock)
            entry_time = signal.get('signal_bar_time', datetime.now(timezone.utc))

            position = OpenPosition(
                trade_id=str(trade_id),
                instrument=instrument,
                direction=direction,
                entry_time=entry_time,
                entry_price=fill_price,
                sl_pips=sl_pips,
            )
            self.positions[instrument] = position
            self.trades_today += 1
            self.total_trades += 1

            log.info(f'ENTRY: {position}')
            log.info(f'  Signal: mtf={signal["mtf_bias"]:.3f} '
                     f'hurst={signal["hurst"]:.3f} vov={signal["vov"]:.3f} '
                     f'window={signal["window_id"]} spread={spread_pips:.1f}')
            return True
        else:
            log.error(f'{instrument}: order failed')
            return False

    def check_timed_exits(self):
        """
        Check all open positions for timed exit (48 M5 bars = 4 hours).

        Source: lines 576-578 (max_exit_bar = entry_bar + hold_bars)

        FIXED: Counts actual completed M5 bars since entry, NOT clock time.
        This correctly handles weekends — if you enter Friday 21:50 UTC,
        the market closes and no M5 bars form until Sunday ~21:00 UTC.
        Clock time would show 240 minutes elapsed on Saturday morning,
        but only ~2 M5 bars actually formed. We count real bars.
        """
        positions_to_close = []

        for instrument, pos in list(self.positions.items()):
            m5 = self.dm.get_m5(instrument)
            if m5 is None or len(m5) == 0:
                continue

            # Count M5 bars that have closed AFTER entry time
            bars_since_entry = (m5.index > pos.entry_time).sum()
            pos.m5_bars_held = bars_since_entry

            if bars_since_entry >= HOLD_BARS_M5:
                positions_to_close.append(instrument)

        for instrument in positions_to_close:
            pos = self.positions[instrument]
            log.info(f'TIMED EXIT: {pos} ({pos.m5_bars_held} bars)')

            result = self.client.close_trade(pos.trade_id)
            if result:
                pl = result.get('pl', '?')
                log.info(f'  Closed: P&L = {pl}')
            else:
                log.error(f'  Failed to close {pos.trade_id}, will retry')
                continue  # Don't remove — will retry next cycle

            del self.positions[instrument]
            self.just_exited[instrument] = True

    def sync_positions(self):
        """
        Sync bot's position tracking with OANDA's actual open trades.
        Handles cases where SL was hit (OANDA closed the trade) while
        the bot was between check cycles.
        """
        oanda_trades = self.client.get_open_trades()
        oanda_trade_ids = set()
        for t in oanda_trades:
            tid = t.get('id', '')
            oanda_trade_ids.add(tid)

        # Check for positions the bot thinks are open but OANDA has closed
        closed_instruments = []
        for instrument, pos in self.positions.items():
            if pos.trade_id not in oanda_trade_ids:
                log.info(f'SL HIT (or external close): {pos}')
                closed_instruments.append(instrument)
                self.just_exited[instrument] = True

        for instrument in closed_instruments:
            del self.positions[instrument]

    def get_status(self):
        """Get current status summary."""
        return {
            'open_positions': len(self.positions),
            'positions': {inst: str(pos) for inst, pos in self.positions.items()},
            'trades_today': self.trades_today,
            'total_trades': self.total_trades,
            'pairs_with_windows': sum(1 for w in self.active_windows.values() if w),
        }

    def reset_daily_counter(self):
        """Reset daily trade counter (call at start of trading day)."""
        self.trades_today = 0
