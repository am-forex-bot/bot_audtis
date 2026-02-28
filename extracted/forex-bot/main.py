#!/usr/bin/env python3
"""
FOREX TRADING BOT — MAIN LOOP
================================
Trades the london_selective_v1 strategy with 5x ATR SL.

Backtest reference:
  15,068 trades | +51,304 pips | +3.405 p/trade | 53.0% WR | 0.9% SL hits

Architecture:
  1. Startup: warm up data (2yr H1 for profiling + indicators)
  2. Profile windows for each pair (rolling 2yr training window)
  3. Main loop (every ~30 seconds):
     a. Check if new M5 bar has closed
     b. Update candle data
     c. Recompute indicators on new data
     d. Check entry signals for all pairs
     e. Execute entries
     f. Check timed exits (48 bars = 4 hours)
     g. Sync with OANDA (detect SL hits)
  4. Daily: reprofile windows at REPROFILE_HOUR_UTC

Usage:
  python main.py
"""

import os
import sys
import time
import logging
import signal as signal_mod
from datetime import datetime, timezone, timedelta

from config import (
    PAIRS, OANDA_ACCOUNT_ID, OANDA_ENVIRONMENT,
    LOG_LEVEL, LOG_DIR, REPROFILE_DAY_UTC, REPROFILE_HOUR_UTC,
    H1_WARMUP_CANDLES, M30_WARMUP_CANDLES,
    HOLD_BARS_M5, SL_ATR_MULT, MAX_SPREAD_PIPS,
    V1_MTF_THRESH, V1_HURST_THRESH, V1_VOV_PCT,
    LONDON_MIN_SHARPE, LONDON_MIN_MEAN_PNL, LONDON_MIN_TRADES,
    ADDON_MIN_SHARPE, ADDON_MIN_MEAN_PNL, ADDON_MIN_TRADES,
    UNITS_PER_TRADE,
)
from oanda_api import OandaClient
from data_manager import DataManager
from indicators import compute_ema_atr, compute_rolling_hurst, compute_vov
from profiler import profile_windows_for_pair
from trader import Trader


# ── LOGGING SETUP ──

def setup_logging():
    """Configure logging to both file and stdout."""
    os.makedirs(LOG_DIR, exist_ok=True)

    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(log_format, date_format))
    root.addHandler(console)

    # File handler (daily rotation)
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f'forex_bot_{today}.log'),
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    root.addHandler(file_handler)

    # Trade-specific log (append-only journal of all trades)
    trade_handler = logging.FileHandler(
        os.path.join(LOG_DIR, 'trades.log'),
        encoding='utf-8'
    )
    trade_handler.setFormatter(logging.Formatter(log_format, date_format))
    trade_handler.setLevel(logging.INFO)
    trade_logger = logging.getLogger('bot.trader')
    trade_logger.addHandler(trade_handler)

    return logging.getLogger('bot.main')


# ── GRACEFUL SHUTDOWN ──

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logging.getLogger('bot.main').info(
        f'Shutdown signal received ({signum}). Finishing current cycle...')


# ── MAIN BOT CLASS ──

class ForexBot:
    """Main bot orchestrator."""

    def __init__(self):
        self.log = logging.getLogger('bot.main')
        self.client = OandaClient()
        self.dm = DataManager(self.client)
        self.trader = Trader(self.client, self.dm)

        self.last_m5_times = {}      # instrument → last processed M5 bar time
        self.last_profile_date = None # date when windows were last profiled
        self.running = False

    def startup(self):
        """
        Full startup sequence:
          1. Verify OANDA connectivity
          2. Warm up data for all pairs
          3. Compute H1 features
          4. Profile windows
        """
        self.log.info('=' * 70)
        self.log.info('  FOREX BOT — STARTING UP')
        self.log.info(f'  Environment: {OANDA_ENVIRONMENT}')
        self.log.info(f'  Account: {OANDA_ACCOUNT_ID}')
        self.log.info(f'  Pairs: {len(PAIRS)}')
        self.log.info(f'  Units/trade: {UNITS_PER_TRADE}')
        self.log.info(f'  Strategy: london_selective_v1')
        self.log.info(f'  SL: {SL_ATR_MULT}x ATR')
        self.log.info(f'  Hold: {HOLD_BARS_M5} M5 bars (4 hours)')
        self.log.info(f'  Max spread: {MAX_SPREAD_PIPS} pips')
        self.log.info(f'  Regime: MTF>={V1_MTF_THRESH} Hurst>{V1_HURST_THRESH} VoV<p{V1_VOV_PCT}')
        self.log.info(f'  London thresholds: S>={LONDON_MIN_SHARPE} '
                       f'M>={LONDON_MIN_MEAN_PNL} N>={LONDON_MIN_TRADES}')
        self.log.info(f'  Addon thresholds: S>={ADDON_MIN_SHARPE} '
                       f'M>={ADDON_MIN_MEAN_PNL} N>={ADDON_MIN_TRADES}')
        self.log.info(f'  Training window: EXPANDING (all available data)')
        self.log.info(f'  H1 warmup: {H1_WARMUP_CANDLES} | M30 warmup: {M30_WARMUP_CANDLES}')
        day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][REPROFILE_DAY_UTC]
        self.log.info(f'  Reprofile: weekly {day_name} {REPROFILE_HOUR_UTC:02d}:00 UTC')
        self.log.info('=' * 70)

        # ── VERIFY CONNECTIVITY ──
        self.log.info('Checking OANDA connectivity...')
        account = self.client.get_account_summary()
        if not account:
            self.log.error('FATAL: Cannot connect to OANDA. Check API token and account ID.')
            sys.exit(1)
        balance = account.get('balance', '?')
        self.log.info(f'Connected. Balance: {balance}')

        # ── WARM UP DATA ──
        self.log.info(f'\nWarming up data (expanding window: '
                       f'{H1_WARMUP_CANDLES} H1 + {M30_WARMUP_CANDLES} M30 per pair)...')
        self.log.info('This will take several minutes on first run.')
        for instrument in PAIRS:
            try:
                self.dm.warm_up(instrument)
                self.dm.compute_h1_features(instrument)
                self.log.info(f'  {instrument}: data ready')
            except Exception as e:
                self.log.error(f'  {instrument}: warm-up failed: {e}', exc_info=True)

        # ── PROFILE WINDOWS ──
        self.profile_all_pairs()

        self.running = True
        self.log.info('\nStartup complete. Entering main loop.')

    def profile_all_pairs(self):
        """Profile windows for all pairs using current training data."""
        self.log.info('\n─── PROFILING WINDOWS ───')
        now = datetime.now(timezone.utc)
        for instrument in PAIRS:
            try:
                data = self.dm.get_data(instrument)
                h1 = self.dm.get_h1(instrument)
                if not data or h1 is None or len(h1) < 1000:
                    self.log.warning(f'{instrument}: insufficient data for profiling')
                    continue

                # Data freshness check: skip profiling on stale data
                latest_h1_time = h1.index[-1]
                staleness = now - latest_h1_time.to_pydatetime().replace(tzinfo=timezone.utc) \
                    if latest_h1_time.tzinfo is None else now - latest_h1_time
                if staleness > timedelta(hours=6):
                    self.log.warning(f'{instrument}: H1 data is stale '
                                     f'(last bar: {latest_h1_time}, {staleness}), skipping profile')
                    continue

                windows, vov_thresh = profile_windows_for_pair(data, h1, instrument)
                self.trader.set_pair_config(instrument, windows, vov_thresh)

                # Trim old data to prevent unbounded memory growth
                self.dm.trim_old_data(instrument)

            except Exception as e:
                self.log.error(f'{instrument}: profiling failed: {e}', exc_info=True)

        status = self.trader.get_status()
        self.log.info(f'Profiling complete. {status["pairs_with_windows"]} pairs have active windows.')
        self.last_profile_date = datetime.now(timezone.utc).date()

    def update_data(self, instrument):
        """Fetch latest candles for all timeframes."""
        for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
            try:
                new_count = self.dm.update_candle(instrument, tf)
                if new_count > 0 and tf in ('H1', 'H4'):
                    # Recompute H1 features when H1 or H4 updates
                    # (MTF bias depends on H4 signal)
                    self.dm.compute_h1_features(instrument)
            except Exception as e:
                self.log.error(f'{instrument} {tf} update failed: {e}')

    def has_new_m5_bar(self, instrument):
        """Check if there's a new completed M5 bar since last check."""
        latest = self.dm.get_latest_m5_time(instrument)
        if latest is None:
            return False
        last_processed = self.last_m5_times.get(instrument)
        if last_processed is None or latest > last_processed:
            self.last_m5_times[instrument] = latest
            return True
        return False

    def should_reprofile(self):
        """Check if it's time for weekly window reprofiling.
        Default: Sunday 22:00 UTC — markets technically open (Wellington)
        but spreads are wide and no signal would fire. Fresh profile
        is ready before Monday London open."""
        now = datetime.now(timezone.utc)
        if self.last_profile_date is None:
            return True
        # Check: is it the right day and hour, and we haven't profiled this week?
        days_since = (now.date() - self.last_profile_date).days
        if (days_since >= 7 and
                now.weekday() == REPROFILE_DAY_UTC and
                now.hour >= REPROFILE_HOUR_UTC):
            return True
        # Also reprofile if it's been >8 days (safety net for missed windows)
        if days_since > 8:
            return True
        return False

    def main_loop(self):
        """
        Main trading loop. Runs every ~30 seconds.

        Cycle:
          1. Check for daily reprofile
          2. For each pair:
             a. Update candle data
             b. If new M5 bar: check signal → execute entry
          3. Check timed exits for all positions
          4. Sync with OANDA (detect SL hits)
          5. Sleep until next cycle
        """
        cycle = 0
        while self.running and not shutdown_requested:
            cycle += 1
            cycle_start = time.time()

            try:
                # ── DAILY REPROFILE ──
                if self.should_reprofile():
                    self.log.info('Daily reprofile triggered')
                    for instrument in PAIRS:
                        self.update_data(instrument)
                    self.profile_all_pairs()
                    self.trader.reset_daily_counter()

                # ── PER-PAIR SCAN ──
                for instrument in PAIRS:
                    try:
                        # Update data
                        self.update_data(instrument)

                        # Capture the previously processed time BEFORE has_new_m5_bar updates it
                        prev_time = self.last_m5_times.get(instrument)

                        # Check if new M5 bar available
                        if not self.has_new_m5_bar(instrument):
                            continue

                        # Pass the previously processed M5 bar time so check_signal
                        # evaluates ALL new bars, not just the latest one
                        signal = self.trader.check_signal(instrument, prev_time)
                        if signal:
                            success = self.trader.execute_entry(instrument, signal)
                            if not success:
                                # Reset prev_active so signal can re-fire next bar
                                # (prevents spread rejection from permanently
                                # consuming the transition — C4 fix)
                                self.trader.on_entry_failed(instrument)

                    except Exception as e:
                        self.log.error(f'{instrument} scan error: {e}', exc_info=True)

                # ── POSITION MANAGEMENT ──
                self.trader.check_timed_exits()
                self.trader.sync_positions()

                # ── STATUS LOG (every 20 cycles ≈ 10 min) ──
                if cycle % 20 == 0:
                    status = self.trader.get_status()
                    self.log.info(
                        f'[Cycle {cycle}] Open: {status["open_positions"]} | '
                        f'Today: {status["trades_today"]} | '
                        f'Total: {status["total_trades"]} | '
                        f'Pairs active: {status["pairs_with_windows"]}')

            except Exception as e:
                self.log.error(f'Main loop error: {e}', exc_info=True)

            # ── SLEEP ──
            elapsed = time.time() - cycle_start
            sleep_time = max(30 - elapsed, 5)  # Target 30s cycles, min 5s
            time.sleep(sleep_time)

        # ── SHUTDOWN ──
        self.shutdown()

    def shutdown(self):
        """Graceful shutdown. Close all positions if configured to do so."""
        self.log.info('\n─── SHUTTING DOWN ───')
        status = self.trader.get_status()
        self.log.info(f'Open positions: {status["open_positions"]}')
        if status['positions']:
            self.log.info('Leaving positions open (managed by OANDA SL/manual close)')
            for inst, desc in status['positions'].items():
                self.log.info(f'  {desc}')
        self.log.info(f'Total trades this session: {status["total_trades"]}')
        self.log.info('Bot stopped.')


def main():
    log = setup_logging()

    # Register signal handlers for graceful shutdown
    signal_mod.signal(signal_mod.SIGTERM, signal_handler)
    signal_mod.signal(signal_mod.SIGINT, signal_handler)

    # Validate config
    if not OANDA_ACCOUNT_ID or not OANDA_ACCOUNT_ID.replace('-', '').replace('X', ''):
        log.error('FATAL: OANDA_ACCOUNT_ID not set. Copy .env.template to .env and configure.')
        sys.exit(1)

    bot = ForexBot()
    bot.startup()
    bot.main_loop()


if __name__ == '__main__':
    main()
