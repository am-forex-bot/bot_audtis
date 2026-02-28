# Forensic Audit Report: Forex Trading Bot (london_selective_v1)

**Date:** 2026-02-28
**Auditor:** Claude Code
**Files Examined:** 10 source files (config.py, main.py, trader.py, indicators.py, profiler.py, oanda_api.py, data_manager.py, .env, requirements.txt, __init__.py)
**Last Modified:** 2026-02-27

---

## Executive Summary

This bot implements the `london_selective_v1` forex strategy, targeting:
- **15,068 trades | +51,304 pips | +3.405 pips/trade | 53% WR | 0.9% SL hits** (backtest reference)

The strategy uses multi-timeframe EMA crossover signals, Hurst exponent trending regime detection, volatility-of-volatility filtering, and M30 window profiling to selectively trade during London session and statistically-validated addon windows. Positions are held for 4 hours max with 5x ATR catastrophe stop losses.

**The bot has 4 CRITICAL/SHOWSTOPPER bugs, 5 HIGH severity issues, and 6 MEDIUM issues.** One critical bug (`MTF_WEIGHTS` not imported in profiler.py) means the **bot cannot profile windows at all and will never place a trade**. A second critical bug means **market orders can be duplicated via retry logic**. A third means **API credentials are exposed**.

---

## Architecture Overview

```
main.py (orchestrator)
  ├── oanda_api.py    → REST API client (candles, orders, trades)
  ├── data_manager.py → Candle storage, EMA/ATR computation
  ├── indicators.py   → MTF bias, Hurst, VoV, regime mask
  ├── profiler.py     → Walk-forward M30 window profiling
  ├── trader.py       → Signal detection, order execution, position mgmt
  └── config.py       → All strategy parameters (locked to backtest)
```

**Flow:** Startup → warm up 5.7 years of H1/M30 data per pair → compute Hurst/VoV/MTF → profile windows → main loop (30s cycles) → detect M5 bar closes → check regime + window + transition → place market order with ATR stop → hold 48 M5 bars → timed exit or SL.

---

## CRITICAL BUGS (Showstoppers)

### C1. `MTF_WEIGHTS` Not Imported in profiler.py — BOT CANNOT TRADE
**File:** `profiler.py:269` | **Impact:** Total failure

`compute_indicators_for_profiling()` references `MTF_WEIGHTS` at line 269, but it is never imported. The import block (lines 28-36) is missing it. This causes a `NameError` crash every time `profile_windows_for_pair()` is called. The exception is caught in `main.py:188`, logged, and silently continues — resulting in **zero active windows for all pairs**, meaning the bot runs but never trades.

```python
# profiler.py line 269 — CRASHES HERE
weights = MTF_WEIGHTS  # NameError: name 'MTF_WEIGHTS' is not defined
```

**Fix:** Add `MTF_WEIGHTS` to the import from config.

---

### C2. Market Order POST Retries Can Cause Duplicate Fills
**File:** `oanda_api.py:57-77` | **Impact:** Double position size, orphaned trades

The `_post()` method retries up to 3 times. If a market order is filled by OANDA but the HTTP response is lost (timeout, network glitch), the bot retries the exact same order. OANDA fills it again. The FOK `timeInForce` does NOT prevent duplicate submissions.

**Scenario:** Order fills → response timeout → retry → second fill → bot only tracks last fill's trade ID → first trade is orphaned and unmanaged.

**Fix:** Set `max_retries=1` for order placement, or use OANDA's `clientRequestID` for idempotency, or check existing positions before retrying.

---

### C3. API Credentials Committed to Repository
**File:** `.env:10-11` | **Impact:** Full account compromise

The `.env` file contains a real OANDA API token and account ID:
```
OANDA_ACCOUNT_ID=101-004-31618463-002
OANDA_API_TOKEN=824b0a45...
```

There is **no `.gitignore` file** in the repository. While the `.env` is currently only inside the zip, anyone with repo access has full trading authority over this account.

**Fix:** Rotate the API token immediately. Add `.gitignore`. Provide `.env.template` with placeholder values.

---

### C4. Spread Rejection Permanently Consumes Signal — No Retry
**File:** `trader.py:175-184, 205-288` | **Impact:** Lost trades, reduced performance

When `check_signal()` detects a valid signal, it sets `prev_active = True` and `just_exited = False` (line 183-184). If `execute_entry()` then rejects due to spread > 5 pips, the signal is consumed forever. On the next bar, `prev_was_active = True` and `just_exited = False`, so the transition check (line 175) returns `None`. The regime must turn OFF and back ON before a new signal can fire.

In the backtest, a spread rejection just skips entry at that bar — the strategy continues scanning. This is a **material divergence** that systematically reduces trade count versus backtest expectations.

**Fix:** Reset `prev_active[instrument] = False` when `execute_entry()` returns `False`.

---

## HIGH Severity Bugs

### H1. Sharpe Ratio Annualization Is Wrong — Windows Selected Too Liberally
**File:** `profiler.py:215-216`

```python
tpy = n  # THIS IS WRONG — n is total trades, not trades-per-year
sharpe = (mean_pnl / std_pnl * np.sqrt(tpy))
```

The annualized Sharpe formula requires `sqrt(trades_per_year)`, not `sqrt(total_trades)`. With ~5.7 years of data, a window with 50 trades/year has `n = 285`, inflating Sharpe by `sqrt(285/50) = 2.4x`. A window with true Sharpe of 0.08 would pass the London threshold (0.2). A window with true Sharpe of 0.29 would pass the addon threshold (0.7). **The bot trades windows whose edge may not actually exist.**

---

### H2. `prev_active` Not Seeded From History on Startup — Spurious Signals
**File:** `trader.py:104-107`

On startup, `prev_active` defaults to `False` for all pairs. If the regime is already active, the first `check_signal` call sees a "transition" and fires a spurious entry. Up to 19 false signals could fire simultaneously after every restart.

---

### H3. Entry Time Uses Wall Clock, Not Bar Time — Holds ~49 Bars Not 48
**File:** `trader.py:273` + `trader.py:310`

```python
entry_time=datetime.now(timezone.utc),  # Wall clock, not bar time
...
bars_since_entry = (m5.index > pos.entry_time).sum()  # Compares against wall clock
```

The entry bar's M5 timestamp (e.g., 10:10:00) is always earlier than `datetime.now()` (e.g., 10:10:03), so the entry bar is never counted. Each trade is held approximately one bar (5 minutes) longer than intended. Over thousands of trades, this systematic shift degrades the backtest's +3.405 pip/trade edge.

---

### H4. No Data Freshness Check Before Profiling
**File:** `main.py:259-261`

The reprofile runs without verifying data is current. If `update_data()` silently fails (API error), profiling runs on stale data, producing stale window selections without any warning.

---

### H5. `get_current_price` Crashes on Empty Price Ladder
**File:** `oanda_api.py:188-189`

```python
'bid': float(price['bids'][0]['price']),
```

If the market is closed and `bids`/`asks` lists are empty, this throws `IndexError` before the `tradeable` check in trader.py can prevent it.

---

## MEDIUM Severity Issues

### M1. `trim_old_data()` Is Never Called — Memory Leak
**File:** `data_manager.py:281`

The method exists but is never invoked anywhere. M5/M15 data grows unboundedly. After a year of continuous operation, M5 would reach ~100K bars per pair. Not catastrophic but wastes ~200MB.

---

### M2. `_put()` Has No HTTP 429 Rate Limit Handling
**File:** `oanda_api.py:79-95`

Unlike `_get` and `_post`, `_put` does not check for 429 responses. Rate-limited trade closes are retried after only 1 second, potentially worsening the rate limit situation. This is used for `close_trade` and `modify_trade_sl` — both critical operations.

---

### M3. Skipped M5 Bars Cause Silent Signal Loss
**File:** `trader.py:163`

```python
i = len(m5) - 1  # Only checks the LAST bar
```

If a network issue causes the bot to miss one cycle, `update_candle` fetches 2 new bars, but `check_signal` only evaluates the last one. A signal on the skipped bar is permanently lost. The backtest evaluates every bar sequentially.

---

### M4. Window ID Attributed to Signal Bar, Not Entry Bar
**File:** `profiler.py:183`

The M30 window ID is recorded from bar `i` (signal bar), but entry occurs at bar `i+1`. At 30-minute boundaries, a signal at window 15 (07:30-07:59) creates an entry at window 16 (08:00-08:29, London open). The trade is attributed to window 15 instead of 16, causing boundary misattribution in the profiling statistics.

---

### M5. Expanding Window Profiling Lacks Recency Weighting
**File:** `profiler.py:324-356`

The VoV percentile and window selection use all available history equally. Old calm/profitable periods dominate the statistics. A window profitable in 2019-2023 but decayed in 2024-2025 may still pass thresholds. The backtest used walk-forward folds to detect this; the live bot does not.

---

### M6. M30 vs M5 Simulation Granularity Discrepancy
**File:** `profiler.py:46-192`

Profiling simulates on M30 bars but live trading uses M5. Regime transitions that occur mid-M30-bar (e.g., MTF bias crosses 0.8 at minute 5 of a 30-min bar) are detected differently. The profiler may undercount trades versus actual M5 live trading, making window selection conservative.

---

## Strategy Analysis

### What the Bot Actually Does

1. **Multi-Timeframe EMA Signal:** EMA(9) vs EMA(21) on M5, M15, H1, H4. Weighted sum (M5: 0.20, M15: 0.30, H1: 0.25, H4: 0.20). Signal requires close above/below fast EMA AND fast above/below slow EMA.

2. **Regime Filter:** `abs(MTF bias) >= 0.8` (strong agreement across TFs) AND `Hurst > 0.55` (trending market via R/S analysis on 504 H1 bars) AND `VoV < p90` (stable volatility).

3. **Window Selection:** Profile M30 windows (0-47, half-hour slots) on 2+ years of training data. London windows (16-25, 08:00-12:59 GMT) need Sharpe >= 0.2, mean P&L >= 0.5 pips, 15+ trades. Non-London "addon" windows need Sharpe >= 0.7, mean >= 2.5 pips, 40+ trades.

4. **Entry:** On M5 bar close, if regime is ON + window is active + transition detected → market order at next tick. Long: buy at ask. Short: sell at bid.

5. **Exit:** Hold for 48 M5 bars (4 hours) then close. Catastrophe SL at 5x ATR (hit rate: 0.9%).

6. **Costs modeled:** 0.2 pip execution slippage, 0.5 pip SL slippage, 0.3 pip/day swap, spread from bid/ask.

### Strategy Strengths
- Comprehensive regime filtering (MTF alignment + Hurst trending + VoV stability)
- Walk-forward window profiling prevents pure curve-fitting
- Conservative position sizing (1000 units = 0.01 lots)
- 55-minute H1 shift correctly prevents look-ahead
- Bar-counting for timed exits correctly handles weekends
- Bid/ask spread data used throughout (not naive mid-price)

### Strategy Concerns
- The +3.405 pip/trade edge is thin — the bugs identified above (especially C4, H1, H3) could easily erode it
- 19 pairs with no correlation/portfolio-level risk management
- No maximum drawdown circuit breaker
- No position sizing adjustment based on account equity
- The Hurst exponent computation is O(n²) and slow for 504-bar windows with 20 log-spaced sub-windows — acceptable but not optimized

---

## Backtest vs Live Fidelity Assessment

| Aspect | Backtest | Live Bot | Match? |
|--------|----------|----------|--------|
| EMA(9,21) | Exact | Exact | Yes |
| ATR(14) | SMA-based | SMA-based | Yes |
| Hurst R/S | 504-bar window, 24-bar step | Same | Yes |
| VoV | std/mean on 168 H1 ATR | Same | Yes |
| MTF weights | M5:0.2, M15:0.3, H1:0.25, H4:0.2 | Same | Yes |
| H1 shift | 55 min for M5, 30 min for M30 | Same | Yes |
| H4 shift | None | None | Yes |
| Regime thresholds | MTF>=0.8, Hurst>0.55, VoV<p90 | Same | Yes |
| Entry pricing | ask_open (long), bid_open (short) | ask/bid at market | Close |
| SL pricing | 5x ATR, checked intrabar | 5x ATR, OANDA-managed | Close |
| Hold period | 48 M5 bars exactly | ~49 bars (bug H3) | **No** |
| Signal transition | Every bar evaluated | Only last bar (bug M3) | **No** |
| Spread rejection | Skips bar, continues scanning | Consumes signal (bug C4) | **No** |
| Window profiling | Walk-forward folds | Single expanding window | Partial |
| Sharpe formula | trades_per_year | total_trades (bug H1) | **No** |
| Window attribution | Signal bar | Same (boundary offset) | Partial |

---

## Recommended Priority Fixes

### Immediate (before deploying)
1. **C1:** Add `MTF_WEIGHTS` to profiler.py imports
2. **C2:** Disable POST retries for market orders (set `max_retries=1`)
3. **C3:** Rotate API token, add `.gitignore`
4. **C4:** Reset `prev_active` when spread rejects entry

### Before live money
5. **H1:** Fix Sharpe annualization: `tpy = n / training_years`
6. **H2:** Seed `prev_active` from historical active_mask on startup
7. **H3:** Use M5 bar timestamp for `entry_time`, not `datetime.now()`
8. **H5:** Add empty price ladder check before indexing

### Soon after
9. **M1:** Call `trim_old_data()` during reprofile
10. **M2:** Add 429 handling to `_put()`
11. **M3:** Evaluate all new bars since last processed, not just the latest
12. **H4:** Add data freshness validation before profiling

---

## Performance Expectations (With Bugs Fixed)

Based on the backtest reference:
- **Trade frequency:** ~15,000 trades over 6 years across 19 pairs = ~2,500/year = ~7/day
- **Expected pip yield:** +3.405 pips/trade (before live-specific costs)
- **Win rate:** 53%
- **SL hit rate:** 0.9% (rare catastrophe stops)
- **At 1000 units (0.01 lots):** ~$0.10/pip on majors = ~$0.34/trade = ~$2.40/day = ~$850/year
- **Live performance will likely be lower** due to: wider live spreads, execution latency, VoV regime shifts, window edge decay, and the systematic differences noted above

---

*End of audit report.*
