"""
OANDA REST API v20 CLIENT
===========================
Handles all communication with OANDA: candles, orders, positions.
"""

import time
import logging
import requests
from datetime import datetime, timezone

from config import (
    OANDA_BASE_URL, OANDA_ACCOUNT_ID, OANDA_API_TOKEN,
    OANDA_GRANULARITY, pip_multiplier,
)

log = logging.getLogger('bot.oanda')


class OandaClient:
    """Thin wrapper around OANDA v20 REST API."""

    def __init__(self):
        self.base_url = OANDA_BASE_URL
        self.account_id = OANDA_ACCOUNT_ID
        self.headers = {
            'Authorization': f'Bearer {OANDA_API_TOKEN}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _get(self, path, params=None, max_retries=3):
        """GET with retry logic."""
        url = f'{self.base_url}{path}'
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    # Rate limited — back off
                    wait = 2 ** attempt
                    log.warning(f'Rate limited, waiting {wait}s (attempt {attempt+1})')
                    time.sleep(wait)
                else:
                    log.error(f'GET {path} → {resp.status_code}: {resp.text}')
                    if attempt < max_retries - 1:
                        time.sleep(1)
            except requests.exceptions.RequestException as e:
                log.error(f'GET {path} failed: {e}')
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _post(self, path, data, max_retries=3):
        """POST with retry logic."""
        url = f'{self.base_url}{path}'
        for attempt in range(max_retries):
            try:
                resp = self.session.post(url, json=data, timeout=30)
                if resp.status_code in (200, 201):
                    return resp.json()
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning(f'Rate limited, waiting {wait}s')
                    time.sleep(wait)
                else:
                    log.error(f'POST {path} → {resp.status_code}: {resp.text}')
                    if attempt < max_retries - 1:
                        time.sleep(1)
            except requests.exceptions.RequestException as e:
                log.error(f'POST {path} failed: {e}')
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _put(self, path, data, max_retries=3):
        """PUT with retry logic."""
        url = f'{self.base_url}{path}'
        for attempt in range(max_retries):
            try:
                resp = self.session.put(url, json=data, timeout=30)
                if resp.status_code in (200, 201):
                    return resp.json()
                else:
                    log.error(f'PUT {path} → {resp.status_code}: {resp.text}')
                    if attempt < max_retries - 1:
                        time.sleep(1)
            except requests.exceptions.RequestException as e:
                log.error(f'PUT {path} failed: {e}')
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    # ── ACCOUNT ──

    def get_account_summary(self):
        """Fetch account summary (balance, unrealised P&L, etc)."""
        data = self._get(f'/v3/accounts/{self.account_id}/summary')
        if data:
            return data.get('account', {})
        return None

    # ── CANDLES ──

    def fetch_candles(self, instrument, granularity, count=None,
                      from_time=None, to_time=None, price='MBA'):
        """
        Fetch candle data from OANDA.

        Args:
            instrument: e.g. 'EUR_USD'
            granularity: OANDA granularity string e.g. 'M5', 'H1'
            count: number of candles (max 5000)
            from_time: RFC3339 datetime string
            to_time: RFC3339 datetime string
            price: 'M' (mid), 'B' (bid), 'A' (ask), or combo like 'MBA'

        Returns:
            list of candle dicts with 'time', 'mid'/'bid'/'ask' OHLC, 'volume'
        """
        params = {
            'granularity': granularity,
            'price': price,
        }
        if count:
            params['count'] = min(count, 5000)
        if from_time:
            params['from'] = from_time
        if to_time:
            params['to'] = to_time

        data = self._get(f'/v3/instruments/{instrument}/candles', params)
        if data:
            return data.get('candles', [])
        return []

    def fetch_candles_bulk(self, instrument, granularity, total_count, price='MBA'):
        """
        Fetch more than 5000 candles by paginating.
        Returns list of candle dicts, oldest first.
        """
        all_candles = []
        remaining = total_count
        to_time = None

        while remaining > 0:
            batch_size = min(remaining, 5000)
            params = {
                'granularity': granularity,
                'price': price,
                'count': batch_size,
            }
            if to_time:
                params['to'] = to_time

            data = self._get(f'/v3/instruments/{instrument}/candles', params)
            if not data:
                log.error(f'Failed to fetch candles for {instrument} {granularity}')
                break

            candles = data.get('candles', [])
            if not candles:
                break

            # Candles come oldest-first. If we're paginating backwards,
            # set to_time to the oldest candle's time for next batch
            all_candles = candles + all_candles
            remaining -= len(candles)

            if len(candles) < batch_size:
                break  # No more data available

            to_time = candles[0]['time']
            time.sleep(0.1)  # Be nice to the API

        return all_candles

    def get_current_price(self, instrument):
        """Fetch current bid/ask price."""
        data = self._get(f'/v3/accounts/{self.account_id}/pricing',
                         params={'instruments': instrument})
        if data and 'prices' in data and data['prices']:
            price = data['prices'][0]
            return {
                'bid': float(price['bids'][0]['price']),
                'ask': float(price['asks'][0]['price']),
                'time': price['time'],
                'tradeable': price.get('tradeable', True),
            }
        return None

    # ── ORDERS ──

    def place_market_order(self, instrument, units, sl_price=None, comment=''):
        """
        Place a market order with optional stop loss.

        Args:
            instrument: e.g. 'EUR_USD'
            units: positive = long, negative = short
            sl_price: stop loss price (optional)
            comment: trade comment for logging

        Returns:
            dict with order fill details, or None on failure
        """
        order_data = {
            'order': {
                'type': 'MARKET',
                'instrument': instrument,
                'units': str(int(units)),
                'timeInForce': 'FOK',
                'positionFill': 'DEFAULT',
            }
        }

        if sl_price is not None:
            pip_mult = pip_multiplier(instrument)
            # Round SL to appropriate precision
            if pip_mult == 100.0:  # JPY pair
                sl_price_str = f'{sl_price:.3f}'
            else:
                sl_price_str = f'{sl_price:.5f}'

            order_data['order']['stopLossOnFill'] = {
                'price': sl_price_str,
                'timeInForce': 'GTC',
            }

        if comment:
            order_data['order']['clientExtensions'] = {
                'comment': comment[:128],
            }

        result = self._post(f'/v3/accounts/{self.account_id}/orders', order_data)
        if result:
            fill = result.get('orderFillTransaction')
            if fill:
                log.info(f'ORDER FILLED: {instrument} {units} units @ {fill.get("price")} '
                         f'| SL={sl_price} | {comment}')
                return fill
            else:
                reject = result.get('orderRejectTransaction')
                if reject:
                    log.error(f'ORDER REJECTED: {instrument} — {reject.get("rejectReason")}')
                else:
                    log.error(f'ORDER UNEXPECTED: {instrument} — {result}')
        return None

    # ── TRADES ──

    def get_open_trades(self, instrument=None):
        """Get all open trades, optionally filtered by instrument."""
        data = self._get(f'/v3/accounts/{self.account_id}/openTrades')
        if not data:
            return []
        trades = data.get('trades', [])
        if instrument:
            trades = [t for t in trades if t['instrument'] == instrument]
        return trades

    def close_trade(self, trade_id, units=None):
        """
        Close a specific trade.

        Args:
            trade_id: OANDA trade ID
            units: partial close units (None = close all)

        Returns:
            close transaction or None
        """
        body = {}
        if units:
            body['units'] = str(int(units))

        result = self._put(
            f'/v3/accounts/{self.account_id}/trades/{trade_id}/close',
            body
        )
        if result:
            close_tx = result.get('orderFillTransaction')
            if close_tx:
                log.info(f'TRADE CLOSED: {trade_id} | P&L={close_tx.get("pl")}')
                return close_tx
            else:
                log.error(f'CLOSE UNEXPECTED for {trade_id}: {result}')
        return None

    def modify_trade_sl(self, trade_id, sl_price, instrument=''):
        """Modify the stop loss on an existing trade."""
        pip_mult = pip_multiplier(instrument) if instrument else 10000.0
        if pip_mult == 100.0:
            sl_str = f'{sl_price:.3f}'
        else:
            sl_str = f'{sl_price:.5f}'

        body = {
            'stopLoss': {
                'price': sl_str,
                'timeInForce': 'GTC',
            }
        }
        return self._put(
            f'/v3/accounts/{self.account_id}/trades/{trade_id}', body)

    # ── CONNECTIVITY CHECK ──

    def ping(self):
        """Check API connectivity."""
        try:
            data = self._get(f'/v3/accounts/{self.account_id}/summary')
            return data is not None
        except Exception:
            return False
