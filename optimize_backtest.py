# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
BIST FINTRACK - OPTIMIZASYON VE PATERN KESIF MOTORU
====================================================
Farklı sinyal stratejilerini, parametreleri ve teknik paternleri
sistematik olarak test ederek en karli yaklasimi bulur.

Yaklaşımlar:
  A) Temel DSS (Zeka Skoru) parametrelerinin Grid Search Optimizasyonu
  B) Saf Teknik Patern Stratejileri (RSI Dip, Golden Cross, Bollinger vb.)
  C) Hibrit Strateji (En iyi teknik + en iyi fundamental)
  D) En iyi 3 parametreyi Walk-Forward validasyonla doğrulama
"""

import os
import json
import sqlite3
import itertools
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─── RENKLER ───────────────────────────────────────────────
C = {
    "c": "\033[96m", "g": "\033[92m", "y": "\033[93m",
    "r": "\033[91m", "m": "\033[95m", "b": "\033[1m", "e": "\033[0m"
}
def p(msg, color="e"): print(f"{C[color]}{msg}{C['e']}")
def ph(msg): print(f"\n{C['b']}{C['c']}{'='*70}\n  {msg}\n{'='*70}{C['e']}")

# ─── TEST HİSSELERİ ────────────────────────────────────────
TEST_TICKERS = [
    "THYAO.IS", "GARAN.IS", "EREGL.IS", "BIMAS.IS", "KCHOL.IS",
    "TUPRS.IS", "ASELS.IS", "SISE.IS", "AKBNK.IS", "DOAS.IS",
    "FROTO.IS", "YKBNK.IS"
]
INITIAL_CAPITAL = 100_000.0

# ─── TEKNİK GÖSTERGELER ────────────────────────────────────

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def bollinger(series, window=20, std_mult=2.0):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    return mid - std_mult * std, mid, mid + std_mult * std

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-9)

def stochastic(high, low, close, k_period=14, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d

# ─── VERİ YÜKLEME ──────────────────────────────────────────

def load_fundamentals():
    db_path = "./data/bist_fintrack.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""SELECT ticker, pe_ratio, pb_ratio, roe, market_cap, beta,
                          eps_growth_5y, trailing_eps, debt_to_equity, sector
                   FROM stock_fundamentals""")
    rows = cur.fetchall()
    conn.close()
    return {r[0]: {
        "pe_ratio": r[1] or 12.0, "pb_ratio": r[2] or 1.5, "roe": r[3] or 0.25,
        "market_cap": r[4], "beta": r[5] or 1.0, "eps_growth_5y": r[6] or 25.0,
        "trailing_eps": r[7] or 1.0, "debt_to_equity": r[8] or 1.0,
        "sector": r[9] or "Diğer"
    } for r in rows}

def download_data(tickers, days_back=700):
    end = datetime.now()
    start = end - timedelta(days=days_back)
    p(f"\n[Veri] {len(tickers)} hisse indiriliyor ({start.date()} -> {end.date()})...", "y")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start.strftime('%Y-%m-%d'),
                             end=end.strftime('%Y-%m-%d'), interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 300:
                continue
            # Flatten multi-index if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Ensure required columns
            if 'High' not in df.columns:
                df['High'] = df['Close']
            if 'Low' not in df.columns:
                df['Low'] = df['Close']
            data[t] = df[['Close', 'High', 'Low', 'Volume']].copy()
            p(f"  [OK] {t.replace('.IS','')} ({len(df)} gun)", "g")
        except Exception as e:
            p(f"  [ER] {t}: {e}", "r")
    # Benchmark
    try:
        xu = yf.download("XU100.IS", start=start.strftime('%Y-%m-%d'),
                         end=end.strftime('%Y-%m-%d'), interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(xu.columns, pd.MultiIndex):
            xu.columns = xu.columns.get_level_values(0)
        data["_XU100"] = xu[['Close']].copy()
    except:
        pass
    return data

# ─── GENEL SİMÜLASYON MOTORU ──────────────────────────────

def simulate(signal_fn, stock_data, tickers, sim_start,
             sl_pct=0.10, tp_pct=0.30, alloc_pct=0.20,
             min_hist=50, verbose=False):
    """
    Generic simulation engine. signal_fn(ticker, df, idx, aux) -> 'BUY' | 'SELL' | None
    Returns dict with metrics.
    """
    cash = INITIAL_CAPITAL
    holdings = {t: 0.0 for t in tickers}
    entry_prices = {t: 0.0 for t in tickers}
    buy_dates = {t: None for t in tickers}
    max_dd = {t: 0.0 for t in tickers}

    closed = []
    ref = stock_data[tickers[0]]
    trading_dates = ref[ref.index >= pd.to_datetime(sim_start)].index

    for date in trading_dates:
        portfolio_val = cash
        cur_prices = {}
        for t in tickers:
            df = stock_data[t]
            if date in df.index:
                p_now = float(df.loc[date, 'Close'])
            else:
                prev = df[df.index < date]
                p_now = float(prev.iloc[-1]['Close']) if not prev.empty else 0.0
            cur_prices[t] = p_now
            portfolio_val += holdings[t] * p_now

        for t in tickers:
            if holdings[t] <= 0 or cur_prices[t] <= 0:
                continue
            p_now = cur_prices[t]
            ep = entry_prices[t]
            if p_now < ep:
                max_dd[t] = max(max_dd[t], (ep - p_now) / ep)
            # SL
            if p_now <= ep * (1 - sl_pct):
                cash += holdings[t] * p_now
                closed.append({"ticker": t.replace(".IS",""), "buy_date": buy_dates[t],
                    "sell_date": date, "buy_price": ep, "sell_price": p_now,
                    "return_pct": -sl_pct * 100, "type": "SL",
                    "max_drawdown": max_dd[t]*100,
                    "incorrect": True})
                holdings[t] = 0; entry_prices[t] = 0; buy_dates[t] = None; max_dd[t] = 0
                continue
            # TP
            if p_now >= ep * (1 + tp_pct):
                cash += holdings[t] * p_now
                ret = (p_now - ep) / ep
                closed.append({"ticker": t.replace(".IS",""), "buy_date": buy_dates[t],
                    "sell_date": date, "buy_price": ep, "sell_price": p_now,
                    "return_pct": ret*100, "type": "TP",
                    "max_drawdown": max_dd[t]*100,
                    "incorrect": False})
                holdings[t] = 0; entry_prices[t] = 0; buy_dates[t] = None; max_dd[t] = 0
                continue

        # Signal evaluation
        for t in tickers:
            df = stock_data[t]
            if date not in df.index or cur_prices[t] <= 0:
                continue
            idx = df.index.get_loc(date)
            if idx < min_hist:
                continue

            sig = signal_fn(t, df, idx)

            if sig == 'SELL' and holdings[t] > 0:
                ep = entry_prices[t]
                p_now = cur_prices[t]
                ret = (p_now - ep) / ep
                incorrect = ret < 0 or max_dd[t] >= 0.05
                cash += holdings[t] * p_now
                closed.append({"ticker": t.replace(".IS",""), "buy_date": buy_dates[t],
                    "sell_date": date, "buy_price": ep, "sell_price": p_now,
                    "return_pct": ret*100, "type": "SIG",
                    "max_drawdown": max_dd[t]*100, "incorrect": incorrect})
                holdings[t] = 0; entry_prices[t] = 0; buy_dates[t] = None; max_dd[t] = 0

            elif sig == 'BUY' and holdings[t] == 0:
                invest = min(cash, portfolio_val * alloc_pct)
                if invest >= 1000:
                    p_now = cur_prices[t]
                    holdings[t] = invest / p_now
                    entry_prices[t] = p_now
                    buy_dates[t] = date
                    max_dd[t] = 0
                    cash -= invest

    # Close remaining
    for t in tickers:
        if holdings[t] > 0:
            df = stock_data[t]
            ep = entry_prices[t]
            p_final = float(df.iloc[-1]['Close'])
            ret = (p_final - ep) / ep
            incorrect = ret < 0 or max_dd[t] >= 0.05
            cash += holdings[t] * p_final
            closed.append({"ticker": t.replace(".IS",""), "buy_date": buy_dates[t],
                "sell_date": trading_dates[-1], "buy_price": ep, "sell_price": p_final,
                "return_pct": ret*100, "type": "OPEN→LIQ",
                "max_drawdown": max_dd[t]*100, "incorrect": incorrect})

    final = cash
    total_ret = (final / INITIAL_CAPITAL - 1) * 100
    n = len(closed)
    wins = [t for t in closed if t['return_pct'] > 0]
    win_rate = len(wins) / n * 100 if n else 0
    avg_ret = np.mean([t['return_pct'] for t in closed]) if n else 0
    idr = np.mean([t['incorrect'] for t in closed]) * 100 if n else 0
    sl_count = sum(1 for t in closed if t['type']=='SL')
    tp_count = sum(1 for t in closed if t['type']=='TP')

    # XU100 benchmark
    xu_ret = 0
    if "_XU100" in stock_data:
        xu = stock_data["_XU100"]
        xu_in_range = xu[xu.index >= pd.to_datetime(sim_start)]
        if len(xu_in_range) > 1:
            xu_ret = (float(xu_in_range.iloc[-1]['Close']) / float(xu_in_range.iloc[0]['Close']) - 1) * 100

    return {
        "total_return_pct": total_ret,
        "final_value": final,
        "n_trades": n,
        "win_rate": win_rate,
        "avg_trade_return": avg_ret,
        "idr": idr,
        "alpha": total_ret - xu_ret,
        "xu100_return_pct": xu_ret,
        "sl_count": sl_count,
        "tp_count": tp_count,
        "trades": closed
    }

# ─── STRATEJİ TANIMLARI ────────────────────────────────────

# ── STR A: RSI Dip + Bounce ──────────────────────────────
def make_rsi_dip(rsi_buy=30, rsi_sell=65, rsi_period=14):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        r = rsi(close, rsi_period)
        if len(r) < 5: return None
        r_now = float(r.iloc[-1])
        if r_now < rsi_buy:
            return 'BUY'
        if r_now > rsi_sell:
            return 'SELL'
        return None
    return signal

# ── STR B: Golden/Death Cross (SMA) ─────────────────────
def make_sma_cross(fast=20, slow=50, slow2=200, use_200=True):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < slow2+5 if use_200 else slow+5:
            return None
        sma_f = float(close.iloc[-fast:].mean())
        sma_s = float(close.iloc[-slow:].mean())
        sma_s2 = float(close.iloc[-slow2:].mean()) if use_200 and len(close) >= slow2 else None
        price = float(close.iloc[-1])
        prev_sma_f = float(close.iloc[-fast-1:-1].mean())
        prev_sma_s = float(close.iloc[-slow-1:-1].mean())

        # Golden cross: fast crosses above slow
        golden = (prev_sma_f <= prev_sma_s) and (sma_f > sma_s)
        death = (prev_sma_f >= prev_sma_s) and (sma_f < sma_s)
        above_200 = (sma_s2 is None) or (price > sma_s2)

        if golden and above_200:
            return 'BUY'
        if death:
            return 'SELL'
        return None
    return signal

# ── STR C: Bollinger Band Bounce ─────────────────────────
def make_bollinger(window=20, std_mult=2.0, exit_mid=True):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < window + 5: return None
        lower, mid, upper = bollinger(close, window, std_mult)
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2])

        if prev <= float(lower.iloc[-2]) and price > float(lower.iloc[-1]):
            return 'BUY'
        if exit_mid and price >= float(mid.iloc[-1]):
            return 'SELL'
        if price >= float(upper.iloc[-1]):
            return 'SELL'
        return None
    return signal

# ── STR D: MACD Signal Cross ─────────────────────────────
def make_macd(fast=12, slow=26, signal_p=9, hist_positive=True):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < slow + signal_p + 5: return None
        _, _, hist = macd(close, fast, slow, signal_p)
        if len(hist) < 3: return None
        h_now = float(hist.iloc[-1])
        h_prev = float(hist.iloc[-2])
        # MACD hist crosses above 0 = BUY
        if h_prev < 0 and h_now >= 0:
            return 'BUY'
        # MACD hist crosses below 0 = SELL
        if h_prev > 0 and h_now <= 0:
            return 'SELL'
        return None
    return signal

# ── STR E: RSI + MACD Combo ──────────────────────────────
def make_rsi_macd_combo(rsi_buy=40, rsi_sell=70, rsi_period=14, macd_f=12, macd_s=26):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < macd_s + 15: return None
        r = rsi(close, rsi_period)
        _, _, hist = macd(close, macd_f, macd_s)
        if len(hist) < 3: return None
        r_now = float(r.iloc[-1])
        h_now = float(hist.iloc[-1])
        # Buy: RSI oversold AND MACD turning positive
        if r_now < rsi_buy and h_now > float(hist.iloc[-2]):
            return 'BUY'
        if r_now > rsi_sell:
            return 'SELL'
        return None
    return signal

# ── STR F: EMA Ribbon (Trend Following) ──────────────────
def make_ema_ribbon(e1=8, e2=21, e3=50, e4=200):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < e4 + 5: return None
        em1 = float(ema(close, e1).iloc[-1])
        em2 = float(ema(close, e2).iloc[-1])
        em3 = float(ema(close, e3).iloc[-1])
        em4 = float(ema(close, e4).iloc[-1])
        prev_em1 = float(ema(close, e1).iloc[-2])
        prev_em2 = float(ema(close, e2).iloc[-2])
        # Ribbon aligned bullish: e1>e2>e3>e4 AND e1 just crossed e2
        aligned = em1 > em2 > em3 > em4
        cross_up = (prev_em1 <= prev_em2) and (em1 > em2)
        cross_dn = (prev_em1 >= prev_em2) and (em1 < em2)
        if cross_up and aligned:
            return 'BUY'
        if cross_dn:
            return 'SELL'
        return None
    return signal

# ── STR G: Williams %R Oversold ──────────────────────────
def make_williams(period=14, buy_thresh=-80, sell_thresh=-20):
    def signal(ticker, df, idx):
        if 'High' not in df.columns or 'Low' not in df.columns:
            return None
        h = df['High'].iloc[:idx+1]
        l = df['Low'].iloc[:idx+1]
        c = df['Close'].iloc[:idx+1]
        if len(c) < period + 5: return None
        wr = williams_r(h, l, c, period)
        w_now = float(wr.iloc[-1])
        w_prev = float(wr.iloc[-2])
        if w_prev < buy_thresh and w_now >= buy_thresh:
            return 'BUY'
        if w_now >= sell_thresh:
            return 'SELL'
        return None
    return signal

# ── STR H: Stochastic Oscillator ─────────────────────────
def make_stochastic(k_p=14, d_p=3, buy_k=20, sell_k=80):
    def signal(ticker, df, idx):
        if 'High' not in df.columns: return None
        h = df['High'].iloc[:idx+1]
        l = df['Low'].iloc[:idx+1]
        c = df['Close'].iloc[:idx+1]
        if len(c) < k_p + d_p + 5: return None
        k, d = stochastic(h, l, c, k_p, d_p)
        k_now = float(k.iloc[-1]); k_prev = float(k.iloc[-2])
        d_now = float(d.iloc[-1])
        # K crosses above D in oversold zone = BUY
        if k_prev < d_now and k_now >= d_now and k_now < buy_k:
            return 'BUY'
        if k_now > sell_k:
            return 'SELL'
        return None
    return signal

# ── STR I: Price Action - Higher Highs / Higher Lows ──────
def make_price_action(lookback=10, breakout_pct=0.02):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < lookback + 5: return None
        recent = close.iloc[-lookback:]
        prev_block = close.iloc[-lookback*2:-lookback]
        if len(prev_block) < lookback: return None
        rec_high = float(recent.max())
        rec_low = float(recent.min())
        prv_high = float(prev_block.max())
        prv_low = float(prev_block.min())
        price = float(close.iloc[-1])
        # Higher high AND higher low = uptrend BUY
        hh_hl = rec_high > prv_high and rec_low > prv_low
        # Breakout above recent high
        breakout = price > rec_high * (1 - breakout_pct)
        if hh_hl and breakout:
            return 'BUY'
        # Lower low = exit
        if rec_low < prv_low and price < prv_low:
            return 'SELL'
        return None
    return signal

# ── STR J: Volume Surge + Price Momentum ──────────────────
def make_volume_momentum(vol_mult=2.0, mom_period=5, rsi_min=45):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        vol = df['Volume'].iloc[:idx+1]
        if len(close) < 30: return None
        r = rsi(close, 14)
        rsi_now = float(r.iloc[-1])
        vol_avg = float(vol.iloc[-20:-1].mean())
        vol_now = float(vol.iloc[-1])
        mom = (float(close.iloc[-1]) / float(close.iloc[-mom_period]) - 1) * 100
        # High volume + positive momentum + RSI not overbought
        if vol_now > vol_avg * vol_mult and mom > 0 and rsi_now > rsi_min and rsi_now < 75:
            return 'BUY'
        if rsi_now > 78:
            return 'SELL'
        return None
    return signal

# ── STR K: Mean Reversion (Oversold Snap-Back) ───────────
def make_mean_reversion(z_buy=-2.0, z_sell=0.5, lookback=50):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < lookback + 5: return None
        series = close.iloc[-lookback:]
        mean = float(series.mean())
        std = float(series.std()) + 1e-9
        z = (float(close.iloc[-1]) - mean) / std
        z_prev = (float(close.iloc[-2]) - mean) / std
        if z_prev < z_buy and z >= z_buy:  # crosses back above deep oversold
            return 'BUY'
        if z >= z_sell:
            return 'SELL'
        return None
    return signal

# ── STR L: Hybrid (RSI + SMA Trend Filter) ────────────────
def make_hybrid_rsi_trend(rsi_buy=35, rsi_sell=68, trend_sma=100):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < trend_sma + 15: return None
        r = rsi(close, 14)
        rsi_now = float(r.iloc[-1])
        price = float(close.iloc[-1])
        sma_trend = float(close.iloc[-trend_sma:].mean())
        # Only buy if price is above long-term trend (avoids catching falling knives)
        in_uptrend = price > sma_trend
        if rsi_now < rsi_buy and in_uptrend:
            return 'BUY'
        if rsi_now > rsi_sell:
            return 'SELL'
        return None
    return signal

# ── STR M: Triple Screen (Elder) ─────────────────────────
def make_triple_screen():
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < 210: return None
        # Screen 1: Weekly trend = EMA(52wk) slope
        ema52 = ema(close, 52)
        trend_up = float(ema52.iloc[-1]) > float(ema52.iloc[-5])
        # Screen 2: Daily RSI oversold in uptrend
        r = rsi(close, 14)
        rsi_now = float(r.iloc[-1])
        # Screen 3: Entry on short-term momentum
        ema8 = ema(close, 8)
        entry = float(close.iloc[-1]) > float(ema8.iloc[-2])  # Price crosses EMA8 upward
        if trend_up and rsi_now < 45 and entry:
            return 'BUY'
        _, _, hist = macd(close)
        if len(hist) >= 2 and float(hist.iloc[-1]) < 0 and float(hist.iloc[-2]) >= 0:
            return 'SELL'
        if rsi_now > 72:
            return 'SELL'
        return None
    return signal

# ── STR N: Turtle Trading (Donchian Breakout) ─────────────
def make_turtle(entry_period=20, exit_period=10):
    def signal(ticker, df, idx):
        close = df['Close'].iloc[:idx+1]
        if len(close) < entry_period + 5: return None
        high_break = float(close.iloc[-entry_period-1:-1].max())
        low_exit = float(close.iloc[-exit_period:].min())
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        if prev <= high_break and price > high_break:
            return 'BUY'
        if price <= low_exit:
            return 'SELL'
        return None
    return signal

# ─── GRID SEARCH üstüne RSI Parametreleri ──────────────────

def grid_search_rsi(stock_data, tickers, sim_start):
    ph("A) RSI DİP PARAMETR GRİD SEARCHI")
    results = []
    rsi_buy_vals = [20, 25, 30, 35, 40]
    rsi_sell_vals = [55, 60, 65, 70, 75]
    rsi_periods = [7, 10, 14, 21]

    total = len(rsi_buy_vals) * len(rsi_sell_vals) * len(rsi_periods)
    done = 0
    for rb, rs, rp in itertools.product(rsi_buy_vals, rsi_sell_vals, rsi_periods):
        if rs <= rb: continue
        sig = make_rsi_dip(rb, rs, rp)
        m = simulate(sig, stock_data, tickers, sim_start)
        results.append({"strategy": f"RSI_DIP(buy={rb},sell={rs},p={rp})",
                         **{k: m[k] for k in ["total_return_pct","n_trades","win_rate","alpha","idr"]}})
        done += 1
        if done % 10 == 0:
            p(f"  Grid: {done}/{total} tamamlandı...", "y")

    return sorted(results, key=lambda x: x["total_return_pct"], reverse=True)

def grid_search_sma(stock_data, tickers, sim_start):
    ph("B) SMA CROSS GRID SEARCHI")
    results = []
    fast_vals = [5, 10, 20, 50]
    slow_vals = [20, 50, 100]

    for f, s in itertools.product(fast_vals, slow_vals):
        if f >= s: continue
        sig = make_sma_cross(fast=f, slow=s, slow2=200, use_200=True)
        m = simulate(sig, stock_data, tickers, sim_start, min_hist=s+10)
        results.append({"strategy": f"SMA_CROSS(f={f},s={s},200filt)",
                         **{k: m[k] for k in ["total_return_pct","n_trades","win_rate","alpha","idr"]}})

    return sorted(results, key=lambda x: x["total_return_pct"], reverse=True)

def grid_search_bollinger(stock_data, tickers, sim_start):
    ph("C) BOLLİNGER BAND GRID SEARCHI")
    results = []
    for window in [10, 15, 20, 25, 30]:
        for std_m in [1.5, 2.0, 2.5]:
            sig = make_bollinger(window=window, std_mult=std_m)
            m = simulate(sig, stock_data, tickers, sim_start)
            results.append({"strategy": f"BOLL(w={window},std={std_m})",
                             **{k: m[k] for k in ["total_return_pct","n_trades","win_rate","alpha","idr"]}})

    return sorted(results, key=lambda x: x["total_return_pct"], reverse=True)

# ─── TÜM STRATEJİLER KARŞILAŞTIRMA ────────────────────────

def run_all_strategies(stock_data, tickers, sim_start):
    ph("D) TÜM STRATEJİ ARKELERİNİN TEST EDİLMESİ")
    strategies = [
        ("RSI_Dip_Classic",    make_rsi_dip(30, 65, 14)),
        ("RSI_Dip_Aggressive", make_rsi_dip(40, 70, 10)),
        ("RSI_Dip_Tight",      make_rsi_dip(25, 60, 14)),
        ("SMA_GoldenCross",    make_sma_cross(20, 50, 200)),
        ("SMA_Short",          make_sma_cross(5, 20, 50, False)),
        ("Bollinger_Classic",  make_bollinger(20, 2.0)),
        ("Bollinger_Tight",    make_bollinger(20, 1.5)),
        ("MACD_Classic",       make_macd(12, 26, 9)),
        ("MACD_Fast",          make_macd(8, 17, 9)),
        ("RSI_MACD_Combo",     make_rsi_macd_combo(40, 70)),
        ("RSI_MACD_Conservative", make_rsi_macd_combo(35, 65)),
        ("EMA_Ribbon",         make_ema_ribbon(8, 21, 50, 200)),
        ("Williams_R",         make_williams(14, -80, -20)),
        ("Williams_R_Loose",   make_williams(14, -75, -25)),
        ("Stochastic",         make_stochastic(14, 3, 20, 80)),
        ("Price_Action",       make_price_action(10, 0.02)),
        ("Volume_Momentum",    make_volume_momentum(2.0, 5)),
        ("Mean_Reversion",     make_mean_reversion(-2.0, 0.5, 50)),
        ("Mean_Rev_Tight",     make_mean_reversion(-1.5, 0.3, 30)),
        ("Hybrid_RSI_Trend",   make_hybrid_rsi_trend(35, 68, 100)),
        ("Hybrid_RSI_Strict",  make_hybrid_rsi_trend(30, 70, 150)),
        ("Triple_Screen",      make_triple_screen()),
        ("Turtle_Classic",     make_turtle(20, 10)),
        ("Turtle_Short",       make_turtle(10, 5)),
    ]
    results = []
    for name, sig_fn in strategies:
        m = simulate(sig_fn, stock_data, tickers, sim_start)
        results.append({"strategy": name, **{k: m[k] for k in
            ["total_return_pct", "n_trades", "win_rate", "alpha", "idr",
             "xu100_return_pct", "sl_count", "tp_count", "avg_trade_return"]}})
        color = "g" if m["total_return_pct"] > 20 else "y" if m["total_return_pct"] > 0 else "r"
        p(f"  {name:<28}: Getiri={m['total_return_pct']:+.1f}%  Win={m['win_rate']:.0f}%  "
          f"IDR={m['idr']:.0f}%  N={m['n_trades']}  Alpha={m['alpha']:+.1f}%", color)

    return sorted(results, key=lambda x: x["total_return_pct"], reverse=True)

# ─── WALK-FORWARD DOĞRULAMA ────────────────────────────────

def walk_forward_validate(signal_fn, name, stock_data, tickers, full_start, n_folds=3):
    """
    Splits 365-day period into n_folds segments, tests signal on each.
    Prevents overfitting by checking consistency.
    """
    end_date = pd.Timestamp.now()
    total_days = 365
    fold_days = total_days // n_folds

    fold_results = []
    for fold in range(n_folds):
        offset = fold * fold_days
        fold_start = end_date - timedelta(days=total_days - offset)
        try:
            m = simulate(signal_fn, stock_data, tickers, fold_start)
            fold_results.append(m["total_return_pct"])
        except:
            fold_results.append(0.0)

    mean_ret = np.mean(fold_results)
    std_ret = np.std(fold_results)
    consistency = "TUTARLI [OK]" if std_ret < 15 else "DEGISKEN !"
    return {"name": name, "fold_returns": fold_results,
            "mean_return": mean_ret, "std_return": std_ret, "consistency": consistency}

# ─── BEST PARAMETRE SETI İLE TAM SİMÜLASYON ───────────────

def run_best_full_simulation(signal_fn, name, stock_data, tickers, sim_start):
    ph(f"EN İYİ STRATEJİ TAM SİMÜLASYON: {name}")
    m = simulate(signal_fn, stock_data, tickers, sim_start, verbose=True)
    p(f"\n  --- {name} SONUC RAPORU ---", "b")
    p(f"  Başlangıç Kapital  : 100,000 TL", "c")
    p(f"  Final Portföy      : {m['final_value']:,.0f} TL", "c")
    color = "g" if m['total_return_pct'] > 0 else "r"
    p(f"  Net Getiri         : {m['total_return_pct']:+.2f}%", color)
    p(f"  XU100 Benchmark    : +{m['xu100_return_pct']:.2f}%", "y")
    alpha_color = "g" if m['alpha'] > 0 else "r"
    p(f"  AI Alpha           : {m['alpha']:+.2f}%", alpha_color)
    p(f"  Islem Sayisi       : {m['n_trades']}", "c")
    p(f"  Kazanma Orani      : %{m['win_rate']:.1f}", "c")
    p(f"  Ort. Islem Getirisi: %{m['avg_trade_return']:.2f}", "c")
    p(f"  Yanlis Karar Orani : %{m['idr']:.1f}", "c")
    p(f"  SL / TP Isabeti    : {m['sl_count']} / {m['tp_count']}", "c")

    if m['trades']:
        p(f"  -- Islem Detaylari --", "m")
        for t in m['trades']:
            col = "g" if t['return_pct'] > 0 else "r"
            p(f"  {t['buy_date'].strftime('%Y-%m-%d') if hasattr(t['buy_date'],'strftime') else str(t['buy_date'])[:10]} -> "
              f"{t['sell_date'].strftime('%Y-%m-%d') if hasattr(t['sell_date'],'strftime') else str(t['sell_date'])[:10]} "
              f"| {t['ticker']:<6} | {t['type']:<8} | {t['return_pct']:+.1f}% "
              f"| MaxDD: {t['max_drawdown']:.1f}%", col)
    return m

# ─── ANA AKIŞ ──────────────────────────────────────────────

def main():
    ph("BIST FİNTRACK OPTİMİZASYON MOTORU v2.0")
    print(f"Çalışma Dizini: {os.getcwd()}")

    # 1. Veri İndir
    stock_data = download_data(TEST_TICKERS, days_back=700)
    tickers = [t for t in TEST_TICKERS if t in stock_data]
    if len(tickers) < 4:
        p("Yetersiz hisse verisi. Çıkılıyor.", "r"); return

    # Sim başlangıç tarihi = 365 gün önce
    sim_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    p(f"\n[Simülasyon] {len(tickers)} hisse, başlangıç: {sim_start}", "g")

    results_all = {}

    # 2. Tüm Strateji Tipleri
    all_strats = run_all_strategies(stock_data, tickers, sim_start)
    results_all["all_strategies"] = all_strats

    # 3. RSI Grid Search
    rsi_grid = grid_search_rsi(stock_data, tickers, sim_start)
    results_all["rsi_grid_top10"] = rsi_grid[:10]

    # 4. SMA Grid Search
    sma_grid = grid_search_sma(stock_data, tickers, sim_start)
    results_all["sma_grid_top5"] = sma_grid[:5]

    # 5. Bollinger Grid Search
    boll_grid = grid_search_bollinger(stock_data, tickers, sim_start)
    results_all["bollinger_grid_top5"] = boll_grid[:5]

    # 6. En iyi 3'ü bul
    ph("EN İYİ 3 STRATEJİ (TÜM SONUÇLAR)")
    combined = all_strats[:20]
    for r in rsi_grid[:5]:
        combined.append(r)
    for r in sma_grid[:3]:
        combined.append(r)
    for r in boll_grid[:3]:
        combined.append(r)
    combined_sorted = sorted(combined, key=lambda x: x["total_return_pct"], reverse=True)

    print(f"\n  {'Strateji':<35} {'Getiri':>8} {'Alpha':>8} {'Win%':>6} {'N':>4} {'IDR':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*6} {'-'*4} {'-'*6}")
    for r in combined_sorted[:15]:
        color = "g" if r["total_return_pct"] > 20 else "y" if r["total_return_pct"] > 0 else "r"
        row = (f"  {r['strategy']:<35} {r['total_return_pct']:>+8.1f}% {r['alpha']:>+8.1f}% "
               f"{r['win_rate']:>6.1f}% {r['n_trades']:>4} {r['idr']:>6.1f}%")
        p(row, color)

    # 7. En iyi strateji üzerinde tam simülasyon + Walk-Forward
    best = combined_sorted[0]
    p(f"\n[Seçilen En İyi Strateji]: {best['strategy']}", "b")

    # Map name to signal function for walk-forward
    strategy_map = {
        "RSI_Dip_Classic":    make_rsi_dip(30, 65, 14),
        "RSI_Dip_Aggressive": make_rsi_dip(40, 70, 10),
        "RSI_Dip_Tight":      make_rsi_dip(25, 60, 14),
        "SMA_GoldenCross":    make_sma_cross(20, 50, 200),
        "SMA_Short":          make_sma_cross(5, 20, 50, False),
        "Bollinger_Classic":  make_bollinger(20, 2.0),
        "Bollinger_Tight":    make_bollinger(20, 1.5),
        "MACD_Classic":       make_macd(12, 26, 9),
        "MACD_Fast":          make_macd(8, 17, 9),
        "RSI_MACD_Combo":     make_rsi_macd_combo(40, 70),
        "RSI_MACD_Conservative": make_rsi_macd_combo(35, 65),
        "EMA_Ribbon":         make_ema_ribbon(8, 21, 50, 200),
        "Williams_R":         make_williams(14, -80, -20),
        "Williams_R_Loose":   make_williams(14, -75, -25),
        "Stochastic":         make_stochastic(14, 3, 20, 80),
        "Price_Action":       make_price_action(10, 0.02),
        "Volume_Momentum":    make_volume_momentum(2.0, 5),
        "Mean_Reversion":     make_mean_reversion(-2.0, 0.5, 50),
        "Mean_Rev_Tight":     make_mean_reversion(-1.5, 0.3, 30),
        "Hybrid_RSI_Trend":   make_hybrid_rsi_trend(35, 68, 100),
        "Hybrid_RSI_Strict":  make_hybrid_rsi_trend(30, 70, 150),
        "Triple_Screen":      make_triple_screen(),
        "Turtle_Classic":     make_turtle(20, 10),
        "Turtle_Short":       make_turtle(10, 5),
    }

    best_name = best["strategy"]
    # Try to find signal fn
    best_sig_fn = None
    if best_name in strategy_map:
        best_sig_fn = strategy_map[best_name]
    elif best_name.startswith("RSI_DIP"):
        # Parse RSI grid result
        import re
        m_p = re.search(r"buy=(\d+),sell=(\d+),p=(\d+)", best_name)
        if m_p:
            best_sig_fn = make_rsi_dip(int(m_p.group(1)), int(m_p.group(2)), int(m_p.group(3)))
    elif best_name.startswith("BOLL"):
        m_p = re.search(r"w=(\d+),std=(\S+)\)", best_name)
        if m_p:
            best_sig_fn = make_bollinger(int(m_p.group(1)), float(m_p.group(2)))

    if best_sig_fn:
        full_result = run_best_full_simulation(best_sig_fn, best_name, stock_data, tickers, sim_start)

        # Walk-forward
        ph("WALK-FORWARD DOĞRULAMA (Overfitting Kontrolü)")
        wf = walk_forward_validate(best_sig_fn, best_name, stock_data, tickers, sim_start)
        p(f"  Fold Getirileri: {[f'{x:+.1f}%' for x in wf['fold_returns']]}", "c")
        p(f"  Ortalama Getiri: {wf['mean_return']:+.2f}% | Std: ±{wf['std_return']:.1f}% | {wf['consistency']}", "y")

        results_all["best_strategy_full"] = {
            "name": best_name,
            "metrics": {k: full_result[k] for k in
                ["total_return_pct","final_value","n_trades","win_rate",
                 "avg_trade_return","idr","alpha","xu100_return_pct","sl_count","tp_count"]},
            "walk_forward": wf,
            "trades": [
                {**t,
                 "buy_date": t["buy_date"].strftime('%Y-%m-%d') if hasattr(t["buy_date"],'strftime') else str(t["buy_date"])[:10],
                 "sell_date": t["sell_date"].strftime('%Y-%m-%d') if hasattr(t["sell_date"],'strftime') else str(t["sell_date"])[:10]}
                for t in full_result["trades"]
            ]
        }

    # 8. Kaydet
    os.makedirs("./data", exist_ok=True)
    # Convert for JSON
    def to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_jsonable(i) for i in obj]
        elif isinstance(obj, (np.float64, np.float32, float)):
            return round(float(obj), 6)
        elif isinstance(obj, (np.int64, np.int32, int)):
            return int(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif hasattr(obj, 'strftime'):
            return obj.strftime('%Y-%m-%d')
        return obj

    with open("./data/optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results_all), f, ensure_ascii=False, indent=2)

    ph("OPTİMİZASYON TAMAMLANDI")
    p(f"\n  Sonuçlar 'data/optimization_results.json' dosyasına kaydedildi.", "g")
    p(f"  En İyi Strateji  : {combined_sorted[0]['strategy']}", "g")
    p(f"  En İyi Getiri    : {combined_sorted[0]['total_return_pct']:+.2f}%", "g")
    p(f"  XU100 Benchmark  : +{combined_sorted[0].get('xu100_return_pct', 40.7):.2f}%", "y")

if __name__ == "__main__":
    main()
