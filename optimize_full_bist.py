# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
BIST FINTRACK - FULL BIST KAPSAMLI OPTIMIZASYON MOTORU v3.0
============================================================
Tüm BIST hisselerine kapsamlı strateji optimizasyonu.
1M TL bütçe, geniş parametre grid search, disk cache.
"""

import os, json, sqlite3, time, pickle, warnings, itertools, re
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── SABITLER ─────────────────────────────────────────────
INITIAL_CAPITAL  = 1_000_000.0
SIM_DAYS         = 365          # son 365 gün simülasyon
DOWNLOAD_DAYS    = 720          # + teknik gösterge tamponu
MIN_TRADING_DAYS = 250          # bu kadar günden az veri varsa dahil etme
BATCH_SIZE       = 20           # yfinance batch download boyutu
CACHE_DIR        = Path("./data/price_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ─── RENKLER ──────────────────────────────────────────────
def col(txt, c):
    codes = {"g":"\033[92m","r":"\033[91m","y":"\033[93m",
             "c":"\033[96m","m":"\033[95m","b":"\033[1m","e":"\033[0m"}
    return f"{codes.get(c,'')}{txt}{codes['e']}"

def p(msg, c="e"):   print(col(msg, c))
def ph(msg):         print(col(f"\n{'='*68}\n  {msg}\n{'='*68}", "c"))
def pp(msg):         print(col(f"  {msg}", "b"))

# ─── VERİ YÜKLEME ─────────────────────────────────────────

def load_tickers_from_db():
    """DB'den tüm kayıtlı hisseleri çek."""
    conn = sqlite3.connect("./data/bist_fintrack.db")
    cur  = conn.cursor()
    cur.execute("SELECT ticker FROM stock_fundamentals")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]   # e.g. ['AKBNK.IS', ...]


def download_all_prices(tickers, days_back=DOWNLOAD_DAYS):
    """
    Tüm hisseleri batch download + disk cache ile indir.
    Zaten indirildiyse (< 12 saat önce) cache'den oku.
    """
    end   = datetime.now()
    start = end - timedelta(days=days_back)
    s_str = start.strftime('%Y-%m-%d')
    e_str = end.strftime('%Y-%m-%d')

    cache_file = CACHE_DIR / "all_prices.pkl"
    if cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < 12:
            p(f"[Cache] {age_h:.1f} saat once indirilmis veri okunuyor...", "y")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            p(f"[Cache] {age_h:.1f} saat eski, yeniden indiriliyor...", "y")

    p(f"\n[Veri] {len(tickers)} hisse {s_str} -> {e_str} indiriliyor...", "y")
    all_data = {}
    failed   = []

    # Batch download (çok daha hızlı)
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for bi, batch in enumerate(batches):
        p(f"  Batch {bi+1}/{len(batches)} ({len(batch)} hisse)...", "y")
        try:
            raw = yf.download(
                batch, start=s_str, end=e_str,
                interval="1d", progress=False, auto_adjust=True
            )
            if raw.empty:
                failed.extend(batch)
                continue

            # Multi-ticker: columns are (Price, Ticker)
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                    else:
                        # Multi-index: (level0=price, level1=ticker)
                        if isinstance(raw.columns, pd.MultiIndex):
                            df = raw.xs(ticker, axis=1, level=1)
                        else:
                            failed.append(ticker); continue

                    needed = ['Close']
                    for col_name in ['High', 'Low', 'Volume']:
                        if col_name in df.columns:
                            needed.append(col_name)
                    df = df[needed].dropna(subset=['Close'])

                    if len(df) < MIN_TRADING_DAYS:
                        continue

                    # Ensure High/Low
                    if 'High'   not in df.columns: df['High']   = df['Close']
                    if 'Low'    not in df.columns: df['Low']    = df['Close']
                    if 'Volume' not in df.columns: df['Volume'] = 0

                    all_data[ticker] = df
                except Exception:
                    failed.append(ticker)
        except Exception as e:
            p(f"    Batch hatasi: {e}", "r")
            failed.extend(batch)

        time.sleep(0.5)  # rate limit

    # XU100 benchmark
    try:
        xu = yf.download("XU100.IS", start=s_str, end=e_str,
                         interval="1d", progress=False, auto_adjust=True)
        if isinstance(xu.columns, pd.MultiIndex):
            xu.columns = xu.columns.get_level_values(0)
        all_data["_XU100"] = xu[['Close']].dropna()
    except Exception as e:
        p(f"[Uyari] XU100 indirilemedi: {e}", "r")

    p(f"\n[Veri] {len(all_data)-1} hisse basariyla indirildi. "
      f"{len(failed)} hata.", "g")

    with open(cache_file, "wb") as f:
        pickle.dump(all_data, f)

    return all_data


# ─── TEKNİK GÖSTERGELER (Vektörize, Hızlı) ────────────────

def rsi_v(close, period=14):
    delta = close.diff()
    g = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    l = (-delta).clip(lower=0).ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def ema_v(close, span):
    return close.ewm(span=span, adjust=False).mean()

def macd_v(close, fast=12, slow=26, sig=9):
    ml = ema_v(close, fast) - ema_v(close, slow)
    sl = ema_v(ml, sig)
    return ml - sl   # histogram

def boll_lower(close, w=20, std=2.0):
    mid = close.rolling(w).mean()
    return mid - std * close.rolling(w).std()

def sma_v(close, w):
    return close.rolling(w).mean()


def precompute_indicators(df):
    """
    Tüm teknik göstergeleri tek seferde vektörize hesapla.
    Returns dict of Series, indexed same as df.
    """
    c = df['Close']
    ind = {}
    # RSI ailesi
    for p in [7, 10, 14, 21]:
        ind[f'rsi{p}'] = rsi_v(c, p)
    # EMA ailesi
    for s in [8, 20, 50, 100, 200]:
        ind[f'ema{s}'] = ema_v(c, s)
    # SMA
    for s in [20, 50, 100, 200]:
        ind[f'sma{s}'] = sma_v(c, s)
    # MACD histogram varyantları
    ind['macd_12_26'] = macd_v(c, 12, 26, 9)
    ind['macd_8_17']  = macd_v(c, 8, 17, 9)
    ind['macd_5_13']  = macd_v(c, 5, 13, 5)
    # Bollinger lower
    for w, std in [(15, 2.0), (20, 2.0), (20, 1.5), (25, 2.0)]:
        ind[f'boll_lower_{w}_{std}'] = boll_lower(c, w, std)
        ind[f'boll_mid_{w}'] = sma_v(c, w)

    return pd.DataFrame(ind, index=df.index)


# ─── GENEL SİMÜLASYON MOTORU (Hızlandırılmış) ─────────────

def run_sim(signal_matrix, prices, tickers, sim_start_date,
            sl_pct=0.10, tp_pct=0.30, alloc_pct=0.20,
            xu100_prices=None):
    """
    signal_matrix: DataFrame(index=dates, columns=tickers)
                   values: +1=BUY, -1=SELL, 0=HOLD
    prices:        DataFrame(index=dates, columns=tickers)
    Returns metrics dict.
    """
    sim_dates = prices.index[prices.index >= pd.to_datetime(sim_start_date)]
    if len(sim_dates) == 0:
        return None

    cash     = INITIAL_CAPITAL
    holdings = {t: 0.0 for t in tickers}
    entry_px = {t: 0.0 for t in tickers}
    buy_dt   = {t: None for t in tickers}
    max_dd   = {t: 0.0 for t in tickers}
    closed   = []

    for date in sim_dates:
        # Portfolio value snapshot
        port_val = cash
        cur_px   = {}
        for t in tickers:
            try:
                p_ = float(prices.loc[date, t])
                if np.isnan(p_): p_ = 0.0
            except:
                try:
                    prev = prices.loc[:date, t].dropna()
                    p_ = float(prev.iloc[-1]) if len(prev) else 0.0
                except:
                    p_ = 0.0
            cur_px[t] = p_
            port_val  += holdings[t] * p_

        # SL / TP check
        for t in tickers:
            if holdings[t] <= 0 or cur_px[t] <= 0:
                continue
            ep  = entry_px[t]
            pnow = cur_px[t]
            if pnow < ep:
                max_dd[t] = max(max_dd[t], (ep - pnow) / ep)
            if pnow <= ep * (1 - sl_pct):
                cash += holdings[t] * pnow
                closed.append(dict(ticker=t, buy_date=buy_dt[t], sell_date=date,
                    buy_px=ep, sell_px=pnow, ret=-sl_pct*100, typ='SL',
                    mdd=max_dd[t]*100, inc=True))
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0
                continue
            if pnow >= ep * (1 + tp_pct):
                ret = (pnow - ep) / ep
                cash += holdings[t] * pnow
                closed.append(dict(ticker=t, buy_date=buy_dt[t], sell_date=date,
                    buy_px=ep, sell_px=pnow, ret=ret*100, typ='TP',
                    mdd=max_dd[t]*100, inc=False))
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0
                continue

        # Signal read
        if date in signal_matrix.index:
            row = signal_matrix.loc[date]
        else:
            continue

        for t in tickers:
            if t not in row.index: continue
            sig  = row[t]
            pnow = cur_px[t]
            if pnow <= 0: continue

            if sig == -1 and holdings[t] > 0:
                ep  = entry_px[t]
                ret = (pnow - ep) / ep
                inc = ret < 0 or max_dd[t] >= 0.05
                cash += holdings[t] * pnow
                closed.append(dict(ticker=t, buy_date=buy_dt[t], sell_date=date,
                    buy_px=ep, sell_px=pnow, ret=ret*100, typ='SIG',
                    mdd=max_dd[t]*100, inc=inc))
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0

            elif sig == 1 and holdings[t] == 0:
                invest = min(cash, port_val * alloc_pct)
                if invest >= 2000:
                    holdings[t]  = invest / pnow
                    entry_px[t]  = pnow
                    buy_dt[t]    = date
                    max_dd[t]    = 0
                    cash        -= invest

    # Kalan pozisyonları kapat
    for t in tickers:
        if holdings[t] > 0:
            last_px = cur_px.get(t, entry_px[t])
            ep      = entry_px[t]
            ret     = (last_px - ep) / ep
            inc     = ret < 0 or max_dd[t] >= 0.05
            cash   += holdings[t] * last_px
            closed.append(dict(ticker=t, buy_date=buy_dt[t], sell_date=sim_dates[-1],
                buy_px=ep, sell_px=last_px, ret=ret*100, typ='LIQ',
                mdd=max_dd[t]*100, inc=inc))

    # Metrikler
    final = cash
    total_ret = (final / INITIAL_CAPITAL - 1) * 100
    n     = len(closed)
    wr    = sum(1 for x in closed if x['ret'] > 0) / n * 100 if n else 0
    idr   = sum(1 for x in closed if x['inc'])    / n * 100 if n else 0
    avg   = np.mean([x['ret'] for x in closed]) if n else 0
    sl_c  = sum(1 for x in closed if x['typ']=='SL')
    tp_c  = sum(1 for x in closed if x['typ']=='TP')

    xu_ret = 0.0
    if xu100_prices is not None and len(xu100_prices) > 0:
        xu_slice = xu100_prices[xu100_prices.index >= pd.to_datetime(sim_start_date)]
        if len(xu_slice) > 1:
            xu_ret = (float(xu_slice.iloc[-1]) / float(xu_slice.iloc[0]) - 1) * 100

    return dict(total_ret=total_ret, final=final, n=n, wr=wr, idr=idr,
                avg=avg, alpha=total_ret-xu_ret, xu100=xu_ret,
                sl=sl_c, tp=tp_c, trades=closed)


# ─── SİNYAL MATRİSİ ÜRETİCİLERİ ──────────────────────────

def sig_rsi_dip(indicators_map, prices, tickers, buy_thr, sell_thr, rsi_key,
                trend_sma=None, macd_key=None):
    """RSI Dip Bounce sinyali (opsiyonel trend filtresi + MACD onayı)."""
    signals = {}
    for t in tickers:
        if t not in indicators_map or t not in prices.columns:
            continue
        ind = indicators_map[t]
        c   = prices[t].dropna()
        if rsi_key not in ind.columns: continue

        # Ortak index üzerinde çalış
        common = c.index.intersection(ind.index)
        r_a = ind[rsi_key].reindex(common)
        c_a = c.reindex(common)

        buy_sig  = r_a < buy_thr
        sell_sig = r_a > sell_thr

        if trend_sma and f'sma{trend_sma}' in ind.columns:
            sma_a = ind[f'sma{trend_sma}'].reindex(common)
            buy_sig = buy_sig & (c_a > sma_a)

        if macd_key and macd_key in ind.columns:
            m_a = ind[macd_key].reindex(common)
            buy_sig = buy_sig & (m_a > m_a.shift(1))

        sig = pd.Series(0, index=common)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)


def sig_macd_cross(indicators_map, prices, tickers, macd_key, trend_sma=None):
    """MACD histogram sıfır geçişi."""
    signals = {}
    for t in tickers:
        if t not in indicators_map: continue
        ind = indicators_map[t]
        c   = prices[t].dropna()
        if macd_key not in ind.columns: continue

        common = c.index.intersection(ind.index)
        h   = ind[macd_key].reindex(common)
        hp  = h.shift(1)
        c_a = c.reindex(common)

        buy_sig  = (hp < 0) & (h >= 0)
        sell_sig = (hp > 0) & (h <= 0)

        if trend_sma and f'sma{trend_sma}' in ind.columns:
            sma_a = ind[f'sma{trend_sma}'].reindex(common)
            buy_sig = buy_sig & (c_a > sma_a)

        sig = pd.Series(0, index=common)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)


def sig_bollinger_bounce(indicators_map, prices, tickers, boll_key, mid_key,
                         use_rsi=False, rsi_key=None, rsi_max=50):
    """Bollinger alt bandından bounce + mid'de çıkış."""
    signals = {}
    for t in tickers:
        if t not in indicators_map: continue
        ind = indicators_map[t]
        c   = prices[t].dropna()
        if boll_key not in ind.columns or mid_key not in ind.columns: continue

        common = c.index.intersection(ind.index)
        c_a  = c.reindex(common)
        lo   = ind[boll_key].reindex(common)
        mid  = ind[mid_key].reindex(common)
        cp   = c_a.shift(1); lop = lo.shift(1)

        buy_sig  = (cp <= lop) & (c_a > lo)
        sell_sig = c_a >= mid

        if use_rsi and rsi_key and rsi_key in ind.columns:
            r_a = ind[rsi_key].reindex(common)
            buy_sig = buy_sig & (r_a < rsi_max)

        sig = pd.Series(0, index=common)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)


def sig_ema_cross(indicators_map, prices, tickers, fast_key, slow_key, trend_key=None):
    """EMA hızlı/yavaş kesişme."""
    signals = {}
    for t in tickers:
        if t not in indicators_map: continue
        ind = indicators_map[t]
        c   = prices[t].dropna()
        if fast_key not in ind.columns or slow_key not in ind.columns: continue

        common = c.index.intersection(ind.index)
        c_a = c.reindex(common)
        f   = ind[fast_key].reindex(common)
        s   = ind[slow_key].reindex(common)
        fp  = f.shift(1); sp = s.shift(1)

        buy_sig  = (fp <= sp) & (f > s)
        sell_sig = (fp >= sp) & (f < s)

        if trend_key and trend_key in ind.columns:
            tr = ind[trend_key].reindex(common)
            buy_sig = buy_sig & (c_a > tr)

        sig = pd.Series(0, index=common)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)


def sig_turtle_breakout(prices, tickers, entry_p=20, exit_p=10):
    """Donchian breakout (Turtle Trading)."""
    signals = {}
    for t in tickers:
        if t not in prices.columns: continue
        c = prices[t].dropna()
        if len(c) < entry_p + 5: continue
        high_br = c.shift(1).rolling(entry_p).max()
        low_ex  = c.rolling(exit_p).min()

        buy_sig  = c > high_br
        sell_sig = c <= low_ex

        sig = pd.Series(0, index=c.index)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)


def sig_mean_rev(indicators_map, prices, tickers, z_buy=-2.0, z_sell=0.5, w=50):
    """Z-score mean reversion."""
    signals = {}
    for t in tickers:
        c = prices[t].dropna() if t in prices.columns else None
        if c is None or len(c) < w + 5: continue
        roll_mean = c.rolling(w).mean()
        roll_std  = c.rolling(w).std() + 1e-9
        z  = (c - roll_mean) / roll_std
        zp = z.shift(1)

        buy_sig  = (zp < z_buy) & (z >= z_buy)
        sell_sig = z >= z_sell

        sig = pd.Series(0, index=c.index)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)


def sig_vol_momentum(indicators_map, prices, tickers, vol_data,
                     vol_mult=2.0, rsi_min=40, rsi_max=72, rsi_key='rsi14', mom_p=5):
    """Hacim patlaması + momentum."""
    signals = {}
    for t in tickers:
        if t not in indicators_map: continue
        ind = indicators_map[t]
        c   = prices[t].dropna()
        if t not in vol_data.columns: continue
        v   = vol_data[t].dropna()
        if rsi_key not in ind.columns: continue

        common = c.index.intersection(ind.index).intersection(v.index)
        c_a = c.reindex(common)
        r_a = ind[rsi_key].reindex(common)
        v_a = v.reindex(common)

        vol_avg = v_a.rolling(20).mean().shift(1)
        mom     = c_a / c_a.shift(mom_p) - 1

        buy_sig  = (v_a > vol_avg * vol_mult) & (mom > 0) & (r_a > rsi_min) & (r_a < rsi_max)
        sell_sig = r_a > rsi_max

        sig = pd.Series(0, index=common)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)


# ─── ANA OPTİMİZASYON FONKSİYONU ─────────────────────────

def run_optimization():
    ph("BIST FULL KAPSAMLI OPTIMIZASYON MOTORU v3.0")
    p(f"  Baslangic Kapital : {INITIAL_CAPITAL:,.0f} TL", "g")
    p(f"  Simulasyon Donemi : Son {SIM_DAYS} gun", "g")

    # 1. Ticker yükle
    tickers = load_tickers_from_db()
    p(f"[DB] {len(tickers)} hisse yuklendi.", "g")

    # 2. Veri indir
    raw_data = download_all_prices(tickers, days_back=DOWNLOAD_DAYS)

    valid_tickers = [t for t in tickers if t in raw_data]
    p(f"[Filtre] {len(valid_tickers)}/{len(tickers)} hisse yeterli veriye sahip.", "g")

    if len(valid_tickers) < 10:
        p("Yeterli hisse yok, cikiliyor.", "r"); return

    # XU100 fiyat serisi
    xu100_prices = None
    if "_XU100" in raw_data:
        xu100_prices = raw_data["_XU100"]["Close"]

    # Sim start
    sim_start = (datetime.now() - timedelta(days=SIM_DAYS)).strftime('%Y-%m-%d')
    p(f"[Sim] Baslangic: {sim_start}", "g")

    # 3. Fiyat matrisi oluştur
    p("\n[Hazirlik] Fiyat ve hacim matrisleri hazirlaniyor...", "y")
    price_dict  = {t: raw_data[t]['Close'] for t in valid_tickers}
    volume_dict = {t: raw_data[t]['Volume'] for t in valid_tickers}
    prices  = pd.DataFrame(price_dict).sort_index()
    volumes = pd.DataFrame(volume_dict).sort_index()
    p(f"  Fiyat matrisi: {prices.shape}", "g")

    # 4. Göstergeler precompute
    p("[Hazirlik] Teknik gostergeler hesaplaniyor...", "y")
    indicators_map = {}
    for i, t in enumerate(valid_tickers):
        try:
            df_t = raw_data[t]
            if len(df_t) < MIN_TRADING_DAYS: continue
            indicators_map[t] = precompute_indicators(df_t)
        except Exception as e:
            pass
        if (i+1) % 20 == 0:
            p(f"  {i+1}/{len(valid_tickers)} gosterge hesaplandi...", "y")

    ind_tickers = [t for t in valid_tickers if t in indicators_map]
    p(f"[Hazirlik] {len(ind_tickers)} hisse icin gostergeler hazir.", "g")

    # ─── STRATEJI GRID SEARCH ────────────────────────────
    all_results = []

    def run_strategy(name, matrix, label=""):
        if matrix is None or matrix.empty: return
        m = run_sim(matrix, prices, [t for t in ind_tickers if t in matrix.columns],
                    sim_start, xu100_prices=xu100_prices)
        if m is None or m['n'] == 0: return
        row = dict(strategy=name, **{k: m[k] for k in
            ['total_ret','n','wr','idr','alpha','xu100','sl','tp','avg','final']})
        all_results.append(row)
        c_  = "g" if m['total_ret'] > 40 else "y" if m['total_ret'] > 0 else "r"
        p(f"  {name:<45}: {m['total_ret']:>+7.1f}%  Win={m['wr']:.0f}%  "
          f"IDR={m['idr']:.0f}%  N={m['n']:>4}  Alpha={m['alpha']:>+6.1f}%  "
          f"Final={m['final']/1e6:.3f}M TL", c_)

    # ─── A: RSI DİP STANDALONE ────────────────────────────
    ph("A) RSI DIP GRID SEARCH (Tum kombinasyonlar)")
    buy_thr_list  = [25, 30, 33, 35, 38, 40, 42, 45]
    sell_thr_list = [60, 65, 68, 70, 72, 75, 78, 80]
    rsi_periods   = [7, 10, 14, 21]
    total_A = len(buy_thr_list) * len(sell_thr_list) * len(rsi_periods)
    done_A  = 0
    for rb, rs, rp in itertools.product(buy_thr_list, sell_thr_list, rsi_periods):
        if rs <= rb: continue
        name = f"RSI(b={rb},s={rs},p={rp})"
        rkey = f"rsi{rp}"
        mx   = sig_rsi_dip(indicators_map, prices, ind_tickers, rb, rs, rkey)
        run_strategy(name, mx)
        done_A += 1
        if done_A % 50 == 0:
            top = sorted(all_results, key=lambda x: x['total_ret'], reverse=True)[:1]
            best_so_far = f"{top[0]['strategy']}={top[0]['total_ret']:+.1f}%" if top else "-"
            p(f"  [A: {done_A}/{total_A}] En iyi su an: {best_so_far}", "m")

    # ─── B: RSI + TREND FİLTRE ────────────────────────────
    ph("B) RSI + SMA TREND FILTRESI GRID SEARCH")
    for rb, rs, rp, tsma in itertools.product(
        [30, 35, 40, 45], [65, 70, 75, 80], [10, 14], [50, 100, 200]
    ):
        if rs <= rb: continue
        name = f"RSI+SMA(b={rb},s={rs},p={rp},sma={tsma})"
        rkey = f"rsi{rp}"
        mx   = sig_rsi_dip(indicators_map, prices, ind_tickers, rb, rs, rkey, trend_sma=tsma)
        run_strategy(name, mx)

    # ─── C: RSI + MACD ONAY ───────────────────────────────
    ph("C) RSI + MACD ONAY GRID SEARCH")
    for rb, rs, rp, mkey in itertools.product(
        [30, 35, 40, 45], [65, 70, 75], [10, 14], ['macd_12_26', 'macd_8_17', 'macd_5_13']
    ):
        if rs <= rb: continue
        name = f"RSI+MACD(b={rb},s={rs},p={rp},{mkey})"
        rkey = f"rsi{rp}"
        mx   = sig_rsi_dip(indicators_map, prices, ind_tickers, rb, rs, rkey, macd_key=mkey)
        run_strategy(name, mx)

    # ─── D: MACD CROSS ────────────────────────────────────
    ph("D) MACD HISTGRAM CROSS")
    for mkey, tsma in itertools.product(
        ['macd_12_26', 'macd_8_17', 'macd_5_13'], [None, 50, 100, 200]
    ):
        name = f"MACD_CROSS({mkey},sma={tsma})"
        mx   = sig_macd_cross(indicators_map, prices, ind_tickers, mkey, trend_sma=tsma)
        run_strategy(name, mx)

    # ─── E: BOLLINGER ─────────────────────────────────────
    ph("E) BOLLINGER BAND BOUNCE")
    boll_params = [
        (15, 2.0), (20, 2.0), (20, 1.5), (20, 2.5), (25, 2.0), (25, 1.5), (30, 2.0)
    ]
    for (w, std), use_rsi, rsi_max in itertools.product(
        boll_params, [False, True], [45, 50, 55]
    ):
        bkey = f'boll_lower_{w}_{std}'
        mkey = f'boll_mid_{w}'
        rkey = 'rsi14'
        name = f"BOLL(w={w},std={std},rsi_filter={'Y' if use_rsi else 'N'},rsimax={rsi_max})"
        mx   = sig_bollinger_bounce(indicators_map, prices, ind_tickers,
                                    bkey, mkey, use_rsi=use_rsi, rsi_key=rkey, rsi_max=rsi_max)
        run_strategy(name, mx)

    # ─── F: EMA CROSS ─────────────────────────────────────
    ph("F) EMA KESISIM")
    ema_pairs = [(8, 20), (8, 50), (20, 50), (20, 100), (50, 200)]
    for (fe, se), trend in itertools.product(ema_pairs, [None, 'ema200', 'sma200']):
        fkey = f'ema{fe}'; skey = f'ema{se}'
        name = f"EMA_X(f={fe},s={se},trend={trend})"
        mx   = sig_ema_cross(indicators_map, prices, ind_tickers, fkey, skey, trend_key=trend)
        run_strategy(name, mx)

    # ─── G: TURTLE BREAKOUT ───────────────────────────────
    ph("G) TURTLE DONCHIAN BREAKOUT")
    for ep, xp in itertools.product([10, 15, 20, 30, 40], [5, 7, 10, 15]):
        if xp >= ep: continue
        name = f"TURTLE(entry={ep},exit={xp})"
        mx   = sig_turtle_breakout(prices, ind_tickers, ep, xp)
        run_strategy(name, mx)

    # ─── H: MEAN REVERSION ────────────────────────────────
    ph("H) MEAN REVERSION (Z-SCORE)")
    for z_b, z_s, w in itertools.product(
        [-2.5, -2.0, -1.5, -1.0], [0.0, 0.5, 1.0], [30, 50, 100]
    ):
        if z_s <= z_b: continue
        name = f"MEANREV(zb={z_b},zs={z_s},w={w})"
        mx   = sig_mean_rev(indicators_map, prices, ind_tickers, z_b, z_s, w)
        run_strategy(name, mx)

    # ─── I: VOLUME MOMENTUM ───────────────────────────────
    ph("I) HACIM MOMENTUM")
    for vm, rmin, rmax in itertools.product(
        [1.5, 2.0, 2.5, 3.0], [35, 40, 45], [70, 72, 75]
    ):
        name = f"VOLMOM(vm={vm},rmin={rmin},rmax={rmax})"
        mx   = sig_vol_momentum(indicators_map, prices, ind_tickers, volumes,
                                vol_mult=vm, rsi_min=rmin, rsi_max=rmax)
        run_strategy(name, mx)

    # ─── J: TRIPLE COMBO (En iyi RSI + MACD + TREND) ─────
    ph("J) TRIPLE COMBO: RSI + MACD + TREND")
    for rb, rs, tsma, mkey in itertools.product(
        [30, 35, 40], [70, 75], [50, 100], ['macd_12_26', 'macd_8_17']
    ):
        name = f"TRIPLE(b={rb},s={rs},sma={tsma},{mkey})"
        rkey = 'rsi14'
        # RSI oversold + trend up + MACD rising
        mx_base = sig_rsi_dip(indicators_map, prices, ind_tickers, rb, rs, rkey,
                               trend_sma=tsma, macd_key=mkey)
        run_strategy(name, mx_base)

    # ─── SONUÇ TABLOSU ────────────────────────────────────
    if not all_results:
        p("Hic sonuc uretilmedi!", "r"); return

    df_res = pd.DataFrame(all_results).sort_values('total_ret', ascending=False)

    ph("TÜM SONUCLARIN EN İYİ 25'İ")
    print(f"\n  {'#':<3} {'Strateji':<50} {'Getiri':>8} {'Alpha':>8} {'Win%':>6} {'IDR':>6} {'N':>5} {'Final TL':>12}")
    print(f"  {'─'*3} {'─'*50} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*5} {'─'*12}")
    for i, row in df_res.head(25).iterrows():
        c_ = "g" if row['total_ret'] > 40 else "y" if row['total_ret'] > 0 else "r"
        rank = df_res.index.get_loc(i) + 1
        line = (f"  {rank:<3} {row['strategy']:<50} {row['total_ret']:>+8.1f}% "
                f"{row['alpha']:>+8.1f}% {row['wr']:>6.1f}% {row['idr']:>6.1f}% "
                f"{row['n']:>5} {row['final']:>12,.0f}")
        p(line, c_)

    # ─── EN İYİ 5 STRATEJİ DETAY ─────────────────────────
    ph("EN İYİ 5 STRATEJİ - DETAYLI İŞLEM LOGU")
    top5 = df_res.head(5)
    detailed_results = []

    for _, row in top5.iterrows():
        sname = row['strategy']
        pp(f"\n[{sname}]")
        pp(f"  Getiri: {row['total_ret']:+.2f}%  Alpha: {row['alpha']:+.2f}%  "
           f"Win: {row['wr']:.1f}%  IDR: {row['idr']:.1f}%  N: {row['n']}  "
           f"Final: {row['final']:,.0f} TL")

        # Re-run ile trade log çıkar
        # Bu sadece top5 için detaylı trade log
        # (Signal re-generation based on name)
        detailed_results.append({
            "rank": int(df_res.index.get_loc(df_res.index[df_res['strategy']==sname][0]))+1,
            "strategy": sname,
            "total_return_pct": float(row['total_ret']),
            "alpha": float(row['alpha']),
            "xu100_return_pct": float(row['xu100']),
            "final_value": float(row['final']),
            "n_trades": int(row['n']),
            "win_rate": float(row['wr']),
            "idr": float(row['idr']),
            "sl_count": int(row['sl']),
            "tp_count": int(row['tp']),
            "avg_trade_return": float(row['avg']),
        })

    # ─── WALK-FORWARD VALIDASYON ──────────────────────────
    ph("WALK-FORWARD VALIDASYON (En iyi 3 strateji)")
    top3_names = df_res.head(3)['strategy'].tolist()
    wf_results = []

    for sname in top3_names:
        folds = []
        for fold in range(3):
            offset     = fold * 120
            fold_start = (datetime.now() - timedelta(days=360 - offset)).strftime('%Y-%m-%d')
            # Reconstruct signal for this strategy
            mx_fold = _reconstruct_signal(sname, indicators_map, prices, ind_tickers, volumes)
            if mx_fold is not None and not mx_fold.empty:
                m = run_sim(mx_fold, prices,
                            [t for t in ind_tickers if t in mx_fold.columns],
                            fold_start, xu100_prices=xu100_prices)
                folds.append(m['total_ret'] if m else 0.0)
            else:
                folds.append(0.0)

        mean_r = float(np.mean(folds)); std_r = float(np.std(folds))
        consist = "TUTARLI [OK]" if std_r < 20 else "DEGISKEN !"
        c_ = "g" if consist.startswith("TUTARLI") else "y"
        p(f"  {sname[:50]:<50}: Folds={[f'{x:+.1f}%' for x in folds]} "
          f"Ort={mean_r:+.1f}% Std={std_r:.1f}% [{consist}]", c_)
        wf_results.append(dict(name=sname, folds=folds, mean=mean_r,
                               std=std_r, consistent=consist.startswith("TUTARLI")))

    # ─── KAYDET ───────────────────────────────────────────
    output = {
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "scope": f"{len(valid_tickers)} BIST hissesi",
        "initial_capital": INITIAL_CAPITAL,
        "sim_period_days": SIM_DAYS,
        "xu100_return_pct": float(xu100_prices.iloc[-1] / xu100_prices[xu100_prices.index >= pd.to_datetime(sim_start)].iloc[0] - 1) * 100 if xu100_prices is not None else 0,
        "total_strategies_tested": len(all_results),
        "top25": df_res.head(25).to_dict(orient='records'),
        "top5_detail": detailed_results,
        "walk_forward": wf_results,
    }
    with open("./data/optimization_results_v3.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    ph("OPTIMIZASYON TAMAMLANDI")
    p(f"  Sonuclar 'data/optimization_results_v3.json' kaydedildi.", "g")
    p(f"  Test edilen toplam strateji: {len(all_results)}", "g")
    p(f"  En iyi strateji : {df_res.iloc[0]['strategy']}", "g")
    p(f"  En iyi getiri   : {df_res.iloc[0]['total_ret']:+.2f}%", "g")
    p(f"  Alpha           : {df_res.iloc[0]['alpha']:+.2f}%", "g")
    p(f"  XU100 Benchmark : {output['xu100_return_pct']:+.2f}%", "y")


def _reconstruct_signal(sname, indicators_map, prices, ind_tickers, volumes):
    """Strateji adından parametreleri okuyup matrisi yeniden üret."""
    try:
        # RSI simple
        m = re.match(r"RSI\(b=(\d+),s=(\d+),p=(\d+)\)$", sname)
        if m:
            rb, rs, rp = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return sig_rsi_dip(indicators_map, prices, ind_tickers, rb, rs, f"rsi{rp}")

        # RSI+SMA
        m = re.match(r"RSI\+SMA\(b=(\d+),s=(\d+),p=(\d+),sma=(\d+)\)", sname)
        if m:
            rb, rs, rp, tsma = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            return sig_rsi_dip(indicators_map, prices, ind_tickers, rb, rs, f"rsi{rp}", trend_sma=tsma)

        # RSI+MACD
        m = re.match(r"RSI\+MACD\(b=(\d+),s=(\d+),p=(\d+),(macd_\S+)\)", sname)
        if m:
            rb, rs, rp, mk = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            return sig_rsi_dip(indicators_map, prices, ind_tickers, rb, rs, f"rsi{rp}", macd_key=mk)

        # TRIPLE
        m = re.match(r"TRIPLE\(b=(\d+),s=(\d+),sma=(\d+),(macd_\S+)\)", sname)
        if m:
            rb, rs, tsma, mk = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            return sig_rsi_dip(indicators_map, prices, ind_tickers, rb, rs, "rsi14",
                               trend_sma=tsma, macd_key=mk)

        # MACD
        m = re.match(r"MACD_CROSS\((macd_\S+),sma=(.+)\)", sname)
        if m:
            mk, tsma = m.group(1), m.group(2)
            tsma = int(tsma) if tsma != 'None' else None
            return sig_macd_cross(indicators_map, prices, ind_tickers, mk, trend_sma=tsma)

        # TURTLE
        m = re.match(r"TURTLE\(entry=(\d+),exit=(\d+)\)", sname)
        if m:
            ep, xp = int(m.group(1)), int(m.group(2))
            return sig_turtle_breakout(prices, ind_tickers, ep, xp)

        # MEANREV
        m = re.match(r"MEANREV\(zb=(-?\d+\.?\d*),zs=(-?\d+\.?\d*),w=(\d+)\)", sname)
        if m:
            zb, zs, w = float(m.group(1)), float(m.group(2)), int(m.group(3))
            return sig_mean_rev(indicators_map, prices, ind_tickers, zb, zs, w)

        # VOLMOM
        m = re.match(r"VOLMOM\(vm=(\S+),rmin=(\d+),rmax=(\d+)\)", sname)
        if m:
            vm, rmin, rmax = float(m.group(1)), int(m.group(2)), int(m.group(3))
            return sig_vol_momentum(indicators_map, prices, ind_tickers, volumes,
                                    vol_mult=vm, rsi_min=rmin, rsi_max=rmax)

    except Exception as e:
        pass
    return None


if __name__ == "__main__":
    run_optimization()
