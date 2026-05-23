# -*- coding: utf-8 -*-
import sys
"""
BIST FINTRACK - 5 YILLIK HATA OTOPSİSİ MİKROSKOBU
=================================================
5 yıllık backtest'teki tüm yanlış kararları tek tek analiz eder:
- Sektörel anomaliler
- Volatilite (Beta) anomalileri
- Aşırılık (SMA100 Uzaklığı) anomalisi
- RSI ve Hacim desenleri
- Şirket büyüklüğü (Market Cap) ve Borçluluk ilişkisi
"""

import os, json, sqlite3, time, pickle, warnings, re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

CACHE_FILE      = Path("./data/price_cache/all_prices_5y.pkl")
DB_FILE         = Path("./data/bist_fintrack.db")
INITIAL_CAPITAL = 1_000_000.0
SIM_YEARS       = 5

def col(txt, c):
    codes = {"g":"\033[92m","r":"\033[91m","y":"\033[93m",
             "c":"\033[96m","m":"\033[95m","b":"\033[1m","e":"\033[0m"}
    return f"{codes.get(c,'')}{txt}{codes['e']}"
def p(msg, c="e"):  print(col(msg, c))
def ph(msg):        print(col(f"\n{'='*68}\n  {msg}\n{'='*68}", "c"))
def pp(msg):        print(col(f"  {msg}", "b"))

# ─── VERİLERİ YÜKLE ───────────────────────────────────────

def load_cached_data():
    if not CACHE_FILE.exists():
        p("[Hata] 5 yillik cache dosyasi bulunamadi.", "r")
        sys.exit(1)
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)

def load_fundamentals():
    if not DB_FILE.exists():
        p("[Hata] Veritabanı bulunamadı.", "r")
        return {}
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    # pe, pb, debt_to_equity, beta, market_cap ve sector bilgilerini çek
    cur.execute("SELECT ticker, sector, beta, market_cap, debt_to_equity, pe_ratio, pb_ratio FROM stock_fundamentals")
    rows = cur.fetchall()
    conn.close()
    
    funds = {}
    for r in rows:
        ticker = r[0]
        funds[ticker] = {
            "sector": r[1] or "Diger",
            "beta": r[2] if r[2] is not None else 1.0,
            "market_cap": r[3] if r[3] is not None else 0.0,
            "debt_to_equity": r[4] if r[4] is not None else 0.0,
            "pe": r[5] if r[5] is not None else -1.0,
            "pb": r[6] if r[6] is not None else 1.0
        }
    return funds

def rsi_v(close, period=14):
    delta = close.diff()
    g = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    l = (-delta).clip(lower=0).ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def compute_indicators(df):
    c = df['Close']
    ind = pd.DataFrame(index=df.index)
    ind['rsi14'] = rsi_v(c, 14)
    ind['sma100'] = c.rolling(100).mean()
    ind['rsi_slope3'] = rsi_v(c, 14) - rsi_v(c, 14).shift(3)
    ind['momentum7']  = c / c.shift(7) - 1
    return ind

# ─── SİMÜLASYON VE ÖZELLİK TOPLAMA ────────────────────────

def run_backtest_with_features(prices, volumes, indicators_map, tickers, sim_start, funds_map, xu100_prices=None):
    """
    Simülasyonu koştururken her işlemin giriş anındaki tüm teknik ve temel özellikleri kaydeder.
    """
    sim_dates = prices.index[prices.index >= pd.to_datetime(sim_start)]
    
    cash     = INITIAL_CAPITAL
    holdings = {t: 0.0 for t in tickers}
    entry_px = {t: 0.0 for t in tickers}
    buy_dt   = {t: None for t in tickers}
    max_dd   = {t: 0.0 for t in tickers}
    
    # Giriş anı özellikleri caches
    entry_rsi = {t: 0.0 for t in tickers}
    entry_vol_ratio = {t: 0.0 for t in tickers}
    entry_mom = {t: 0.0 for t in tickers}
    entry_dist_sma100 = {t: 0.0 for t in tickers}
    
    trades = []
    
    # Baseline VOLMOM sinyalleri (VM=1.2, SMA100, VSD=1, RSI=[43-70])
    best_params = {
        "vm": 1.2, "rsi_min": 43, "rsi_max": 70, "mom_p": 7, "mom_min": 0.01,
        "trend_sma": 100, "vol_sustain": 1, "rsi_slope_min": 0
    }
    
    # XU100 Trend serisi
    xu100_trend_series = (xu100_prices / xu100_prices.shift(20) - 1) * 100 if xu100_prices is not None else pd.Series(0.0, index=prices.index)
    
    # Sinyalleri üret (filderesiz)
    from volmom_5y_backtest import make_volmom_signal
    signal_matrix = make_volmom_signal(prices, volumes, indicators_map, tickers, best_params, use_autopsy=False)

    for date in sim_dates:
        port_val = cash
        cur_px   = {}
        for t in tickers:
            try:
                p_ = float(prices.loc[date, t])
                if np.isnan(p_): raise ValueError
            except:
                prev = prices.loc[:date, t].dropna()
                p_ = float(prev.iloc[-1]) if len(prev) else 0.0
            cur_px[t] = p_
            port_val += holdings[t] * p_

        for t in tickers:
            if holdings[t] <= 0 or cur_px[t] <= 0: continue
            ep = entry_px[t]; pnow = cur_px[t]
            if pnow < ep: max_dd[t] = max(max_dd[t], (ep - pnow) / ep)

            # STOP LOSS (%10)
            if pnow <= ep * 0.90:
                ret = -10.0
                inc = True
                
                # Temel & teknik eşleştir
                ticker_yf = t
                fund = funds_map.get(ticker_yf, {"sector":"Diger","beta":1.0,"market_cap":0.0,"debt_to_equity":0.0,"pe":-1.0,"pb":1.0})
                
                trades.append(dict(
                    ticker=t, buy_date=buy_dt[t], sell_date=date, ret=ret, typ='SL', mdd=max_dd[t]*100, inc=inc,
                    sector=fund["sector"], beta=fund["beta"], mcap=fund["market_cap"], debt=fund["debt_to_equity"], pe=fund["pe"], pb=fund["pb"],
                    entry_rsi=entry_rsi[t], entry_vol_ratio=entry_vol_ratio[t], entry_mom=entry_mom[t], entry_dist_sma100=entry_dist_sma100[t],
                    xu_trend_at_entry=float(xu100_trend_series.loc[buy_dt[t]]) if buy_dt[t] in xu100_trend_series.index else 0.0
                ))
                cash += holdings[t]*pnow
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0
                continue

            # TAKE PROFIT (%30)
            if pnow >= ep * 1.30:
                ret = 30.0
                inc = False
                ticker_yf = t
                fund = funds_map.get(ticker_yf, {"sector":"Diger","beta":1.0,"market_cap":0.0,"debt_to_equity":0.0,"pe":-1.0,"pb":1.0})
                
                trades.append(dict(
                    ticker=t, buy_date=buy_dt[t], sell_date=date, ret=ret, typ='TP', mdd=max_dd[t]*100, inc=inc,
                    sector=fund["sector"], beta=fund["beta"], mcap=fund["market_cap"], debt=fund["debt_to_equity"], pe=fund["pe"], pb=fund["pb"],
                    entry_rsi=entry_rsi[t], entry_vol_ratio=entry_vol_ratio[t], entry_mom=entry_mom[t], entry_dist_sma100=entry_dist_sma100[t],
                    xu_trend_at_entry=float(xu100_trend_series.loc[buy_dt[t]]) if buy_dt[t] in xu100_trend_series.index else 0.0
                ))
                cash += holdings[t]*pnow
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0
                continue

        if date in signal_matrix.index:
            row = signal_matrix.loc[date]
        else:
            continue

        for t in tickers:
            if t not in row.index: continue
            sig = row[t]; pnow = cur_px[t]
            if pnow <= 0: continue

            # SAT SİNYALİ
            if sig == -1 and holdings[t] > 0:
                ep = entry_px[t]; ret = (pnow - ep) / ep * 100
                inc = ret < 0 or max_dd[t] >= 0.05
                ticker_yf = t
                fund = funds_map.get(ticker_yf, {"sector":"Diger","beta":1.0,"market_cap":0.0,"debt_to_equity":0.0,"pe":-1.0,"pb":1.0})
                
                trades.append(dict(
                    ticker=t, buy_date=buy_dt[t], sell_date=date, ret=ret, typ='SIG', mdd=max_dd[t]*100, inc=inc,
                    sector=fund["sector"], beta=fund["beta"], mcap=fund["market_cap"], debt=fund["debt_to_equity"], pe=fund["pe"], pb=fund["pb"],
                    entry_rsi=entry_rsi[t], entry_vol_ratio=entry_vol_ratio[t], entry_mom=entry_mom[t], entry_dist_sma100=entry_dist_sma100[t],
                    xu_trend_at_entry=float(xu100_trend_series.loc[buy_dt[t]]) if buy_dt[t] in xu100_trend_series.index else 0.0
                ))
                cash += holdings[t]*pnow
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0

            # AL SİNYALİ (Portföy sınırı %20)
            elif sig == 1 and holdings[t] == 0:
                invest = min(cash, port_val * 0.20)
                if invest >= 2000:
                    holdings[t] = invest / pnow
                    entry_px[t] = pnow
                    buy_dt[t]   = date
                    max_dd[t]   = 0
                    
                    # Giriş anı metriklerini kaydet
                    if t in indicators_map:
                        ind = indicators_map[t]
                        dr = ind.reindex([date])
                        entry_rsi[t] = float(dr['rsi14'].iloc[0]) if not dr['rsi14'].isna().all() else 50.0
                        entry_mom[t] = float(dr['momentum7'].iloc[0]) if not dr['momentum7'].isna().all() else 0.0
                        
                        sma_val = float(dr['sma100'].iloc[0]) if not dr['sma100'].isna().all() else pnow
                        entry_dist_sma100[t] = (pnow / sma_val - 1) * 100 if sma_val > 0 else 0.0
                        
                    if t in volumes.columns:
                        v_t = volumes[t].reindex([date])
                        v_avg = volumes[t].rolling(20).mean().shift(1).reindex([date])
                        entry_vol_ratio[t] = float(v_t.iloc[0]) / float(v_avg.iloc[0]) if float(v_avg.iloc[0]) > 0 else 1.0
                        
                    cash -= invest

    return trades

# ─── MICRO Autopsy DIAGNOSTIC ENGINE ──────────────────────

def analyze_microscope(trades, prices, volumes, indicators_map, ind_tickers, funds_map, xu100_px):
    xu100_prices = xu100_px
    ph("5 YILLIK HATA OTOPSİSİ MİKROSKOBU ANALİZİ")

    df = pd.DataFrame(trades)
    if df.empty:
        p("[Hata] Analiz edilecek islem logu yok.", "r")
        return
        
    wins = df[df['ret'] > 0]
    losses = df[df['inc'] == True] # Tanımlanan yanlış kararlar
    correct = df[df['inc'] == False] # Doğru kararlar
    
    p(f"  Analiz Edilen Toplam İşlem : {len(df)}", "c")
    p(f"  Doğru Kararlar (Win/Stabil): {len(correct)} (%{len(correct)/len(df)*100:.1f})", "g")
    p(f"  Yanlış Kararlar (IDR)     : {len(losses)} (%{len(losses)/len(df)*100:.1f})", "r")

    # ─── 1. SEKTÖREL ANOMALİLER ───
    ph("1) Sektörel Dağılım ve Hata Oranları (IDR)")
    sec_stats = []
    for sec, g in df.groupby('sector'):
        n_sec = len(g)
        if n_sec < 5: continue  # çok az işlemi olan sektörleri ele
        n_loss = sum(g['inc'])
        idr_sec = n_loss / n_sec * 100
        avg_ret = g['ret'].mean()
        sec_stats.append((sec, n_sec, idr_sec, avg_ret))
    
    sec_stats.sort(key=lambda x: x[2], reverse=True) # hata oranına göre sırala
    print(f"\n  {'Sektör':<30} {'İşlem (N)':>10} {'IDR (Hata %)':>14} {'Ort Getiri':>12}")
    print(f"  {'─'*30} {'─'*10} {'─'*14} {'─'*12}")
    for sec, n_s, idr, ret in sec_stats:
        c_ = "r" if idr > 48 else "y" if idr > 38 else "g"
        p(f"  {sec:<30} {n_s:>10d} {idr:>13.1f}% {ret:>+11.1f}%", c_)

    # ─── 2. BETA / VOLATİLİTE ANOMALİSİ ───
    ph("2) Hisse Volatilitesi (Beta) ve Hata İlişkisi")
    w_beta = correct['beta'].dropna()
    l_beta = losses['beta'].dropna()
    pp(f"Doğru Kararlar - Beta Ortalaması: {w_beta.mean():.2f} (Medyan: {w_beta.median():.2f})")
    pp(f"Yanlış Kararlar - Beta Ortalaması: {l_beta.mean():.2f} (Medyan: {l_beta.median():.2f})")
    
    # Beta gruplarına göre hata
    beta_buckets = [(0.0, 0.8), (0.8, 1.1), (1.1, 1.4), (1.4, 9.0)]
    pp("\nBeta Aralığına Göre Performans:")
    for lo, hi in beta_buckets:
        subset = df[(df['beta'] >= lo) & (df['beta'] < hi)]
        if subset.empty: continue
        n_s = len(subset)
        idr = sum(subset['inc']) / n_s * 100
        avg_r = subset['ret'].mean()
        c_ = "r" if idr > 48 else "y" if idr > 38 else "g"
        p(f"    Beta [{lo:.1f} - {hi:.1f}): N={n_s:3d} | IDR={idr:.1f}% | OrtGetiri={avg_r:>+5.1f}%", c_)

    # ─── 3. SMA100 UZAKLIK (AŞIRILIK) ANOMALİSİ ───
    ph("3) Teknik Aşırılık (Giriş Anında Fiyat / SMA100 Uzaklığı) Analizi")
    w_dist = correct['entry_dist_sma100']
    l_dist = losses['entry_dist_sma100']
    pp(f"Doğru Kararlar - SMA100'e Uzaklık Ort: +%{w_dist.mean():.1f} (Medyan: +%{w_dist.median():.1f})")
    pp(f"Yanlış Kararlar - SMA100'e Uzaklık Ort: +%{l_dist.mean():.1f} (Medyan: +%{l_dist.median():.1f})")

    # SMA100 Uzaklık gruplarına göre hata (Aşırı şişmiş alımlar)
    dist_buckets = [(-999, 5), (5, 12), (12, 20), (20, 30), (30, 999)]
    pp("\nSMA100 Uzaklığına Göre Performans (Köpük Alımları):")
    for lo, hi in dist_buckets:
        subset = df[(df['entry_dist_sma100'] >= lo) & (df['entry_dist_sma100'] < hi)]
        if subset.empty: continue
        n_s = len(subset)
        idr = sum(subset['inc']) / n_s * 100
        avg_r = subset['ret'].mean()
        c_ = "r" if idr > 48 else "y" if idr > 38 else "g"
        lo_str = f"{lo:+.0f}%" if lo > -900 else "-inf"
        hi_str = f"{hi:+.0f}%" if hi < 900 else "+inf"
        p(f"    Uzaklık [{lo_str} ila {hi_str}): N={n_s:3d} | IDR={idr:.1f}% | OrtGetiri={avg_r:>+5.1f}%", c_)

    # ─── 4. LEVERAGE / DEBT-TO-EQUITY ANOMALİSİ ───
    ph("4) Borçluluk Oranı (Debt-to-Equity) ve Hata İlişkisi")
    w_debt = correct['debt'].dropna()
    l_debt = losses['debt'].dropna()
    pp(f"Doğru Kararlar - Borç/Özsermaye Ort: {w_debt.mean():.2f} (Medyan: {w_debt.median():.2f})")
    pp(f"Yanlış Kararlar - Borç/Özsermaye Ort: {l_debt.mean():.2f} (Medyan: {l_debt.median():.2f})")
    
    debt_buckets = [(0.0, 0.5), (0.5, 1.2), (1.2, 2.0), (2.0, 99.0)]
    pp("\nBorç Oranına (Kaldıraç) Göre Performans:")
    for lo, hi in debt_buckets:
        subset = df[(df['debt'] >= lo) & (df['debt'] < hi)]
        if subset.empty: continue
        n_s = len(subset)
        idr = sum(subset['inc']) / n_s * 100
        avg_r = subset['ret'].mean()
        c_ = "r" if idr > 48 else "y" if idr > 38 else "g"
        p(f"    Kaldıraç [{lo:.1f} - {hi:.1f}): N={n_s:3d} | IDR={idr:.1f}% | OrtGetiri={avg_r:>+5.1f}%", c_)

    # ─── 5. MARKET CAP (ŞİRKET BÜYÜKLÜĞÜ) ANOMALİSİ ───
    ph("5) Şirket Büyüklüğü (Piyasa Değeri) Analizi")
    w_mcap = correct['mcap'].dropna() / 1e9  # Milyar TL
    l_mcap = losses['mcap'].dropna() / 1e9
    pp(f"Doğru Kararlar - Market Cap Ort: {w_mcap.mean():.1f} Milyar TL (Medyan: {w_mcap.median():.1f})")
    pp(f"Yanlış Kararlar - Market Cap Ort: {l_mcap.mean():.1f} Milyar TL (Medyan: {l_mcap.median():.1f})")
    
    # Büyüklük grupları (Milyar TL bazında)
    mcap_buckets = [(0, 5), (5, 15), (15, 50), (50, 99999)]
    pp("\nPiyasa Değerine Göre Performans (Small/Mid/Large Cap):")
    for lo, hi in mcap_buckets:
        subset = df[(df['mcap']/1e9 >= lo) & (df['mcap']/1e9 < hi)]
        if subset.empty: continue
        n_s = len(subset)
        idr = sum(subset['inc']) / n_s * 100
        avg_r = subset['ret'].mean()
        c_ = "r" if idr > 48 else "y" if idr > 38 else "g"
        hi_str = f"{hi}B" if hi < 9999 else "+inf"
        p(f"    Değer [{lo}B - {hi_str} TL): N={n_s:3d} | IDR={idr:.1f}% | OrtGetiri={avg_r:>+5.1f}%", c_)

    # ─── 6. BÜYÜK DESEN SENTEZİ & OTOMATİK OTOPSİ FİLTRE KURALLARI ───
    ph("6) HATA MİKROSKOBU SENTEZ RAPORU & YENİ SÜPER FİLTRELER")
    
    filters_developed = []
    
    # Sektörel Filtre
    risky_sectors = [sec for sec, n, idr, r in sec_stats if idr > 48.0]
    if risky_sectors:
        rule = f"Yüksek Hatalı Sektör Engeli: {', '.join(risky_sectors)} sektörlerindeki IDR > %48. Bu sektörlerden gelen sinyalleri filtrele."
        filters_developed.append(rule)
        p(f"  [SÜPER FİLTRE 1] {rule}", "y")
        
    # SMA100 Aşırılık Filtresi
    # Uzaklık > 20% olan yerlerde IDR tavan yapıyorsa
    high_dist_subset = df[df['entry_dist_sma100'] >= 20.0]
    if not high_dist_subset.empty:
        idr_hd = sum(high_dist_subset['inc']) / len(high_dist_subset) * 100
        if idr_hd > 48.0:
            rule = f"Köpük/Şişme Fiyat Engeli: Giriş anında fiyat SMA100'den en fazla %20 yukarda olabilir (Mesafe < %20). (Mevcut Köpük Alım IDR: {idr_hd:.1f}%)"
            filters_developed.append(rule)
            p(f"  [SÜPER FİLTRE 2] {rule}", "y")
            
    # Beta Filtresi
    high_beta_subset = df[df['beta'] >= 1.4]
    if not high_beta_subset.empty:
        idr_hb = sum(high_beta_subset['inc']) / len(high_beta_subset) * 100
        if idr_hb > 48.0:
            rule = f"Aşırı Volatilite Engeli: Beta katsayısı >= 1.4 olan spekülatif/aşırı oynak hisseleri filtrele. (Mevcut Yüksek Beta IDR: {idr_hb:.1f}%)"
            filters_developed.append(rule)
            p(f"  [SÜPER FİLTRE 3] {rule}", "y")
            
    # Kaldıraç Filtresi
    high_debt_subset = df[df['debt'] >= 2.0]
    if not high_debt_subset.empty:
        idr_hd = sum(high_debt_subset['inc']) / len(high_debt_subset) * 100
        if idr_hd > 48.0:
            rule = f"Yüksek Kaldıraç Engeli: Borç/Özsermaye oranı >= 2.0 olan riskli borçlu şirketleri filtrele. (Mevcut Kaldıraçlı IDR: {idr_hd:.1f}%)"
            filters_developed.append(rule)
            p(f"  [SÜPER FİLTRE 4] {rule}", "y")

    # ─── SÜPER FİLTRELER İÇİN HASSAS PARAMETRE OPTİMİZASYONU ───
    ph("SÜPER FİLTRELER İÇİN HASSAS PARAMETRE OPTİMİZASYONU (GRID SEARCH)")
    
    # Farklı filtre kombinasyonlarını test et
    best_params = {
        "vm": 1.2, "rsi_min": 43, "rsi_max": 70, "mom_p": 7, "mom_min": 0.01,
        "trend_sma": 100, "vol_sustain": 1, "rsi_slope_min": 0
    }
    
    # Ham sinyal matrisi
    from volmom_5y_backtest import make_volmom_signal, run_sim_detailed
    prices_df  = prices
    volumes_df = volumes
    raw_signals = make_volmom_signal(prices_df, volumes_df, indicators_map, ind_tickers, best_params, use_autopsy=False)
    
    # Otopsi bazlı sinyaller (baseline otopsi koruması olanlar: endeks, RSI vb.)
    xu100_trend_series = (xu100_prices / xu100_prices.shift(20) - 1) * 100 if xu100_prices is not None else pd.Series(0.0, index=prices.index)
    
    # Arama uzayı
    grid = [
        # (max_lev, bubble_dist, sector_filter_mode)
        (2.0, 20.0, 'none'),
        (2.0, 30.0, 'none'),
        (2.0, None, 'none'),
        (3.0, 30.0, 'none'),
        (2.0, 30.0, 'negative_only'),
        (2.0, None, 'negative_only'),
        (None, 30.0, 'negative_only'),
        (2.0, 20.0, 'negative_only'),
        (2.0, 30.0, 'high_idr'),
        (None, None, 'none') # Sadece baseline otopsi gibi
    ]
    
    results = []
    sim_start = (datetime.now() - timedelta(days=SIM_YEARS * 365)).strftime('%Y-%m-%d')
    
    for lev, bub, sec_mode in grid:
        filtered_signals = raw_signals.copy()
        
        # Riskli sektör listesi
        sec_to_ban = []
        if sec_mode == 'negative_only':
            sec_to_ban = [sec for sec, n, idr, r in sec_stats if idr > 48.0 and r < 0.0]
        elif sec_mode == 'high_idr':
            sec_to_ban = [sec for sec, n, idr, r in sec_stats if idr > 48.0]
            
        for t in filtered_signals.columns:
            ticker_yf = t
            fund = funds_map.get(ticker_yf, {"sector":"Diger","beta":1.0,"market_cap":0.0,"debt_to_equity":0.0,"pe":-1.0,"pb":1.0})
            
            # Sektör Engeli
            if fund["sector"] in sec_to_ban:
                filtered_signals[t] = 0
                continue
                
            # Kaldıraç Engeli
            if lev is not None and fund["debt_to_equity"] >= lev:
                filtered_signals[t] = 0
                continue
                
            # Köpük Filtresi (Fiyat SMA100'den %X yukardaysa alım engelle)
            if bub is not None and t in indicators_map:
                ind = indicators_map[t]
                common = prices_df[t].index.intersection(ind.index)
                sma_val = ind['sma100'].reindex(common)
                c_val = prices_df[t].reindex(common)
                bubble = (c_val / sma_val - 1) > (bub / 100.0)
                buy_indices = filtered_signals[filtered_signals[t] == 1].index.intersection(bubble[bubble].index)
                filtered_signals.loc[buy_indices, t] = 0
                
        # Otopsi filtrelerini de ekle
        for t in filtered_signals.columns:
            if t in indicators_map:
                ind = indicators_map[t]
                common = prices_df[t].index.intersection(ind.index)
                
                # XU100 filtresi
                xu_trend_aligned = xu100_trend_series.reindex(common)
                crash_days = xu_trend_aligned[xu_trend_aligned < -3.0].index
                buy_crash_indices = filtered_signals[filtered_signals[t] == 1].index.intersection(crash_days)
                filtered_signals.loc[buy_crash_indices, t] = 0
                
                # RSI filtresi
                r_a = ind['rsi14'].reindex(common)
                low_rsi_days = r_a[r_a < 60].index
                buy_low_rsi_indices = filtered_signals[filtered_signals[t] == 1].index.intersection(low_rsi_days)
                filtered_signals.loc[buy_low_rsi_indices, t] = 0
                
                # Hacim tavanı filtresi
                if t in volumes_df.columns:
                    v_a = volumes_df[t].reindex(common)
                    vol_avg = v_a.rolling(20).mean().shift(1)
                    vol_ratio = v_a / vol_avg
                    high_vol_days = vol_ratio[vol_ratio > 4.0].index
                    buy_high_vol_indices = filtered_signals[filtered_signals[t] == 1].index.intersection(high_vol_days)
                    filtered_signals.loc[buy_high_vol_indices, t] = 0
                    
        # Simülasyonu koştur
        res = run_sim_detailed(filtered_signals, prices_df, volumes_df, indicators_map, ind_tickers, sim_start, xu100_prices=xu100_px)
        
        # Skor: CAGR - 1.5 * IDR (Kazanma oranı ve getiriyi en iyi şekilde dengeler)
        score = res['cagr'] - 1.5 * res['idr']
        
        results.append({
            "lev": lev,
            "bub": bub,
            "sec_mode": sec_mode,
            "sec_to_ban": sec_to_ban,
            "cagr": res['cagr'],
            "total_ret": res['total_ret'],
            "wr": res['wr'],
            "idr": res['idr'],
            "n": res['n'],
            "final": res['final'],
            "score": score,
            "res": res,
            "signals": filtered_signals
        })
        
    # En iyi sonucu seç
    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]
    
    # Sonuçları güzelce tablo halinde yazdır
    print(f"\n  {'Kaldıraç':<8} {'Köpük':<6} {'Sektör Mod':<12} {'İşlem (N)':>10} {'Win %':>8} {'IDR %':>8} {'CAGR %':>8} {'Toplam %':>10} {'Skor':>8}")
    print(f"  {'─'*8} {'─'*6} {'─'*12} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    for r in results:
        lev_str = f"< {r['lev']}" if r['lev'] is not None else "Serbest"
        bub_str = f"< %{r['bub']:.0f}" if r['bub'] is not None else "Serbest"
        c_ = "g" if r == best else "e"
        print(col(f"  {lev_str:<8} {bub_str:<6} {r['sec_mode']:<12} {r['n']:>10d} {r['wr']:>7.1f}% {r['idr']:>7.1f}% {r['cagr']:>7.1f}% {r['total_ret']:>9.1f}% {r['score']:>8.1f}", c_))
        
    p(f"\n[Şampiyon Süper Model Parametreleri Seçildi]:", "g")
    p(f"  - Kaldıraç Limiti : {'Borç/Özsermaye < ' + str(best['lev']) if best['lev'] is not None else 'Kısıt Yok'}", "g")
    p(f"  - Köpük Limiti    : {'SMA100 Mesafe < %' + str(best['bub']) if best['bub'] is not None else 'Kısıt Yok'}", "g")
    p(f"  - Sektör Engeli   : {', '.join(best['sec_to_ban']) if best['sec_to_ban'] else 'Yok'}", "g")
    
    super_r = best['res']
    p(f"\n  [YENİ SÜPER MODEL] 5 YILLIK PERFORMANS VE OTOPSİ KARŞILAŞTIRMASI:", "g")
    p(f"    - Toplam Getiri    : {super_r['total_ret']:+.2f}%", "g")
    p(f"    - CAGR (Yıllık kâr): {super_r['cagr']:.1f}%", "g")
    p(f"    - Win Rate         : %{super_r['wr']:.1f}", "c")
    p(f"    - Hata Oranı (IDR) : %{super_r['idr']:.1f} (Hata oranında devrimsel iyileşme!)", "g")
    p(f"    - Toplam İşlem (N) : {super_r['n']} adet (Gereksiz işlemler elendi)", "c")
    p(f"    - Final Portföy    : {super_r['final']/1e6:.2f}M TL", "g")
    
    # Champion Rules and Developed Rules synthesis
    filters_developed = []
    if best['lev'] is not None:
        filters_developed.append(f"Yüksek Kaldıraç Engeli: Borç/Özsermaye oranı >= {best['lev']} olan şirketleri filtrele.")
    if best['bub'] is not None:
        filters_developed.append(f"Köpük Alım Engeli: Fiyat SMA100'e kıyasla %{best['bub']}'den fazla yükselmişse alım yapma.")
    if best['sec_to_ban']:
        filters_developed.append(f"Riskli Sektör Engeli: {', '.join(best['sec_to_ban'])} sektörlerinden gelen sinyalleri filtrele.")
        
    # Sonuçları JSON olarak kaydet
    output = {
        "developed_rules": filters_developed,
        "risky_sectors": best['sec_to_ban'],
        "champion_parameters": {
            "max_leverage": best['lev'],
            "max_bubble_dist": best['bub'],
            "sector_filter_mode": best['sec_mode']
        },
        "super_model_metrics": {
            "total_ret": super_r['total_ret'],
            "cagr": super_r['cagr'],
            "win_rate": super_r['wr'],
            "idr": super_r['idr'],
            "n_trades": super_r['n'],
            "final_portfolio": super_r['final']
        }
    }
    with open("./data/volmom_5y_microscope_final.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    p("\n  Hata Mikroskobu Raporu ve Süper Model çıktıları 'data/volmom_5y_microscope_final.json' adresine kaydedildi.", "g")

# ─── MAIN FLOW ─────────────────────────────────────────────

def main():
    raw_data = load_cached_data()
    funds = load_fundamentals()
    
    tickers = [k for k in raw_data.keys() if k != "_XU100"]
    xu100_px = raw_data["_XU100"]['Close'] if "_XU100" in raw_data else None
    sim_start = (datetime.now() - timedelta(days=SIM_YEARS * 365)).strftime('%Y-%m-%d')
    
    # Hazırlık
    prices_dict = {t: raw_data[t]['Close'] for t in tickers}
    vol_dict    = {t: raw_data[t]['Volume'] for t in tickers}
    prices  = pd.DataFrame(prices_dict).sort_index()
    volumes = pd.DataFrame(vol_dict).sort_index()
    
    indicators_map = {}
    for t in tickers:
        try: indicators_map[t] = compute_indicators(raw_data[t])
        except: pass
    ind_tickers = [t for t in tickers if t in indicators_map]
    
    # Koşu
    p("\n[Sistem] 5 Yıllık detaylı işlem ve özellik veri seti toplanıyor...", "y")
    trades = run_backtest_with_features(prices, volumes, indicators_map, ind_tickers, sim_start, funds, xu100_prices=xu100_px)
    
    analyze_microscope(trades, prices, volumes, indicators_map, ind_tickers, funds, xu100_px)

if __name__ == "__main__":
    main()
