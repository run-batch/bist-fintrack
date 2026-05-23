# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
BIST FINTRACK - AUTOPSY FILTER TESTING & IDR MINIMIZATION
==========================================================
Tests the addition of systematic autopsy-based filters:
1) XU100 Trend Floor (e.g., >= -3%, >= -1%, >= 0%)
2) RSI Floor (e.g., >= 50, >= 55, >= 60 to avoid falling knives)
3) Volume Multiplier Cap (e.g., <= 3.0x, <= 3.5x to avoid exhaustion spikes)
"""

import os, json, sqlite3, time, pickle, warnings, itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

INITIAL_CAPITAL = 1_000_000.0
SIM_DAYS        = 365
CACHE_FILE      = Path("./data/price_cache/all_prices.pkl")
OPTIMIZED_FILE  = Path("./data/volmom_optimized.json")

def col(txt, c):
    codes = {"g":"\033[92m","r":"\033[91m","y":"\033[93m",
             "c":"\033[96m","m":"\033[95m","b":"\033[1m","e":"\033[0m"}
    return f"{codes.get(c,'')}{txt}{codes['e']}"
def p(msg, c="e"):  print(col(msg, c))
def ph(msg):        print(col(f"\n{'='*68}\n  {msg}\n{'='*68}", "c"))
def pp(msg):        print(col(f"  {msg}", "b"))

# ─── VERİ YÜKLEME ──────────────────────────────────────────

def load_cached_data():
    if not CACHE_FILE.exists():
        p("[Hata] Cache dosyasi yok.", "r")
        sys.exit(1)
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)

def load_fundamentals():
    conn = sqlite3.connect("./data/bist_fintrack.db")
    cur  = conn.cursor()
    cur.execute("SELECT ticker, sector, beta, market_cap FROM stock_fundamentals")
    rows = cur.fetchall()
    conn.close()
    return {r[0]: {"sector": r[1] or "Diger", "beta": r[2] or 1.0, "mktcap": r[3]} for r in rows}

def rsi_v(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def compute_indicators(df):
    c = df['Close']
    ind = pd.DataFrame(index=df.index)
    ind['rsi14'] = rsi_v(c, 14)
    for span in [100]:
        ind[f'sma{span}'] = c.rolling(span).mean()
    ind['rsi_slope3'] = rsi_v(c, 14) - rsi_v(c, 14).shift(3)
    ind['momentum7']  = c / c.shift(7) - 1
    return ind

# ─── AUTOPSY-FILTERED SİNYAL ÜRETİCİ ────────────────────────

def make_autopsy_signal(prices, vol_data, indicators_map, tickers,
                         best_params,
                         xu100_trend_20g_min=None,
                         rsi_floor=None,
                         vol_ratio_max=None,
                         xu100_prices=None):
    """
    Sinyal üreticisine otopsi filtrelerini de ekler.
    """
    signals = {}
    vm        = best_params['vm']
    rsi_min   = best_params['rsi_min']
    rsi_max   = best_params['rsi_max']
    mom_p     = best_params['mom_p']
    mom_min   = best_params['mom_min']
    tsma      = best_params['trend_sma']
    vsd       = best_params['vol_sustain']
    slope_min = best_params['rsi_slope_min']

    # XU100 20 günlük trend serisini hazırla
    xu100_trend_series = None
    if xu100_prices is not None and xu100_trend_20g_min is not None:
        xu100_trend_series = (xu100_prices / xu100_prices.shift(20) - 1) * 100

    for t in tickers:
        if t not in indicators_map or t not in prices.columns: continue
        if t not in vol_data.columns: continue
        ind = indicators_map[t]
        c   = prices[t].dropna()
        v   = vol_data[t].dropna()

        common = c.index.intersection(ind.index).intersection(v.index)
        if len(common) < 60: continue

        c_a = c.reindex(common)
        v_a = v.reindex(common)
        r_a = ind['rsi14'].reindex(common)
        vol_avg = v_a.rolling(20).mean().shift(1)
        mom_a   = (c_a / c_a.shift(mom_p) - 1)

        # Baseline VOLMOM koşulları
        buy_sig  = (v_a > vol_avg * vm) & (mom_a > mom_min) & (r_a > rsi_min) & (r_a < rsi_max)
        sell_sig = r_a > rsi_max

        # 100-SMA filtresi
        if tsma and f'sma{tsma}' in ind.columns:
            sma_a = ind[f'sma{tsma}'].reindex(common)
            buy_sig = buy_sig & (c_a > sma_a)

        # RSI eğimi filtresi
        slope_key = 'rsi_slope3'
        if slope_key in ind.columns:
            slope_a = ind[slope_key].reindex(common)
            buy_sig = buy_sig & (slope_a > slope_min)

        # Hacim sürdürülebilirliği
        if vsd > 0:
            high_vol = v_a > vol_avg * (vm * 0.7)
            sustained = high_vol.rolling(vsd).sum() >= vsd
            buy_sig = buy_sig & sustained.shift(1)

        # ─── OTOPSİ FİLTRELERİ ───
        
        # A) XU100 Endeks Trend Filtresi
        if xu100_trend_series is not None and xu100_trend_20g_min is not None:
            xu_trend_aligned = xu100_trend_series.reindex(common)
            buy_sig = buy_sig & (xu_trend_aligned >= xu100_trend_20g_min)

        # B) RSI Düşen Bıçak Filtresi (Yeni RSI tabanı)
        if rsi_floor is not None:
            buy_sig = buy_sig & (r_a >= rsi_floor)

        # C) Hacim Aşırılık Sınırı (VolRatio Cap)
        if vol_ratio_max is not None:
            vol_ratio_aligned = (v_a / vol_avg).reindex(common)
            buy_sig = buy_sig & (vol_ratio_aligned <= vol_ratio_max)

        sig = pd.Series(0, index=common)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)

# ─── SİMÜLASYON MOTORU (IDR odaklı) ──────────────────────

def run_sim(signal_matrix, prices, vol_data, indicators_map, tickers, sim_start, xu100_prices=None):
    sim_dates = prices.index[prices.index >= pd.to_datetime(sim_start)]
    if len(sim_dates) == 0: return None

    cash     = INITIAL_CAPITAL
    holdings = {t: 0.0 for t in tickers}
    entry_px = {t: 0.0 for t in tickers}
    buy_dt   = {t: None for t in tickers}
    max_dd   = {t: 0.0 for t in tickers}
    closed   = []

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
                closed.append(dict(ticker=t, buy_date=buy_dt[t], buy_px=ep, sell_date=date, sell_px=pnow, ret=-10.0, typ='SL', mdd=max_dd[t]*100, inc=True))
                cash += holdings[t]*pnow
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0
                continue

            # TAKE PROFIT (%30)
            if pnow >= ep * 1.30:
                closed.append(dict(ticker=t, buy_date=buy_dt[t], buy_px=ep, sell_date=date, sell_px=pnow, ret=30.0, typ='TP', mdd=max_dd[t]*100, inc=False))
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
                closed.append(dict(ticker=t, buy_date=buy_dt[t], buy_px=ep, sell_date=date, sell_px=pnow, ret=ret, typ='SIG', mdd=max_dd[t]*100, inc=inc))
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
                    cash -= invest

    # Kalanları kapat
    for t in tickers:
        if holdings[t] > 0:
            ep = entry_px[t]
            last_px = float(prices[t].dropna().iloc[-1])
            ret = (last_px - ep) / ep * 100
            inc = ret < 0 or max_dd[t] >= 0.05
            closed.append(dict(ticker=t, buy_date=buy_dt[t], buy_px=ep, sell_date=sim_dates[-1], sell_px=last_px, ret=ret, typ='LIQ', mdd=max_dd[t]*100, inc=inc))
            cash += holdings[t] * last_px

    final = cash
    total_ret = (final / INITIAL_CAPITAL - 1) * 100
    n = len(closed)
    wr  = sum(1 for x in closed if x['ret'] > 0) / n * 100 if n else 0
    idr = sum(1 for x in closed if x['inc'])    / n * 100 if n else 0
    avg = np.mean([x['ret'] for x in closed]) if n else 0

    return dict(total_ret=total_ret, final=final, n=n, wr=wr, idr=idr, avg=avg, trades=closed)

# ─── MAİN FLOW ─────────────────────────────────────────────

def main():
    ph("BIST FINTRACK - OTOPSİ FİLTRESİ OPTİMİZASYON VE IDR DÜŞÜRME ÇALIŞMASI")

    raw_data   = load_cached_data()
    funds      = load_fundamentals()
    sim_start  = (datetime.now() - timedelta(days=SIM_DAYS)).strftime('%Y-%m-%d')

    tickers    = [k for k in raw_data.keys() if k != "_XU100"]
    xu100_px   = raw_data["_XU100"]['Close'] if "_XU100" in raw_data else None

    # En iyi parametreleri yükle
    if not OPTIMIZED_FILE.exists():
        p("[Hata] volmom_optimized.json dosyasi yok.", "r")
        sys.exit(1)
    with open(OPTIMIZED_FILE, "r", encoding="utf-8") as f:
        best_params = json.load(f)["best_params"]

    # Parametrelerin float olmamasını garanti et
    best_params['trend_sma'] = 100
    best_params['vol_sustain'] = 1
    best_params['rsi_slope_min'] = 0
    best_params['mom_p'] = 7

    p(f"\n[Sistem] Baseline Parametreleri: vm={best_params['vm']} | RSI={best_params['rsi_min']}-{best_params['rsi_max']} | TrendSMA=100\n", "y")

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

    # Baseline testi
    p("  Baseline Simülasyonu koşturuluyor...", "y")
    base_mx = make_autopsy_signal(prices, volumes, indicators_map, ind_tickers, best_params)
    base_r  = run_sim(base_mx, prices, volumes, indicators_map, ind_tickers, sim_start)
    
    p(f"  [BASELINE] Getiri: {base_r['total_ret']:+.2f}% | Win Rate: %{base_r['wr']:.1f} | IDR: %{base_r['idr']:.1f} | N: {base_r['n']}", "b")

    # ─── OTOPSİ FİLTRELERİ PARAMETRE GRID SEARCH ───
    ph("OTOPSİ FİLTRELERİ GRID SEARCH")

    autopsy_grid = list(itertools.product(
        [-3.0, -1.0, 0.0, 1.0, None],       # xu100_trend_20g_min (Endeks filtresi)
        [43, 50, 53, 55, 60, None],          # rsi_floor (Düşen bıçak filtresi)
        [3.0, 3.5, 4.0, None]                # vol_ratio_max (Aşırı hacim filtresi)
    ))

    results = []
    
    for xu_trend, rsi_fl, vol_max in autopsy_grid:
        # Sinyalleri üret
        mx = make_autopsy_signal(
            prices, volumes, indicators_map, ind_tickers, best_params,
            xu100_trend_20g_min=xu_trend, rsi_floor=rsi_fl, vol_ratio_max=vol_max,
            xu100_prices=xu100_px
        )
        if mx.empty: continue
        
        m = run_sim(mx, prices, volumes, indicators_map, ind_tickers, sim_start, xu100_prices=xu100_px)
        if m is None or m['n'] < 15: continue  # Yetersiz işlem sayısını ele

        results.append(dict(
            xu_trend=xu_trend,
            rsi_floor=rsi_fl,
            vol_max=vol_max,
            total_ret=m['total_ret'],
            wr=m['wr'],
            idr=m['idr'],
            n=m['n'],
            avg=m['avg']
        ))

    df = pd.DataFrame(results)
    
    # IDR'ye göre sırala (en düşük IDR en başta)
    df_by_idr = df.sort_values('idr', ascending=True)

    ph("YENİ YOL HARİTASI: EN DÜŞÜK YANLIŞ KARAR ORANLI (IDR) STRATEJİLER")
    print(f"\n  {'#':<3} {'XU Trend':>8} {'RSI Flr':>7} {'Vol Max':>7} {'IDR (Düşük)':>12} {'Win Rate':>9} {'Getiri':>8} {'N':>5}")
    print(f"  {'-'*3} {'-'*8} {'-'*7} {'-'*7} {'-'*12} {'-'*9} {'-'*8} {'-'*5}")
    
    for i, row in df_by_idr.head(15).iterrows():
        rank = df_by_idr.index.get_loc(i) + 1
        c_ = "g" if row['idr'] < 30 else "y" if row['idr'] < 35 else "e"
        
        xu_str = f">={row['xu_trend']}%" if row['xu_trend'] is not None else "None"
        rsi_str = f">={row['rsi_floor']}" if row['rsi_floor'] is not None else "None"
        vol_str = f"<={row['vol_max']}x" if row['vol_max'] is not None else "None"
        
        line = (f"  {rank:<3} {xu_str:>8} {rsi_str:>7} {vol_str:>7} "
                f"{row['idr']:>11.1f}% {row['wr']:>8.1f}% {row['total_ret']:>+7.1f}% {row['n']:>5}")
        p(line, c_)

    # Getiriye göre sırala ama IDR < %38 filtrele
    df_filtered_ret = df[df['idr'] <= 38.0].sort_values('total_ret', ascending=False)
    
    ph("HEM DÜŞÜK HATA (IDR <= %35) HEM EN YÜKSEK GETİRİLİ STRATEJİLER")
    print(f"\n  {'#':<3} {'XU Trend':>8} {'RSI Flr':>7} {'Vol Max':>7} {'IDR':>6} {'Win Rate':>9} {'Getiri (Yüksek)':>15} {'N':>5}")
    print(f"  {'-'*3} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*9} {'-'*15} {'-'*5}")
    
    for i, row in df_filtered_ret.head(15).iterrows():
        rank = df_filtered_ret.index.get_loc(i) + 1
        c_ = "g" if row['total_ret'] > 120 else "y" if row['total_ret'] > 90 else "e"
        
        xu_str = f">={row['xu_trend']}%" if row['xu_trend'] is not None else "None"
        rsi_str = f">={row['rsi_floor']}" if row['rsi_floor'] is not None else "None"
        vol_str = f"<={row['vol_max']}x" if row['vol_max'] is not None else "None"
        
        line = (f"  {rank:<3} {xu_str:>8} {rsi_str:>7} {vol_str:>7} "
                f"{row['idr']:>5.1f}% {row['wr']:>8.1f}% {row['total_ret']:>+14.1f}% {row['n']:>5}")
        p(line, c_)

    # Şampiyon Autopsy Stratejiyi Çalıştır ve JSON'a Kaydet
    if not df_filtered_ret.empty:
        best_autopsy = df_filtered_ret.iloc[0].to_dict()
        
        # Clean best_autopsy to convert np.nan/NaN to None
        for key in ['xu_trend', 'rsi_floor', 'vol_max']:
            val = best_autopsy.get(key)
            if pd.isna(val) or val is None:
                best_autopsy[key] = None

        ph("ŞAMPİYON OTOPSİ STRATEJİSİ DETAYLI ANALİZİ")
        p(f"  Endeks Filtresi : XU100 20G Trend >= {best_autopsy['xu_trend']}%" if best_autopsy['xu_trend'] is not None else "  Endeks Filtresi : Yok", "g")
        p(f"  RSI Tabanı      : Giriş RSI >= {best_autopsy['rsi_floor']}" if best_autopsy['rsi_floor'] is not None else "  RSI Tabanı      : Yok", "g")
        p(f"  Hacim Tavanı    : Giriş VolRatio <= {best_autopsy['vol_max']}x" if best_autopsy['vol_max'] is not None else "  Hacim Tavanı    : Yok", "g")
        
        best_mx = make_autopsy_signal(
            prices, volumes, indicators_map, ind_tickers, best_params,
            xu100_trend_20g_min=best_autopsy['xu_trend'],
            rsi_floor=best_autopsy['rsi_floor'],
            vol_ratio_max=best_autopsy['vol_max'],
            xu100_prices=xu100_px
        )
        
        sim_res = run_sim(best_mx, prices, volumes, indicators_map, ind_tickers, sim_start, xu100_prices=xu100_px)
        
        # Sonuçları JSON olarak kaydet
        output = {
            "autopsy_filters_applied": {
                "xu100_trend_min": best_autopsy['xu_trend'],
                "rsi_floor": best_autopsy['rsi_floor'],
                "vol_ratio_max": best_autopsy['vol_max']
            },
            "metrics": {
                "total_ret": sim_res['total_ret'],
                "win_rate": sim_res['wr'],
                "idr": sim_res['idr'],
                "n_trades": sim_res['n'],
                "final_portfolio": sim_res['final']
            },
            "trade_log": [
                {k: (str(v) if isinstance(v, pd.Timestamp) else
                     v.strftime('%Y-%m-%d') if hasattr(v,'strftime') else v)
                 for k, v in t.items()} for t in sim_res['trades']
            ]
        }
        
        with open("./data/volmom_autopsy_final.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            
        p("\n  Otopsi filtreli yuksek performansli sonuclar 'data/volmom_autopsy_final.json' kaydedildi.", "g")

if __name__ == "__main__":
    main()
