# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
BIST FINTRACK - 5 YILLIK KAPSAMLI ALGORİTMİK BACKTEST MOTORU
============================================================
Şampiyon otopsi ve baseline modellerimizi 5 yıllık (2021 - 2026)
farklı piyasa rejimleri altında (Boğa, Ayı, Yüksek/Düşük Faiz) test eder.
"""

import os, json, sqlite3, time, pickle, warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

INITIAL_CAPITAL = 1_000_000.0
SIM_YEARS       = 5
DOWNLOAD_DAYS   = 5 * 365 + 200  # 5 yıl + teknik gösterge tamponu
BATCH_SIZE      = 20
CACHE_FILE      = Path("./data/price_cache/all_prices_5y.pkl")

def col(txt, c):
    codes = {"g":"\033[92m","r":"\033[91m","y":"\033[93m",
             "c":"\033[96m","m":"\033[95m","b":"\033[1m","e":"\033[0m"}
    return f"{codes.get(c,'')}{txt}{codes['e']}"
def p(msg, c="e"):  print(col(msg, c))
def ph(msg):        print(col(f"\n{'='*68}\n  {msg}\n{'='*68}", "c"))
def pp(msg):        print(col(f"  {msg}", "b"))

# ─── VERİ BATCH DOWNLOADER (5 YIL) ─────────────────────────

def load_tickers_from_db():
    conn = sqlite3.connect("./data/bist_fintrack.db")
    cur  = conn.cursor()
    cur.execute("SELECT ticker FROM stock_fundamentals")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def download_5y_prices(tickers):
    end   = datetime.now()
    start = end - timedelta(days=DOWNLOAD_DAYS)
    s_str = start.strftime('%Y-%m-%d')
    e_str = end.strftime('%Y-%m-%d')

    if CACHE_FILE.exists():
        age_h = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age_h < 24: # 24 saat cache ömrü
            p(f"[Cache] 5 Yillik fiyat verileri local cache'den okunuyor ({age_h:.1f} saat eski)...", "y")
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)

    p(f"\n[Veri] {len(tickers)} hissenin 5 YILLIK verisi {s_str} -> {e_str} indiriliyor...", "y")
    all_data = {}
    failed   = []

    # Batch download
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

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                    else:
                        if isinstance(raw.columns, pd.MultiIndex):
                            df = raw.xs(ticker, axis=1, level=1)
                        else:
                            failed.append(ticker); continue

                    needed = ['Close']
                    for col_name in ['High', 'Low', 'Volume']:
                        if col_name in df.columns:
                            needed.append(col_name)
                    df = df[needed].dropna(subset=['Close'])

                    if len(df) < 500:  # 5 yıllık analiz için en az 500 trading günü şartı
                        continue

                    if 'High'   not in df.columns: df['High']   = df['Close']
                    if 'Low'    not in df.columns: df['Low']    = df['Close']
                    if 'Volume' not in df.columns: df['Volume'] = 0

                    all_data[ticker] = df
                except Exception:
                    failed.append(ticker)
        except Exception as e:
            failed.extend(batch)
        time.sleep(0.5)

    # XU100
    try:
        xu = yf.download("XU100.IS", start=s_str, end=e_str,
                          interval="1d", progress=False, auto_adjust=True)
        if isinstance(xu.columns, pd.MultiIndex):
            xu.columns = xu.columns.get_level_values(0)
        all_data["_XU100"] = xu[['Close']].dropna()
    except Exception as e:
        p(f"[Hata] XU100 indirilemedi: {e}", "r")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(all_data, f)
    return all_data

# ─── GÖSTERGE HESAPLAMA ───────────────────────────────────

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

# ─── SİNYAL ÜRETİCİ (OTOPSE VE STANDART) ──────────────────

def make_volmom_signal(prices, vol_data, indicators_map, tickers,
                       best_params, xu100_px=None,
                       use_autopsy=False,
                       xu_trend_min=-3.0,
                       rsi_floor=60,
                       vol_ratio_max=4.0):
    signals = {}
    vm        = best_params['vm']
    rsi_min   = best_params['rsi_min']
    rsi_max   = best_params['rsi_max']
    mom_p     = best_params['mom_p']
    mom_min   = best_params['mom_min']
    tsma      = best_params['trend_sma']
    vsd       = best_params['vol_sustain']
    slope_min = best_params['rsi_slope_min']

    # XU100 20 günlük trend
    xu100_trend_series = None
    if xu100_px is not None and use_autopsy and xu_trend_min is not None:
        xu100_trend_series = (xu100_px / xu100_px.shift(20) - 1) * 100

    for t in tickers:
        if t not in indicators_map or t not in prices.columns: continue
        if t not in vol_data.columns: continue
        ind = indicators_map[t]
        c   = prices[t].dropna()
        v   = vol_data[t].dropna()

        common = c.index.intersection(ind.index).intersection(v.index)
        if len(common) < 120: continue

        c_a = c.reindex(common)
        v_a = v.reindex(common)
        r_a = ind['rsi14'].reindex(common)
        vol_avg = v_a.rolling(20).mean().shift(1)
        mom_a   = (c_a / c_a.shift(mom_p) - 1)

        buy_sig  = (v_a > vol_avg * vm) & (mom_a > mom_min) & (r_a > rsi_min) & (r_a < rsi_max)
        sell_sig = r_a > rsi_max

        # SMA
        if tsma and f'sma{tsma}' in ind.columns:
            sma_a = ind[f'sma{tsma}'].reindex(common)
            buy_sig = buy_sig & (c_a > sma_a)

        # RSI eğimi
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
        if use_autopsy:
            if xu100_trend_series is not None and xu_trend_min is not None:
                xu_trend_aligned = xu100_trend_series.reindex(common)
                buy_sig = buy_sig & (xu_trend_aligned >= xu_trend_min)
            if rsi_floor is not None:
                buy_sig = buy_sig & (r_a >= rsi_floor)
            if vol_ratio_max is not None:
                vol_ratio_aligned = (v_a / vol_avg).reindex(common)
                buy_sig = buy_sig & (vol_ratio_aligned <= vol_ratio_max)

        sig = pd.Series(0, index=common)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)

# ─── SİMÜLASYON MOTORU (5 YIL) ───────────────────────────

def run_sim_detailed(signal_matrix, prices, vol_data, indicators_map, tickers, sim_start, xu100_prices=None):
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
                closed.append(dict(ticker=t, buy_date=buy_dt[t], sell_date=date, ret=-10.0, typ='SL', mdd=max_dd[t]*100, inc=True))
                cash += holdings[t]*pnow
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0
                continue

            # TAKE PROFIT (%30)
            if pnow >= ep * 1.30:
                closed.append(dict(ticker=t, buy_date=buy_dt[t], sell_date=date, ret=30.0, typ='TP', mdd=max_dd[t]*100, inc=False))
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

            if sig == -1 and holdings[t] > 0:
                ep = entry_px[t]; ret = (pnow - ep) / ep * 100
                inc = ret < 0 or max_dd[t] >= 0.05
                closed.append(dict(ticker=t, buy_date=buy_dt[t], sell_date=date, ret=ret, typ='SIG', mdd=max_dd[t]*100, inc=inc))
                cash += holdings[t]*pnow
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0

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
            closed.append(dict(ticker=t, buy_date=buy_dt[t], sell_date=sim_dates[-1], ret=ret, typ='LIQ', mdd=max_dd[t]*100, inc=inc))
            cash += holdings[t] * last_px

    final = cash
    total_ret = (final / INITIAL_CAPITAL - 1) * 100
    n = len(closed)
    wr  = sum(1 for x in closed if x['ret'] > 0) / n * 100 if n else 0
    idr = sum(1 for x in closed if x['inc'])    / n * 100 if n else 0
    avg = np.mean([x['ret'] for x in closed]) if n else 0
    sl_c  = sum(1 for x in closed if x['typ']=='SL')
    tp_c  = sum(1 for x in closed if x['typ']=='TP')

    # CAGR (Compound Annual Growth Rate)
    n_years = SIM_YEARS
    cagr = ((final / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100

    xu_ret = 0.0
    xu_cagr = 0.0
    if xu100_prices is not None and len(xu100_prices) > 0:
        xu_slice = xu100_prices[xu100_prices.index >= pd.to_datetime(sim_start)]
        if len(xu_slice) > 1:
            xu_ret = (float(xu_slice.iloc[-1]) / float(xu_slice.iloc[0]) - 1) * 100
            xu_cagr = ((float(xu_slice.iloc[-1]) / float(xu_slice.iloc[0])) ** (1 / n_years) - 1) * 100

    return dict(total_ret=total_ret, final=final, cagr=cagr, n=n, wr=wr, idr=idr, avg=avg,
                alpha=total_ret-xu_ret, alpha_cagr=cagr-xu_cagr, xu100=xu_ret, xu100_cagr=xu_cagr,
                sl=sl_c, tp=tp_c, trades=closed)

# ─── MAIN FLOW ─────────────────────────────────────────────

def main():
    ph("BIST FINTRACK - 5 YILLIK KAPSAMLI BACKTEST & MAKRO PERFORMANS ANALIZI")
    p(f"  Analiz Araligi  : Mayis 2021 - Mayis 2026 (5 Yil)", "g")
    p(f"  Baslangic Bütce : {INITIAL_CAPITAL:,.0f} TL", "g")

    tickers = load_tickers_from_db()
    raw_data = download_5y_prices(tickers)
    
    valid_tickers = [t for t in tickers if t in raw_data]
    p(f"  {len(valid_tickers)} hisse 5 yillik veriye sahip.", "g")

    xu100_px = raw_data["_XU100"]['Close'] if "_XU100" in raw_data else None
    sim_start = (datetime.now() - timedelta(days=SIM_YEARS * 365)).strftime('%Y-%m-%d')

    # Hazırlık
    prices_dict = {t: raw_data[t]['Close'] for t in valid_tickers}
    vol_dict    = {t: raw_data[t]['Volume'] for t in valid_tickers}
    prices  = pd.DataFrame(prices_dict).sort_index()
    volumes = pd.DataFrame(vol_dict).sort_index()

    indicators_map = {}
    for t in valid_tickers:
        try: indicators_map[t] = compute_indicators(raw_data[t])
        except: pass
    ind_tickers = [t for t in valid_tickers if t in indicators_map]

    # Baseline parametrelerini yükle
    best_params = {
        "vm": 1.2, "rsi_min": 43, "rsi_max": 70, "mom_p": 7, "mom_min": 0.01,
        "trend_sma": 100, "vol_sustain": 1, "rsi_slope_min": 0, "atr_filter": False
    }

    # 🚀 KOŞU 1: En İyi VOLMOM (Baseline - Optimizasyondan Çıkan)
    p("\n[Koşu 1] 5 Yıllık Baseline VOLMOM simülasyonu başlatılıyor...", "y")
    base_mx = make_volmom_signal(prices, volumes, indicators_map, ind_tickers, best_params, use_autopsy=False)
    base_r  = run_sim_detailed(base_mx, prices, volumes, indicators_map, ind_tickers, sim_start, xu100_prices=xu100_px)

    # 🚀 KOŞU 2: YZ Otopsi-VOLMOM (Otopsi Korumalı Şampiyon)
    p("[Koşu 2] 5 Yıllık YZ Otopsi-VOLMOM (Şampiyon) simülasyonu başlatılıyor...", "y")
    autopsy_mx = make_volmom_signal(
        prices, volumes, indicators_map, ind_tickers, best_params, xu100_px=xu100_px,
        use_autopsy=True, xu_trend_min=-3.0, rsi_floor=60, vol_ratio_max=4.0
    )
    autopsy_r = run_sim_detailed(autopsy_mx, prices, volumes, indicators_map, ind_tickers, sim_start, xu100_prices=xu100_px)

    # 🚀 KOŞU 3: Agresif YZ Modeli (Fırsatçı Breakout)
    p("[Koşu 3] 5 Yıllık Agresif YZ Modeli (Fırsatçı Breakout) simülasyonu başlatılıyor...", "y")
    agg_params = {
        "vm": 1.0, "rsi_min": 40, "rsi_max": 75, "mom_p": 5, "mom_min": 0.02,
        "trend_sma": 50, "vol_sustain": 1, "rsi_slope_min": 0, "atr_filter": False
    }
    agg_mx = make_volmom_signal(prices, volumes, indicators_map, ind_tickers, agg_params, use_autopsy=False)
    agg_r  = run_sim_detailed(agg_mx, prices, volumes, indicators_map, ind_tickers, sim_start, xu100_prices=xu100_px)

    # ─── RAPORLAMA ───────────────────────────────────────────
    ph("5 YILLIK ENFLASYON HESAPLAŞMASI VE PERFORMANS RAPORU (2021 - 2026)")
    
    # TÜİK TÜFE Endeks Değişimi: Mayıs 2021 (537.05) -> Nisan 2026 (4028.47)
    cpi_may_2021 = 537.05
    cpi_apr_2026 = 4028.47
    cum_inflation = (cpi_apr_2026 / cpi_may_2021 - 1) * 100
    inf_cagr = ((cpi_apr_2026 / cpi_may_2021) ** (1 / SIM_YEARS) - 1) * 100

    print(f"\n  {'Metrik':<28} {'TÜİK TÜFE':>12} {'XU100 Al-Tut':>14} {'Korumacı YZ':>14} {'En İyi VOLMOM':>15} {'⚡ Agresif YZ':>15}")
    print(f"  {'─'*28} {'─'*12} {'─'*14} {'─'*14} {'─'*15} {'─'*15}")
    
    metrics = [
        ('Toplam Net Getiri', f"+{cum_inflation:.1f}%", f"+{base_r['xu100']:.1f}%", f"+{autopsy_r['total_ret']:.1f}%", f"+{base_r['total_ret']:.1f}%", f"+{agg_r['total_ret']:.1f}%"),
        ('Bileşik Yıllık Büyüme (CAGR)', f"+{inf_cagr:.1f}%", f"+{base_r['xu100_cagr']:.1f}%", f"+{autopsy_r['cagr']:.1f}%", f"+{base_r['cagr']:.1f}%", f"+{agg_r['cagr']:.1f}%"),
        ('Reel CAGR (Enflasyon Üstü)', "Benchmark", f"{base_r['xu100_cagr'] - inf_cagr:>+5.1f}%", f"{autopsy_r['cagr'] - inf_cagr:>+5.1f}%", f"{base_r['cagr'] - inf_cagr:>+5.1f}%", f"{agg_r['cagr'] - inf_cagr:>+5.1f}%"),
        ('Kazanma Oranı (Win Rate)', "-", "-", f"%{autopsy_r['wr']:.1f}", f"%{base_r['wr']:.1f}", f"%{agg_r['wr']:.1f}"),
        ('Yanlış Karar Oranı (IDR)', "-", "-", f"%{autopsy_r['idr']:.1f}", f"%{base_r['idr']:.1f}", f"%{agg_r['idr']:.1f}"),
        ('Toplam İşlem Sayısı (N)', "-", "-", f"{autopsy_r['n']} ad", f"{base_r['n']} ad", f"{agg_r['n']} ad"),
        ('Final Portföy Değeri', f"{(INITIAL_CAPITAL * (1+cum_inflation/100))/1e6:.2f}M TL", f"{(INITIAL_CAPITAL * (1+base_r['xu100']/100))/1e6:.2f}M TL", f"{autopsy_r['final']/1e6:.2f}M TL", f"{base_r['final']/1e6:.2f}M TL", f"{agg_r['final']/1e6:.2f}M TL"),
    ]
    
    for name, inf_val, xu, aut, base, agg_val in metrics:
        c_ = "e"
        if "Getiri" in name or "CAGR" in name or "Reel" in name:
            c_ = "g"
        elif "IDR" in name:
            c_ = "y"
        p(f"  {name:<28} {inf_val:>12} {xu:>14} {aut:>14} {base:>15} {agg_val:>15}", c_)

    # Detaylı analiz
    p(f"\n  [Makro Enflasyon Analizi]:", "c")
    p(f"    - Resmi Enflasyon (TÜİK TÜFE) 5 yılda fiyatları {cpi_apr_2026/cpi_may_2021:.2f} katına çıkarmıştır.", "y")
    p(f"    - BIST 100 Endeksi (XU100) reel bazda yıllık %{base_r['xu100_cagr'] - inf_cagr:.1f} getiri sunmuştur.", "g")
    
    for name, res in [('Korumacı YZ', autopsy_r), ('En İyi VOLMOM', base_r), ('Agresif YZ', agg_r)]:
        beats = "YENMİŞTİR" if res['cagr'] > inf_cagr else "YENİLMİŞTİR"
        c_ = "g" if res['cagr'] > inf_cagr else "r"
        p(f"    - {name:<15} : Enflasyonu yıllık %{res['cagr'] - inf_cagr:+.1f} farkla {beats}.", c_)

    # Sonuçları JSON olarak kaydet
    output = {
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "analysis_period_years": SIM_YEARS,
        "initial_capital": INITIAL_CAPITAL,
        "metrics_summary": {
            "inflation": {
                "total_return_pct": cum_inflation,
                "cagr_pct": inf_cagr,
                "final_portfolio": INITIAL_CAPITAL * (1 + cum_inflation/100)
            },
            "xu100": {
                "total_return_pct": base_r['xu100'],
                "cagr_pct": base_r['xu100_cagr'],
                "final_portfolio": INITIAL_CAPITAL * (1 + base_r['xu100']/100)
            },
            "best_volmom": {
                "total_return_pct": base_r['total_ret'],
                "cagr_pct": base_r['cagr'],
                "alpha_cagr_pct": base_r['alpha_cagr'],
                "win_rate": base_r['wr'],
                "idr": base_r['idr'],
                "n_trades": base_r['n'],
                "final_portfolio": base_r['final']
            },
            "autopsy_volmom": {
                "total_return_pct": autopsy_r['total_ret'],
                "cagr_pct": autopsy_r['cagr'],
                "alpha_cagr_pct": autopsy_r['alpha_cagr'],
                "win_rate": autopsy_r['wr'],
                "idr": autopsy_r['idr'],
                "n_trades": autopsy_r['n'],
                "final_portfolio": autopsy_r['final']
            },
            "aggressive_ai": {
                "total_return_pct": agg_r['total_ret'],
                "cagr_pct": agg_r['cagr'],
                "alpha_cagr_pct": agg_r['cagr'] - base_r['xu100_cagr'],
                "win_rate": agg_r['wr'],
                "idr": agg_r['idr'],
                "n_trades": agg_r['n'],
                "final_portfolio": agg_r['final']
            }
        }
    }
    with open("./data/backtest_5y_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    p("\n  5 yillik makro analiz sonuclari 'data/backtest_5y_results.json' kaydedildi.", "g")

if __name__ == "__main__":
    main()
