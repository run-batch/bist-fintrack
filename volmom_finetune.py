# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
BIST FINTRACK - VOLMOM INCE AYAR + HATA OTOPSISI
=================================================
1) VOLMOM stratejisini cok daha ince parametrelerle optimize et
2) Kaybedilen her islemi sistematik olarak analiz et:
   - Giris aninda piyasa rejimi nasil?
   - RSI egimi hangi yonde?
   - Hacim surekliligi var mi yoksa tek gunluk spike mi?
   - Giris sonrasi kac gunde zarar materiallesdi?
   - Hangi sektorler daha fazla hata uretdi?
3) "Kacin" paternlerini tanimla -> filtreler onerDir
"""

import os, json, sqlite3, time, pickle, warnings, itertools, re
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

INITIAL_CAPITAL = 1_000_000.0
SIM_DAYS        = 365
DOWNLOAD_DAYS   = 720
MIN_DAYS        = 250
CACHE_FILE      = Path("./data/price_cache/all_prices.pkl")

def col(txt, c):
    codes = {"g":"\033[92m","r":"\033[91m","y":"\033[93m",
             "c":"\033[96m","m":"\033[95m","b":"\033[1m","e":"\033[0m"}
    return f"{codes.get(c,'')}{txt}{codes['e']}"
def p(msg, c="e"):  print(col(msg, c))
def ph(msg):        print(col(f"\n{'='*68}\n  {msg}\n{'='*68}", "c"))
def pp(msg):        print(col(f"  {msg}", "b"))

# ─── VERİ ─────────────────────────────────────────────────

def load_cached_data():
    if not CACHE_FILE.exists():
        p("[Hata] Cache dosyasi yok. Once optimize_full_bist.py calistirin.", "r")
        sys.exit(1)
    age_h = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
    p(f"[Cache] {age_h:.1f} saat once indirilmis veri okunuyor...", "y")
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)

def load_fundamentals():
    conn = sqlite3.connect("./data/bist_fintrack.db")
    cur  = conn.cursor()
    cur.execute("SELECT ticker, sector, beta, market_cap FROM stock_fundamentals")
    rows = cur.fetchall()
    conn.close()
    return {r[0]: {"sector": r[1] or "Diger", "beta": r[2] or 1.0, "mktcap": r[3]} for r in rows}

# ─── TEKNİK GÖSTERGELER ───────────────────────────────────

def rsi_v(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def ema_v(c, s): return c.ewm(span=s, adjust=False).mean()
def sma_v(c, w): return c.rolling(w).mean()

def compute_indicators(df):
    c = df['Close']
    ind = pd.DataFrame(index=df.index)
    for period in [7, 10, 14, 21]:
        ind[f'rsi{period}'] = rsi_v(c, period)
    for span in [8, 20, 50, 100, 200]:
        ind[f'ema{span}'] = ema_v(c, span)
    for w in [20, 50, 100, 200]:
        ind[f'sma{w}'] = sma_v(c, w)
    ind['rsi_slope3']  = rsi_v(c, 14) - rsi_v(c, 14).shift(3)  # RSI 3-gun egimi
    ind['rsi_slope5']  = rsi_v(c, 14) - rsi_v(c, 14).shift(5)
    ind['rsi_slope10'] = rsi_v(c, 14) - rsi_v(c, 14).shift(10)
    ind['momentum5']   = c / c.shift(5) - 1
    ind['momentum10']  = c / c.shift(10) - 1
    ind['momentum20']  = c / c.shift(20) - 1
    ind['atr14']       = (df['High'] - df['Low']).rolling(14).mean() if 'High' in df.columns else pd.Series(0, index=df.index)
    ind['atr_pct']     = ind['atr14'] / c
    return ind

# ─── VOLMOM SİNYAL ÜRETİCİ (GENİŞLETİLMİŞ) ──────────────

def make_volmom_signal(prices, vol_data, indicators_map, tickers,
                       vm=1.5, rsi_min=45, rsi_max=75,
                       mom_p=5, mom_min=0.0,
                       vol_sustain_days=0,    # ardışık yüksek hacim günü şartı
                       rsi_slope_min=-999,    # RSI son N gün en az bu kadar artmali
                       rsi_slope_period=3,
                       trend_sma=None,        # fiyat bu SMA üzerinde olmalı
                       atr_filter=False):     # ATR volatilite filtresi
    """
    Genişletilmiş VOLMOM sinyal üreticisi.
    Tüm ek filtreler açılıp kapanabilir.
    """
    signals = {}
    for t in tickers:
        if t not in indicators_map or t not in prices.columns: continue
        if t not in vol_data.columns: continue
        ind = indicators_map[t]
        c   = prices[t].dropna()
        v   = vol_data[t].dropna()
        if 'rsi14' not in ind.columns: continue

        common = c.index.intersection(ind.index).intersection(v.index)
        if len(common) < 60: continue

        c_a = c.reindex(common)
        v_a = v.reindex(common)
        r_a = ind['rsi14'].reindex(common)
        vol_avg = v_a.rolling(20).mean().shift(1)
        mom_a   = (c_a / c_a.shift(mom_p) - 1)

        buy_sig  = (v_a > vol_avg * vm) & (mom_a > mom_min) & (r_a > rsi_min) & (r_a < rsi_max)
        sell_sig = r_a > rsi_max

        # RSI eğimi filtresi
        if rsi_slope_min > -999:
            slope_key = f'rsi_slope{rsi_slope_period}'
            if slope_key in ind.columns:
                slope_a = ind[slope_key].reindex(common)
                buy_sig = buy_sig & (slope_a > rsi_slope_min)

        # Trend filtresi
        if trend_sma and f'sma{trend_sma}' in ind.columns:
            sma_a = ind[f'sma{trend_sma}'].reindex(common)
            buy_sig = buy_sig & (c_a > sma_a)

        # Hacim sürdürülebilirlik filtresi: son N günde de yüksek hacim olmalı
        if vol_sustain_days > 0:
            high_vol = v_a > vol_avg * (vm * 0.7)  # biraz daha gevşek eşik
            sustained = high_vol.rolling(vol_sustain_days).sum() >= vol_sustain_days
            buy_sig = buy_sig & sustained.shift(1)  # dünden beri sürekli

        # ATR volatilite filtresi: aşırı oynak hisseleri at
        if atr_filter and 'atr_pct' in ind.columns:
            atr_a = ind['atr_pct'].reindex(common)
            atr_avg = atr_a.rolling(30).mean()
            buy_sig = buy_sig & (atr_a < atr_avg * 2.5)

        sig = pd.Series(0, index=common)
        sig.loc[buy_sig[buy_sig].index]  = 1
        sig.loc[sell_sig[sell_sig].index] = -1
        signals[t] = sig.reindex(prices.index, fill_value=0)

    return pd.DataFrame(signals)


# ─── SİMÜLASYON MOTORU (İşlem Detaylı) ───────────────────

def run_sim_detailed(signal_matrix, prices, vol_data, indicators_map, tickers,
                     sim_start, xu100_prices=None,
                     sl_pct=0.10, tp_pct=0.30, alloc_pct=0.20):
    """
    Tam simülasyon + her işlem için giris-cikis detay kaydı.
    Hata analizi için zengin bilgi toplar.
    """
    sim_dates = prices.index[prices.index >= pd.to_datetime(sim_start)]
    if len(sim_dates) == 0: return None

    cash     = INITIAL_CAPITAL
    holdings = {t: 0.0 for t in tickers}
    entry_px = {t: 0.0 for t in tickers}
    buy_dt   = {t: None for t in tickers}
    max_dd   = {t: 0.0 for t in tickers}
    entry_rsi = {t: 0.0 for t in tickers}
    entry_vol_ratio = {t: 0.0 for t in tickers}
    entry_mom = {t: 0.0 for t in tickers}
    closed   = []

    xu100_series = xu100_prices if xu100_prices is not None else pd.Series(dtype=float)

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

        # XU100 durumu
        xu_now  = 0.0
        xu_past = 0.0
        if len(xu100_series) > 0:
            xu_before = xu100_series[xu100_series.index <= date]
            if len(xu_before) >= 20:
                xu_now  = float(xu_before.iloc[-1])
                xu_past = float(xu_before.iloc[-20])

        for t in tickers:
            if holdings[t] <= 0 or cur_px[t] <= 0: continue
            ep = entry_px[t]; pnow = cur_px[t]
            if pnow < ep: max_dd[t] = max(max_dd[t], (ep - pnow) / ep)

            if pnow <= ep * (1 - sl_pct):
                ret = -sl_pct * 100
                xu_ret_since_entry = 0.0
                if len(xu100_series) and buy_dt[t] is not None:
                    xu_at_entry = xu100_series[xu100_series.index <= buy_dt[t]]
                    xu_at_exit  = xu100_series[xu100_series.index <= date]
                    if len(xu_at_entry) and len(xu_at_exit):
                        xu_ret_since_entry = (float(xu_at_exit.iloc[-1]) / float(xu_at_entry.iloc[-1]) - 1) * 100

                closed.append(dict(
                    ticker=t, buy_date=buy_dt[t], sell_date=date,
                    buy_px=ep, sell_px=pnow, ret=ret, typ='SL',
                    mdd=max_dd[t]*100, inc=True,
                    entry_rsi=entry_rsi[t], entry_vol_ratio=entry_vol_ratio[t],
                    entry_mom=entry_mom[t],
                    hold_days=(date - buy_dt[t]).days if buy_dt[t] else 0,
                    xu_ret_during=xu_ret_since_entry,
                    xu_trend_at_entry=(xu_now / xu_past - 1)*100 if xu_past else 0,
                ))
                cash += holdings[t]*pnow
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0
                continue

            if pnow >= ep * (1 + tp_pct):
                ret = (pnow - ep) / ep * 100
                xu_ret_since_entry = 0.0
                if len(xu100_series) and buy_dt[t] is not None:
                    xu_at_entry = xu100_series[xu100_series.index <= buy_dt[t]]
                    xu_at_exit  = xu100_series[xu100_series.index <= date]
                    if len(xu_at_entry) and len(xu_at_exit):
                        xu_ret_since_entry = (float(xu_at_exit.iloc[-1]) / float(xu_at_entry.iloc[-1]) - 1) * 100
                closed.append(dict(
                    ticker=t, buy_date=buy_dt[t], sell_date=date,
                    buy_px=ep, sell_px=pnow, ret=ret, typ='TP',
                    mdd=max_dd[t]*100, inc=False,
                    entry_rsi=entry_rsi[t], entry_vol_ratio=entry_vol_ratio[t],
                    entry_mom=entry_mom[t],
                    hold_days=(date - buy_dt[t]).days if buy_dt[t] else 0,
                    xu_ret_during=xu_ret_since_entry,
                    xu_trend_at_entry=(xu_now / xu_past - 1)*100 if xu_past else 0,
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

            if sig == -1 and holdings[t] > 0:
                ep = entry_px[t]; ret = (pnow - ep) / ep * 100
                inc = ret < 0 or max_dd[t] >= 0.05
                xu_ret_since_entry = 0.0
                if len(xu100_series) and buy_dt[t] is not None:
                    xu_at_entry = xu100_series[xu100_series.index <= buy_dt[t]]
                    xu_at_exit  = xu100_series[xu100_series.index <= date]
                    if len(xu_at_entry) and len(xu_at_exit):
                        xu_ret_since_entry = (float(xu_at_exit.iloc[-1]) / float(xu_at_entry.iloc[-1]) - 1) * 100
                closed.append(dict(
                    ticker=t, buy_date=buy_dt[t], sell_date=date,
                    buy_px=ep, sell_px=pnow, ret=ret, typ='SIG',
                    mdd=max_dd[t]*100, inc=inc,
                    entry_rsi=entry_rsi[t], entry_vol_ratio=entry_vol_ratio[t],
                    entry_mom=entry_mom[t],
                    hold_days=(date - buy_dt[t]).days if buy_dt[t] else 0,
                    xu_ret_during=xu_ret_since_entry,
                    xu_trend_at_entry=(xu_now / xu_past - 1)*100 if xu_past else 0,
                ))
                cash += holdings[t]*pnow
                holdings[t]=0; entry_px[t]=0; buy_dt[t]=None; max_dd[t]=0

            elif sig == 1 and holdings[t] == 0:
                invest = min(cash, port_val * alloc_pct)
                if invest >= 2000:
                    holdings[t] = invest / pnow
                    entry_px[t] = pnow
                    buy_dt[t]   = date
                    max_dd[t]   = 0
                    # Girişi kaydet
                    if t in indicators_map:
                        ind_at_entry = indicators_map[t]
                        date_row = ind_at_entry.reindex([date])
                        entry_rsi[t]       = float(date_row['rsi14'].iloc[0]) if not date_row['rsi14'].isna().all() else 0
                        mom_col = f'momentum5'
                        entry_mom[t]       = float(date_row[mom_col].iloc[0]) if mom_col in date_row and not date_row[mom_col].isna().all() else 0
                    if t in vol_data.columns:
                        v_t = vol_data[t].reindex([date])
                        v_avg = vol_data[t].rolling(20).mean().shift(1).reindex([date])
                        entry_vol_ratio[t] = float(v_t.iloc[0]) / float(v_avg.iloc[0]) if float(v_avg.iloc[0]) > 0 else 0
                    cash -= invest

    # Kalan pozisyonlar
    for t in tickers:
        if holdings[t] > 0:
            ep = entry_px[t]
            last_px = float(prices[t].dropna().iloc[-1])
            ret = (last_px - ep) / ep * 100
            inc = ret < 0 or max_dd[t] >= 0.05
            closed.append(dict(
                ticker=t, buy_date=buy_dt[t], sell_date=sim_dates[-1],
                buy_px=ep, sell_px=last_px, ret=ret, typ='LIQ',
                mdd=max_dd[t]*100, inc=inc,
                entry_rsi=entry_rsi[t], entry_vol_ratio=entry_vol_ratio[t],
                entry_mom=entry_mom[t],
                hold_days=(sim_dates[-1] - buy_dt[t]).days if buy_dt[t] else 0,
                xu_ret_during=0.0, xu_trend_at_entry=0.0,
            ))
            cash += holdings[t] * last_px

    final = cash
    total_ret = (final / INITIAL_CAPITAL - 1) * 100
    n   = len(closed)
    wr  = sum(1 for x in closed if x['ret'] > 0) / n * 100 if n else 0
    idr = sum(1 for x in closed if x['inc'])    / n * 100 if n else 0
    avg = np.mean([x['ret'] for x in closed]) if n else 0
    sl_c = sum(1 for x in closed if x['typ']=='SL')
    tp_c = sum(1 for x in closed if x['typ']=='TP')
    xu_ret = 0.0
    if xu100_prices is not None and len(xu100_prices):
        xu_s = xu100_prices[xu100_prices.index >= pd.to_datetime(sim_start)]
        if len(xu_s) > 1:
            xu_ret = (float(xu_s.iloc[-1]) / float(xu_s.iloc[0]) - 1) * 100

    return dict(total_ret=total_ret, final=final, n=n, wr=wr, idr=idr,
                avg=avg, alpha=total_ret-xu_ret, xu100=xu_ret,
                sl=sl_c, tp=tp_c, trades=closed)


# ─── HATA OTOPSİSİ ANALİZİ ───────────────────────────────

def analyze_failures(trades, fundamentals, prices, xu100_prices, sim_start):
    """
    Kaybedilen işlemleri sistematik analiz eder.
    Kazanç vs kayıp örüntülerini karşılaştırır.
    Filtre önerileri üretir.
    """
    ph("HATA OTOPSI ANALIZI")

    wins   = [t for t in trades if t['ret'] > 0]
    losses = [t for t in trades if t['ret'] <= 0]

    p(f"  Toplam islem : {len(trades)}", "c")
    p(f"  Kazancli     : {len(wins)} (%{len(wins)/len(trades)*100:.1f})", "g")
    p(f"  Zarar        : {len(losses)} (%{len(losses)/len(trades)*100:.1f})", "r")

    if not losses:
        p("  [Tebrikler! Hic zararli islem yok.]", "g")
        return {}

    # ─── 1. RSI GIRIS ANALIZI ─────────────────────────────
    ph("1) RSI Giris Degeri Analizi")
    w_rsi = [t['entry_rsi'] for t in wins   if t['entry_rsi'] > 0]
    l_rsi = [t['entry_rsi'] for t in losses if t['entry_rsi'] > 0]

    if w_rsi and l_rsi:
        pp(f"Kazananlar - RSI  Ort: {np.mean(w_rsi):.1f}  Std: {np.std(w_rsi):.1f}  "
           f"[{np.percentile(w_rsi,25):.0f} - {np.percentile(w_rsi,75):.0f}]")
        pp(f"Kaybedenler - RSI Ort: {np.mean(l_rsi):.1f}  Std: {np.std(l_rsi):.1f}  "
           f"[{np.percentile(l_rsi,25):.0f} - {np.percentile(l_rsi,75):.0f}]")

        # RSI bucket analizi
        pp("\nRSI araligina gore kazanc orani:")
        buckets = [(45,50),(50,55),(55,60),(60,65),(65,70),(70,75)]
        for lo, hi in buckets:
            in_bucket = [t for t in trades if lo <= t['entry_rsi'] < hi]
            if not in_bucket: continue
            wr_b = sum(1 for x in in_bucket if x['ret'] > 0) / len(in_bucket) * 100
            avg_b = np.mean([x['ret'] for x in in_bucket])
            c_  = "g" if wr_b >= 55 else "y" if wr_b >= 45 else "r"
            p(f"    RSI [{lo}-{hi}): N={len(in_bucket):3d}  Win={wr_b:.0f}%  "
              f"OrtRet={avg_b:+.1f}%", c_)

    # ─── 2. HACİM ORANI ANALİZİ ───────────────────────────
    ph("2) Hacim Orani Analizi (Giris Aninda V / 20G-Ort)")
    w_vr = [t['entry_vol_ratio'] for t in wins   if t['entry_vol_ratio'] > 0]
    l_vr = [t['entry_vol_ratio'] for t in losses if t['entry_vol_ratio'] > 0]

    if w_vr and l_vr:
        pp(f"Kazananlar - VolRatio  Ort: {np.mean(w_vr):.2f}  Medyan: {np.median(w_vr):.2f}")
        pp(f"Kaybedenler - VolRatio Ort: {np.mean(l_vr):.2f}  Medyan: {np.median(l_vr):.2f}")

        vol_buckets = [(1.0,1.5),(1.5,2.0),(2.0,3.0),(3.0,5.0),(5.0,99.9)]
        pp("\nHacim carpanina gore kazanc orani:")
        for lo, hi in vol_buckets:
            in_b = [t for t in trades if lo <= t['entry_vol_ratio'] < hi]
            if not in_b: continue
            wr_b = sum(1 for x in in_b if x['ret'] > 0) / len(in_b) * 100
            avg_b = np.mean([x['ret'] for x in in_b])
            c_  = "g" if wr_b >= 55 else "y" if wr_b >= 45 else "r"
            p(f"    VolRatio [{lo:.1f}-{hi:.1f}): N={len(in_b):3d}  Win={wr_b:.0f}%  "
              f"OrtRet={avg_b:+.1f}%", c_)

    # ─── 3. MOMENTUM ANALİZİ ──────────────────────────────
    ph("3) Giris Anindaki 5 Gunluk Momentum Analizi")
    w_mom = [t['entry_mom']*100 for t in wins   if abs(t['entry_mom']) < 1]
    l_mom = [t['entry_mom']*100 for t in losses if abs(t['entry_mom']) < 1]

    if w_mom and l_mom:
        pp(f"Kazananlar - Momentum  Ort: {np.mean(w_mom):+.1f}%  "
           f"Std: {np.std(w_mom):.1f}%")
        pp(f"Kaybedenler - Momentum Ort: {np.mean(l_mom):+.1f}%  "
           f"Std: {np.std(l_mom):.1f}%")

    # ─── 4. XU100 PİYASA REJİMİ ───────────────────────────
    ph("4) Piyasa Rejimi Analizi (XU100 Giris Anindaki 20G Trendi)")
    w_xu = [t['xu_trend_at_entry'] for t in wins]
    l_xu = [t['xu_trend_at_entry'] for t in losses]

    if w_xu and l_xu:
        pp(f"Kazananlar - XU100 trend:  Ort={np.mean(w_xu):+.1f}%  "
           f"Negatif={sum(1 for x in w_xu if x<0)}/{len(w_xu)}")
        pp(f"Kaybedenler - XU100 trend: Ort={np.mean(l_xu):+.1f}%  "
           f"Negatif={sum(1 for x in l_xu if x<0)}/{len(l_xu)}")

        xu_buckets = [(-20,-5),(-5,-1),(-1,0),(0,1),(1,5),(5,20)]
        pp("\nXU100 20g trendine gore kazanc orani:")
        for lo, hi in xu_buckets:
            in_b = [t for t in trades if lo <= t['xu_trend_at_entry'] < hi]
            if not in_b: continue
            wr_b = sum(1 for x in in_b if x['ret'] > 0) / len(in_b) * 100
            avg_b = np.mean([x['ret'] for x in in_b])
            c_  = "g" if wr_b >= 55 else "y" if wr_b >= 45 else "r"
            p(f"    XU100 trend [{lo:+.0f}% to {hi:+.0f}%): N={len(in_b):3d}  "
              f"Win={wr_b:.0f}%  OrtRet={avg_b:+.1f}%", c_)

    # ─── 5. ELDE TUTMA SÜRESİ ─────────────────────────────
    ph("5) Tutma Suresi Analizi")
    w_hd = [t['hold_days'] for t in wins]
    l_hd = [t['hold_days'] for t in losses]

    if w_hd and l_hd:
        pp(f"Kazananlar - Ort tutma suresi: {np.mean(w_hd):.0f} gun  "
           f"Medyan: {np.median(w_hd):.0f} gun")
        pp(f"Kaybedenler - Ort tutma suresi: {np.mean(l_hd):.0f} gun  "
           f"Medyan: {np.median(l_hd):.0f} gun")

    # ─── 6. SEKTOR ANALİZİ ────────────────────────────────
    ph("6) Sektor Bazli Basari Analizi")
    sector_stats = defaultdict(lambda: {"wins":0, "losses":0, "total_ret":0.0})
    for t in trades:
        ticker = t['ticker'] + '.IS'
        sector = fundamentals.get(ticker, {}).get('sector', 'Diger')
        if t['ret'] > 0:
            sector_stats[sector]['wins'] += 1
        else:
            sector_stats[sector]['losses'] += 1
        sector_stats[sector]['total_ret'] += t['ret']

    sec_rows = []
    for sector, s in sector_stats.items():
        n_s = s['wins'] + s['losses']
        wr_s = s['wins'] / n_s * 100
        sec_rows.append((sector, n_s, wr_s, s['total_ret']/n_s))
    sec_rows.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  {'Sektor':<35} {'N':>4} {'Win%':>6} {'OrtRet':>8}")
    print(f"  {'─'*35} {'─'*4} {'─'*6} {'─'*8}")
    for sector, n_s, wr_s, avg_r in sec_rows:
        c_ = "g" if wr_s >= 55 else "y" if wr_s >= 45 else "r"
        p(f"  {sector:<35} {n_s:>4} {wr_s:>6.1f}% {avg_r:>+8.1f}%", c_)

    # ─── 7. KAY'IN AYLAR ANALİZİ ──────────────────────────
    ph("7) Ay Bazli Basari Analizi")
    month_stats = defaultdict(lambda: {"wins":0,"losses":0,"total_ret":0.0})
    for t in trades:
        try:
            m = t['buy_date'].month if hasattr(t['buy_date'],'month') else pd.to_datetime(t['buy_date']).month
            month_stats[m]['wins' if t['ret']>0 else 'losses'] += 1
            month_stats[m]['total_ret'] += t['ret']
        except:
            pass

    month_names = {1:"Oca",2:"Sub",3:"Mar",4:"Nis",5:"May",6:"Haz",
                   7:"Tem",8:"Agu",9:"Eyl",10:"Eki",11:"Kas",12:"Ara"}
    print(f"\n  {'Ay':<6} {'N':>4} {'Win%':>6} {'OrtRet':>8}")
    for m in range(1, 13):
        if m not in month_stats: continue
        s = month_stats[m]
        n_m = s['wins'] + s['losses']
        wr_m = s['wins'] / n_m * 100
        avg_m = s['total_ret'] / n_m
        c_ = "g" if wr_m >= 55 else "y" if wr_m >= 45 else "r"
        p(f"  {month_names.get(m,str(m)):<6} {n_m:>4} {wr_m:>6.1f}% {avg_m:>+8.1f}%", c_)

    # ─── 8. FİLTRE ÖNERİSİ ────────────────────────────────
    ph("8) OTOMATIK FILTRE ONERILERI")
    filters_found = []

    # XU100 trend filtresi
    xu_neg = [t for t in trades if t['xu_trend_at_entry'] < -3]
    if xu_neg:
        wr_xu_neg = sum(1 for x in xu_neg if x['ret'] > 0) / len(xu_neg) * 100
        if wr_xu_neg < 45:
            suggestion = f"XU100 son 20 gun < -3%: Win orani {wr_xu_neg:.0f}% -> bu koşulda ALIM YAPMA"
            filters_found.append(suggestion)
            p(f"  [FiLTRE ONERISI] {suggestion}", "y")

    # RSI üst bandı
    rsi_high = [t for t in trades if t['entry_rsi'] > 68]
    if rsi_high:
        wr_rh = sum(1 for x in rsi_high if x['ret'] > 0) / len(rsi_high) * 100
        if wr_rh < 45:
            suggestion = f"Giris RSI > 68: Win orani {wr_rh:.0f}% -> RSI ust sınırı 68'e indir"
            filters_found.append(suggestion)
            p(f"  [FILTRE ONERISI] {suggestion}", "y")

    # Momentum negatif
    mom_neg = [t for t in trades if t['entry_mom'] < 0]
    if mom_neg:
        wr_mn = sum(1 for x in mom_neg if x['ret'] > 0) / len(mom_neg) * 100
        if wr_mn < 45:
            suggestion = f"5G momentum negatif: Win orani {wr_mn:.0f}% -> negatif momentumda ALIM YAPMA"
            filters_found.append(suggestion)
            p(f"  [FILTRE ONERISI] {suggestion}", "y")

    if not filters_found:
        p("  Otomatik filtre onerisi bulunamadi, mevcut parametreler iyi gorunuyor.", "g")

    return {"filters": filters_found}


# ─── VOLMOM FINE-TUNE ─────────────────────────────────────

def run_volmom_finetune(prices, vol_data, indicators_map, tickers, sim_start, xu100_prices):
    ph("VOLMOM INCE AYAR GRID SEARCH")

    results = []

    param_grid = list(itertools.product(
        [1.2, 1.3, 1.5, 1.7, 2.0],        # vm
        [40, 43, 45, 48, 50],              # rsi_min
        [70, 72, 74, 75, 78],              # rsi_max
        [3, 5, 7, 10],                     # mom_p
        [0.005, 0.01, 0.02],               # mom_min
        [None, 50, 100],                   # trend_sma
        [0, 1],                            # vol_sustain_days
        [-999, -3, 0, 3],                  # rsi_slope_min (3-gun)
        [False, True],                     # atr_filter
    ))

    total = len(param_grid)
    p(f"  Test edilecek kombinasyon sayisi: {total}", "y")
    done = 0

    for (vm, rmin, rmax, mom_p, mom_min, tsma, vsd, slope_min, atr_f) in param_grid:
        if rmax <= rmin: continue
        try:
            mx = make_volmom_signal(
                prices, vol_data, indicators_map, tickers,
                vm=vm, rsi_min=rmin, rsi_max=rmax,
                mom_p=mom_p, mom_min=mom_min,
                trend_sma=tsma, vol_sustain_days=vsd,
                rsi_slope_min=slope_min, rsi_slope_period=3,
                atr_filter=atr_f
            )
            if mx.empty: continue
            m = run_sim_detailed(mx, prices, vol_data, indicators_map,
                                 [t for t in tickers if t in mx.columns],
                                 sim_start, xu100_prices=xu100_prices)
            if m is None or m['n'] < 10: continue

            results.append(dict(
                vm=vm, rsi_min=rmin, rsi_max=rmax, mom_p=mom_p,
                mom_min=mom_min, trend_sma=tsma, vol_sustain=vsd,
                rsi_slope_min=slope_min, atr_filter=atr_f,
                total_ret=m['total_ret'], final=m['final'],
                n=m['n'], wr=m['wr'], idr=m['idr'],
                alpha=m['alpha'], xu100=m['xu100'],
                sl=m['sl'], tp=m['tp'], avg=m['avg']
            ))
        except Exception as e:
            pass

        done += 1
        if done % 500 == 0:
            top = sorted(results, key=lambda x: x['total_ret'], reverse=True)[:1]
            best_str = f"{top[0]['total_ret']:+.1f}%" if top else "-"
            p(f"  [{done}/{total}] En iyi su an: {best_str}", "m")

    if not results:
        p("Hic sonuc uretilmedi.", "r"); return [], None

    df = pd.DataFrame(results).sort_values('total_ret', ascending=False)

    ph("VOLMOM INCE AYAR - EN IYI 20")
    print(f"\n  {'#':<3} {'vm':>4} {'rmin':>5} {'rmax':>5} {'momP':>5} {'momMin':>7} "
          f"{'SMA':>5} {'VSD':>4} {'Slope':>6} {'ATR':>4} "
          f"{'Ret':>7} {'Alpha':>7} {'Win%':>5} {'IDR':>5} {'N':>5}")
    print(f"  {'-'*3} {'-'*4} {'-'*5} {'-'*5} {'-'*5} {'-'*7} "
          f"{'-'*5} {'-'*4} {'-'*6} {'-'*4} "
          f"{'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*5}")
    for i, row in df.head(20).iterrows():
        rank = df.index.get_loc(i) + 1
        c_  = "g" if row['total_ret'] > 80 else "y" if row['total_ret'] > 50 else "e"
        line = (f"  {rank:<3} {row['vm']:>4.1f} {row['rsi_min']:>5} {row['rsi_max']:>5} "
                f"{row['mom_p']:>5} {row['mom_min']:>7.3f} "
                f"{str(row['trend_sma']):>5} {row['vol_sustain']:>4} "
                f"{row['rsi_slope_min']:>6.0f} {'E' if row['atr_filter'] else 'H':>4} "
                f"{row['total_ret']:>+7.1f}% {row['alpha']:>+7.1f}% "
                f"{row['wr']:>5.1f}% {row['idr']:>5.1f}% {row['n']:>5}")
        p(line, c_)

    return df.to_dict('records'), df.iloc[0].to_dict()


# ─── ANA AKIŞ ─────────────────────────────────────────────

def main():
    ph("VOLMOM INCE AYAR + HATA OTOPSI MOTORU")

    raw_data   = load_cached_data()
    funds      = load_fundamentals()
    sim_start  = (datetime.now() - timedelta(days=SIM_DAYS)).strftime('%Y-%m-%d')

    tickers    = [k for k in raw_data.keys() if k != "_XU100"]
    xu100_px   = raw_data["_XU100"]['Close'] if "_XU100" in raw_data else None

    # Matrisler
    p("\n[Hazirlik] Matrisler olusturuluyor...", "y")
    prices_dict = {t: raw_data[t]['Close'] for t in tickers}
    vol_dict    = {t: raw_data[t]['Volume'] for t in tickers}
    prices  = pd.DataFrame(prices_dict).sort_index()
    volumes = pd.DataFrame(vol_dict).sort_index()

    p("[Hazirlik] Gostergeler hesaplaniyor...", "y")
    indicators_map = {}
    for i, t in enumerate(tickers):
        try:
            indicators_map[t] = compute_indicators(raw_data[t])
        except:
            pass
    ind_tickers = [t for t in tickers if t in indicators_map]
    p(f"  {len(ind_tickers)} hisse hazir.", "g")

    # FAZA 1: VOLMOM fine-tune (Eğer cache varsa oradan yükle, yoksa sıfırdan çalıştır)
    opt_file = Path("./data/volmom_optimized.json")
    if opt_file.exists():
        p("[Sistem] Mevcut optimizasyon cache verileri 'data/volmom_optimized.json' dosyasından okunuyor...", "g")
        with open(opt_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        all_results = cached_data.get("top20_results", [])
        best_params = cached_data.get("best_params", {})
    else:
        all_results, best_params = run_volmom_finetune(
            prices, volumes, indicators_map, ind_tickers, sim_start, xu100_px
        )

    # FAZA 2: En iyi parametrelerle TAM simülasyon + hata otopsisi
    if best_params:
        # Pandas DataFrame dönüşümünden kaynaklanan float64 tiplerini temizle
        tsma = best_params['trend_sma']
        if tsma is not None and not pd.isna(tsma):
            tsma = int(tsma)
            best_params['trend_sma'] = tsma
        else:
            tsma = None
            best_params['trend_sma'] = None

        best_params['rsi_min'] = int(best_params['rsi_min'])
        best_params['rsi_max'] = int(best_params['rsi_max'])
        best_params['mom_p'] = int(best_params['mom_p'])
        best_params['vol_sustain'] = int(best_params['vol_sustain'])
        best_params['rsi_slope_min'] = int(best_params['rsi_slope_min'])
        best_params['atr_filter'] = bool(best_params['atr_filter'])

        ph(f"EN IYI VOLMOM PARAMETRELERI ILE TAM SIMULASYON")
        pp(f"vm={best_params['vm']}  rsi=[{best_params['rsi_min']}-{best_params['rsi_max']}]  "
           f"mom_p={best_params['mom_p']}  mom_min={best_params['mom_min']:.3f}")
        pp(f"trend_sma={best_params['trend_sma']}  vol_sustain={best_params['vol_sustain']}  "
           f"rsi_slope_min={best_params['rsi_slope_min']}  atr_filter={best_params['atr_filter']}")

        best_mx = make_volmom_signal(
            prices, volumes, indicators_map, ind_tickers,
            vm=best_params['vm'], rsi_min=best_params['rsi_min'],
            rsi_max=best_params['rsi_max'], mom_p=best_params['mom_p'],
            mom_min=best_params['mom_min'], trend_sma=best_params['trend_sma'],
            vol_sustain_days=best_params['vol_sustain'],
            rsi_slope_min=best_params['rsi_slope_min'], rsi_slope_period=3,
            atr_filter=best_params['atr_filter']
        )
        best_m = run_sim_detailed(best_mx, prices, volumes, indicators_map,
                                  [t for t in ind_tickers if t in best_mx.columns],
                                  sim_start, xu100_prices=xu100_px)
        if best_m:
            p(f"\n  Getiri   : {best_m['total_ret']:+.2f}%", "g")
            p(f"  Alpha    : {best_m['alpha']:+.2f}%", "g")
            p(f"  Final    : {best_m['final']:,.0f} TL", "g")
            p(f"  Win Rate : %{best_m['wr']:.1f}", "c")
            p(f"  IDR      : %{best_m['idr']:.1f}", "c")
            p(f"  N Islem  : {best_m['n']}", "c")

            # FAZA 3: Hata otopsisi
            filter_suggestions = analyze_failures(
                best_m['trades'], funds, prices, xu100_px, sim_start
            )

            # FAZA 4: Baseline VOLMOM ile de karşılaştır
            ph("KARSILASTIRMA: En iyi vs Baseline VOLMOM(1.5, 45-75)")
            baseline_mx = make_volmom_signal(
                prices, volumes, indicators_map, ind_tickers,
                vm=1.5, rsi_min=45, rsi_max=75
            )
            baseline_m = run_sim_detailed(baseline_mx, prices, volumes, indicators_map,
                                          [t for t in ind_tickers if t in baseline_mx.columns],
                                          sim_start, xu100_prices=xu100_px)
            if baseline_m:
                print(f"\n  {'Metrik':<25} {'Baseline':>12} {'En Iyi':>12} {'Fark':>10}")
                print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*10}")
                metrics = [
                    ('Getiri', baseline_m['total_ret'], best_m['total_ret'], '%'),
                    ('Alpha', baseline_m['alpha'], best_m['alpha'], '%'),
                    ('Win Rate', baseline_m['wr'], best_m['wr'], '%'),
                    ('IDR', baseline_m['idr'], best_m['idr'], '%'),
                    ('N Islem', baseline_m['n'], best_m['n'], ''),
                    ('Ortalama Islem', baseline_m['avg'], best_m['avg'], '%'),
                ]
                for name, bv, ov, unit in metrics:
                    diff = ov - bv
                    c_diff = "g" if diff > 0 else "r"
                    print(f"  {name:<25} {bv:>11.2f}{unit} {ov:>11.2f}{unit}", end="")
                    p(f" {diff:>+10.2f}{unit}", c_diff)

            # KAYDET
            output = {
                "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else
                                    int(v) if isinstance(v, (np.integer, int)) else
                                    v) for k, v in best_params.items()},
                "best_metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else
                                     int(v) if isinstance(v, (np.integer, int)) else v)
                                 for k, v in best_m.items() if k != 'trades'},
                "filter_suggestions": filter_suggestions.get('filters', []),
                "top20_results": [
                    {k: (float(v) if isinstance(v, (np.floating, float)) else
                         int(v) if isinstance(v, (np.integer, int)) else v)
                     for k, v in r.items()} for r in all_results[:20]
                ],
                "trade_log": [
                    {k: (str(v) if isinstance(v, pd.Timestamp) else
                         v.strftime('%Y-%m-%d') if hasattr(v,'strftime') else v)
                     for k, v in t.items()} for t in best_m['trades']
                ]
            }
            with open("./data/volmom_optimized.json", "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            p("\n  Sonuclar 'data/volmom_optimized.json' kaydedildi.", "g")

    ph("TAMAMLANDI")

if __name__ == "__main__":
    main()
