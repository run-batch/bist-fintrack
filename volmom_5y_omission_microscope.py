# -*- coding: utf-8 -*-
"""
BIST FINTRACK - KAÇAN FIRSATLAR OTOPSİ MİKROSKOBU (OMISSION ERROR AUDIT)
========================================================================
5 yıllık BIST fiyat arşivi ve temel veriler üzerinde geriye dönük tarama yaparak:
- Gerçekleşen büyük yükselişleri (60 günde >= %40 artan) tespit eder.
- Algoritmamızın bu yükselişleri neden alım sinyaliyle yakalayamadığını analiz eder.
- Hangi koruma filtrelerimizin (Endeks trendi, RSI, Hacim, Kaldıraç, SMA100)
  kazanan işlemleri "haksız yere engellediğini" (Omission Error) ölçer.
"""

import os, json, sqlite3, time, pickle, warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

CACHE_FILE      = Path("./data/price_cache/all_prices_5y.pkl")
DB_FILE         = Path("./data/bist_fintrack.db")
SIM_YEARS       = 5

def col(c, txt):
    codes = {"g":"\033[92m","r":"\033[91m","y":"\033[93m",
             "c":"\033[96m","m":"\033[95m","b":"\033[1m","e":"\033[0m"}
    return f"{codes.get(c,'')}{txt}{codes['e']}"
def p(msg, c="e"):  print(col(c, msg))
def ph(msg):        print(col("c", f"\n{'='*68}\n  {msg}\n{'='*68}"))
def pp(msg):        print(col("b", f"  {msg}"))

# ─── VERİLERİ YÜKLE ───────────────────────────────────────

def load_cached_data():
    if not CACHE_FILE.exists():
        p("[Hata] 5 yillik cache dosyasi bulunamadi.", "r")
        return {}
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)

def load_fundamentals():
    if not DB_FILE.exists():
        return {}
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("SELECT ticker, sector, beta, market_cap, debt_to_equity FROM stock_fundamentals")
    rows = cur.fetchall()
    conn.close()
    
    funds = {}
    for r in rows:
        ticker = r[0]
        funds[ticker] = {
            "sector": r[1] or "Diger",
            "beta": r[2] if r[2] is not None else 1.0,
            "market_cap": r[3] if r[3] is not None else 0.0,
            "debt_to_equity": r[4] if r[4] is not None else 0.0
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

# ─── FIRSAT ANALİZİ ────────────────────────────────────────

def main():
    ph("BIST 5-YILLIK KAÇAN FIRSATLAR OTOPSİ ÇALIŞMASI (OMISSION ERROR AUDIT)")
    
    raw_data = load_cached_data()
    funds = load_fundamentals()
    
    tickers = [k for k in raw_data.keys() if k != "_XU100"]
    xu100_px = raw_data["_XU100"]['Close'] if "_XU100" in raw_data else None
    
    # Fiyat ve hacim matrisleri
    prices_dict = {t: raw_data[t]['Close'] for t in tickers}
    vol_dict    = {t: raw_data[t]['Volume'] for t in tickers}
    prices  = pd.DataFrame(prices_dict).sort_index()
    volumes = pd.DataFrame(vol_dict).sort_index()
    
    # Göstergeler
    indicators_map = {}
    for t in tickers:
        try: indicators_map[t] = compute_indicators(raw_data[t])
        except: pass
    ind_tickers = [t for t in tickers if t in indicators_map]
    
    # ─── 1. BÜYÜK YÜKSELİŞLERİN (FIRSATLARIN) TESPİTİ ───
    # 60 trading günü içinde >= %40 net yükseliş yaşayan dönemleri tara
    p("\n[1] 5 Yıllık geçmişte BIST hisselerinin yaşadığı tüm büyük ralli fırsatları taranıyor...", "y")
    
    surges = []
    for t in ind_tickers:
        c_series = prices[t].dropna()
        if len(c_series) < 120: continue
        
        # 60 gün sonraki getiri
        fwd_ret = c_series.shift(-60) / c_series - 1
        
        # Yerel düşük noktalarda yükseliş başlangıcı yakalamak için koşullar
        # fwd_ret >= 40% olan ve son 30 günün en düşüğüne yakın olan günleri ralli başlangıcı kabul et
        rolling_min = c_series.rolling(30).min()
        is_local_low = c_series <= rolling_min * 1.08
        
        opportunity_days = fwd_ret[(fwd_ret >= 0.40) & is_local_low].index
        
        # Üst üste binen günleri grupla ve her gruptaki ilk günü (breakout başlangıcını) al
        if len(opportunity_days) > 0:
            filtered_days = []
            last_day = None
            for d in opportunity_days:
                if last_day is None or (d - last_day).days > 45: # 45 gün tampon
                    filtered_days.append(d)
                    last_day = d
            
            for d in filtered_days:
                # 60 günlük gerçek zirve getiriyi bul
                idx_loc = c_series.index.get_loc(d)
                slice_60 = c_series.iloc[idx_loc:min(len(c_series), idx_loc+60)]
                peak_ret = (slice_60.max() / c_series.loc[d] - 1) * 100
                
                surges.append({
                    "ticker": t,
                    "date": d,
                    "price_at_start": float(c_series.loc[d]),
                    "peak_return_60d": float(peak_ret),
                    "sector": funds.get(t, {}).get("sector", "Diger"),
                    "debt_to_equity": funds.get(t, {}).get("debt_to_equity", 0.0),
                    "beta": funds.get(t, {}).get("beta", 1.0)
                })
                
    p(f"  Toplam Tespit Edilen Büyük Fırsat (N): {len(surges)} adet", "g")
    if not surges:
        p("[Uyarı] Hiç büyük yükseliş fırsatı tespit edilemedi.", "y")
        return
        
    df_surges = pd.DataFrame(surges)
    
    # ─── 2. ALGORİTMİK SİNYAL KONTROLÜ ───
    # Şampiyon VOLMOM Parametreleri
    best_params = {
        "vm": 1.2, "rsi_min": 43, "rsi_max": 70, "mom_p": 7, "mom_min": 0.01,
        "trend_sma": 100, "vol_sustain": 1, "rsi_slope_min": 0
    }
    
    from volmom_5y_backtest import make_volmom_signal
    # Filtresiz ham sinyaller
    raw_signals = make_volmom_signal(prices, volumes, indicators_map, ind_tickers, best_params, use_autopsy=False)
    
    # XU100 Trend serisi
    xu100_trend_series = (xu100_px / xu100_px.shift(20) - 1) * 100 if xu100_px is not None else pd.Series(0.0, index=prices.index)
    
    p("\n[2] Tespit edilen büyük fırsatların kaçırılma nedenleri analiz ediliyor...", "y")
    
    missed_reasons = []
    # Neden sayaçları
    no_core_signal = 0  # Hacim/Momentum algoritması hiç alım üretmedi
    blocked_by_filters = 0 # Algoritma alım üretti ama koruma filtreleri engelledi
    
    # Hangi filtrenin kaç kazananı engellediğini sayan sözlük
    filter_block_counts = {
        "XU100_Trend_Negatif": 0,
        "RSI_Eşik_Altı": 0,
        "Hacim_Artışı_Aşırı": 0,
        "SMA100_Altında": 0,
        "Kaldıraç_Limit_Üstü": 0
    }
    
    for idx, row in df_surges.iterrows():
        t = row['ticker']
        d = row['date']
        
        # Bu fırsatın başladığı tarihin ±5 gününde alım sinyali üretildi mi?
        start_search = d - timedelta(days=5)
        end_search   = d + timedelta(days=5)
        
        sig_slice = raw_signals.loc[start_search:end_search, t] if t in raw_signals.columns else pd.Series()
        signal_generated = (sig_slice == 1).any()
        
        if not signal_generated:
            # Algoritmanın kendi çekirdeği (hacim patlaması + momentum) bu hisseyi fark edemedi
            no_core_signal += 1
            missed_reasons.append("Çekirdek Algoritma Sinyal Üretmedi (Yetersiz Hacim/Momentum Kırılımı)")
        else:
            # Sinyal üretildi ama hangi filtre(ler) engelledi?
            blocked_by_filters += 1
            blocks = []
            
            # 1. XU100 Filtresi (Trend < -3%)
            xu_trend = float(xu100_trend_series.loc[d]) if d in xu100_trend_series.index else 0.0
            if xu_trend < -3.0:
                blocks.append("XU100 Düşüş Filtresi (Endeks Trendi < -%3)")
                filter_block_counts["XU100_Trend_Negatif"] += 1
                
            # 2. RSI Eşiği (RSI < 60 ise alımı engeller)
            ind = indicators_map[t]
            rsi_val = float(ind.loc[d, 'rsi14']) if d in ind.index else 50.0
            if rsi_val < 60.0:
                blocks.append("RSI Sınırı (RSI < 60)")
                filter_block_counts["RSI_Eşik_Altı"] += 1
                
            # 3. Hacim Tavanı (Hacim Çarpanı > 4x)
            v_day = float(volumes.loc[d, t]) if d in volumes.index else 0.0
            v_avg = float(volumes[t].rolling(20).mean().shift(1).loc[d]) if d in volumes.index else 1.0
            vol_ratio = v_day / v_avg if v_avg > 0 else 1.0
            if vol_ratio > 4.0:
                blocks.append("Aşırı Hacim Filtresi (VolRatio > 4x)")
                filter_block_counts["Hacim_Artışı_Aşırı"] += 1
                
            # 4. SMA100 Altında Kalma
            sma_val = float(ind.loc[d, 'sma100']) if d in ind.index else 0.0
            px_val = float(prices.loc[d, t])
            if px_val < sma_val:
                blocks.append("SMA100 Altı Filtresi (Düşüş Trendi)")
                filter_block_counts["SMA100_Altında"] += 1
                
            # 5. Kaldıraç (Borç/Özsermaye >= 2.0)
            debt = row['debt_to_equity']
            if debt >= 2.0:
                blocks.append("Kaldıraç Engeli (Borç/Özsermaye >= 2.0)")
                filter_block_counts["Kaldıraç_Limit_Üstü"] += 1
                
            if not blocks:
                # Aslında sinyal filtrelenmedi, muhtemelen portföyde nakit kalmadığı için alınamadı (Kapital sınırı)
                missed_reasons.append("Kapital Limiti / Alokasyon Doluluğu (Nakit yetersizliği)")
            else:
                missed_reasons.append(f"Filtre Engeli: {', '.join(blocks)}")
                
    df_surges['missed_reason'] = missed_reasons
    
    # ─── 3. ANALİZ RAPORLAMASI ──────────────────────────────────
    ph("KAÇAN FIRSATLAR OTOPSİ ANALİZ RAPORU (5 YILLIK DETAY)")
    
    p(f"  Toplam Fırsat İşlemi (N)            : {len(df_surges)} adet")
    p(f"  Fırsatların Ort. Rallileşme Gücü   : %{df_surges['peak_return_60d'].mean():.1f} getiri")
    p(f"  Çekirdek Algoritmanın Fark Etmediği : {no_core_signal} adet (%{no_core_signal/len(df_surges)*100:.1f})", "r")
    p(f"  Filtrelerin Korumak Adına Elediği   : {blocked_by_filters} adet (%{blocked_by_filters/len(df_surges)*100:.1f})", "y")
    
    # Filtrelerin engelleme istatistikleri
    ph("Hangi Koruma Filtresi Kaç Tane Yükselen Şampiyonu Engelledi?")
    for filter_name, count in sorted(filter_block_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(df_surges) * 100
        p(f"  - {filter_name:<22} : {count:3d} adet kazanan hisse elendi (%{pct:.1f})", "y")
        
    p("\n  *Analitik Yorum: Bu veriler koruma filtrelerinin getirdiği Omission Error (Fırsat Kaçırma) maliyetidir.", "c")
    p("   Hata oranını (IDR) %43'ten %38'e çekmek için koyduğumuz her kural, bu kazananların filtrelenmesine yol açtı.", "c")
    
    # Sektörel olarak en çok hangi sektörlerde yükseliş kaçırdık?
    ph("Kaçan Fırsatların Sektörel Dağılımı")
    sec_counts = df_surges['sector'].value_counts()
    for sec, count in sec_counts.items():
        pct = count / len(df_surges) * 100
        p(f"  - {sec:<30} : {count:3d} adet yükseliş kaçtı (%{pct:.1f})", "e")

    # En yüksek getiriyle kaçırdığımız ilk 5 fırsat
    ph("Kaçırılan En Devasa 5 Yükseliş Fırsatı")
    top_5 = df_surges.sort_values(by='peak_return_60d', ascending=False).head(5)
    print(f"\n  {'Hisse':<10} {'Tarih':<12} {'Sektör':<25} {'60g Getiri %':>12} {'Kaçırma Nedeni'}")
    print(f"  {'─'*10} {'─'*12} {'─'*25} {'─'*12} {'─'*35}")
    for _, row in top_5.iterrows():
        print(f"  {row['ticker']:<10} {row['date'].strftime('%Y-%m-%d'):<12} {row['sector']:<25} {row['peak_return_60d']:>11.1f}%  {row['missed_reason']}")

    # Sonuçları JSON olarak kaydet
    output = {
        "total_missed_opportunities": len(df_surges),
        "average_surge_return_pct": df_surges['peak_return_60d'].mean(),
        "unnoticed_by_core_pct": no_core_signal/len(df_surges)*100,
        "blocked_by_filters_pct": blocked_by_filters/len(df_surges)*100,
        "filter_block_stats": {k: int(v) for k, v in filter_block_counts.items()},
        "top_5_missed_opportunities": top_5.to_dict(orient='records')
    }
    with open("./data/volmom_5y_omission_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    p("\n  Kaçan Fırsatlar Mikroskobu Raporu './data/volmom_5y_omission_results.json' adresine kaydedildi.", "g")

if __name__ == "__main__":
    main()
