# -*- coding: utf-8 -*-
import sys
import io
import os
import json
import pickle
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from valuation_engine import calculate_rsi, get_dss_score, get_aggressive_score

# Set console encoding to UTF-8 for beautiful Turkish characters on Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Styles & Colors for premium terminal output
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_RED = "\033[91m"
CLR_MAGENTA = "\033[95m"
CLR_BOLD = "\033[1m"
CLR_RESET = "\033[0m"

def load_db_fundamentals():
    db_path = "./data/bist_fintrack.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Run app.py to seed first.")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    SELECT ticker, name, pb_ratio, roe, beta, eps_growth_5y, trailing_eps, debt_to_equity, sector, market 
    FROM stock_fundamentals WHERE market='BIST'
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    fundamentals = {}
    for r in rows:
        ticker = r[0]
        fundamentals[ticker] = {
            "name": r[1],
            "pb_ratio": r[2] or 1.5,
            "roe": r[3] or 0.25,
            "beta": r[4] or 1.0,
            "eps_growth_5y": r[5] or 25.0,
            "trailing_eps": r[6] or 5.0,
            "debt_to_equity": r[7] or 1.0,
            "sector": r[8] or "Diğer",
            "market": r[9]
        }
    return fundamentals

def main():
    print(f"{CLR_CYAN}{CLR_BOLD}==========================================================================")
    print("      BIST FINTRACK - TÜM BIST KAPSAMLI TRADING ROBOT SİMÜLASYONU        ")
    print(f"=========================================================================={CLR_RESET}")
    print(f"[1/4] Veritabanından şirket temel analiz verileri yükleniyor...")
    
    try:
        db_funds = load_db_fundamentals()
        print(f"{CLR_GREEN}[OK] DB'den {len(db_funds)} BIST şirketi başarıyla yüklendi.{CLR_RESET}")
    except Exception as e:
        print(f"{CLR_RED}[HATA] DB yüklenemedi: {e}{CLR_RESET}")
        return

    print(f"\n[2/4] Fiyat arşivi disk cache'inden (all_prices.pkl) okunuyor...")
    cache_path = "./data/price_cache/all_prices.pkl"
    if not os.path.exists(cache_path):
        print(f"{CLR_RED}[HATA] Fiyat cache dosyası bulunamadı: {cache_path}{CLR_RESET}")
        return
        
    with open(cache_path, "rb") as f:
        stock_data = pickle.load(f)
    print(f"{CLR_GREEN}[OK] {len(stock_data)} hissenin tarihsel fiyat serisi başarıyla yüklendi.{CLR_RESET}")

    # Determine valid BIST tickers
    valid_tickers = [t for t in db_funds.keys() if t in stock_data and not stock_data[t].empty]
    print(f"Aktif işlem taranacak BIST hisse sayısı: {len(valid_tickers)}")

    # Extract all trading dates from ASTOR or other liquid stock
    sample_ticker = "ASTOR.IS" if "ASTOR.IS" in valid_tickers else valid_tickers[0]
    all_dates = pd.to_datetime(stock_data[sample_ticker].index)
    
    # We take the last 365 days for trading simulation
    trading_dates = all_dates[all_dates >= (all_dates[-1] - timedelta(days=365))]
    print(f"Simülasyon Tarih Aralığı: {trading_dates[0].strftime('%Y-%m-%d')} -> {trading_dates[-1].strftime('%Y-%m-%d')} ({len(trading_dates)} İşlem Günü)")

    # SIMULATION PARAMETERS (HIGH FREQUENCY - TRADING BOT ACTIVE)
    INITIAL_CAPITAL = 1_000_000.0
    cash = INITIAL_CAPITAL
    holdings = {t: 0.0 for t in valid_tickers}
    entry_prices = {t: 0.0 for t in valid_tickers}
    peak_prices = {t: 0.0 for t in valid_tickers}
    buy_dates = {t: None for t in valid_tickers}
    max_holding_drawdown = {t: 0.0 for t in valid_tickers}

    # Highly active thresholds
    BUY_SCORE_THRESHOLD = 60      # Relaxed from 70
    SELL_SCORE_THRESHOLD = 40
    ALLOCATION_PCT = 0.05         # 5% max budget per trade to allow up to 20 parallel positions!
    STOP_LOSS_PCT = 0.08          # 8% tight stop loss
    TAKE_PROFIT_PCT = 0.25        # 25% take profit
    
    trades = []
    
    print(f"\n{CLR_YELLOW}{CLR_BOLD}[3/4] Trading Bot Simülasyonu Başlıyor... (İşlemler canlı olarak akacaktır){CLR_RESET}")
    print("-" * 100)
    print(f" {'Tarih':10} | {'Hisse':7} | {'İşlem Türü':11} | {'Fiyat':9} | {'Değişim':8} | {'Net Kar/Zarar (%)':17} | {'Bakiye (TL)':13}")
    print("-" * 100)

    # Simulation loop
    for date_idx, current_date in enumerate(trading_dates):
        # A. Update portfolio value
        portfolio_val = cash
        current_prices = {}
        
        for t in valid_tickers:
            df_t = stock_data[t]
            if current_date in df_t.index:
                p = float(df_t.loc[current_date, 'Close'])
                current_prices[t] = p
                portfolio_val += holdings[t] * p
            else:
                prev_rows = df_t[df_t.index < current_date]
                if not prev_rows.empty:
                    p = float(prev_rows.iloc[-1]['Close'])
                    current_prices[t] = p
                    portfolio_val += holdings[t] * p
                else:
                    current_prices[t] = 0.0

        # Update peak prices for trailing metrics
        for t in valid_tickers:
            if holdings[t] > 0.0 and current_prices[t] > peak_prices[t]:
                peak_prices[t] = current_prices[t]

        # B. Check active positions for TP/SL
        for t in valid_tickers:
            if holdings[t] > 0.0:
                p_close = current_prices[t]
                if p_close <= 0.0:
                    continue
                    
                entry_p = entry_prices[t]
                ret_pct = (p_close - entry_p) / entry_p
                
                # Drawdown from peak
                p_peak = peak_prices[t]
                dd = (p_peak - p_close) / p_peak
                if dd > max_holding_drawdown[t]:
                    max_holding_drawdown[t] = dd
                    
                # 1. Stop-Loss check (8% loss)
                if ret_pct <= -STOP_LOSS_PCT:
                    cash_gained = holdings[t] * p_close
                    cash += cash_gained
                    trades.append({
                        'ticker': t.replace('.IS', ''),
                        'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                        'sell_date': current_date.strftime('%Y-%m-%d'),
                        'buy_price': entry_p,
                        'sell_price': p_close,
                        'return_pct': ret_pct * 100,
                        'type': 'STOP-LOSS'
                    })
                    print(f" {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):7} | {CLR_RED}{CLR_BOLD}STOP-LOSS  {CLR_RESET} | {p_close:9.2f} | {CLR_RED}%{ret_pct*100:.1f}{CLR_RESET} | {cash_gained:11.2f} TL | {portfolio_val:11.2f} TL")
                    holdings[t] = 0.0
                    entry_prices[t] = 0.0
                    buy_dates[t] = None
                    max_holding_drawdown[t] = 0.0
                    peak_prices[t] = 0.0
                    
                # 2. Take-Profit check (25% profit)
                elif ret_pct >= TAKE_PROFIT_PCT:
                    cash_gained = holdings[t] * p_close
                    cash += cash_gained
                    trades.append({
                        'ticker': t.replace('.IS', ''),
                        'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                        'sell_date': current_date.strftime('%Y-%m-%d'),
                        'buy_price': entry_p,
                        'sell_price': p_close,
                        'return_pct': ret_pct * 100,
                        'type': 'TAKE-PROFIT'
                    })
                    print(f" {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):7} | {CLR_GREEN}{CLR_BOLD}TAKE-PROFIT{CLR_RESET} | {p_close:9.2f} | {CLR_GREEN}%{ret_pct*100:.1f}{CLR_RESET} | {cash_gained:11.2f} TL | {portfolio_val:11.2f} TL")
                    holdings[t] = 0.0
                    entry_prices[t] = 0.0
                    buy_dates[t] = None
                    max_holding_drawdown[t] = 0.0
                    peak_prices[t] = 0.0

        # C. Pre-calculate Daily Sector Stats for Z-Scores
        fundamentals_on_day = []
        for t in valid_tickers:
            fund = db_funds[t]
            p_close = current_prices.get(t, 0.0)
            if p_close > 0:
                eps_val = fund["trailing_eps"]
                pb_ratio = fund["pb_ratio"]
                pe_ratio = p_close / eps_val if eps_val > 0 else None
                eveb_ratio = pe_ratio * 0.7 if (pe_ratio and pe_ratio < 100) else 15.0
                
                fundamentals_on_day.append({
                    'sector': fund["sector"] or "Diğer",
                    'pe': pe_ratio if (pe_ratio and pe_ratio > 0) else None,
                    'pb': pb_ratio if (pb_ratio and pb_ratio > 0) else None,
                    'eveb': eveb_ratio
                })
        df_funds_day = pd.DataFrame(fundamentals_on_day)
        
        sector_stats_day = {}
        if not df_funds_day.empty:
            for sect, group in df_funds_day.groupby('sector'):
                sector_stats_day[sect] = {
                    'pe_mean': float(group['pe'].dropna().mean()) if not group['pe'].dropna().empty else 12.0,
                    'pe_std': float(group['pe'].dropna().std()) if (len(group['pe'].dropna()) > 1 and group['pe'].dropna().std() > 0.01) else 1.0,
                    'pb_mean': float(group['pb'].dropna().mean()) if not group['pb'].dropna().empty else 1.5,
                    'pb_std': float(group['pb'].dropna().std()) if (len(group['pb'].dropna()) > 1 and group['pb'].dropna().std() > 0.01) else 0.5,
                    'evebitda_mean': float(group['eveb'].dropna().mean()) if not group['eveb'].dropna().empty else 8.0,
                    'evebitda_std': float(group['eveb'].dropna().std()) if (len(group['eveb'].dropna()) > 1 and group['eveb'].dropna().std() > 0.01) else 2.0
                }

        # D. Signal Evaluation & Trading
        daily_scores = {}
        for t in valid_tickers:
            df_t = stock_data[t]
            if current_date not in df_t.index or current_prices[t] <= 0:
                continue
                
            p_close = current_prices[t]
            idx_in_history = df_t.index.get_loc(current_date)
            
            if idx_in_history < 200:
                continue
                
            fund = db_funds[t]
            sect_stats_day = sector_stats_day.get(fund["sector"] or "Diğer", {
                'pe_mean': 12.0, 'pe_std': 1.0,
                'pb_mean': 1.5, 'pb_std': 0.5,
                'evebitda_mean': 8.0, 'evebitda_std': 2.0
            })
            
            # Run get_aggressive_score for our active trading bot
            score = get_aggressive_score(
                price=p_close,
                eps=fund["trailing_eps"],
                bvps=p_close / fund["pb_ratio"],
                roe=fund["roe"],
                sector=fund["sector"],
                debt_to_equity=fund["debt_to_equity"],
                beta=fund["beta"],
                eps_growth_5y=fund["eps_growth_5y"],
                close_history=df_t['Close'],
                vol_history=df_t['Volume'],
                idx=idx_in_history,
                eps_growth_multiplier=1.5,
                market="BIST",
                sector_stats=sect_stats_day
            )
            daily_scores[t] = score

        # Sort by score descending to pick the best opportunities first
        sorted_opportunities = sorted(daily_scores.items(), key=lambda item: item[1], reverse=True)
        
        for t, score in sorted_opportunities:
            # Check if active positions sell check
            if holdings[t] > 0.0 and score <= SELL_SCORE_THRESHOLD:
                p_close = current_prices[t]
                entry_p = entry_prices[t]
                ret_pct = (p_close - entry_p) / entry_p
                cash_gained = holdings[t] * p_close
                cash += cash_gained
                trades.append({
                    'ticker': t.replace('.IS', ''),
                    'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                    'sell_date': current_date.strftime('%Y-%m-%d'),
                    'buy_price': entry_p,
                    'sell_price': p_close,
                    'return_pct': ret_pct * 100,
                    'type': 'SIGNAL'
                })
                print(f" {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):7} | {CLR_MAGENTA}{CLR_BOLD}SİNYAL SAT  {CLR_RESET} | {p_close:9.2f} | {CLR_GREEN if ret_pct>=0 else CLR_RED}%{ret_pct*100:.1f}{CLR_RESET} | {cash_gained:11.2f} TL | {portfolio_val:11.2f} TL")
                holdings[t] = 0.0
                entry_prices[t] = 0.0
                buy_dates[t] = None
                max_holding_drawdown[t] = 0.0
                peak_prices[t] = 0.0

            # Buy check
            elif holdings[t] == 0.0 and score >= BUY_SCORE_THRESHOLD:
                # Count current holdings
                active_pos_count = len([x for x in holdings.values() if x > 0.0])
                if active_pos_count >= 20: # max 20 positions
                    continue
                    
                p_close = current_prices[t]
                investment = min(cash, portfolio_val * ALLOCATION_PCT)
                if investment >= 1000.0: # min investment threshold
                    shares = investment / p_close
                    cash -= investment
                    holdings[t] = shares
                    entry_prices[t] = p_close
                    peak_prices[t] = p_close
                    buy_dates[t] = current_date
                    max_holding_drawdown[t] = 0.0
                    print(f" {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):7} | {CLR_CYAN}{CLR_BOLD}SİNYAL AL   {CLR_RESET} | {p_close:9.2f} |  -      | {investment:11.2f} TL | {portfolio_val:11.2f} TL")

        # Live Progress indicator (print every 20 days to show it's alive if no trades)
        if date_idx % 20 == 0:
            print(f" >>> Gelişim Takip: {current_date.strftime('%Y-%m-%d')} | Portföy Değeri: {portfolio_val:,.2f} TL | Nakit: {cash:,.2f} TL")

    # Finalize open positions
    final_val = cash
    last_date = trading_dates[-1]
    for t in valid_tickers:
        if holdings[t] > 0.0:
            p_close = current_prices[t]
            final_val += holdings[t] * p_close
            ret_pct = (p_close - entry_prices[t]) / entry_prices[t]
            trades.append({
                'ticker': t.replace('.IS', ''),
                'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                'sell_date': last_date.strftime('%Y-%m-%d'),
                'buy_price': entry_prices[t],
                'sell_price': p_close,
                'return_pct': ret_pct * 100,
                'type': 'FORCE_CLOSE'
            })

    # Save results to full bist JSON
    win_trades = [t for t in trades if t['return_pct'] > 0]
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0.0
    total_return_pct = ((final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print("-" * 100)
    print(f"\n{CLR_GREEN}{CLR_BOLD}[4/4] Trading Bot Simülasyonu Başarıyla Tamamlandı! Rapor Hesaplanıyor...{CLR_RESET}")
    print(f"{CLR_CYAN}{CLR_BOLD}==========================================================================")
    print("                     TRADING BOT PERFORMANS RAPORU                        ")
    print(f"=========================================================================={CLR_RESET}")
    print(f"  Başlangıç Kapitali             : {INITIAL_CAPITAL:,.2f} TL")
    print(f"  Final Portföy Değeri           : {final_val:,.2f} TL")
    
    ret_color = CLR_GREEN if total_return_pct >= 0 else CLR_RED
    print(f"  Toplam Net Portföy Getirisi    : {ret_color}%{total_return_pct:.2f}{CLR_RESET}")
    print(f"  Toplam Yapılan İşlem Adeti     : {len(trades)} adet")
    print(f"  Kazanma Oranı (Win Rate)       : %{win_rate:.2f}")
    
    avg_ret = np.mean([t['return_pct'] for t in trades]) if trades else 0.0
    avg_color = CLR_GREEN if avg_ret >= 0 else CLR_RED
    print(f"  Ortalama İşlem Getirisi        : {avg_color}%{avg_ret:.2f}{CLR_RESET}")
    print(f"{CLR_CYAN}{CLR_BOLD}=========================================================================={CLR_RESET}\n")

    results_payload = {
        "initial_capital": INITIAL_CAPITAL,
        "final_value": final_val,
        "total_return_pct": total_return_pct,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_ret,
        "trades": trades
    }
    
    out_file = "./data/backtest_results_full_bist.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f"[Rapor] Tüm detaylı işlem logları '{out_file}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
