# -*- coding: utf-8 -*-
import sys
import io
import os
import json
import pickle
import sqlite3
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from valuation_engine import calculate_rsi, get_dss_score, get_aggressive_score, get_usd_try_rates

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
        raise FileNotFoundError(f"Database not found at {db_path}. Please run app.py to seed the database first.")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    SELECT ticker, name, pb_ratio, roe, beta, eps_growth_5y, trailing_eps, debt_to_equity, sector, market 
    FROM stock_fundamentals
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
    print("      BIST FINTRACK - KÜRESEL PORTFÖY TRADING ROBOT SİMÜLASYONU (USD)      ")
    print(f"=========================================================================={CLR_RESET}")
    print(f"[1/5] Veritabanından küresel temel veriler yükleniyor...")
    
    try:
        db_funds = load_db_fundamentals()
        print(f"{CLR_GREEN}[OK] DB'den {len(db_funds)} şirket verisi başarıyla yüklendi.{CLR_RESET}")
    except Exception as e:
        print(f"{CLR_RED}[HATA] DB yüklenemedi: {e}{CLR_RESET}")
        return

    # Determine date range for last 365 days
    end_date = datetime.now()
    start_trading_date = end_date - timedelta(days=365)
    start_download_date = start_trading_date - timedelta(days=280) # indicator buffer
    
    s_str = start_download_date.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')

    print(f"\n[2/5] Dinamik faiz verileri (^IRX) indiriliyor...")
    try:
        irx_df = yf.download("^IRX", start=s_str, end=e_str, interval="1d", progress=False)
        irx_rates = irx_df['Close'].squeeze()
        print(f"{CLR_GREEN}[OK] Gecelik faiz oranları arşivi yüklendi.{CLR_RESET}")
    except Exception as e:
        print(f"{CLR_RED}[HATA] Faiz verileri indirilemedi, %4.5 sabit faiz fallback aktif ediliyor: {e}{CLR_RESET}")
        irx_rates = pd.Series(4.5, index=pd.date_range(s_str, e_str))

    print(f"\n[3/5] Dinamik döviz kuru verileri (USD/TRY) güncelleniyor...")
    try:
        usd_try_rates = get_usd_try_rates(s_str, e_str)
        print(f"{CLR_GREEN}[OK] Döviz kuru arşivi başarıyla güncellendi ve önbellekten okundu.{CLR_RESET}")
    except Exception as e:
        print(f"{CLR_RED}[HATA] Kur arşivi güncellenemedi: {e}{CLR_RESET}")
        return

    # Load BIST Price cache
    print(f"\n[4/5] BIST Fiyat Arşivi (all_prices.pkl) önbellekten okunuyor...")
    bist_cache_path = "./data/price_cache/all_prices.pkl"
    if not os.path.exists(bist_cache_path):
        print(f"{CLR_RED}[HATA] BIST fiyat önbellek dosyası bulunamadı: {bist_cache_path}{CLR_RESET}")
        return
    with open(bist_cache_path, "rb") as f:
        bist_prices = pickle.load(f)
    print(f"{CLR_GREEN}[OK] BIST fiyat arşivi yüklendi.{CLR_RESET}")

    # Load SP500 Prices
    sp500_tickers = [t for t, f in db_funds.items() if f["market"] == "SP500"]
    print(f"\n[5/5] S&P 500 hisse fiyatları indiriliyor (30 Dev Şirket)...")
    sp500_prices = {}
    for bi, ticker in enumerate(sp500_tickers):
        print(f"  ({bi+1}/30) {ticker} indiriliyor...")
        try:
            df = yf.download(ticker, start=s_str, end=e_str, interval="1d", progress=False)
            if not df.empty and len(df) >= 250:
                close = df['Close'].squeeze()
                vol = df['Volume'].squeeze()
                sp500_prices[ticker] = pd.DataFrame({'Close': close, 'Volume': vol})
        except Exception as e:
            print(f"    {CLR_RED}[UYARI] {ticker} indirilemedi: {e}{CLR_RESET}")

    # Combine BIST & S&P 500 universes
    all_tickers = []
    stock_data = {}
    
    # 1. Add BIST and convert prices to USD dynamically
    for t, f in db_funds.items():
        if f["market"] == "BIST" and t in bist_prices and not bist_prices[t].empty:
            df_try = bist_prices[t].copy()
            # Convert Close price to USD
            df_usd = pd.DataFrame(index=df_try.index)
            close_usd = []
            for d, val in df_try['Close'].items():
                d_str = d.strftime("%Y-%m-%d")
                rate = usd_try_rates.get(d_str, 32.5)
                close_usd.append(float(val) / rate)
            df_usd['Close'] = close_usd
            df_usd['Volume'] = df_try['Volume']
            
            stock_data[t] = df_usd
            all_tickers.append(t)
            
    # 2. Add S&P 500 directly in USD
    for t in sp500_tickers:
        if t in sp500_prices and not sp500_prices[t].empty:
            stock_data[t] = sp500_prices[t]
            all_tickers.append(t)

    print(f"\n{CLR_GREEN}[OK] Küresel Evren Kuruldu: {len(all_tickers)} Hisse (Dolar Bazlı Simülasyon){CLR_RESET}")

    # Align dates
    sample_ticker = sp500_tickers[0] if sp500_tickers else all_tickers[0]
    all_dates = pd.to_datetime(stock_data[sample_ticker].index)
    trading_dates = all_dates[all_dates >= pd.to_datetime(start_trading_date)]
    
    print(f"Simülasyon Dönemi: {trading_dates[0].strftime('%Y-%m-%d')} -> {trading_dates[-1].strftime('%Y-%m-%d')} ({len(trading_dates)} İşlem Günü)")

    # SIMULATION STATE (USD BASE)
    INITIAL_CAPITAL = 100_000.0   # $100,000 USD
    cash = INITIAL_CAPITAL
    holdings = {t: 0.0 for t in all_tickers}
    entry_prices = {t: 0.0 for t in all_tickers}
    peak_prices = {t: 0.0 for t in all_tickers}
    buy_dates = {t: None for t in all_tickers}
    max_holding_drawdown = {t: 0.0 for t in all_tickers}
    
    # Active Quant parameters
    BUY_SCORE_THRESHOLD = 60
    SELL_SCORE_THRESHOLD = 40
    ALLOCATION_PCT = 0.05         # 5% max allocation to allow 20 diversified positions
    TAKE_PROFIT_PCT = 0.25        # 25% take profit target
    
    trades = []
    repo_income_total = 0.0

    print(f"\n{CLR_YELLOW}{CLR_BOLD}--- KÜRESEL TRADING SIMÜLASYONU BAŞLIYOR (CANLI İŞLEM AKIŞI) ---{CLR_RESET}")
    print("-" * 110)
    print(f" {'Tarih':10} | {'Pazar':5} | {'Hisse':7} | {'İşlem Türü':11} | {'Fiyat':9} | {'Değişim':8} | {'SL Limit':8} | {'Portföy (USD)':13}")
    print("-" * 110)

    for date_idx, current_date in enumerate(trading_dates):
        # A. Calculate Daily Portfolio Value & Update cash with dynamic overnight repo yield
        portfolio_val = cash
        current_prices = {}
        d_str = current_date.strftime("%Y-%m-%d")
        
        # Calculate daily Fed Treasury Bill rate
        if current_date in irx_rates.index:
            try:
                annual_yield = float(irx_rates.loc[current_date].iloc[0])
            except Exception:
                try:
                    annual_yield = float(irx_rates.loc[current_date])
                except Exception:
                    annual_yield = 4.5
        else:
            prev_yield = irx_rates[irx_rates.index < current_date]
            annual_yield = float(prev_yield.iloc[-1]) if not prev_yield.empty else 4.5
            
        daily_interest = (annual_yield / 100.0) / 252.0
        interest_gained = cash * daily_interest
        cash += interest_gained
        repo_income_total += interest_gained
        portfolio_val += interest_gained
        
        for t in all_tickers:
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
        for t in all_tickers:
            if holdings[t] > 0.0 and current_prices[t] > peak_prices[t]:
                peak_prices[t] = current_prices[t]

        # B. Check positions for dynamic Stop-Loss & Take-Profit
        for t in all_tickers:
            if holdings[t] > 0.0:
                p_close = current_prices[t]
                if p_close <= 0.0:
                    continue
                    
                entry_p = entry_prices[t]
                ret_pct = (p_close - entry_p) / entry_p
                
                # Volatility-adjusted dynamic stop loss based on 14-day rolling returns std dev
                df_t = stock_data[t]
                idx_in_history = np.searchsorted(df_t.index, current_date, side='right') - 1
                
                # Dynamic volatility check
                if idx_in_history >= 0:
                    close_history = df_t['Close'].iloc[:idx_in_history+1]
                    if len(close_history) >= 15:
                        returns = close_history.pct_change().dropna().iloc[-14:]
                        volatility = returns.std()
                        # Map volatility to SL between 4% and 15%
                        dynamic_sl = max(0.04, min(0.15, volatility * 2.0))
                    else:
                        dynamic_sl = 0.08
                else:
                    dynamic_sl = 0.08
                
                p_peak = peak_prices[t]
                dd = (p_peak - p_close) / p_peak
                if dd > max_holding_drawdown[t]:
                    max_holding_drawdown[t] = dd
                    
                # 1. Dynamic Volatility-adjusted Stop-Loss Check
                if ret_pct <= -dynamic_sl:
                    cash_gained = holdings[t] * p_close
                    cash += cash_gained
                    trades.append({
                        'ticker': t.replace('.IS', ''),
                        'market': db_funds[t]["market"],
                        'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                        'sell_date': d_str,
                        'buy_price': entry_p,
                        'sell_price': p_close,
                        'return_pct': ret_pct * 100,
                        'type': 'STOP-LOSS'
                    })
                    print(f" {d_str} | {db_funds[t]['market']:5} | {t.replace('.IS', ''):7} | {CLR_RED}{CLR_BOLD}STOP-LOSS  {CLR_RESET} | {p_close:9.2f} | {CLR_RED}%{ret_pct*100:.1f}{CLR_RESET} | {CLR_YELLOW}%{dynamic_sl*100:.1f}{CLR_RESET} | {portfolio_val:11.2f} $")
                    holdings[t] = 0.0
                    entry_prices[t] = 0.0
                    buy_dates[t] = None
                    max_holding_drawdown[t] = 0.0
                    peak_prices[t] = 0.0
                    
                # 2. Take-Profit Check
                elif ret_pct >= TAKE_PROFIT_PCT:
                    cash_gained = holdings[t] * p_close
                    cash += cash_gained
                    trades.append({
                        'ticker': t.replace('.IS', ''),
                        'market': db_funds[t]["market"],
                        'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                        'sell_date': d_str,
                        'buy_price': entry_p,
                        'sell_price': p_close,
                        'return_pct': ret_pct * 100,
                        'type': 'TAKE-PROFIT'
                    })
                    print(f" {d_str} | {db_funds[t]['market']:5} | {t.replace('.IS', ''):7} | {CLR_GREEN}{CLR_BOLD}TAKE-PROFIT{CLR_RESET} | {p_close:9.2f} | {CLR_GREEN}%{ret_pct*100:.1f}{CLR_RESET} |    -     | {portfolio_val:11.2f} $")
                    holdings[t] = 0.0
                    entry_prices[t] = 0.0
                    buy_dates[t] = None
                    max_holding_drawdown[t] = 0.0
                    peak_prices[t] = 0.0

        # C. Daily Sectoral Stats calculation for Z-Scores
        fundamentals_on_day = {}
        for market_filter in ["BIST", "SP500"]:
            mkt_tickers = [x for x in all_tickers if db_funds[x]["market"] == market_filter]
            fundamentals_on_day[market_filter] = []
            
            for t in mkt_tickers:
                fund = db_funds[t]
                p_close = current_prices.get(t, 0.0)
                if p_close > 0:
                    eps_val = fund["trailing_eps"]
                    if fund["market"] == "BIST":
                        rate = usd_try_rates.get(d_str, 32.5)
                        eps_val = eps_val / rate
                    pb_ratio = fund["pb_ratio"]
                    pe_ratio = p_close / eps_val if eps_val > 0 else None
                    eveb_ratio = pe_ratio * 0.7 if (pe_ratio and pe_ratio < 100) else 15.0
                    
                    fundamentals_on_day[market_filter].append({
                        'sector': fund["sector"] or "Diğer",
                        'pe': pe_ratio if (pe_ratio and pe_ratio > 0) else None,
                        'pb': pb_ratio if (pb_ratio and pb_ratio > 0) else None,
                        'eveb': eveb_ratio
                    })
                    
        sector_stats_day = {}
        for market_filter in ["BIST", "SP500"]:
            df_funds_day = pd.DataFrame(fundamentals_on_day[market_filter])
            if not df_funds_day.empty:
                for sect, group in df_funds_day.groupby('sector'):
                    sector_stats_day[(market_filter, sect)] = {
                        'pe_mean': float(group['pe'].dropna().mean()) if not group['pe'].dropna().empty else 12.0,
                        'pe_std': float(group['pe'].dropna().std()) if (len(group['pe'].dropna()) > 1 and group['pe'].dropna().std() > 0.01) else 1.0,
                        'pb_mean': float(group['pb'].dropna().mean()) if not group['pb'].dropna().empty else 1.5,
                        'pb_std': float(group['pb'].dropna().std()) if (len(group['pb'].dropna()) > 1 and group['pb'].dropna().std() > 0.01) else 0.5,
                        'evebitda_mean': float(group['eveb'].dropna().mean()) if not group['eveb'].dropna().empty else 8.0,
                        'evebitda_std': float(group['eveb'].dropna().std()) if (len(group['eveb'].dropna()) > 1 and group['eveb'].dropna().std() > 0.01) else 2.0
                    }

        # D. Daily Z-Score valuation scoring
        daily_scores = {}
        for t in all_tickers:
            df_t = stock_data[t]
            if current_date not in df_t.index or current_prices[t] <= 0:
                continue
                
            p_close = current_prices[t]
            idx_in_history = df_t.index.get_loc(current_date)
            
            if idx_in_history < 200:
                continue
                
            fund = db_funds[t]
            sect_stats_day = sector_stats_day.get((fund["market"], fund["sector"] or "Diğer"), {
                'pe_mean': 12.0, 'pe_std': 1.0,
                'pb_mean': 1.5, 'pb_std': 0.5,
                'evebitda_mean': 8.0, 'evebitda_std': 2.0
            })
            
            # Convert BIST EPS to USD for consistent scoring
            eps_val = fund["trailing_eps"]
            if fund["market"] == "BIST":
                rate = usd_try_rates.get(d_str, 32.5)
                eps_val = eps_val / rate

            score = get_aggressive_score(
                price=p_close,
                eps=eps_val,
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
                market=fund["market"],
                sector_stats=sect_stats_day
            )
            daily_scores[t] = score

        # Trading Signals Execution
        sorted_opportunities = sorted(daily_scores.items(), key=lambda item: item[1], reverse=True)
        
        for t, score in sorted_opportunities:
            # Check for Sell Signal
            if holdings[t] > 0.0 and score <= SELL_SCORE_THRESHOLD:
                p_close = current_prices[t]
                entry_p = entry_prices[t]
                ret_pct = (p_close - entry_p) / entry_p
                cash_gained = holdings[t] * p_close
                cash += cash_gained
                trades.append({
                    'ticker': t.replace('.IS', ''),
                    'market': db_funds[t]["market"],
                    'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                    'sell_date': d_str,
                    'buy_price': entry_p,
                    'sell_price': p_close,
                    'return_pct': ret_pct * 100,
                    'type': 'SIGNAL'
                })
                print(f" {d_str} | {db_funds[t]['market']:5} | {t.replace('.IS', ''):7} | {CLR_MAGENTA}{CLR_BOLD}SİNYAL SAT  {CLR_RESET} | {p_close:9.2f} | {CLR_GREEN if ret_pct>=0 else CLR_RED}%{ret_pct*100:.1f}{CLR_RESET} |    -     | {portfolio_val:11.2f} $")
                holdings[t] = 0.0
                entry_prices[t] = 0.0
                buy_dates[t] = None
                max_holding_drawdown[t] = 0.0
                peak_prices[t] = 0.0

            # Buy Check
            elif holdings[t] == 0.0 and score >= BUY_SCORE_THRESHOLD:
                # Count current holdings
                active_pos_count = len([x for x in holdings.values() if x > 0.0])
                if active_pos_count >= 20: # max 20 positions
                    continue
                    
                # Market allocation check (max 50% BIST, max 50% S&P 500)
                mkt = db_funds[t]["market"]
                mkt_holdings_val = sum([holdings[x] * current_prices[x] for x in all_tickers if db_funds[x]["market"] == mkt])
                if mkt_holdings_val >= portfolio_val * 0.50:
                    continue
                    
                p_close = current_prices[t]
                investment = min(cash, portfolio_val * ALLOCATION_PCT)
                if investment >= 500.0:
                    shares = investment / p_close
                    cash -= investment
                    holdings[t] = shares
                    entry_prices[t] = p_close
                    peak_prices[t] = p_close
                    buy_dates[t] = current_date
                    max_holding_drawdown[t] = 0.0
                    print(f" {d_str} | {db_funds[t]['market']:5} | {t.replace('.IS', ''):7} | {CLR_CYAN}{CLR_BOLD}SİNYAL AL   {CLR_RESET} | {p_close:9.2f} |  -      |    -     | {portfolio_val:11.2f} $")

        # Live Progress indicator
        if date_idx % 20 == 0:
            print(f" >>> Gelişim Takip: {d_str} | Portföy Değeri: {portfolio_val:,.2f} $ | Gecelik Faiz Getirisi (^IRX): %{annual_yield:.2f}")

    # Finalize remaining open positions
    final_val = cash
    last_date = trading_dates[-1]
    for t in all_tickers:
        if holdings[t] > 0.0:
            p_close = current_prices[t]
            final_val += holdings[t] * p_close
            ret_pct = (p_close - entry_prices[t]) / entry_prices[t]
            trades.append({
                'ticker': t.replace('.IS', ''),
                'market': db_funds[t]["market"],
                'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                'sell_date': last_date.strftime('%Y-%m-%d'),
                'buy_price': entry_prices[t],
                'sell_price': p_close,
                'return_pct': ret_pct * 100,
                'type': 'FORCE_CLOSE'
            })

    win_trades = [t for t in trades if t['return_pct'] > 0]
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0.0
    total_return_pct = ((final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    print("-" * 110)
    print(f"\n{CLR_GREEN}{CLR_BOLD}Küresel Quant Portföy Simülasyonu Başarıyla Tamamlandı! Rapor Hesaplanıyor...{CLR_RESET}")
    print(f"{CLR_CYAN}{CLR_BOLD}==========================================================================")
    print("                     KÜRESEL QUANT PORTFÖY RAPORU (USD)                   ")
    print(f"=========================================================================={CLR_RESET}")
    print(f"  Başlangıç Kapitali             : {INITIAL_CAPITAL:,.2f} $")
    print(f"  Final Portföy Değeri           : {final_val:,.2f} $")
    
    ret_color = CLR_GREEN if total_return_pct >= 0 else CLR_RED
    print(f"  Toplam Net Dolar Getirisi      : {ret_color}%{total_return_pct:.2f}{CLR_RESET}")
    print(f"  Gecelik Repo/Para Piyasası Geliri: {CLR_GREEN}{repo_income_total:,.2f} ${CLR_RESET}")
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
        "repo_interest_earned": repo_income_total,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_ret,
        "trades": trades
    }
    
    out_file = "./data/backtest_results_global_portfolio.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f"[Rapor] Tüm detaylı küresel işlem logları '{out_file}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
