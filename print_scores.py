import os
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from backtest_simulation import load_db_fundamentals, get_dss_score

def analyze_scores():
    db_funds = load_db_fundamentals()
    test_tickers = [
        "THYAO.IS", "GARAN.IS", "EREGL.IS", "BIMAS.IS", "KCHOL.IS", 
        "TUPRS.IS", "ASELS.IS", "SASA.IS", "SISE.IS", "AKBNK.IS",
        "ASTOR.IS", "DOAS.IS"
    ]
    
    end_date = datetime.now()
    start_trading_date = end_date - timedelta(days=365)
    start_download_date = start_trading_date - timedelta(days=280)
    
    stock_data = {}
    for t in test_tickers:
        df = yf.download(t, start=start_download_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d", progress=False)
        if df.empty or len(df) < 250:
            continue
        close_series = df['Close'].squeeze()
        volume_series = df['Volume'].squeeze()
        stock_data[t] = pd.DataFrame({'Close': close_series, 'Volume': volume_series})
        
    valid_tickers = list(stock_data.keys())
    df_ref = stock_data[valid_tickers[0]]
    trading_dates = df_ref[df_ref.index >= pd.to_datetime(start_trading_date)].index
    
    print("Ticker | Max Score | Min Score | Avg Score | Last Price | Start Price")
    print("-" * 75)
    
    for t in valid_tickers:
        df_t = stock_data[t]
        fund = db_funds[t]
        scores = []
        prices = []
        
        for current_date in trading_dates:
            if current_date not in df_t.index:
                continue
            p_close = float(df_t.loc[current_date, 'Close'])
            idx_in_history = df_t.index.get_loc(current_date)
            if idx_in_history < 200:
                continue
                
            score = get_dss_score(
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
                idx=idx_in_history
            )
            scores.append(score)
            prices.append(p_close)
            
        if scores:
            print(f"{t.replace('.IS', ''):6} | {max(scores):9} | {min(scores):9} | {np.mean(scores):9.1f} | {prices[-1]:10.2f} | {prices[0]:11.2f}")

if __name__ == "__main__":
    analyze_scores()
