import os
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Style colors for premium console output
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_RED = "\033[91m"
CLR_MAGENTA = "\033[95m"
CLR_BOLD = "\033[1m"
CLR_RESET = "\033[0m"

from valuation_engine import calculate_rsi, get_dss_score, get_aggressive_score


def load_db_fundamentals():
    """Loads current fundamental parameters for seeded stocks from the local SQLite database."""
    db_path = "./data/bist_fintrack.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run app.py to seed the database first.")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query details
    query = """
    SELECT ticker, name, pe_ratio, pb_ratio, ev_ebitda, dividend_yield, 
           roe, market_cap, beta, eps_growth_5y, trailing_eps, debt_to_equity, sector, market 
    FROM stock_fundamentals
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    
    fundamentals = {}
    for r in rows:
        ticker = r[0]
        fundamentals[ticker] = {
            "name": r[1],
            "pe_ratio": r[2] or 12.0,
            "pb_ratio": r[3] or 1.5,
            "ev_ebitda": r[4],
            "dividend_yield": r[5],
            "roe": r[6] or 0.25,
            "market_cap": r[7],
            "beta": r[8] or 1.0,
            "eps_growth_5y": r[9] or 25.0,
            "trailing_eps": r[10] or 1.0,
            "debt_to_equity": r[11] or 1.0,
            "sector": r[12] or "Diğer",
            "market": r[13] if len(r) > 13 else "BIST"
        }
        
    conn.close()
    return fundamentals


def run_single_backtest_scenario(config, db_funds, valid_tickers, stock_data, xu100, trading_dates, start_trading_date, market="BIST"):
    """Runs a single portfolio backtest simulation using the provided configuration parameters."""
    initial_capital = 100000.0  # 100,000 TRY or USD
    cash = initial_capital
    holdings = {t: 0.0 for t in valid_tickers}
    entry_prices = {t: 0.0 for t in valid_tickers}
    buy_dates = {t: None for t in valid_tickers}
    max_holding_drawdown = {t: 0.0 for t in valid_tickers}
    peak_prices = {t: 0.0 for t in valid_tickers} # Track peak price achieved since purchase
    curr = "$" if market == "SP500" else "TL"
    
    # Parameters from config
    BUY_SCORE_THRESHOLD = config["buy_threshold"]
    SELL_SCORE_THRESHOLD = config["sell_threshold"]
    STOP_LOSS_PCT = config["stop_loss_pct"]
    TAKE_PROFIT_PCT = config["take_profit_pct"]
    ALLOCATION_PCT = config["allocation_pct"]
    
    tr_bond_yield = config["tr_bond_yield"]
    value_weight = config["value_weight"]
    momentum_weight = config["momentum_weight"]
    eps_growth_multiplier = config["eps_growth_multiplier"]
    
    # Extract learning/adaptive flags from configuration
    TRAILING_STOP = config.get("trailing_stop", False)
    DYNAMIC_TP = config.get("dynamic_tp", False)
    INDEX_SHIELD = config.get("index_shield", False)
    ADAPTIVE_ALLOC = config.get("adaptive_alloc", False)
    
    closed_trades = []
    daily_portfolio_value = []
    
    try:
        print(f"\n{CLR_CYAN}{CLR_BOLD}--- [SİMÜLASYON BAŞLADI: {config['name'].upper()} ({market})] ---{CLR_RESET}")
    except UnicodeEncodeError:
        # Strip non-ASCII characters / emojis for safe console printing on Windows terminals
        safe_name = config['name'].replace("✨ ", "").replace("⚡ ", "").replace("🏆 ", "")
        # replace Turkish characters to be extremely safe if CP1254 maps fail
        safe_name = safe_name.replace("ı", "i").replace("Ş", "S").replace("ş", "s").replace("ğ", "g").replace("Ğ", "G").replace("ü", "u").replace("Ü", "U").replace("ö", "o").replace("Ö", "O").replace("ç", "c").replace("Ç", "C")
        print(f"\n{CLR_CYAN}{CLR_BOLD}--- [SIMULASYON BASLADI: {safe_name.upper()} ({market})] ---{CLR_RESET}")
    
    # Day-by-Day Simulation Loop
    for date_idx, current_date in enumerate(trading_dates):
        # A. Calculate total portfolio value at the start of the day
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
                    
        daily_portfolio_value.append({'date': current_date, 'value': portfolio_val})
        
        # A.2. Calculate Index Trend if INDEX_SHIELD is active
        index_trend_neg = False
        if INDEX_SHIELD:
            if current_date in xu100.index:
                idx_idx = xu100.index.get_loc(current_date)
                if idx_idx >= 20:
                    prev_index_prices = xu100['Close'].iloc[idx_idx-20 : idx_idx+1]
                    index_sma20 = prev_index_prices.mean()
                    index_cur = float(xu100['Close'].iloc[idx_idx])
                    if index_cur < index_sma20:
                        index_trend_neg = True

        # A.3. Pre-calculate Daily Scores for all tickers on this day
        # Compute dynamic sectoral statistics on this day to avoid look-ahead bias
        fundamentals_on_day = []
        for t in valid_tickers:
            fund = db_funds[t]
            p_close = current_prices.get(t, 0.0)
            if p_close > 0:
                eps_val = fund["trailing_eps"]
                pb_ratio = fund["pb_ratio"] or 1.5
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
                
        daily_scores = {}
        for t in valid_tickers:
            df_t = stock_data[t]
            if current_date not in df_t.index or current_prices[t] <= 0:
                continue
                
            p_close = current_prices[t]
            idx_in_history = df_t.index.get_loc(current_date)
            
            # Ensure we have enough history to calculate SMA200
            if idx_in_history < 200:
                continue
                
            fund = db_funds[t]
            sect_stats_day = sector_stats_day.get(fund["sector"] or "Diğer", {
                'pe_mean': 12.0, 'pe_std': 1.0,
                'pb_mean': 1.5, 'pb_std': 0.5,
                'evebitda_mean': 8.0, 'evebitda_std': 2.0
            })
            
            # Reconstruct Daily Zeka Skoru with custom strategy params and daily sector stats
            if config["id"] in ["aggressive_ai", "autopsy_volmom"]:
                score = get_aggressive_score(
                    price=p_close,
                    eps=fund["trailing_eps"],
                    bvps=p_close / (fund["pb_ratio"] or 1.5),
                    roe=fund["roe"],
                    sector=fund["sector"],
                    debt_to_equity=fund["debt_to_equity"],
                    beta=fund["beta"],
                    eps_growth_5y=fund["eps_growth_5y"],
                    close_history=df_t['Close'],
                    vol_history=df_t['Volume'],
                    idx=idx_in_history,
                    eps_growth_multiplier=eps_growth_multiplier,
                    market=market,
                    sector_stats=sect_stats_day
                )
            else:
                score = get_dss_score(
                    price=p_close,
                    eps=fund["trailing_eps"],
                    bvps=p_close / (fund["pb_ratio"] or 1.5),
                    roe=fund["roe"],
                    sector=fund["sector"],
                    debt_to_equity=fund["debt_to_equity"],
                    beta=fund["beta"],
                    eps_growth_5y=fund["eps_growth_5y"],
                    close_history=df_t['Close'],
                    vol_history=df_t['Volume'],
                    idx=idx_in_history,
                    tr_bond_yield=tr_bond_yield,
                    value_weight=value_weight,
                    momentum_weight=momentum_weight,
                    eps_growth_multiplier=eps_growth_multiplier,
                    market=market,
                    sector_stats=sect_stats_day
                )
            daily_scores[t] = score

        # Update peak prices for active holdings
        for t in valid_tickers:
            if holdings[t] > 0.0 and current_prices[t] > peak_prices[t]:
                peak_prices[t] = current_prices[t]
        
        # B. Check Stop-Loss and Take-Profit for active holdings
        for t in valid_tickers:
            if holdings[t] > 0.0:
                p_close = current_prices[t]
                if p_close <= 0.0:
                    continue
                entry_p = entry_prices[t]
                score = daily_scores.get(t, 50)
                
                # Update Max Drawdown while holding
                if p_close < entry_p:
                    drawdown = (entry_p - p_close) / entry_p
                    max_holding_drawdown[t] = max(max_holding_drawdown[t], drawdown)
                
                # Sektörel Stop-Loss Kalibrasyonu
                fund = db_funds[t]
                sec_lower = fund.get("sector", "Diğer").lower()
                current_sl_pct = STOP_LOSS_PCT
                if config["id"] in ["autopsy_volmom", "aggressive_ai"]:
                    if any(x in sec_lower for x in ["teknoloji", "yazılım", "tech", "software", "enerji", "energy"]):
                        current_sl_pct = 0.08  # Tighten standard SL for volatile assets
                
                # Determine Stop Price (Trailing vs. Entry-based)
                if TRAILING_STOP:
                    stop_price = peak_prices[t] * (1 - current_sl_pct)
                else:
                    stop_price = entry_p * (1 - current_sl_pct)
                
                # Check Stop-Loss
                if p_close <= stop_price:
                    cash_gained = holdings[t] * p_close
                    cash += cash_gained
                    
                    ret_pct = (p_close - entry_p) / entry_p
                    closed_trades.append({
                        'ticker': t.replace('.IS', ''),
                        'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                        'buy_price': entry_p,
                        'sell_date': current_date.strftime('%Y-%m-%d'),
                        'sell_price': p_close,
                        'return_pct': ret_pct * 100,
                        'type': 'STOP-LOSS',
                        'max_drawdown': max_holding_drawdown[t] * 100,
                        'incorrect_decision': (ret_pct < 0.0)
                    })
                    
                    print(f"  [{CLR_RED}STOP-LOSS{CLR_RESET}] {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):6} satıldı! Fiyat: {p_close:7.2f} {curr} (Maliyet: {entry_p:.2f} {curr}, Net: %{ret_pct*100:.1f})")
                    holdings[t] = 0.0
                    entry_prices[t] = 0.0
                    buy_dates[t] = None
                    max_holding_drawdown[t] = 0.0
                    peak_prices[t] = 0.0
                    continue
                    
                # Check Take-Profit
                elif p_close >= entry_p * (1 + TAKE_PROFIT_PCT):
                    if DYNAMIC_TP and score >= 68:
                        # Yapay Zeka skoru çok yüksek olduğu için kazananın koşmasına izin ver, kâr alma!
                        continue
                        
                    cash_gained = holdings[t] * p_close
                    cash += cash_gained
                    
                    ret_pct = (p_close - entry_p) / entry_p
                    closed_trades.append({
                        'ticker': t.replace('.IS', ''),
                        'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                        'buy_price': entry_p,
                        'sell_date': current_date.strftime('%Y-%m-%d'),
                        'sell_price': p_close,
                        'return_pct': ret_pct * 100,
                        'type': 'TAKE-PROFIT',
                        'max_drawdown': max_holding_drawdown[t] * 100,
                        'incorrect_decision': False
                    })
                    
                    print(f"  [{CLR_GREEN}TAKE-PROFIT{CLR_RESET}] {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):6} satıldı! Fiyat: {p_close:7.2f} {curr} (Maliyet: {entry_p:.2f} {curr}, Net: %{ret_pct*100:.1f})")
                    holdings[t] = 0.0
                    entry_prices[t] = 0.0
                    buy_dates[t] = None
                    max_holding_drawdown[t] = 0.0
                    peak_prices[t] = 0.0
                    continue
        
        # C. Re-evaluate Score Signals (Only perform 1 action per stock per day)
        for t in valid_tickers:
            if current_prices[t] <= 0 or t not in daily_scores:
                continue
                
            p_close = current_prices[t]
            score = daily_scores[t]
            
            fund = db_funds[t]
            sec_lower = fund.get("sector", "Diğer").lower()
            
            # Sektörel SL Belirle (IDR kontrolü için)
            current_sl_pct = STOP_LOSS_PCT
            if config["id"] in ["autopsy_volmom", "aggressive_ai"]:
                if any(x in sec_lower for x in ["teknoloji", "yazılım", "tech", "software", "enerji", "energy"]):
                    current_sl_pct = 0.08
            
            # DECISION RULE:
            # 1. SELL: Hold stock, and score <= SELL_SCORE_THRESHOLD
            if holdings[t] > 0.0 and score <= SELL_SCORE_THRESHOLD:
                entry_p = entry_prices[t]
                cash_gained = holdings[t] * p_close
                cash += cash_gained
                
                ret_pct = (p_close - entry_p) / entry_p
                # Robust IDR threshold logic: must close in net loss OR trigger SL limit drawdown
                incorrect = (ret_pct < 0.0) or (max_holding_drawdown[t] >= current_sl_pct)
                
                closed_trades.append({
                    'ticker': t.replace('.IS', ''),
                    'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                    'buy_price': entry_p,
                    'sell_date': current_date.strftime('%Y-%m-%d'),
                    'sell_price': p_close,
                    'return_pct': ret_pct * 100,
                    'type': 'SIGNAL',
                    'max_drawdown': max_holding_drawdown[t] * 100,
                    'incorrect_decision': incorrect
                })
                
                log_color = CLR_GREEN if ret_pct >= 0 else CLR_RED
                print(f"  [SİNYAL SATIŞ] {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):6} satıldı! Fiyat: {p_close:7.2f} {curr} (Zeka Skoru: {score:2}, Getiri: {log_color}%{ret_pct*100:.1f}{CLR_RESET}, Max Geri: %{max_holding_drawdown[t]*100:.1f})")
                
                holdings[t] = 0.0
                entry_prices[t] = 0.0
                buy_dates[t] = None
                max_holding_drawdown[t] = 0.0
                peak_prices[t] = 0.0
                
            # 2. BUY: Do not hold stock, cash is available, and score >= BUY_SCORE_THRESHOLD (Index-Shield-Adjusted)
            elif holdings[t] == 0.0:
                # Sektörel İnce Ayarlar (Dynamic Sector Calibration)
                sector_buy_adjust = 0
                if config["id"] in ["autopsy_volmom", "aggressive_ai"]:
                    # Kararlı ve Trendi Sağlam Sektörler
                    if any(x in sec_lower for x in ["savunma", "telekom", "otomotiv", "defense", "telecom", "automotive"]):
                        sector_buy_adjust = -2 # Barajı esnet
                    # Volatilitesi ve Hata Riski Yüksek Sektörler
                    elif any(x in sec_lower for x in ["teknoloji", "yazılım", "tech", "software", "enerji", "energy"]):
                        sector_buy_adjust = 3  # Barajı sıkılaştır
                    # GYO ve Holding iştirak NAV iskontolu yapılar
                    elif any(x in sec_lower for x in ["gyo", "reit", "holding"]):
                        sector_buy_adjust = 2  # Barajı yükselt
                
                current_buy_threshold = BUY_SCORE_THRESHOLD + sector_buy_adjust
                if index_trend_neg:
                    current_buy_threshold += 5 # Downtrend kalkanı
                    
                # Patern Algılama Kalkanları (Pattern Detection Shields)
                pattern_blocked = False
                if config["id"] in ["autopsy_volmom", "aggressive_ai"]:
                    # 1. Hacim Tükeniş Kalkanı (Blowoff Volume Check)
                    df_t = stock_data[t]
                    idx_in_history = df_t.index.get_loc(current_date)
                    vol_so_far = df_t['Volume'].iloc[:idx_in_history+1]
                    if len(vol_so_far) >= 21:
                        v_day = float(vol_so_far.iloc[-1])
                        v_avg = float(vol_so_far.iloc[-21:-1].mean())
                        vol_ratio = v_day / (v_avg + 1e-9)
                        if vol_ratio > 3.5:
                            pattern_blocked = True
                    
                    # 2. Köpük Kalkanı (SMA100 Bubble Shield)
                    close_so_far = df_t['Close'].iloc[:idx_in_history+1]
                    if len(close_so_far) >= 100:
                        sma100 = float(close_so_far.iloc[-100:].mean())
                        bubble_ratio = p_close / (sma100 + 1e-9)
                        if bubble_ratio > 1.25:
                            pattern_blocked = True
                            
                    # 3. RSI Eğim Teyidi (RSI Slope Check)
                    if len(close_so_far) >= 20:
                        rsi_series = calculate_rsi(close_so_far)
                        if len(rsi_series) >= 4:
                            rsi_today = float(rsi_series.iloc[-1])
                            rsi_prev = float(rsi_series.iloc[-4])
                            rsi_slope = rsi_today - rsi_prev
                            if rsi_slope < 0.0:
                                pattern_blocked = True
                
                if score >= current_buy_threshold:
                    if pattern_blocked:
                        continue # Pattern shield block active
                        
                    # Adaptive Capital Allocation based on conviction
                    current_alloc = ALLOCATION_PCT
                    if ADAPTIVE_ALLOC:
                        if score >= 85:
                            current_alloc = 0.25 # Overweight
                        elif score >= 75:
                            current_alloc = 0.20
                        else:
                            current_alloc = 0.15 # Underweight
                            
                    investment = min(cash, portfolio_val * current_alloc)
                    if investment >= 1000.0:  # must be a meaningful amount
                        shares = investment / p_close
                        cash -= investment
                        
                        holdings[t] = shares
                        entry_prices[t] = p_close
                        peak_prices[t] = p_close
                        buy_dates[t] = current_date
                        max_holding_drawdown[t] = 0.0
                        
                        print(f"  [SİNYAL ALIŞ ] {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):6} alındı!  Fiyat: {p_close:7.2f} {curr} (Zeka Skoru: {score:2}, Bütçe: {investment:.2f} {curr})")

    # Finalize Open Positions at last available prices
    last_date = trading_dates[-1]
    final_portfolio_val = cash
    
    for t in valid_tickers:
        if holdings[t] > 0.0:
            df_t = stock_data[t]
            p_final = float(df_t.iloc[-1]['Close'])
            cash_gained = holdings[t] * p_final
            final_portfolio_val += cash_gained
            
            fund = db_funds[t]
            sec_lower = fund.get("sector", "Diğer").lower()
            current_sl_pct = STOP_LOSS_PCT
            if config["id"] in ["autopsy_volmom", "aggressive_ai"]:
                if any(x in sec_lower for x in ["teknoloji", "yazılım", "tech", "software", "enerji", "energy"]):
                    current_sl_pct = 0.08
            
            entry_p = entry_prices[t]
            ret_pct = (p_final - entry_p) / entry_p
            incorrect = (ret_pct < 0.0) or (max_holding_drawdown[t] >= current_sl_pct)
            
            closed_trades.append({
                'ticker': t.replace('.IS', ''),
                'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                'buy_price': entry_p,
                'sell_date': last_date.strftime('%Y-%m-%d'),
                'sell_price': p_final,
                'return_pct': ret_pct * 100,
                'type': 'LİKİDE (AÇIK)',
                'max_drawdown': max_holding_drawdown[t] * 100,
                'incorrect_decision': incorrect
            })
            holdings[t] = 0.0

    # Calculate Benchmark Index Return
    xu100_start_price = float(xu100[xu100.index >= pd.to_datetime(start_trading_date)].iloc[0]['Close'])
    xu100_end_price = float(xu100.iloc[-1]['Close'])
    xu100_return_pct = (xu100_end_price / xu100_start_price - 1) * 100

    # Compute simulation metrics
    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t['return_pct'] > 0]
    losing_trades = [t for t in closed_trades if t['return_pct'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    incorrect_trades = [t for t in closed_trades if t['incorrect_decision']]
    incorrect_decision_rate = (len(incorrect_trades) / total_trades * 100) if total_trades > 0 else 0
    
    avg_trade_return = np.mean([t['return_pct'] for t in closed_trades]) if total_trades > 0 else 0
    avg_max_drawdown = np.mean([t['max_drawdown'] for t in closed_trades]) if total_trades > 0 else 0
    
    total_return_pct = (final_portfolio_val / initial_capital - 1) * 100
    alpha = total_return_pct - xu100_return_pct
    
    stop_loss_count = len([t for t in closed_trades if t['type'] == 'STOP-LOSS'])
    take_profit_count = len([t for t in closed_trades if t['type'] == 'TAKE-PROFIT'])
    signal_sell_count = len([t for t in closed_trades if t['type'] == 'SIGNAL'])

    return {
        "scenario_name": config["name"],
        "initial_capital": initial_capital,
        "final_value": final_portfolio_val,
        "total_return_pct": total_return_pct,
        "xu100_return_pct": xu100_return_pct,
        "alpha": alpha,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "incorrect_decision_rate": incorrect_decision_rate,
        "avg_trade_return": avg_trade_return,
        "avg_max_drawdown": avg_max_drawdown,
        "stop_loss_count": stop_loss_count,
        "take_profit_count": take_profit_count,
        "signal_sell_count": signal_sell_count,
        "trades": closed_trades
    }

def execute_market_backtest(market="BIST"):
    print(f"\n{CLR_CYAN}{CLR_BOLD}==========================================================================")
    print(f"      {market} FinTrack Karar Destek ve Geriye Dönük Simülasyon Raporu")
    print(f"=========================================================================={CLR_RESET}\n")
    
    # 1. Load Tickers from DB
    try:
        db_funds = load_db_fundamentals()
        print(f"[Veritabanı] SQLite üzerinden {len(db_funds)} şirketin finansal verileri yüklendi.")
    except Exception as e:
        print(f"{CLR_RED}[HATA] Veritabanı okunurken hata oluştu: {e}{CLR_RESET}")
        return

    # Select representative major BIST or SP500 stocks
    if market == "SP500":
        test_tickers = [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "LLY", "AVGO", "JPM", "UNH"
        ]
        benchmark_ticker = "^GSPC"
        output_file = "./data/backtest_results_sp500.json"
        curr = "$"
    else:
        test_tickers = [
            "THYAO.IS", "GARAN.IS", "EREGL.IS", "BIMAS.IS", "KCHOL.IS", 
            "TUPRS.IS", "ASELS.IS", "SASA.IS", "SISE.IS", "AKBNK.IS",
            "ASTOR.IS", "DOAS.IS"
        ]
        benchmark_ticker = "XU100.IS"
        output_file = "./data/backtest_results.json"
        curr = "TL"
        
    # Verify tickers exist in database
    tickers = [t for t in test_tickers if t in db_funds]
    if not tickers:
        print(f"{CLR_YELLOW}[UYARI] Belirlenen test hisseleri veritabanında bulunamadı. DB'den ilk 10 hisse alınıyor...{CLR_RESET}")
        tickers = [t for t, f in db_funds.items() if f.get("market", "BIST") == market][:10]
        if not tickers:
            tickers = list(db_funds.keys())[:10]
        
    print(f"[Simülasyon] Test Grubu Hisseleri: {', '.join([t.replace('.IS', '') for t in tickers])}")
    
    # Define Date Ranges
    end_date = datetime.now()
    start_trading_date = end_date - timedelta(days=365)
    start_download_date = start_trading_date - timedelta(days=280)
    
    print(f"[Tarih] Simülasyon Dönemi: {start_trading_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print(f"[Tarih] Gösterge Tampon Başlangıcı: {start_download_date.strftime('%Y-%m-%d')}")
    
    # 2. Download Price History once for high performance
    print(f"\n[yfinance] Tarihsel hisse fiyatları indiriliyor ({market})...")
    stock_data = {}
    
    for t in tickers:
        print(f"  > {t.replace('.IS', '')} verisi indiriliyor...")
        df = yf.download(t, start=start_download_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d", progress=False)
        if df.empty or len(df) < 250:
            print(f"    {CLR_RED}[UYARI] {t} için yetersiz veri bulundu, test dışı bırakıldı.{CLR_RESET}")
            continue
            
        close_series = df['Close'].squeeze()
        volume_series = df['Volume'].squeeze()
        stock_data[t] = pd.DataFrame({'Close': close_series, 'Volume': volume_series})
        
    valid_tickers = list(stock_data.keys())
    if not valid_tickers:
        print(f"{CLR_RED}[HATA] Hiçbir hisse için fiyat verisi indirilemedi. Test durduruldu.{CLR_RESET}")
        return
        
    # Download Benchmark Index
    print(f"  > {benchmark_ticker} (Endeks) verisi indiriliyor...")
    xu100 = yf.download(benchmark_ticker, start=start_download_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d", progress=False)
    xu_close = xu100['Close'].squeeze()
    xu100 = pd.DataFrame({'Close': xu_close})
        
    # Align Trading Dates
    df_ref = stock_data[valid_tickers[0]]
    trading_dates = df_ref[df_ref.index >= pd.to_datetime(start_trading_date)].index
        # 3. Define Backtesting Strategies (calibrated for US or BIST)
    if market == "SP500":
        scenarios_config = [
            {
                "id": "autopsy_volmom",
                "name": "✨ YZ Otopsi-VOLMOM (Şampiyon)",
                "buy_threshold": 68,
                "sell_threshold": 40,
                "tr_bond_yield": 3.5,
                "value_weight": 0.40,
                "momentum_weight": 0.60,
                "eps_growth_multiplier": 1.2,
                "stop_loss_pct": 0.08,
                "take_profit_pct": 0.25,
                "allocation_pct": 0.20,
                "trailing_stop": True,
                "dynamic_tp": True,
                "index_shield": True,
                "adaptive_alloc": True
            },
            {
                "id": "conservative",
                "name": "Muhafazakar Klasik",
                "buy_threshold": 75,
                "sell_threshold": 45,
                "tr_bond_yield": 4.25,        # US 4.25% bond yield
                "value_weight": 0.70,
                "momentum_weight": 0.30,
                "eps_growth_multiplier": 1.0,
                "stop_loss_pct": 0.08,        # tighter stop-loss for lower volatility US market
                "take_profit_pct": 0.20,
                "allocation_pct": 0.20
            },
            {
                "id": "balanced",
                "name": "Enflasyon Dengeli (Önerilen)",
                "buy_threshold": 70,
                "sell_threshold": 45,
                "tr_bond_yield": 3.75,
                "value_weight": 0.60,
                "momentum_weight": 0.40,
                "eps_growth_multiplier": 1.1,
                "stop_loss_pct": 0.08,
                "take_profit_pct": 0.25,
                "allocation_pct": 0.20
            },
            {
                "id": "active",
                "name": "Aktif Taktik Momentum",
                "buy_threshold": 65,
                "sell_threshold": 40,
                "tr_bond_yield": 3.5,
                "value_weight": 0.50,
                "momentum_weight": 0.50,
                "eps_growth_multiplier": 1.2,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.30,
                "allocation_pct": 0.20
            },
            {
                "id": "aggressive_ai",
                "name": "Agresif Zeka Modeli",
                "buy_threshold": 65,
                "sell_threshold": 40,
                "tr_bond_yield": 3.25,
                "value_weight": 0.50,
                "momentum_weight": 0.50,
                "eps_growth_multiplier": 1.2,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.35,
                "allocation_pct": 0.20,
                "trailing_stop": True,
                "dynamic_tp": True,
                "index_shield": False,
                "adaptive_alloc": True
            }
        ]
    else:
        scenarios_config = [
            {
                "id": "autopsy_volmom",
                "name": "✨ YZ Otopsi-VOLMOM (Şampiyon)",
                "buy_threshold": 68,
                "sell_threshold": 40,
                "tr_bond_yield": 30.0,
                "value_weight": 0.40,
                "momentum_weight": 0.60,
                "eps_growth_multiplier": 1.4,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.30,
                "allocation_pct": 0.20,
                "trailing_stop": True,
                "dynamic_tp": True,
                "index_shield": True,
                "adaptive_alloc": True
            },
            {
                "id": "conservative",
                "name": "Muhafazakar Klasik",
                "buy_threshold": 75,
                "sell_threshold": 45,
                "tr_bond_yield": 45.0,
                "value_weight": 0.70,
                "momentum_weight": 0.30,
                "eps_growth_multiplier": 1.0,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.30,
                "allocation_pct": 0.20
            },
            {
                "id": "balanced",
                "name": "Enflasyon Dengeli (Önerilen)",
                "buy_threshold": 70,
                "sell_threshold": 45,
                "tr_bond_yield": 35.0,
                "value_weight": 0.60,
                "momentum_weight": 0.40,
                "eps_growth_multiplier": 1.3,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.30,
                "allocation_pct": 0.20
            },
            {
                "id": "active",
                "name": "Aktif Taktik Momentum",
                "buy_threshold": 65,
                "sell_threshold": 40,
                "tr_bond_yield": 30.0,
                "value_weight": 0.50,
                "momentum_weight": 0.50,
                "eps_growth_multiplier": 1.5,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.30,
                "allocation_pct": 0.20
            },
            {
                "id": "aggressive_ai",
                "name": "Agresif Zeka Modeli",
                "buy_threshold": 65,
                "sell_threshold": 40,
                "tr_bond_yield": 25.0,
                "value_weight": 0.50,
                "momentum_weight": 0.50,
                "eps_growth_multiplier": 1.5,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.35,
                "allocation_pct": 0.20,
                "trailing_stop": True,
                "dynamic_tp": True,
                "index_shield": False,
                "adaptive_alloc": True
            }
        ]
    
    results = {}
    for config in scenarios_config:
        results[config["id"]] = run_single_backtest_scenario(
            config=config,
            db_funds=db_funds,
            valid_tickers=valid_tickers,
            stock_data=stock_data,
            xu100=xu100,
            trading_dates=trading_dates,
            start_trading_date=start_trading_date,
            market=market
        )
        
    # 4. PRINT SIDE-BY-SIDE STRATEGY COMPARISON REPORT
    print(f"\n{CLR_CYAN}{CLR_BOLD}==========================================================================================")
    print(f"                    {market} STRATEJİ KARŞILAŞTIRMA VE PERFORMANS RAPORU                           ")
    print(f"=========================================================================={CLR_RESET}")
    print(f"  Metrik                       | Muhafazakar Klasik | Enflasyon Dengeli  | Aktif Taktik Momentum")
    print("-" * 90)
    
    c = results["conservative"]
    b = results["balanced"]
    a = results["active"]
    
    print(f"  Başlangıç Kapitali           | {c['initial_capital']:,.2f} {curr}   | {b['initial_capital']:,.2f} {curr}   | {a['initial_capital']:,.2f} {curr}")
    print(f"  Final Portföy Değeri         | {c['final_value']:,.2f} {curr}   | {b['final_value']:,.2f} {curr}   | {a['final_value']:,.2f} {curr}")
    
    c_ret_color = CLR_GREEN if c['total_return_pct'] >= 0 else CLR_RED
    b_ret_color = CLR_GREEN if b['total_return_pct'] >= 0 else CLR_RED
    a_ret_color = CLR_GREEN if a['total_return_pct'] >= 0 else CLR_RED
    print(f"  Toplam Net Getiri            | {c_ret_color}%{c['total_return_pct']:.2f}{CLR_RESET}            | {b_ret_color}%{b['total_return_pct']:.2f}{CLR_RESET}            | {a_ret_color}%{a['total_return_pct']:.2f}{CLR_RESET}")
    
    print(f"  Endeks Benchmark Getirisi    | %{c['xu100_return_pct']:.2f}            | %{b['xu100_return_pct']:.2f}            | %{a['xu100_return_pct']:.2f}")
    
    c_alp_color = CLR_GREEN if c['alpha'] >= 0 else CLR_RED
    b_alp_color = CLR_GREEN if b['alpha'] >= 0 else CLR_RED
    a_alp_color = CLR_GREEN if a['alpha'] >= 0 else CLR_RED
    print(f"  AI ALPHA (Piyasa Üstünlüğü)  | {c_alp_color}%{c['alpha']:.2f}{CLR_RESET}           | {b_alp_color}%{b['alpha']:.2f}{CLR_RESET}           | {a_alp_color}%{a['alpha']:.2f}{CLR_RESET}")
    print("-" * 90)
    print(f"  Toplam Yapılan İşlem         | {c['total_trades']:2} adet           | {b['total_trades']:2} adet           | {a['total_trades']:2} adet")
    print(f"  Kazançlı Kapanan             | {len([t for t in c['trades'] if t['return_pct'] > 0]):2} adet           | {len([t for t in b['trades'] if t['return_pct'] > 0]):2} adet           | {len([t for t in a['trades'] if t['return_pct'] > 0]):2} adet")
    print(f"  Kazanma Oranı (Win Rate)     | %{c['win_rate']:.1f}              | %{b['win_rate']:.1f}              | %{a['win_rate']:.1f}")
    print(f"  Ortalama İşlem Getirisi      | %{c['avg_trade_return']:.2f}             | %{b['avg_trade_return']:.2f}             | %{a['avg_trade_return']:.2f}")
    print("-" * 90)
    print(f"  Yanlış Karar Oranı (IDR)     | {CLR_RED}%{c['incorrect_decision_rate']:.1f}{CLR_RESET}            | {CLR_GREEN if b['incorrect_decision_rate']<35 else CLR_RED}%{b['incorrect_decision_rate']:.1f}{CLR_RESET}            | {CLR_GREEN if a['incorrect_decision_rate']<35 else CLR_RED}%{a['incorrect_decision_rate']:.1f}{CLR_RESET}")
    print(f"  Doğru Karar Oranı            | %{100 - c['incorrect_decision_rate']:.1f}            | %{100 - b['incorrect_decision_rate']:.1f}            | %{100 - a['incorrect_decision_rate']:.1f}")
    print(f"  Ortalama Satış Sonrası Max DD| %{c['avg_max_drawdown']:.2f}             | %{b['avg_max_drawdown']:.2f}             | %{a['avg_max_drawdown']:.2f}")
    print(f"  Stop-Loss İsabeti (SL Hit)   | {c['stop_loss_count']:2} adet           | {b['stop_loss_count']:2} adet           | {a['stop_loss_count']:2} adet")
    print(f"  Kar Al İsabeti (TP Hit)      | {c['take_profit_count']:2} adet           | {b['take_profit_count']:2} adet           | {a['take_profit_count']:2} adet")
    print(f"  Sinyal Bazlı Satış (Signal)  | {c['signal_sell_count']:2} adet           | {b['signal_sell_count']:2} adet           | {a['signal_sell_count']:2} adet")
    print(f"{CLR_CYAN}{CLR_BOLD}=========================================================================================={CLR_RESET}\n")
 
    # 5. Save all results to a structured JSON file using autopsy_volmom as default top-level data
    champ = results["autopsy_volmom"]
    results_payload = {
        "initial_capital": champ["initial_capital"],
        "final_value": champ["final_value"],
        "total_return_pct": champ["total_return_pct"],
        "xu100_return_pct": champ["xu100_return_pct"],
        "alpha": champ["alpha"],
        "total_trades": champ["total_trades"],
        "win_rate": champ["win_rate"],
        "incorrect_decision_rate": champ["incorrect_decision_rate"],
        "avg_trade_return": champ["avg_trade_return"],
        "avg_max_drawdown": champ["avg_max_drawdown"],
        "stop_loss_count": champ["stop_loss_count"],
        "take_profit_count": champ["take_profit_count"],
        "signal_sell_count": champ["signal_sell_count"],
        "trades": champ["trades"],
        "scenarios": {
            "autopsy_volmom": champ,
            "conservative": c,
            "balanced": b,
            "active": a,
            "aggressive_ai": results["aggressive_ai"]
        }
    }
    
    os.makedirs("./data", exist_ok=True)
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f"[Rapor] Tüm simülasyon detayları '{output_file}' dosyasına kaydedildi.")

def run_historical_backtest():
    print(f"\n{CLR_CYAN}{CLR_BOLD}==========================================================================")
    print("      KÜRESEL PIYASALAR GERİYE DÖNÜK SİMÜLASYON MOTORU (365 GÜN)")
    print(f"=========================================================================={CLR_RESET}\n")
    
    # Run BIST Backtests
    execute_market_backtest("BIST")
    
    # Run S&P 500 Backtests
    execute_market_backtest("SP500")

if __name__ == "__main__":
    run_historical_backtest()

