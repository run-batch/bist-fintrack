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

def calculate_rsi(series, period=14):
    """Safely calculates the 14-period RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_dss_score(price, eps, bvps, roe, sector, debt_to_equity, beta, eps_growth_5y, close_history, vol_history, idx,
                  tr_bond_yield=45.0, value_weight=0.7, momentum_weight=0.3, eps_growth_multiplier=1.0):
    """
    Reconstructs the exact BIST Radar intelligence score logic day-by-day.
    """
    if price <= 0 or eps <= 0:
        return 5  # Floor minimum score
    
    # 1. VALUATION MODELS (FUNDAMENTAL)
    # A. DCF (Discounted Cash Flow)
    TR_BOND_YIELD = tr_bond_yield
    ERP = 6.0
    
    # Apply high leverage penalty if Debt/Equity > 2.0
    g_rate = eps_growth_5y * eps_growth_multiplier
    if debt_to_equity > 2.0:
        g_rate *= 0.6
        
    g_decimal = g_rate / 100.0
    cost_of_equity = (TR_BOND_YIELD + beta * ERP) / 100.0
    
    dcf_val = 0.0
    fcf = eps
    for i in range(1, 6):
        fcf *= (1 + g_decimal)
        dcf_val += fcf / ((1 + cost_of_equity) ** i)
        
    terminal_growth_cap = 12.0
    terminal_growth_pct = min((g_rate * 0.4), terminal_growth_cap)
    terminal_g = terminal_growth_pct / 100.0
    
    if cost_of_equity > terminal_g:
        terminal_val = (fcf * (1 + terminal_g)) / (cost_of_equity - terminal_g)
        dcf_val += terminal_val / ((1 + cost_of_equity) ** 5)
        
    # B. Benjamin Graham Formülü
    graham_multiplier = 15.0 if sector == 'Banka' else 22.5
    if eps > 0 and bvps > 0:
        fair_graham = np.sqrt(graham_multiplier * eps * bvps)
    else:
        fair_graham = 0.0
        
    # C. Sektörel Çarpan Analizi (Multiples)
    if sector == 'Banka': target_pe = 8.0
    elif sector == 'Havacılık': target_pe = 14.0
    elif sector in ['Holding', 'Demir-Çelik / Çimento']: target_pe = 12.0
    else: target_pe = 16.0
    
    fair_multiples = eps * target_pe
    
    # D. Composite Fair Price
    if sector == 'Banka':
        weights = {'dcf': 0.1, 'graham': 0.6, 'multiples': 0.3}
    elif sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım']:
        weights = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
    else:
        weights = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
        
    intrinsic_avg = (dcf_val * weights['dcf'] + fair_graham * weights['graham'] + fair_multiples * weights['multiples'])
    
    # E. Value Score based on MOS (Margin of Safety)
    mos = (intrinsic_avg / price) - 1.0
    value_score = int(50 + np.tanh(mos) * 50)
    
    # 2. TECHNICAL TREND & MOMENTUM
    prices_so_far = close_history.iloc[:idx+1]
    vol_so_far = vol_history.iloc[:idx+1]
    
    momentum_score = 50
    
    if len(prices_so_far) >= 200:
        sma50 = float(prices_so_far.iloc[-50:].mean())
        sma200 = float(prices_so_far.iloc[-200:].mean())
        rsi = float(calculate_rsi(prices_so_far).iloc[-1])
        
        v_current = vol_so_far.iloc[-5:].mean()
        v_prev = vol_so_far.iloc[-25:-5].mean()
        v_change = (v_current / v_prev - 1) * 100 if v_prev > 0 else 0
        
        # Trend conditions
        if price > sma50 > sma200:
            momentum_score += 25
        elif price > sma50:
            momentum_score += 15
        elif price < sma50 < sma200:
            momentum_score -= 25
        elif price < sma50:
            momentum_score -= 15
            
        # RSI bounds
        if rsi < 30:
            momentum_score += 10
        elif rsi > 70:
            momentum_score -= 10
            
        # Volume spikes
        if v_change > 30:
            momentum_score += 5
            
    # Clamp scores
    momentum_score = max(0, min(100, momentum_score))
    
    # 3. COMPOSITE INTELLIGENCE SCORE
    score = int(value_score * value_weight + momentum_score * momentum_weight)
    return max(5, min(95, score))

def get_aggressive_score(price, eps, bvps, roe, sector, debt_to_equity, beta, eps_growth_5y, close_history, vol_history, idx,
                         eps_growth_multiplier=1.5):
    """
    Reconstructs the exact BIST Radar aggressive intelligence score logic day-by-day.
    """
    if price <= 0 or eps <= 0:
        return 5
    
    is_bank_holding = sector in ['Banka']
    is_loss_making = (eps <= 0) or (roe < 0)
    
    # 1. VALUATION MODELS (FUNDAMENTAL)
    if is_bank_holding:
        roe_val = roe
        nim_val = 0.05 + 0.02 * roe_val
        npl_val = max(0.005, 0.025 - 0.01 * roe_val)
        syr_val = 0.16 + 0.04 * beta
        
        roe_score = min(100.0, max(0.0, (roe_val / 0.40) * 100))
        nim_score = min(100.0, max(0.0, (nim_val / 0.07) * 100))
        npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.005) / 0.04) * 100))
        syr_score = min(100.0, max(0.0, ((syr_val - 0.12) / 0.08) * 100))
        
        bank_score = int(roe_score * 0.30 + nim_score * 0.25 + npl_score * 0.25 + syr_score * 0.20)
        bank_score = max(5, min(95, bank_score))
        
        value_score = bank_score
    else:
        # A. Benjamin Graham Formülü (Enflasyon Uyumlu - %15 beklenen enflasyon)
        expected_inflation_agg = 0.15
        base_k_agg = 11.5 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 22.5
        graham_multiplier_agg = base_k_agg / (1.0 + expected_inflation_agg)
        if eps > 0 and bvps > 0:
            fair_graham_agg = np.sqrt(graham_multiplier_agg * eps * bvps)
        else:
            fair_graham_agg = 0.0
            
        # B. DCF with lower discount rate (rf* = 18%)
        rf_star_agg = 0.18
        delta_weight_agg = 0.01
        debt_penalty_agg = delta_weight_agg * max(0.0, np.log(1.0 + debt_to_equity))
        cost_of_equity_agg = rf_star_agg + beta * 0.08 + debt_penalty_agg
        
        eps_g_agg = eps_growth_5y * eps_growth_multiplier
        if debt_to_equity > 4.0:
            eps_g_agg *= 0.8
            
        g_decimal_agg = eps_g_agg / 100.0
        dcf_val_agg = 0.0
        fcf_agg = eps
        for i in range(1, 6):
            fcf_agg *= (1 + g_decimal_agg)
            dcf_val_agg += fcf_agg / ((1 + cost_of_equity_agg) ** i)
            
        terminal_growth_cap = 12.0
        terminal_growth_pct_agg = min((eps_g_agg * 0.4), terminal_growth_cap)
        terminal_g_agg = terminal_growth_pct_agg / 100.0
        if cost_of_equity_agg > terminal_g_agg:
            terminal_val_agg = (fcf_agg * (1 + terminal_g_agg)) / (cost_of_equity_agg - terminal_g_agg)
            dcf_val_agg += terminal_val_agg / ((1 + cost_of_equity_agg) ** 5)
            
        # C. Sektörel Çarpan Analizi
        if sector == 'Havacılık': target_pe = 14.0
        elif sector == 'Demir-Çelik / Çimento': target_pe = 12.0
        else: target_pe = 16.0
        fair_multiples = eps * target_pe
        
        if is_loss_making:
            dcf_val_agg = 0.0
            fair_graham_agg *= 0.5
            intrinsic_avg_agg = fair_graham_agg
        else:
            if sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım']:
                weights_agg = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
            else:
                weights_agg = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
            intrinsic_avg_agg = (dcf_val_agg * weights_agg['dcf'] + fair_graham_agg * weights_agg['graham'] + fair_multiples * weights_agg['multiples'])
            
        # GYO and Holding Value Score adjustment
        if sector in ['GYO', 'Holding', 'Holding / Enerji']:
            intrinsic_avg_agg *= 0.60
            
        mos_agg = (intrinsic_avg_agg / price) - 1.0 if price > 0 else 0.0
        value_score = int(50 + np.tanh(mos_agg) * 50)
        
    # 2. TECHNICAL TREND & MOMENTUM
    prices_so_far = close_history.iloc[:idx+1]
    vol_so_far = vol_history.iloc[:idx+1]
    
    momentum_score = 50
    if len(prices_so_far) >= 200:
        sma50 = float(prices_so_far.iloc[-50:].mean())
        sma200 = float(prices_so_far.iloc[-200:].mean())
        rsi = float(calculate_rsi(prices_so_far).iloc[-1])
        
        v_current = vol_so_far.iloc[-5:].mean()
        v_prev = vol_so_far.iloc[-25:-5].mean()
        v_change = (v_current / v_prev - 1) * 100 if v_prev > 0 else 0
        
        if price > sma50 > sma200:
            momentum_score += 25
        elif price > sma50:
            momentum_score += 15
        elif price < sma50 < sma200:
            momentum_score -= 25
        elif price < sma50:
            momentum_score -= 15
            
        if rsi < 30:
            momentum_score += 10
        elif rsi > 70:
            momentum_score -= 10
            
        if v_change > 30:
            momentum_score += 5
            
        # Aggressive Turn-around Bonus
        if rsi < 45:
            momentum_score += 15
            
    momentum_score = max(5, min(95, momentum_score))
    
    # 3. COMPOSITE INTELLIGENCE SCORE
    score = int(value_score * 0.5 + momentum_score * 0.5)
    return max(5, min(95, score))

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
           roe, market_cap, beta, eps_growth_5y, trailing_eps, debt_to_equity, sector 
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
            "sector": r[12] or "Diğer"
        }
        
    conn.close()
    return fundamentals

def run_single_backtest_scenario(config, db_funds, valid_tickers, stock_data, xu100, trading_dates, start_trading_date):
    """Runs a single portfolio backtest simulation using the provided configuration parameters."""
    initial_capital = 100000.0  # 100,000 TRY
    cash = initial_capital
    holdings = {t: 0.0 for t in valid_tickers}
    entry_prices = {t: 0.0 for t in valid_tickers}
    buy_dates = {t: None for t in valid_tickers}
    max_holding_drawdown = {t: 0.0 for t in valid_tickers}
    
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
    
    closed_trades = []
    daily_portfolio_value = []
    
    print(f"\n{CLR_CYAN}{CLR_BOLD}--- [SİMÜLASYON BAŞLADI: {config['name'].upper()}] ---{CLR_RESET}")
    
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
        
        # B. Check Stop-Loss and Take-Profit for active holdings
        for t in valid_tickers:
            if holdings[t] > 0.0:
                p_close = current_prices[t]
                entry_p = entry_prices[t]
                
                # Update Max Drawdown while holding
                if p_close < entry_p:
                    drawdown = (entry_p - p_close) / entry_p
                    max_holding_drawdown[t] = max(max_holding_drawdown[t], drawdown)
                
                # Check Stop-Loss
                if p_close <= entry_p * (1 - STOP_LOSS_PCT):
                    cash_gained = holdings[t] * p_close
                    cash += cash_gained
                    
                    ret_pct = -STOP_LOSS_PCT
                    closed_trades.append({
                        'ticker': t.replace('.IS', ''),
                        'buy_date': buy_dates[t].strftime('%Y-%m-%d'),
                        'buy_price': entry_p,
                        'sell_date': current_date.strftime('%Y-%m-%d'),
                        'sell_price': p_close,
                        'return_pct': ret_pct * 100,
                        'type': 'STOP-LOSS',
                        'max_drawdown': max_holding_drawdown[t] * 100,
                        'incorrect_decision': True
                    })
                    
                    print(f"  [{CLR_RED}STOP-LOSS{CLR_RESET}] {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):6} satıldı! Fiyat: {p_close:7.2f} TL (Maliyet: {entry_p:.2f} TL, Net: %{ret_pct*100:.1f})")
                    holdings[t] = 0.0
                    entry_prices[t] = 0.0
                    buy_dates[t] = None
                    max_holding_drawdown[t] = 0.0
                    continue
                    
                # Check Take-Profit
                elif p_close >= entry_p * (1 + TAKE_PROFIT_PCT):
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
                    
                    print(f"  [{CLR_GREEN}TAKE-PROFIT{CLR_RESET}] {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):6} satıldı! Fiyat: {p_close:7.2f} TL (Maliyet: {entry_p:.2f} TL, Net: %{ret_pct*100:.1f})")
                    holdings[t] = 0.0
                    entry_prices[t] = 0.0
                    buy_dates[t] = None
                    max_holding_drawdown[t] = 0.0
                    continue
        
        # C. Re-evaluate Score Signals (Only perform 1 action per stock per day)
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
            
            # Reconstruct Daily Zeka Skoru with custom strategy params
            if config["id"] == "aggressive_ai":
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
                    eps_growth_multiplier=eps_growth_multiplier
                )
            else:
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
                    idx=idx_in_history,
                    tr_bond_yield=tr_bond_yield,
                    value_weight=value_weight,
                    momentum_weight=momentum_weight,
                    eps_growth_multiplier=eps_growth_multiplier
                )
            
            # DECISION RULE:
            # 1. SELL: Hold stock, and score <= SELL_SCORE_THRESHOLD
            if holdings[t] > 0.0 and score <= SELL_SCORE_THRESHOLD:
                entry_p = entry_prices[t]
                cash_gained = holdings[t] * p_close
                cash += cash_gained
                
                ret_pct = (p_close - entry_p) / entry_p
                incorrect = (ret_pct < 0.0) or (max_holding_drawdown[t] >= 0.05)
                
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
                print(f"  [SİNYAL SATIŞ] {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):6} satıldı! Fiyat: {p_close:7.2f} TL (Zeka Skoru: {score:2}, Getiri: {log_color}%{ret_pct*100:.1f}{CLR_RESET}, Max Geri: %{max_holding_drawdown[t]*100:.1f})")
                
                holdings[t] = 0.0
                entry_prices[t] = 0.0
                buy_dates[t] = None
                max_holding_drawdown[t] = 0.0
                
            # 2. BUY: Do not hold stock, cash is available, and score >= BUY_SCORE_THRESHOLD
            elif holdings[t] == 0.0 and score >= BUY_SCORE_THRESHOLD:
                # Allocate 20% of total portfolio value
                investment = min(cash, portfolio_val * ALLOCATION_PCT)
                if investment >= 1000.0:  # must be a meaningful amount (at least 1,000 TL)
                    shares = investment / p_close
                    cash -= investment
                    
                    holdings[t] = shares
                    entry_prices[t] = p_close
                    buy_dates[t] = current_date
                    max_holding_drawdown[t] = 0.0
                    
                    print(f"  [{CLR_CYAN}SİNYAL ALIŞ {CLR_RESET}] {current_date.strftime('%Y-%m-%d')} | {t.replace('.IS', ''):6} alındı!  Fiyat: {p_close:7.2f} TL (Zeka Skoru: {score:2}, Bütçe: {investment:.2f} TL)")

    # Finalize Open Positions at last available prices
    last_date = trading_dates[-1]
    final_portfolio_val = cash
    
    for t in valid_tickers:
        if holdings[t] > 0.0:
            df_t = stock_data[t]
            p_final = float(df_t.iloc[-1]['Close'])
            cash_gained = holdings[t] * p_final
            final_portfolio_val += cash_gained
            
            entry_p = entry_prices[t]
            ret_pct = (p_final - entry_p) / entry_p
            incorrect = (ret_pct < 0.0) or (max_holding_drawdown[t] >= 0.05)
            
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

    # Calculate Benchmark Index XU100 Return
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

def run_historical_backtest():
    print(f"\n{CLR_CYAN}{CLR_BOLD}==========================================================================")
    print("      BIST RADAR KARAR DESTEK VE GERİYE DÖNÜK SİMÜLASYON MOTORU (365 GÜN)")
    print(f"=========================================================================={CLR_RESET}\n")
    
    # 1. Load Tickers from DB
    try:
        db_funds = load_db_fundamentals()
        print(f"[Veritabanı] SQLite üzerinden {len(db_funds)} şirketin finansal verileri yüklendi.")
    except Exception as e:
        print(f"{CLR_RED}[HATA] Veritabanı okunurken hata oluştu: {e}{CLR_RESET}")
        return

    # Select representative major BIST stocks from different sectors to backtest
    test_tickers = [
        "THYAO.IS", "GARAN.IS", "EREGL.IS", "BIMAS.IS", "KCHOL.IS", 
        "TUPRS.IS", "ASELS.IS", "SASA.IS", "SISE.IS", "AKBNK.IS",
        "ASTOR.IS", "DOAS.IS"
    ]
    
    # Verify tickers exist in database
    tickers = [t for t in test_tickers if t in db_funds]
    if not tickers:
        print(f"{CLR_YELLOW}[UYARI] Belirlenen test hisseleri veritabanında bulunamadı. DB'den ilk 10 hisse alınıyor...{CLR_RESET}")
        tickers = list(db_funds.keys())[:10]
        
    print(f"[Simülasyon] Test Grubu Hisseleri: {', '.join([t.replace('.IS', '') for t in tickers])}")
    
    # Define Date Ranges
    end_date = datetime.now()
    start_trading_date = end_date - timedelta(days=365)
    start_download_date = start_trading_date - timedelta(days=280)
    
    print(f"[Tarih] Simülasyon Dönemi: {start_trading_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print(f"[Tarih] Gösterge Tampon Başlangıcı: {start_download_date.strftime('%Y-%m-%d')}")
    
    # 2. Download Price History once for high performance
    print("\n[yfinance] Tarihsel hisse fiyatları indiriliyor...")
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
        
    # Download Benchmark Index XU100.IS once
    print("  > XU100 (BIST 100 Endeksi) verisi indiriliyor...")
    xu100 = yf.download("XU100.IS", start=start_download_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d", progress=False)
    xu_close = xu100['Close'].squeeze()
    xu100 = pd.DataFrame({'Close': xu_close})
        
    # Align Trading Dates
    df_ref = stock_data[valid_tickers[0]]
    trading_dates = df_ref[df_ref.index >= pd.to_datetime(start_trading_date)].index
    print(f"\n[Analiz] Toplam işlem yapılacak iş günü sayısı: {len(trading_dates)} gün.")
    
    # 3. Define 4 Backtesting Strategies
    scenarios_config = [
        {
            "id": "conservative",
            "name": "Muhafazakar Klasik",
            "buy_threshold": 75,
            "sell_threshold": 45,
            "tr_bond_yield": 45.0,        # conservative high bond rate
            "value_weight": 0.70,
            "momentum_weight": 0.30,
            "eps_growth_multiplier": 1.0, # seeded growth
            "stop_loss_pct": 0.10,
            "take_profit_pct": 0.30,
            "allocation_pct": 0.20
        },
        {
            "id": "balanced",
            "name": "Enflasyon Dengeli (Önerilen)",
            "buy_threshold": 70,          # moderately lower threshold for wider coverage
            "sell_threshold": 45,
            "tr_bond_yield": 35.0,        # 35% discount yield (takes tax shelter & nominal buffer)
            "value_weight": 0.60,
            "momentum_weight": 0.40,
            "eps_growth_multiplier": 1.3, # 1.3x nominal growth adjustment (high inflation adaptation)
            "stop_loss_pct": 0.10,
            "take_profit_pct": 0.30,
            "allocation_pct": 0.20
        },
        {
            "id": "active",
            "name": "Aktif Taktik Momentum",
            "buy_threshold": 65,          # tactical coverage
            "sell_threshold": 40,
            "tr_bond_yield": 30.0,        # 30% discount rate
            "value_weight": 0.50,
            "momentum_weight": 0.50,      # balanced fundamental & momentum weight
            "eps_growth_multiplier": 1.5, # 1.5x nominal growth adaptation
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
            "take_profit_pct": 0.35,      # higher take-profit for aggressive run
            "allocation_pct": 0.20
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
            start_trading_date=start_trading_date
        )
        
    # 4. PRINT SIDE-BY-SIDE STRATEGY COMPARISON REPORT
    print(f"\n{CLR_CYAN}{CLR_BOLD}==========================================================================================")
    print("                    STRATEJİ KARŞILAŞTIRMA VE PERFORMANS RAPORU                           ")
    print(f"=========================================================================={CLR_RESET}")
    print(f"  Metrik                       | Muhafazakar Klasik | Enflasyon Dengeli  | Aktif Taktik Momentum")
    print("-" * 90)
    
    c = results["conservative"]
    b = results["balanced"]
    a = results["active"]
    
    print(f"  Başlangıç Kapitali           | {c['initial_capital']:,.2f} TRY   | {b['initial_capital']:,.2f} TRY   | {a['initial_capital']:,.2f} TRY")
    print(f"  Final Portföy Değeri         | {c['final_value']:,.2f} TRY   | {b['final_value']:,.2f} TRY   | {a['final_value']:,.2f} TRY")
    
    c_ret_color = CLR_GREEN if c['total_return_pct'] >= 0 else CLR_RED
    b_ret_color = CLR_GREEN if b['total_return_pct'] >= 0 else CLR_RED
    a_ret_color = CLR_GREEN if a['total_return_pct'] >= 0 else CLR_RED
    print(f"  Toplam Net Getiri            | {c_ret_color}%{c['total_return_pct']:.2f}{CLR_RESET}            | {b_ret_color}%{b['total_return_pct']:.2f}{CLR_RESET}            | {a_ret_color}%{a['total_return_pct']:.2f}{CLR_RESET}")
    
    print(f"  XU100 Endeks Getirisi        | %{c['xu100_return_pct']:.2f}            | %{b['xu100_return_pct']:.2f}            | %{a['xu100_return_pct']:.2f}")
    
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

    # 5. Save all results to a structured JSON file
    # We choose the "balanced" scenario (Enflasyon Dengeli) as the primary/default payload for backward compatibility,
    # and save all scenarios in a sub-dictionary.
    results_payload = {
        "initial_capital": b["initial_capital"],
        "final_value": b["final_value"],
        "total_return_pct": b["total_return_pct"],
        "xu100_return_pct": b["xu100_return_pct"],
        "alpha": b["alpha"],
        "total_trades": b["total_trades"],
        "win_rate": b["win_rate"],
        "incorrect_decision_rate": b["incorrect_decision_rate"],
        "avg_trade_return": b["avg_trade_return"],
        "avg_max_drawdown": b["avg_max_drawdown"],
        "stop_loss_count": b["stop_loss_count"],
        "take_profit_count": b["take_profit_count"],
        "signal_sell_count": b["signal_sell_count"],
        "trades": b["trades"],
        "scenarios": {
            "conservative": c,
            "balanced": b,
            "active": a,
            "aggressive_ai": results["aggressive_ai"]
        }
    }
    
    os.makedirs("./data", exist_ok=True)
    import json
    with open("./data/backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print("[Rapor] Tüm simülasyon detayları 'data/backtest_results.json' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_historical_backtest()
