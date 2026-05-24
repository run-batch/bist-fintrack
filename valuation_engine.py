# -*- coding: utf-8 -*-
"""
BIST FINTRACK - MATHEMATICAL VALUATION ENGINE (FORMÜL OMURGASI)
==============================================================
Bu dosya projenin değerleme ve puanlama kalbini oluşturur.
Tüm DCF, Graham, DuPont bankacılık ve kompozit Yapay Zeka Zeka Skorları buradadır.
app.py ve backtest_simulation.py bu dosyayı import eder.
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_usd_try_rates(start_date_str, end_date_str):
    """
    Retrieves USD/TRY exchange rates dynamically and caches them locally to avoid repeating downloads.
    Only downloads missing date ranges between the last cached date and the end date.
    Returns a dictionary mapping date strings to USD/TRY rates.
    """
    cache_dir = "./data/price_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "usd_try_rates.json")
    
    rates = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                rates = json.load(f)
        except Exception:
            rates = {}
            
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Check if we need to fetch missing rates
    target_dates = []
    curr_dt = start_dt
    while curr_dt <= end_dt:
        d_str = curr_dt.strftime("%Y-%m-%d")
        if d_str not in rates:
            target_dates.append(curr_dt)
        curr_dt += timedelta(days=1)
        
    if target_dates:
        fetch_start = target_dates[0].strftime("%Y-%m-%d")
        fetch_end = (target_dates[-1] + timedelta(days=2)).strftime("%Y-%m-%d")
        
        try:
            df = yf.download("TRY=X", start=fetch_start, end=fetch_end, interval="1d", progress=False)
            if not df.empty:
                close = df['Close'].squeeze()
                if isinstance(close, pd.Series):
                    for d, val in close.items():
                        d_str = d.strftime("%Y-%m-%d")
                        if not np.isnan(val) and val > 0:
                            rates[d_str] = float(val)
                elif isinstance(close, float) or isinstance(close, np.float64):
                    d_str = target_dates[0].strftime("%Y-%m-%d")
                    rates[d_str] = float(close)
            
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(rates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[UYARI] Döviz kuru indirilemedi fallback uygulanıyor: {e}")
            
    res_rates = {}
    curr_dt = start_dt
    last_known_rate = 32.50
    
    if rates:
        sorted_rates = sorted(rates.items())
        last_known_rate = sorted_rates[-1][1]
        
    while curr_dt <= end_dt:
        d_str = curr_dt.strftime("%Y-%m-%d")
        if d_str in rates:
            last_known_rate = rates[d_str]
        res_rates[d_str] = last_known_rate
        curr_dt += timedelta(days=1)
        
    return res_rates

def calculate_rsi(series, period=14):
    """Safely calculates the 14-period RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_dss_score(price, eps, bvps, roe, sector, debt_to_equity, beta, eps_growth_5y, close_history, vol_history, idx,
                  tr_bond_yield=45.0, value_weight=0.7, momentum_weight=0.3, eps_growth_multiplier=1.0, market="BIST",
                  sector_stats: dict = None):
    """
    Reconstructs the exact intelligence score logic day-by-day.
    Uses dynamic sector Z-Scores and the non-linear 2-Stage Pipeline Trigger.
    """
    if price <= 0 or eps <= 0:
        return 5
        
    is_bank_holding = (sector == 'Banka') or (market == 'SP500' and sector == 'Finans / Banka')
    is_loss_making = (eps <= 0) or (roe < 0)
    debt_val = debt_to_equity or 0.0
    
    # 1. VALUATION MODELS (FUNDAMENTAL)
    z_scores = []
    pe_ratio = price / eps if eps > 0 else 999.0
    pb_ratio = price / bvps if bvps > 0 else 1.5
    ev_ebitda_val = pe_ratio * 0.7 if pe_ratio < 100 else 15.0
    
    if sector_stats:
        if pe_ratio > 0 and sector_stats.get('pe_std', 0) > 0.01:
            z_pe = (pe_ratio - sector_stats['pe_mean']) / sector_stats['pe_std']
            z_scores.append(z_pe)
        if pb_ratio > 0 and sector_stats.get('pb_std', 0) > 0.01:
            z_pb = (pb_ratio - sector_stats['pb_mean']) / sector_stats['pb_std']
            z_scores.append(z_pb)
        if ev_ebitda_val > 0 and sector_stats.get('evebitda_std', 0) > 0.01:
            z_eveb = (ev_ebitda_val - sector_stats['evebitda_mean']) / sector_stats['evebitda_std']
            z_scores.append(z_eveb)
            
    # Calculate Earnings Yield & ROIC
    ey = eps / price if price > 0 else 0.05
    roic = roe / (1.0 + debt_val) if roe else 0.12
    
    ey_score = int(50 + np.tanh((ey - 0.08) / 0.06) * 50)
    roic_score = int(50 + np.tanh((roic - 0.12) / 0.10) * 50)
    
    ey_score = max(5, min(95, ey_score))
    roic_score = max(5, min(95, roic_score))
    
    # Classical/Fallback Valuation (in case Z-scores aren't available)
    if is_bank_holding:
        roe_val = roe or 0.25
        if market == "SP500":
            nim_val = 0.02 + 0.01 * roe_val
            npl_val = max(0.002, 0.015 - 0.005 * roe_val)
            syr_val = 0.12 + 0.02 * beta
            
            roe_score = min(100.0, max(0.0, (roe_val / 0.18) * 100))
            nim_score = min(100.0, max(0.0, (nim_val / 0.04) * 100))
            npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.002) / 0.015) * 100))
            syr_score = min(100.0, max(0.0, ((syr_val - 0.10) / 0.06) * 100))
        else:
            nim_val = 0.05 + 0.02 * roe_val
            npl_val = max(0.005, 0.025 - 0.01 * roe_val)
            syr_val = 0.16 + 0.04 * beta
            
            roe_score = min(100.0, max(0.0, (roe_val / 0.40) * 100))
            nim_score = min(100.0, max(0.0, (nim_val / 0.07) * 100))
            npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.005) / 0.04) * 100))
            syr_score = min(100.0, max(0.0, ((syr_val - 0.12) / 0.08) * 100))
            
        bank_score = int(roe_score * 0.30 + nim_score * 0.25 + npl_score * 0.25 + syr_score * 0.20)
        classical_value_score = max(5, min(95, bank_score))
    else:
        if is_loss_making:
            # Turn-around Fallback
            growth_puan = max(0, min(40, int(eps_growth_5y * eps_growth_multiplier * 0.8)))
            leverage_safety = 40 if debt_val < 1.5 else (25 if debt_val < 3.0 else 10)
            classical_value_score = max(10, min(90, growth_puan + leverage_safety))
        else:
            # DCF
            TR_BOND_YIELD = tr_bond_yield
            ERP = 5.0 if market == "SP500" else 6.0
            g_rate = eps_growth_5y * eps_growth_multiplier
            if market == "SP500":
                if debt_val > 4.0: g_rate *= 0.8
            else:
                if debt_val > 2.0: g_rate *= 0.6
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
                
            # Graham
            if market == "SP500":
                base_k = 11.2 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 21.9
                graham_multiplier = base_k / (1.0 + 0.025)
            else:
                graham_multiplier = 15.0 if sector == 'Banka' else 22.5
            fair_graham = np.sqrt(graham_multiplier * eps * bvps) if (eps > 0 and bvps > 0) else 0.0
            
            # Multiples
            if market == "SP500":
                if sector == 'Teknoloji': target_pe = 25.0
                elif sector == 'Perakende': target_pe = 20.0
                elif sector == 'İletişim / Medya': target_pe = 20.0
                elif sector == 'Otomotiv': target_pe = 18.0
                elif sector == 'Finans / Banka': target_pe = 15.0
                elif sector == 'Holding': target_pe = 14.0
                elif sector == 'Sağlık': target_pe = 18.0
                elif sector == 'Enerji': target_pe = 15.0
                else: target_pe = 16.0
            else:
                if sector == 'Banka': target_pe = 8.0
                elif sector == 'Havacılık': target_pe = 14.0
                elif sector in ['Holding', 'Demir-Çelik / Çimento']: target_pe = 12.0
                else: target_pe = 16.0
            fair_multiples = eps * target_pe
            
            if sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım', 'Teknoloji']:
                weights = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
            else:
                weights = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
            intrinsic_avg = (dcf_val * weights['dcf'] + fair_graham * weights['graham'] + fair_multiples * weights['multiples'])
            if sector in ['GYO', 'Holding', 'Holding / Enerji']:
                intrinsic_avg *= 0.60
                
            mos = (intrinsic_avg / price) - 1.0
            classical_value_score = int(50 + np.tanh(mos) * 50)
            
    # Combine Z-Score value (if available), Earnings Yield, and ROIC quality
    if z_scores:
        composite_z = np.mean(z_scores)
        z_value_score = int(50 - np.tanh(composite_z * 0.7) * 50)
        value_score = int(z_value_score * 0.5 + ey_score * 0.3 + roic_score * 0.2)
    else:
        value_score = int(classical_value_score * 0.5 + ey_score * 0.3 + roic_score * 0.2)
        
    value_score = max(5, min(95, value_score))
    
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
        
        if price > sma50 > sma200: momentum_score += 25
        elif price > sma50: momentum_score += 15
        elif price < sma50 < sma200: momentum_score -= 25
        elif price < sma50: momentum_score -= 15
            
        if rsi < 30: momentum_score += 10
        elif rsi > 70: momentum_score -= 10
        if v_change > 30: momentum_score += 5
        
    momentum_score = max(5, min(95, momentum_score))
    
    # 3. PIPELINE TRIGGER (NON-LINEAR)
    if value_score >= 70:
        if momentum_score >= 50:
            score = int(85 + (value_score - 70) * 0.5 + (momentum_score - 50) * 0.3)
        else:
            score = int(65 + (value_score - 70) * 0.5)
    elif value_score < 45:
        score = int(value_score * 0.7 + momentum_score * 0.1)
    else:
        score = int(value_score * 0.6 + momentum_score * 0.4)
        
    return max(5, min(95, score))


def get_aggressive_score(price, eps, bvps, roe, sector, debt_to_equity, beta, eps_growth_5y, close_history, vol_history, idx,
                          eps_growth_multiplier=1.5, market="BIST", sector_stats: dict = None):
    """
    Reconstructs the exact aggressive intelligence score logic day-by-day.
    Uses dynamic sector Z-Scores and the non-linear 2-Stage Pipeline Trigger.
    """
    if price <= 0 or eps <= 0:
        return 5
        
    is_bank_holding = (sector == 'Banka') or (market == 'SP500' and sector == 'Finans / Banka')
    is_loss_making = (eps <= 0) or (roe < 0)
    debt_val = debt_to_equity or 0.0
    
    # 1. VALUATION MODELS (FUNDAMENTAL)
    z_scores = []
    pe_ratio = price / eps if eps > 0 else 999.0
    pb_ratio = price / bvps if bvps > 0 else 1.5
    ev_ebitda_val = pe_ratio * 0.7 if pe_ratio < 100 else 15.0
    
    if sector_stats:
        if pe_ratio > 0 and sector_stats.get('pe_std', 0) > 0.01:
            z_pe = (pe_ratio - sector_stats['pe_mean']) / sector_stats['pe_std']
            z_scores.append(z_pe)
        if pb_ratio > 0 and sector_stats.get('pb_std', 0) > 0.01:
            z_pb = (pb_ratio - sector_stats['pb_mean']) / sector_stats['pb_std']
            z_scores.append(z_pb)
        if ev_ebitda_val > 0 and sector_stats.get('evebitda_std', 0) > 0.01:
            z_eveb = (ev_ebitda_val - sector_stats['evebitda_mean']) / sector_stats['evebitda_std']
            z_scores.append(z_eveb)
            
    # Calculate Earnings Yield & ROIC
    ey = eps / price if price > 0 else 0.05
    roic = roe / (1.0 + debt_val) if roe else 0.12
    
    ey_score = int(50 + np.tanh((ey - 0.08) / 0.06) * 50)
    roic_score = int(50 + np.tanh((roic - 0.12) / 0.10) * 50)
    
    ey_score = max(5, min(95, ey_score))
    roic_score = max(5, min(95, roic_score))
    
    if is_bank_holding:
        roe_val = roe or 0.25
        if market == "SP500":
            nim_val = 0.02 + 0.01 * roe_val
            npl_val = max(0.002, 0.015 - 0.005 * roe_val)
            syr_val = 0.12 + 0.02 * beta
            
            roe_score = min(100.0, max(0.0, (roe_val / 0.18) * 100))
            nim_score = min(100.0, max(0.0, (nim_val / 0.04) * 100))
            npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.002) / 0.015) * 100))
            syr_score = min(100.0, max(0.0, ((syr_val - 0.10) / 0.06) * 100))
        else:
            nim_val = 0.05 + 0.02 * roe_val
            npl_val = max(0.005, 0.025 - 0.01 * roe_val)
            syr_val = 0.16 + 0.04 * beta
            
            roe_score = min(100.0, max(0.0, (roe_val / 0.40) * 100))
            nim_score = min(100.0, max(0.0, (nim_val / 0.07) * 100))
            npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.005) / 0.04) * 100))
            syr_score = min(100.0, max(0.0, ((syr_val - 0.12) / 0.08) * 100))
            
        bank_score = int(roe_score * 0.30 + nim_score * 0.25 + npl_score * 0.25 + syr_score * 0.20)
        classical_value_score_agg = max(5, min(95, bank_score))
    else:
        if is_loss_making:
            # Turn-around Engine (Aggressive is more receptive to growth and technical momentum)
            growth_puan = max(0, min(50, int(eps_growth_5y * eps_growth_multiplier * 1.0)))
            leverage_safety = 35 if debt_val < 2.0 else (20 if debt_val < 4.0 else 5)
            classical_value_score_agg = max(15, min(90, growth_puan + leverage_safety))
        else:
            # Graham
            if market == "SP500":
                expected_inflation_agg = 0.015
                base_k_agg = 11.2 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 21.9
            else:
                expected_inflation_agg = 0.15
                base_k_agg = 11.5 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 22.5
            graham_multiplier_agg = base_k_agg / (1.0 + expected_inflation_agg)
            fair_graham_agg = np.sqrt(graham_multiplier_agg * eps * bvps) if (eps > 0 and bvps > 0) else 0.0
            
            # DCF
            if market == "SP500":
                rf_star_agg = 0.035
                erp_agg = 0.04
            else:
                rf_star_agg = 0.18
                erp_agg = 0.08
            delta_weight_agg = 0.01
            debt_penalty_agg = delta_weight_agg * max(0.0, np.log(1.0 + debt_val))
            cost_of_equity_agg = rf_star_agg + beta * erp_agg + debt_penalty_agg
            
            eps_g_agg = eps_growth_5y * eps_growth_multiplier
            if market == "SP500":
                if debt_val > 4.0: eps_g_agg *= 0.8
            else:
                if debt_val > 4.0: eps_g_agg *= 0.8
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
                
            # Multiples
            if market == "SP500":
                if sector == 'Teknoloji': target_pe = 25.0
                elif sector == 'Perakende': target_pe = 20.0
                elif sector == 'İletişim / Medya': target_pe = 20.0
                elif sector == 'Otomotiv': target_pe = 18.0
                elif sector == 'Finans / Banka': target_pe = 15.0
                elif sector == 'Holding': target_pe = 14.0
                elif sector == 'Sağlık': target_pe = 18.0
                elif sector == 'Enerji': target_pe = 15.0
                else: target_pe = 16.0
            else:
                if sector == 'Banka': target_pe = 8.0
                elif sector == 'Havacılık': target_pe = 14.0
                elif sector in ['Holding', 'Demir-Çelik / Çimento']: target_pe = 12.0
                else: target_pe = 16.0
            fair_multiples = eps * target_pe
            
            if sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım', 'Teknoloji']:
                weights_agg = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
            else:
                weights_agg = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
            intrinsic_avg_agg = (dcf_val_agg * weights_agg['dcf'] + fair_graham_agg * weights_agg['graham'] + fair_multiples * weights_agg['multiples'])
            if sector in ['GYO', 'Holding', 'Holding / Enerji']:
                intrinsic_avg_agg *= 0.60
                
            mos_agg = (intrinsic_avg_agg / price) - 1.0 if price > 0 else 0.0
            classical_value_score_agg = int(50 + np.tanh(mos_agg) * 50)
            
    # Combine Z-Score value (if available), Earnings Yield, and ROIC quality
    if z_scores:
        composite_z = np.mean(z_scores)
        z_value_score_agg = int(50 - np.tanh(composite_z * 0.7) * 50)
        value_score_agg = int(z_value_score_agg * 0.5 + ey_score * 0.3 + roic_score * 0.2)
    else:
        value_score_agg = int(classical_value_score_agg * 0.5 + ey_score * 0.3 + roic_score * 0.2)
        
    value_score_agg = max(5, min(95, value_score_agg))
    
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
        
        if price > sma50 > sma200: momentum_score += 25
        elif price > sma50: momentum_score += 15
        elif price < sma50 < sma200: momentum_score -= 25
        elif price < sma50: momentum_score -= 15
            
        if rsi < 30: momentum_score += 10
        elif rsi > 70: momentum_score -= 10
        if v_change > 30: momentum_score += 5
        
        # Aggressive Turn-around Bonus
        if rsi < 45:
            momentum_score += 15
            
    momentum_score = max(5, min(95, momentum_score))
    
    # 3. PIPELINE TRIGGER (NON-LINEAR AGGRESSIVE)
    if value_score_agg >= 60:
        if momentum_score >= 45:
            score_agg = int(80 + (value_score_agg - 60) * 0.6 + (momentum_score - 45) * 0.4)
        else:
            score_agg = int(65 + (value_score_agg - 60) * 0.5)
    elif value_score_agg < 35:
        score_agg = int(value_score_agg * 0.7 + momentum_score * 0.2)
    else:
        score_agg = int(value_score_agg * 0.5 + momentum_score * 0.5)
        
    return max(5, min(95, score_agg))


def evaluate_stock_valuations(
    ticker_code, name, sector, market, cur_price,
    pe_ratio, pb_ratio, roe, ev_ebitda, dividend_yield,
    beta, trailing_eps, debt_to_equity, eps_growth_5y,
    rsi, sma50, sma200, v_change, trend,
    sector_stats: dict = None
):
    """
    Core mathematical valuation engine for a single asset.
    Computes dynamic Sectoral Z-Scores, ROIC quality, and FCF/Earnings Yield.
    Applies the non-linear 2-Stage Pipeline Trigger to calculate the final composite score.
    Returns a unified dict with all results.
    """
    curr = "$" if market == "SP500" else "TL"
    
    # VALUATION MODELS
    pe = pe_ratio or -1.0
    pb = pb_ratio or 1.5
    roe_val = roe or 0.25
    eps_g = eps_growth_5y or 25.0
    debt_val = debt_to_equity or 0.0
    
    # Calculate EPS
    if trailing_eps and trailing_eps > 0:
        eps = trailing_eps
    elif pe > 0:
        eps = cur_price / pe
    else:
        eps = cur_price * (roe_val * 0.4) / pb
        
    if eps <= 0:
        eps = cur_price * 0.05
        
    # GYO and Holding EPS adjustment (ROE momentum scaling)
    if sector in ['GYO', 'Holding', 'Holding / Enerji']:
        roe_factor = max(0.05, min(1.0, roe_val * 2.0))
        eps *= roe_factor
        
    bvps = cur_price / pb
    is_loss_making = (pe < 0) or (roe_val < 0)
    
    rationale_parts = []
    is_bank_holding = (sector == 'Banka') or (market == 'SP500' and sector == 'Finans / Banka' and ticker_code in ['JPM', 'BAC'])
    
    # 1. DYNAMIC SECTORAL Z-SCORE CALCULATION
    z_scores = []
    if sector_stats:
        if pe > 0 and sector_stats.get('pe_std', 0) > 0.01:
            z_pe = (pe - sector_stats['pe_mean']) / sector_stats['pe_std']
            z_scores.append(z_pe)
        if pb > 0 and sector_stats.get('pb_std', 0) > 0.01:
            z_pb = (pb - sector_stats['pb_mean']) / sector_stats['pb_std']
            z_scores.append(z_pb)
        if ev_ebitda and ev_ebitda > 0 and sector_stats.get('evebitda_std', 0) > 0.01:
            z_eveb = (ev_ebitda - sector_stats['evebitda_mean']) / sector_stats['evebitda_std']
            z_scores.append(z_eveb)
            
    # Calculate Earnings Yield & ROIC
    ey = eps / cur_price if cur_price > 0 else 0.05
    roic = roe_val / (1.0 + debt_val)
    
    ey_score = int(50 + np.tanh((ey - 0.08) / 0.06) * 50)
    roic_score = int(50 + np.tanh((roic - 0.12) / 0.10) * 50)
    
    ey_score = max(5, min(95, ey_score))
    roic_score = max(5, min(95, roic_score))
    
    # A. DUPONT / NIM BANKACILIK MODÜLÜ
    if is_bank_holding:
        if market == "SP500":
            nim_val = 0.02 + 0.01 * roe_val
            npl_val = max(0.002, 0.015 - 0.005 * roe_val)
            syr_val = 0.12 + 0.02 * (beta or 1.0)
            
            roe_score = min(100.0, max(0.0, (roe_val / 0.18) * 100))
            nim_score = min(100.0, max(0.0, (nim_val / 0.04) * 100))
            npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.002) / 0.015) * 100))
            syr_score = min(100.0, max(0.0, ((syr_val - 0.10) / 0.06) * 100))
            
            bank_score = int(roe_score * 0.30 + nim_score * 0.25 + npl_score * 0.25 + syr_score * 0.20)
            bank_score = max(5, min(95, bank_score))
            target_pe = 14.0
        else:
            nim_val = 0.05 + 0.02 * roe_val
            npl_val = max(0.005, 0.025 - 0.01 * roe_val)
            syr_val = 0.16 + 0.04 * (beta or 1.0)
            
            roe_score = min(100.0, max(0.0, (roe_val / 0.40) * 100))
            nim_score = min(100.0, max(0.0, (nim_val / 0.07) * 100))
            npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.005) / 0.04) * 100))
            syr_score = min(100.0, max(0.0, ((syr_val - 0.12) / 0.08) * 100))
            
            bank_score = int(roe_score * 0.30 + nim_score * 0.25 + npl_score * 0.25 + syr_score * 0.20)
            bank_score = max(5, min(95, bank_score))
            target_pe = 8.0 if sector == 'Banka' else 12.0
            
        fair_multiples = eps * target_pe
        intrinsic_avg = fair_multiples * (bank_score / 70.0)
        
        dcf_val = 0.0
        fair_graham = 0.0
        classical_value_score = bank_score
        mos = (intrinsic_avg / cur_price) - 1.0 if cur_price > 0 else 0.0
        opt_val = intrinsic_avg * 1.25
        pes_val = intrinsic_avg * 0.75
        
        rationale_parts.append(f"DuPont/NIM Derecelendirmesi: ROE %{roe_val*100:.1f}, NIM %{nim_val*100:.1f}, NPL %{npl_val*100:.1f}, SYR %{syr_val*100:.1f} rasyolarına göre finansal zeka ratingi: {bank_score}/100.")
        
    else:
        # B. SANAYİ VE DİĞER ŞIRKETLER MODÜLÜ (GRAHAM + DCF / TURN-AROUND)
        if is_loss_making:
            # Turn-around Engine (Distress & Recovery Model)
            growth_puan = max(0, min(40, int(eps_g * 0.8)))
            leverage_safety = 40 if debt_val < 1.5 else (25 if debt_val < 3.0 else 10)
            rsi_bonus = 20 if rsi < 45 else 0
            
            classical_value_score = max(10, min(90, growth_puan + leverage_safety + rsi_bonus))
            
            dcf_val = 0.0
            fair_multiples = 0.0
            fair_graham = cur_price * (classical_value_score / 100.0)
            intrinsic_avg = fair_graham
            opt_val = intrinsic_avg * 1.25
            pes_val = intrinsic_avg * 0.75
            mos = (intrinsic_avg / cur_price) - 1.0 if cur_price > 0 else 0.0
            
            rationale_parts.append(f"Zarar Açıklayan Şirket / Turn-around Analizi: Negatif kâra rağmen büyüme beklentisi ({eps_g:.1f}%) ve kaldıraç emniyeti (Borç/Öz: {debt_val:.2f}) kapsamında turn-around değeri {classical_value_score}/100 olarak hesaplandı.")
        else:
            # Standard Fundamental Valuations
            if market == "SP500":
                expected_inflation = 0.025
                base_k = 11.2 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 21.9
                graham_multiplier = base_k / (1.0 + expected_inflation)
            else:
                expected_inflation = 0.25
                base_k = 11.5 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 22.5
                graham_multiplier = base_k / (1.0 + expected_inflation)
            
            if eps > 0 and bvps > 0:
                fair_graham = np.sqrt(graham_multiplier * eps * bvps)
            else:
                fair_graham = 0.0
                
            # DCF
            if market == "SP500":
                rf_star = 0.0425
                erp_rate = 0.05
                delta_weight = 0.01
            else:
                rf_star = 0.25
                erp_rate = 0.08
                delta_weight = 0.03
                
            debt_penalty = delta_weight * max(0.0, np.log(1.0 + debt_val))
            cost_of_equity = rf_star + (beta or 1.0) * erp_rate + debt_penalty
            
            if market == "SP500":
                if debt_val > 4.0: eps_g *= 0.8
            else:
                if debt_val > 2.0: eps_g *= 0.6
                
            g_decimal = eps_g / 100.0
            dcf_val = 0.0
            fcf = eps
            for i in range(1, 6):
                fcf *= (1 + g_decimal)
                dcf_val += fcf / ((1 + cost_of_equity) ** i)
                
            terminal_growth_cap = 12.0
            terminal_growth_pct = min((eps_g * 0.4), terminal_growth_cap)
            terminal_g = terminal_growth_pct / 100.0
            if cost_of_equity > terminal_g:
                terminal_val = (fcf * (1 + terminal_g)) / (cost_of_equity - terminal_g)
                dcf_val += terminal_val / ((1 + cost_of_equity) ** 5)
                
            # Sektörel Çarpan Analizi
            if market == "SP500":
                if sector == 'Teknoloji': target_pe = 25.0
                elif sector == 'Perakende': target_pe = 20.0
                elif sector == 'İletişim / Medya': target_pe = 20.0
                elif sector == 'Otomotiv': target_pe = 18.0
                elif sector == 'Finans / Banka': target_pe = 15.0
                elif sector == 'Holding': target_pe = 14.0
                elif sector == 'Sağlık': target_pe = 18.0
                elif sector == 'Enerji': target_pe = 15.0
                else: target_pe = 16.0
            else:
                if sector == 'Havacılık': target_pe = 14.0
                elif sector == 'Demir-Çelik / Çimento': target_pe = 12.0
                else: target_pe = 16.0
            fair_multiples = eps * target_pe
            
            if sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım', 'Teknoloji']:
                weights = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
            else:
                weights = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
                
            intrinsic_avg = (dcf_val * weights['dcf'] + fair_graham * weights['graham'] + fair_multiples * weights['multiples'])
            
            # Senaryolar
            opt_g = g_decimal * 1.2
            opt_r = cost_of_equity - 0.01
            opt_dcf = 0.0
            fcf_opt = eps
            for i in range(1, 6):
                fcf_opt *= (1 + opt_g)
                opt_dcf += fcf_opt / ((1 + opt_r) ** i)
            opt_term = min((eps_g * 1.2 * 0.4), terminal_growth_cap) / 100.0
            if opt_r > opt_term:
                opt_dcf += ((fcf_opt * (1 + opt_term)) / (opt_r - opt_term)) / ((1 + opt_r) ** 5)
            opt_val = opt_dcf * 1.1
            
            pes_g = g_decimal * 0.8
            pes_r = cost_of_equity + 0.02
            pes_dcf = 0.0
            fcf_pes = eps
            for i in range(1, 6):
                fcf_pes *= (1 + pes_g)
                pes_dcf += fcf_pes / ((1 + pes_r) ** i)
            pes_term = min((eps_g * 0.8 * 0.4), terminal_growth_cap) / 100.0
            if pes_r > pes_term:
                pes_dcf += ((fcf_pes * (1 + pes_term)) / (pes_r - pes_term)) / ((1 + pes_r) ** 5)
            pes_val = pes_dcf * 0.9
            
            if pes_val > intrinsic_avg: pes_val = intrinsic_avg * 0.8
            if opt_val < intrinsic_avg: opt_val = intrinsic_avg * 1.3
            
            mos = (intrinsic_avg / cur_price) - 1.0 if cur_price > 0 else 0.0
            classical_value_score = int(50 + np.tanh(mos) * 50)
            
            if sector in ['GYO', 'Holding', 'Holding / Enerji']:
                classical_value_score = int(classical_value_score * 0.60)
                if sector == 'GYO':
                    rationale_parts.append("GYO Varlık İskontosu Uyarısı: Kalıcı varlık iskontosu nedeniyle değerleme skoru %40 oranında düşürüldü. Sanal yeniden değerleme kârları elendi.")
                else:
                    rationale_parts.append("Holding Varlık İskontosu Uyarısı: İştirak NAV iskontosu nedeniyle değerleme skoru %40 cezalandırıldı.")
                    
    # Combine Z-Score value (if available), Earnings Yield, and ROIC quality
    if z_scores:
        composite_z = np.mean(z_scores)
        z_value_score = int(50 - np.tanh(composite_z * 0.7) * 50)
        value_score = int(z_value_score * 0.5 + ey_score * 0.3 + roic_score * 0.2)
        rationale_parts.append(f"Sektörel Z-Skoru Rölatif Analizi: Sektör akranlarına kıyasla Z-puanı: {composite_z:.2f} (Değer Skoru: {value_score}/100).")
    else:
        value_score = int(classical_value_score * 0.5 + ey_score * 0.3 + roic_score * 0.2)
        
    value_score = max(5, min(95, value_score))
    
    # 2. TECHNICAL TREND & MOMENTUM
    momentum_score = 50
    if cur_price > sma50 > sma200: momentum_score += 25
    elif cur_price > sma50: momentum_score += 15
    elif cur_price < sma50 < sma200: momentum_score -= 25
    elif cur_price < sma50: momentum_score -= 15
        
    if rsi < 30:
        momentum_score += 10
        rationale_parts.append("Teknik aşırı satım bölgesinde (rsi < 30) - Toplama fırsatı olabilir.")
    elif rsi > 70:
        momentum_score -= 10
        rationale_parts.append("Teknik aşırı alım bölgesinde (rsi > 70) - Kar realizasyonu düşünülebilir.")
        
    if v_change > 30:
        momentum_score += 5
        rationale_parts.append("Hisse hacminde ani artış var - Kurumsal ilgi yükseliyor.")
        
    momentum_score = max(5, min(95, momentum_score))
    
    # 3. PIPELINE TRIGGER (NON-LINEAR COMPOSITE RATING)
    if value_score >= 70:
        if momentum_score >= 50:
            score = int(85 + (value_score - 70) * 0.5 + (momentum_score - 50) * 0.3)
            rationale_parts.append("Sıralı Tetikleme: Hisse son derece ucuz ve teknik olarak güçlü yükseliş trendi teyit edildi (Güçlü AL).")
        else:
            score = int(65 + (value_score - 70) * 0.5)
            rationale_parts.append("Sıralı Tetikleme: Hisse temel olarak çok cazip ancak momentum henüz teyit edilmedi (Dip Toplama / AL).")
    elif value_score < 45:
        score = int(value_score * 0.7 + momentum_score * 0.1)
        rationale_parts.append("Sıralı Tetikleme: Hisse temel çarpanlarına göre aşırı değerli. Zirvede alımı engellemek amacıyla momentum puanı baskılandı (SAT).")
    else:
        score = int(value_score * 0.6 + momentum_score * 0.4)
        
    score = max(5, min(95, score))
    
    # --- AGGRESSIVE INTELLIGENCE MODEL WITH PIPELINE TRIGGER ---
    if is_bank_holding:
        momentum_score_agg = momentum_score + 15 if rsi < 45 else momentum_score
        momentum_score_agg = max(5, min(95, momentum_score_agg))
        
        value_score_agg = bank_score
        if value_score_agg >= 60:
            if momentum_score_agg >= 45:
                score_agg = int(80 + (value_score_agg - 60) * 0.6 + (momentum_score_agg - 45) * 0.4)
            else:
                score_agg = int(65 + (value_score_agg - 60) * 0.5)
        elif value_score_agg < 35:
            score_agg = int(value_score_agg * 0.7 + momentum_score_agg * 0.2)
        else:
            score_agg = int(value_score_agg * 0.5 + momentum_score_agg * 0.5)
            
        score_agg = max(5, min(95, score_agg))
        
        rationale_agg_parts = [
            f"DuPont/NIM Agresif Değerleme: DuPont Puanı: {bank_score}/100, Agresif Turn-around Puanı: {momentum_score_agg}/100. Sıralı Tetikleme Skoru: {score_agg}/100."
        ]
        if rsi < 45:
            rationale_agg_parts.append("Düşük RSI (<45) dipten dönüş sinyaliyle ekstra momentum bonusu uygulandı.")
        rationale_agg = " ".join(rationale_agg_parts)
    else:
        if is_loss_making:
            growth_puan = max(0, min(50, int(eps_g * 1.2 * 1.0)))
            leverage_safety = 35 if debt_val < 2.0 else (20 if debt_val < 4.0 else 5)
            value_score_agg = max(15, min(90, growth_puan + leverage_safety))
        else:
            if market == "SP500":
                expected_inflation_agg = 0.015
                base_k_agg = 11.2 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 21.9
            else:
                expected_inflation_agg = 0.15
                base_k_agg = 11.5 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 22.5
                
            graham_multiplier_agg = base_k_agg / (1.0 + expected_inflation_agg)
            fair_graham_agg = np.sqrt(graham_multiplier_agg * eps * bvps) if (eps > 0 and bvps > 0) else 0.0
            
            if market == "SP500":
                rf_star_agg = 0.035
                erp_agg = 0.04
            else:
                rf_star_agg = 0.18
                erp_agg = 0.08
                
            delta_weight_agg = 0.01
            debt_penalty_agg = delta_weight_agg * max(0.0, np.log(1.0 + debt_val))
            cost_of_equity_agg = rf_star_agg + (beta or 1.0) * erp_agg + debt_penalty_agg
            
            eps_g_agg = eps_growth_5y or 25.0
            if market == "SP500":
                if debt_val > 4.0: eps_g_agg *= 0.8
            else:
                if debt_val > 4.0: eps_g_agg *= 0.8
                
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
                
            if sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım', 'Teknoloji']:
                weights_agg = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
            else:
                weights_agg = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
            intrinsic_avg_agg = (dcf_val_agg * weights_agg['dcf'] + fair_graham_agg * weights_agg['graham'] + fair_multiples * weights_agg['multiples'])
            
            if sector in ['GYO', 'Holding', 'Holding / Enerji']:
                intrinsic_avg_agg *= 0.60
                
            mos_agg = (intrinsic_avg_agg / cur_price) - 1.0 if cur_price > 0 else 0.0
            value_score_agg = int(50 + np.tanh(mos_agg) * 50)
            
        # Combine Z-Score value (if available), Earnings Yield, and ROIC quality
        if z_scores:
            composite_z = np.mean(z_scores)
            z_value_score_agg = int(50 - np.tanh(composite_z * 0.7) * 50)
            value_score_agg = int(z_value_score_agg * 0.5 + ey_score * 0.3 + roic_score * 0.2)
            
        value_score_agg = max(5, min(95, value_score_agg))
        
        momentum_score_agg = momentum_score + 15 if rsi < 45 else momentum_score
        momentum_score_agg = max(5, min(95, momentum_score_agg))
        
        # Pipeline trigger (non-linear) for aggressive model
        if value_score_agg >= 60:
            if momentum_score_agg >= 45:
                score_agg = int(80 + (value_score_agg - 60) * 0.6 + (momentum_score_agg - 45) * 0.4)
            else:
                score_agg = int(65 + (value_score_agg - 60) * 0.5)
        elif value_score_agg < 35:
            score_agg = int(value_score_agg * 0.7 + momentum_score_agg * 0.2)
        else:
            score_agg = int(value_score_agg * 0.5 + momentum_score_agg * 0.5)
            
        score_agg = max(5, min(95, score_agg))
        
        rationale_agg_parts = [
            f"Agresif Değerleme: Değer Skoru: {value_score_agg}/100, Momentum: {momentum_score_agg}/100. Sıralı Tetikleme Skoru: {score_agg}/100."
        ]
        if rsi < 45:
            rationale_agg_parts.append("Turn-around: RSI 45'in altında aşırı satım bölgesinden çıkış potansiyeli ödüllendirildi (+15 Bonus).")
        if sector in ['GYO', 'Holding', 'Holding / Enerji']:
            if sector == 'GYO':
                rationale_agg_parts.append("GYO Varlık İskontosu: Sanal kârlar ROE ile elendi ve %40 kalıcı iskonto uygulandı.")
            else:
                rationale_agg_parts.append("Holding Varlık İskontosu: NAV iskontosu nedeniyle değerleme skoru %40 cezalandırıldı.")
                
        if not is_loss_making:
            if mos_agg > 0.2:
                rationale_agg_parts.append(f"Hisse agresif hedef değerine göre %{mos_agg*100:.1f} iskontolu görünmektedir.")
            elif mos_agg < -0.1:
                rationale_agg_parts.append(f"Agresif senaryoda dahi %{-mos_agg*100:.1f} primlidir.")
                
        rationale_agg = " ".join(rationale_agg_parts)
        
    label_agg = "TUT"
    if score_agg >= 85: label_agg = "Güçlü AL / Şampiyon"
    elif score_agg >= 70: label_agg = "Ucuz / AL"
    elif score_agg <= 30: label_agg = "Çok Pahalı / SAT"
    elif score_agg <= 45: label_agg = "Pahalı / TUT"
    
    label = "TUT"
    if score >= 85: label = "Güçlü AL / Şampiyon"
    elif score >= 70: label = "Ucuz / AL"
    elif score <= 30: label = "Çok Pahalı / SAT"
    elif score <= 45: label = "Pahalı / TUT"
    
    if score >= 80 and "Yükseliş" in trend:
        rationale_parts.append("Mükemmel uyum! Şirket hem finansal olarak çok cazip hem de teknik olarak güçlü bir yükseliş trendinde.")
    elif score >= 80 and "Düşüş" in trend:
        rationale_parts.append("Hisse aşırı ucuzlamış ancak teknik momentum hala negatif. Kademeli alım / dip toplama düşünülebilir.")
    elif score <= 40 and "Yükseliş" in trend:
        rationale_parts.append("Hisse temel çarpanlarına göre oldukça şişmiş fakat güçlü bir momentumla yükseliyor. Dikkatli olunmalıdır.")
        
    if not rationale_parts:
        rationale_parts.append("Fiyat temel değerleme rasyoları ile uyumlu. Dengeli ve stabil seyir bekleniyor.")
        
    return {
        "intrinsic_value_dcf": dcf_val if 'dcf_val' in locals() else 0.0,
        "intrinsic_value_graham": fair_graham if 'fair_graham' in locals() else 0.0,
        "fair_price_multiples": fair_multiples if 'fair_multiples' in locals() else 0.0,
        "intrinsic_value_optimistic": opt_val if 'opt_val' in locals() else 0.0,
        "intrinsic_value_pessimistic": pes_val if 'pes_val' in locals() else 0.0,
        "intrinsic_value_avg": intrinsic_avg if 'intrinsic_avg' in locals() else cur_price,
        "margin_of_safety": mos if 'mos' in locals() else 0.0,
        "value_score": value_score,
        "momentum_score": momentum_score,
        "intelligence_score": score,
        "valuation_label": label,
        "intelligence_score_aggressive": score_agg,
        "valuation_label_aggressive": label_agg,
        "rationale_aggressive": rationale_agg,
        "rationale": " ".join(rationale_parts)
    }
