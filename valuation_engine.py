# -*- coding: utf-8 -*-
"""
BIST FINTRACK - MATHEMATICAL VALUATION ENGINE (FORMÜL OMURGASI)
==============================================================
Bu dosya projenin değerleme ve puanlama kalbini oluşturur.
Tüm DCF, Graham, DuPont bankacılık ve kompozit Yapay Zeka Zeka Skorları buradadır.
app.py ve backtest_simulation.py bu dosyayı import eder.
"""

import numpy as np
import pandas as pd

def calculate_rsi(series, period=14):
    """Safely calculates the 14-period RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_dss_score(price, eps, bvps, roe, sector, debt_to_equity, beta, eps_growth_5y, close_history, vol_history, idx,
                  tr_bond_yield=45.0, value_weight=0.7, momentum_weight=0.3, eps_growth_multiplier=1.0, market="BIST"):
    """
    Reconstructs the exact intelligence score logic day-by-day.
    """
    if price <= 0 or eps <= 0:
        return 5  # Floor minimum score
    
    # 1. VALUATION MODELS (FUNDAMENTAL)
    # A. DCF (Discounted Cash Flow)
    TR_BOND_YIELD = tr_bond_yield
    ERP = 5.0 if market == "SP500" else 6.0
    
    # Apply high leverage penalty based on market standards
    g_rate = eps_growth_5y * eps_growth_multiplier
    if market == "SP500":
        if debt_to_equity > 4.0:
            g_rate *= 0.8
    else:
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
    if market == "SP500":
        base_k = 11.2 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 21.9
        expected_inflation = 0.025 # 2.5% expected inflation
        graham_multiplier = base_k / (1.0 + expected_inflation)
    else:
        graham_multiplier = 15.0 if sector == 'Banka' else 22.5
        
    if eps > 0 and bvps > 0:
        fair_graham = np.sqrt(graham_multiplier * eps * bvps)
    else:
        fair_graham = 0.0
        
    # C. Sektörel Çarpan Analizi (Multiples)
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
    
    # D. Composite Fair Price
    if market != "SP500" and sector == 'Banka':
        weights = {'dcf': 0.1, 'graham': 0.6, 'multiples': 0.3}
    elif sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım', 'Teknoloji']:
        weights = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
    else:
        weights = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
        
    intrinsic_avg = (dcf_val * weights['dcf'] + fair_graham * weights['graham'] + fair_multiples * weights['multiples'])
    
    # Apply valuation discounts for Holdings and REITs
    if sector in ['GYO', 'Holding', 'Holding / Enerji']:
        intrinsic_avg *= 0.60
        
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
                          eps_growth_multiplier=1.5, market="BIST"):
    """
    Reconstructs the exact aggressive intelligence score logic day-by-day.
    """
    if price <= 0 or eps <= 0:
        return 5
    
    is_bank_holding = (sector == 'Banka') or (market == 'SP500' and sector == 'Finans / Banka' and sector in ['JPM', 'BAC'])
    is_loss_making = (eps <= 0) or (roe < 0)
    
    # 1. VALUATION MODELS (FUNDAMENTAL)
    if is_bank_holding:
        roe_val = roe
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
        bank_score = max(5, min(95, bank_score))
        
        value_score_agg = bank_score
    else:
        # A. Benjamin Graham Formülü (Enflasyon Uyumlu)
        if market == "SP500":
            expected_inflation_agg = 0.015
            base_k_agg = 11.2 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 21.9
        else:
            expected_inflation_agg = 0.15
            base_k_agg = 11.5 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 22.5
            
        graham_multiplier_agg = base_k_agg / (1.0 + expected_inflation_agg)
        if eps > 0 and bvps > 0:
            fair_graham_agg = np.sqrt(graham_multiplier_agg * eps * bvps)
        else:
            fair_graham_agg = 0.0
            
        # B. DCF with lower discount rate
        if market == "SP500":
            rf_star_agg = 0.035
            erp_agg = 0.04
        else:
            rf_star_agg = 0.18
            erp_agg = 0.08
            
        delta_weight_agg = 0.01
        debt_penalty_agg = delta_weight_agg * max(0.0, np.log(1.0 + debt_to_equity))
        cost_of_equity_agg = rf_star_agg + beta * erp_agg + debt_penalty_agg
        
        eps_g_agg = eps_growth_5y * eps_growth_multiplier
        if market == "SP500":
            if debt_to_equity > 4.0:
                eps_g_agg *= 0.8
        else:
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
        
        if is_loss_making:
            dcf_val_agg = 0.0
            fair_graham_agg *= 0.5
            intrinsic_avg_agg = fair_graham_agg
        else:
            if sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım', 'Teknoloji']:
                weights_agg = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
            else:
                weights_agg = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
            intrinsic_avg_agg = (dcf_val_agg * weights_agg['dcf'] + fair_graham_agg * weights_agg['graham'] + fair_multiples * weights_agg['multiples'])
            
        # GYO and Holding Value Score adjustment
        if sector in ['GYO', 'Holding', 'Holding / Enerji']:
            intrinsic_avg_agg *= 0.60
            
        mos_agg = (intrinsic_avg_agg / price) - 1.0 if price > 0 else 0.0
        value_score_agg = int(50 + np.tanh(mos_agg) * 50)
        
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
    score_agg = int(value_score_agg * 0.5 + momentum_score * 0.5)
    return max(5, min(95, score_agg))


def evaluate_stock_valuations(
    ticker_code, name, sector, market, cur_price,
    pe_ratio, pb_ratio, roe, ev_ebitda, dividend_yield,
    beta, trailing_eps, debt_to_equity, eps_growth_5y,
    rsi, sma50, sma200, v_change, trend
):
    """
    Core mathematical valuation engine for a single asset.
    Computes DCF, Graham, Multiples, bank ratings, and both Korumacı and Agresif scores.
    Returns a unified dict with all results to store in the DB.
    """
    curr = "$" if market == "SP500" else "TL"
    
    # VALUATION MODELS
    pe = pe_ratio or -1.0
    pb = pb_ratio or 1.0
    roe_val = roe or 0.25
    eps_g = eps_growth_5y or 25.0
    
    # Calculate EPS
    if trailing_eps and trailing_eps > 0:
        eps = trailing_eps
    elif pe > 0:
        eps = cur_price / pe
    else:
        eps = cur_price * (roe_val * 0.4) / (pb or 1.0) # Estimated synthetic EPS
        
    if eps <= 0:
        eps = cur_price * 0.05 # safe fallback floor
        
    # GYO and Holding EPS adjustment (ROE momentum scaling)
    if sector in ['GYO', 'Holding', 'Holding / Enerji']:
        roe_factor = max(0.05, min(1.0, roe_val * 2.0))
        eps *= roe_factor
        
    bvps = cur_price / (pb if pb > 0 else 1.0)
    is_loss_making = (pe < 0) or (roe_val < 0)
    
    rationale_parts = []
    is_bank_holding = (sector == 'Banka') or (market == 'SP500' and sector == 'Finans / Banka' and ticker_code in ['JPM', 'BAC'])
    
    # A. DUPONT / NIM BANKACILIK MODÜLÜ
    if is_bank_holding:
        if market == "SP500":
            nim_val = 0.02 + 0.01 * roe_val
            npl_val = max(0.002, 0.015 - 0.005 * roe_val)
            syr_val = 0.12 + 0.02 * (beta or 1.0)
            
            # US banking benchmarks
            roe_score = min(100.0, max(0.0, (roe_val / 0.18) * 100)) # %18 ROE tam puan
            nim_score = min(100.0, max(0.0, (nim_val / 0.04) * 100)) # %4 NIM tam puan
            npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.002) / 0.015) * 100))
            syr_score = min(100.0, max(0.0, ((syr_val - 0.10) / 0.06) * 100))
            
            bank_score = int(roe_score * 0.30 + nim_score * 0.25 + npl_score * 0.25 + syr_score * 0.20)
            bank_score = max(5, min(95, bank_score))
            
            target_pe = 14.0
        else:
            nim_val = 0.05 + 0.02 * roe_val
            npl_val = max(0.005, 0.025 - 0.01 * roe_val)
            syr_val = 0.16 + 0.04 * (beta or 1.0)
            
            roe_score = min(100.0, max(0.0, (roe_val / 0.40) * 100)) # %40 ROE tam puan
            nim_score = min(100.0, max(0.0, (nim_val / 0.07) * 100)) # %7 NIM tam puan
            npl_score = min(100.0, max(0.0, (1.0 - (npl_val - 0.005) / 0.04) * 100)) # %0.5 NPL 100, %4.5+ 0 puan
            syr_score = min(100.0, max(0.0, ((syr_val - 0.12) / 0.08) * 100)) # %12 SYR 0, %20+ tam puan
            
            bank_score = int(roe_score * 0.30 + nim_score * 0.25 + npl_score * 0.25 + syr_score * 0.20)
            bank_score = max(5, min(95, bank_score))
            
            target_pe = 8.0 if sector == 'Banka' else 12.0
            
        fair_multiples = eps * target_pe
        intrinsic_avg = fair_multiples * (bank_score / 70.0)
        
        dcf_val = 0.0
        fair_graham = 0.0
        value_score = bank_score
        mos = (intrinsic_avg / cur_price) - 1.0 if cur_price > 0 else 0.0
        
        opt_val = intrinsic_avg * 1.25
        pes_val = intrinsic_avg * 0.75
        
        rationale_parts.append(f"DuPont/NIM Derecelendirmesi: ROE %{roe_val*100:.1f}, NIM %{nim_val*100:.1f}, NPL %{npl_val*100:.1f}, SYR %{syr_val*100:.1f} rasyolarına göre finansal zeka ratingi: {bank_score}/100.")
        
    else:
        # B. SANAYİ VE DİĞER ŞIRKETLER MODÜLÜ (GRAHAM + DCF)
        if market == "SP500":
            expected_inflation = 0.025 # %2.5 expected inflation
            base_k = 11.2 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 21.9
            graham_multiplier = base_k / (1.0 + expected_inflation)
        else:
            expected_inflation = 0.25 # %25 beklenen enflasyon
            base_k = 11.5 if sector in ['GYO', 'Holding', 'Holding / Enerji'] else 22.5
            graham_multiplier = base_k / (1.0 + expected_inflation)
        
        if eps > 0 and bvps > 0:
            fair_graham = np.sqrt(graham_multiplier * eps * bvps)
        else:
            fair_graham = 0.0
            
        # 2. DCF (Normalize risksiz oran r*)
        if market == "SP500":
            rf_star = 0.0425 # US 4.25% bond yield
            erp_rate = 0.05 # 5% ERP
            delta_weight = 0.01 # lower penalty for US
        else:
            rf_star = 0.25
            erp_rate = 0.08
            delta_weight = 0.03
            
        debt_val = debt_to_equity or 0.0
        debt_penalty = delta_weight * max(0.0, np.log(1.0 + debt_val))
        cost_of_equity = rf_star + (beta or 1.0) * erp_rate + debt_penalty
        
        if market == "SP500":
            if debt_val > 4.0:
                eps_g *= 0.8
                rationale_parts.append(f"DİKKAT: Yüksek borç yükü (D/E: {debt_val:.2f}) sebebiyle büyüme beklentisi ve DCF değeri cezalandırıldı.")
        else:
            if debt_val > 2.0:
                eps_g *= 0.6
                rationale_parts.append(f"DİKKAT: Yüksek borç yükü (D/E: {debt_val:.2f}) sebebiyle büyüme beklentisi ve DCF değeri cezalandırıldı.")
            
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
            
        # 3. Sektörel Çarpan Analizi
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
        
        # Kompozit içsel değer hesaplama
        if is_loss_making:
            dcf_val = 0.0
            fair_multiples = 0.0
            fair_graham *= 0.3
            intrinsic_avg = fair_graham
            rationale_parts.append("!!! ŞİRKET ZARAR AÇIKLAMIŞ: Net nakit akışı negatif olduğu için DCF iptal edilmiştir. Son derece yüksek risklidir.")
        else:
            if sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım', 'Teknoloji']:
                weights = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
            else:
                weights = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
            intrinsic_avg = (dcf_val * weights['dcf'] + fair_graham * weights['graham'] + fair_multiples * weights['multiples'])
            
        # Senaryo Analizleri
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
        
        # Senaryo emniyet koruması
        if pes_val > intrinsic_avg: pes_val = intrinsic_avg * 0.8
        if opt_val < intrinsic_avg: opt_val = intrinsic_avg * 1.3
        
        mos = (intrinsic_avg / cur_price) - 1.0 if cur_price > 0 else 0.0
        value_score = int(50 + np.tanh(mos) * 50)
        
        # GYO and Holding Value Score adjustment
        if sector in ['GYO', 'Holding', 'Holding / Enerji']:
            value_score = int(value_score * 0.60)
            if sector == 'GYO':
                rationale_parts.append("GYO Varlık İskontosu Uyarısı: Kalıcı varlık iskontosu nedeniyle değerleme skoru %40 oranında düşürüldü. Sanal yeniden değerleme kârları elendi.")
            else:
                rationale_parts.append("Holding Varlık İskontosu Uyarısı: İştirak NAV iskontosu nedeniyle değerleme skoru %40 cezalandırıldı.")
    
    # Momentum Score (Technicals base)
    momentum_score = 50
    if cur_price > sma50 > sma200:
        momentum_score += 25
    elif cur_price > sma50:
        momentum_score += 15
    elif cur_price < sma50 < sma200:
        momentum_score -= 25
    elif cur_price < sma50:
        momentum_score -= 15
        
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
        
    # Composite Intelligence Score
    score = int(value_score * 0.7 + momentum_score * 0.3)
    score = max(5, min(95, score))

    # --- AGGRESSIVE INTELLIGENCE MODEL ---
    if is_bank_holding:
        momentum_score_agg = momentum_score + 15 if rsi < 45 else momentum_score
        momentum_score_agg = max(5, min(95, momentum_score_agg))
        score_agg = int(bank_score * 0.5 + momentum_score_agg * 0.5)
        score_agg = max(5, min(95, score_agg))
        
        rationale_agg_parts = []
        rationale_agg_parts.append(f"DuPont/NIM Agresif Değerleme: DuPont Puanı: {bank_score}/100, Agresif Turn-around Puanı: {momentum_score_agg}/100 (%50-%50 dengeli kompozit).")
        if rsi < 45:
            rationale_agg_parts.append("Düşük RSI (<45) dipten dönüş sinyaliyle ekstra momentum bonusu uygulandı.")
        rationale_agg = " ".join(rationale_agg_parts)
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
            if debt_val > 4.0:
                eps_g_agg *= 0.8
        else:
            if debt_val > 4.0:
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
            
        if is_loss_making:
            dcf_val_agg = 0.0
            fair_graham_agg *= 0.5
            intrinsic_avg_agg = fair_graham_agg
        else:
            if sector in ['Aviation', 'Havacılık', 'Teknoloji / Yazılım', 'Teknoloji']:
                weights_agg = {'dcf': 0.6, 'graham': 0.1, 'multiples': 0.3}
            else:
                weights_agg = {'dcf': 0.4, 'graham': 0.3, 'multiples': 0.3}
            intrinsic_avg_agg = (dcf_val_agg * weights_agg['dcf'] + fair_graham_agg * weights_agg['graham'] + fair_multiples * weights_agg['multiples'])
            
        mos_agg = (intrinsic_avg_agg / cur_price) - 1.0 if cur_price > 0 else 0.0
        value_score_agg = int(50 + np.tanh(mos_agg) * 50)
        
        momentum_score_agg = momentum_score + 15 if rsi < 45 else momentum_score
        momentum_score_agg = max(5, min(95, momentum_score_agg))
        
        score_agg = int(value_score_agg * 0.5 + momentum_score_agg * 0.5)
        score_agg = max(5, min(95, score_agg))
        
        rationale_agg_parts = []
        rationale_agg_parts.append(f"Agresif Değerleme: İçsel Değer: {intrinsic_avg_agg:.2f} {curr} (%50 Temel, %50 Momentum).")
        if rsi < 45:
            rationale_agg_parts.append("Turn-around: RSI 45'in altında aşırı satım bölgesinden çıkış potansiyeli ödüllendirildi (+15 Bonus).")
            
        # GYO and Holding aggressive score adjustment
        if sector in ['GYO', 'Holding', 'Holding / Enerji']:
            value_score_agg = int(value_score_agg * 0.60)
            score_agg = int(value_score_agg * 0.5 + momentum_score_agg * 0.5)
            score_agg = max(5, min(95, score_agg))
            if sector == 'GYO':
                rationale_agg_parts.append("GYO Varlık İskontosu: Sanal kârlar ROE ile elendi ve %40 kalıcı iskonto uygulandı.")
            else:
                rationale_agg_parts.append("Holding Varlık İskontosu: NAV iskontosu nedeniyle değerleme skoru %40 cezalandırıldı.")
        
        if mos_agg > 0.2:
            rationale_agg_parts.append(f"Hisse agresif hedef değerine göre %{mos_agg*100:.1f} iskontolu görünmektedir.")
        elif mos_agg < -0.1:
            rationale_agg_parts.append(f"Agresif senaryoda dahi %{-mos_agg*100:.1f} primlidir.")
        if debt_val > 3.0:
            rationale_agg_parts.append(f"Kaldıraç oranı ({debt_val:.1f}) yüksek olsa da turn-around ralli beklentisi ön plandadır.")
        if not rationale_agg_parts:
            rationale_agg_parts.append("Dengeli agresif seyir beklentisi.")
        rationale_agg = " ".join(rationale_agg_parts)
        
    label_agg = "TUT"
    if score_agg >= 85: label_agg = "Güçlü AL / Şampiyon"
    elif score_agg >= 70: label_agg = "Ucuz / AL"
    elif score_agg <= 30: label_agg = "Çok Pahalı / SAT"
    elif score_agg <= 45: label_agg = "Pahalı / TUT"
    
    # Format label
    label = "TUT"
    if score >= 85: label = "Güçlü AL / Şampiyon"
    elif score >= 70: label = "Ucuz / AL"
    elif score <= 30: label = "Çok Pahalı / SAT"
    elif score <= 45: label = "Pahalı / TUT"
    
    # Additional rationale details
    if score >= 80 and "Yükseliş" in trend:
        rationale_parts.append(f"Mükemmel uyum! Şirket hem finansal olarak çok cazip hem de teknik olarak güçlü bir yükseliş trendinde.")
    elif score >= 80 and "Düşüş" in trend:
        rationale_parts.append("Hisse aşırı ucuzlamış ancak teknik momentum hala negatif. Kademeli alım / dip toplama düşünülebilir.")
    elif score <= 40 and "Yükseliş" in trend:
        rationale_parts.append("Hisse temel çarpanlarına göre oldukça şişmiş fakat güçlü bir momentumla yükseliyor. Dikkatli olunmalıdır.")
        
    if not rationale_parts:
        rationale_parts.append("Fiyat temel değerleme rasyoları ile uyumlu. Dengeli ve stabil seyir bekleniyor.")
        
    return {
        "intrinsic_value_dcf": dcf_val,
        "intrinsic_value_graham": fair_graham,
        "fair_price_multiples": fair_multiples,
        "intrinsic_value_optimistic": opt_val,
        "intrinsic_value_pessimistic": pes_val,
        "intrinsic_value_avg": intrinsic_avg,
        "margin_of_safety": mos,
        "value_score": value_score,
        "momentum_score": momentum_score,
        "intelligence_score": score,
        "valuation_label": label,
        "intelligence_score_aggressive": score_agg,
        "valuation_label_aggressive": label_agg,
        "rationale_aggressive": rationale_agg,
        "rationale": " ".join(rationale_parts)
    }
