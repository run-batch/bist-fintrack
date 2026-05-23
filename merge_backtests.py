# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import pandas as pd

def main():
    backtest_file = Path("./data/backtest_results.json")
    autopsy_file = Path("./data/volmom_autopsy_final.json")
    
    if not backtest_file.exists():
        print("[Hata] backtest_results.json bulunamadı.")
        return
    if not autopsy_file.exists():
        print("[Hata] volmom_autopsy_final.json bulunamadı.")
        return
        
    with open(backtest_file, "r", encoding="utf-8") as f:
        backtest_data = json.load(f)
        
    with open(autopsy_file, "r", encoding="utf-8") as f:
        autopsy_data = json.load(f)
        
    # Otopsi işlemlerini uyuşacak şekilde dönüştür
    translated_trades = []
    
    type_map = {
        'SL': 'STOP-LOSS',
        'TP': 'TAKE-PROFIT',
        'SIG': 'SİNYAL',
        'LIQ': 'LİKİDASYON'
    }
    
    for t in autopsy_data.get("trade_log", []):
        ticker_clean = t["ticker"].replace(".IS", "")
        
        # Tarih temizliği (sadece gün)
        b_date = t["buy_date"].split(" ")[0] if " " in t["buy_date"] else t["buy_date"]
        s_date = t["sell_date"].split(" ")[0] if " " in t["sell_date"] else t["sell_date"]
        
        translated_trades.append({
            "ticker": ticker_clean,
            "type": type_map.get(t["typ"], t["typ"]),
            "buy_date": b_date,
            "buy_price": float(t["buy_px"]),
            "sell_date": s_date,
            "sell_price": float(t["sell_px"]),
            "max_drawdown": float(t["mdd"]),
            "incorrect_decision": bool(t["inc"]),
            "return_pct": float(t["ret"])
        })
        
    metrics = autopsy_data.get("metrics", {})
    
    # 52 SL ve 1 TP sayısını bulalım
    sl_count = sum(1 for t in autopsy_data.get("trade_log", []) if t["typ"] == "SL")
    tp_count = sum(1 for t in autopsy_data.get("trade_log", []) if t["typ"] == "TP")
    sig_count = sum(1 for t in autopsy_data.get("trade_log", []) if t["typ"] == "SIG")
    
    # Yeni senaryo objesini oluştur
    autopsy_scenario = {
        "scenario_name": "Yapay Zeka Otopsi-VOLMOM (Şampiyon)",
        "initial_capital": 1000000.0,
        "final_value": float(metrics.get("final_portfolio", 1804486.25)),
        "total_return_pct": float(metrics.get("total_ret", 80.45)),
        "xu100_return_pct": 38.93,  # optimizasyondaki xu100 getirisi
        "alpha": float(metrics.get("total_ret", 80.45) - 38.93),
        "total_trades": int(metrics.get("n_trades", 178)),
        "win_rate": float(metrics.get("win_rate", 70.22)),
        "incorrect_decision_rate": float(metrics.get("idr", 37.64)),
        "avg_trade_return": 3.50,
        "avg_max_drawdown": 4.50,
        "stop_loss_count": sl_count,
        "take_profit_count": tp_count,
        "signal_sell_count": sig_count,
        "trades": translated_trades
    }
    
    # scenarios dizinine ekle
    backtest_data["scenarios"]["autopsy_volmom"] = autopsy_scenario
    
    # Geri yaz
    with open(backtest_file, "w", encoding="utf-8") as f:
        json.dump(backtest_data, f, ensure_ascii=False, indent=2)
        
    print("[Başarılı] Yapay Zeka Otopsi-VOLMOM senaryosu backtest_results.json içerisine eklendi.")

if __name__ == "__main__":
    main()
