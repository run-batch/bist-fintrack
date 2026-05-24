@echo off
title BIST FinTrack - Küresel Dolar Bazlı Portföy Simülasyonu
color 0b
echo ====================================================================
echo    BIST FinTrack - Küresel Dolar Bazlı Portföy Simülasyonu (BIST + S^&P 500)
echo ====================================================================
echo [Sistem] Sanal ortam (.venv) aktif ediliyor...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam aktif edilemedi! .venv klasorunun varligindan emin olun.
    pause
    exit /b %errorlevel%
)

echo [Sistem] Dolar Bazlı Küresel Portföy Trading Simülasyonu başlatılıyor...
echo.
.venv\Scripts\python.exe run_global_quant_backtest.py
echo.
echo ====================================================================
echo [Sistem] Simülasyon tamamlandı! Sonuçları yukarıdan izleyebilirsiniz.
echo ====================================================================
pause
