@echo off
chcp 65001 > nul
title BIST FinTrack - Kuresel Dolar Bazli Portfoy Simulasyonu
color 0b
echo ====================================================================
echo    BIST FinTrack - Kuresel Dolar Bazli Portfoy Simulasyonu (BIST + S^&P 500)
echo ====================================================================
echo [Sistem] Sanal ortam (.venv) aktif ediliyor...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam aktif edilemedi! .venv klasorunun varligindan emin olun.
    pause
    exit /b %errorlevel%
)

echo [Sistem] Dolar Bazli Kuresel Portfoy Trading Simulasyonu baslatiliyor...
echo.
.venv\Scripts\python.exe run_global_quant_backtest.py
echo.
echo ====================================================================
echo [Sistem] Simulasyon tamamlandi! Sonuclari yukaridan izleyebilirsiniz.
echo ====================================================================
pause
