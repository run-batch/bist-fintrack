@echo off
chcp 65001 > nul
title BIST FinTrack - Trading Bot Simülasyonu (Tüm BIST)
color 0b
echo ====================================================================
echo             BIST FinTrack - Trading Bot Simülasyonu
echo ====================================================================
echo [Sistem] Sanal ortam (.venv) aktif ediliyor...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam aktif edilemedi! .venv klasorunun varligindan emin olun.
    pause
    exit /b %errorlevel%
)

echo [Sistem] Trading Bot ve Tum BIST Geriye Donuk Testi baslatiliyor...
echo.
.venv\Scripts\python.exe run_full_bist_backtest.py
echo.
echo ====================================================================
echo [Sistem] Simülasyon tamamlandı! Sonuclari yukarıdan izleyebilirsiniz.
echo ====================================================================
pause
