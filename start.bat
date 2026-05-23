@echo off
title BIST FinTrack Launcher
color 0A
echo ===================================================
echo             BIST FINTRACK RADAR SITESI             
echo ===================================================
echo.
echo Bu betik, gerekli bagliliklari kontrol edecek ve
echo BIST Firsat Radari uygulamasini baslatacaktir.
echo.

:: Check python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [HATA] Sisteminizde Python yuklu degil veya PATH degiskenine eklenmemis!
    echo Lutfen Python yukleyin ve tekrar deneyin.
    pause
    exit /b 1
)

:: Virtual environment check
if not exist .venv (
    echo [.venv] Sanal ortam bulunamadi. Yeni bir sanal ortam olusturuluyor...
    python -m venv .venv
    if %errorlevel% neq 0 (
        color 0C
        echo [HATA] Sanal ortam olusturulamadi!
        pause
        exit /b 1
    )
    echo Sanal ortam basariyla olusturuldu.
)

:: Activate virtual environment
echo [.venv] Sanal ortam aktif ediliyor...
call .venv\Scripts\activate

:: Install requirements
echo [Pip] Gerekli kutuphaneler kontrol ediliyor ve yukleniyor...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    color 0C
    echo [HATA] Gerekli paketler yuklenirken bir hata olustu!
    pause
    exit /b 1
)

echo.
echo ===================================================
echo  Sistem basariyla hazirlandi! FastAPI baslatiliyor...
echo  Uygulamaya erismek icin tarayicinizdan sunu acin:
echo  Halka Acik Radar: http://127.0.0.1:8000
echo  Gizli Yonetici Paneli: http://127.0.0.1:8000/admin.html
echo ===================================================
echo.

python app.py

pause
