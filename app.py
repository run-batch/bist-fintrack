import os
import uvicorn
import re
import json
import logging
import asyncio
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bist-fintrack")

# Database setup
DATABASE_URL = "sqlite:///./data/bist_fintrack.db"
os.makedirs("./data", exist_ok=True)
os.makedirs("./static", exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class Asset(Base):
    __tablename__ = 'assets'
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)  # e.g., THYAO
    name = Column(String, nullable=True)            # e.g., Turk Hava Yollari
    market = Column(String, default="BIST")
    is_bist30 = Column(Boolean, default=False)
    is_bist100 = Column(Boolean, default=False)
    last_updated = Column(DateTime, nullable=True)

class StockFundamental(Base):
    __tablename__ = 'stock_fundamentals'
    ticker = Column(String, primary_key=True)       # e.g., THYAO.IS
    name = Column(String, nullable=True)
    market = Column(String, default="BIST")
    pe_ratio = Column(Float, nullable=True)         # F/K
    pb_ratio = Column(Float, nullable=True)         # PD/DD
    ev_ebitda = Column(Float, nullable=True)        # FD/FAVÖK
    dividend_yield = Column(Float, nullable=True)   # Temettü Verimi (decimal, e.g. 0.05)
    roe = Column(Float, nullable=True)              # Özsermaye Karlılığı (decimal, e.g. 0.25)
    market_cap = Column(Float, nullable=True)       # Piyasa Değeri
    beta = Column(Float, default=1.0)               # Risk katsayısı
    eps_growth_5y = Column(Float, default=25.0)     # Beklenen 5 yıllık büyüme %
    trailing_eps = Column(Float, nullable=True)     # EPS
    debt_to_equity = Column(Float, nullable=True)   # Borç/Özsermaye (decimal, e.g. 1.5)
    sector = Column(String, default="Diğer")
    data_source = Column(String, default="yfinance") # yfinance, scraping, cached
    last_updated = Column(DateTime)


class ValuationResult(Base):
    __tablename__ = 'valuation_results'
    ticker = Column(String, primary_key=True)       # e.g., THYAO.IS
    name = Column(String, nullable=True)
    sector = Column(String, default="Diğer")
    market = Column(String, default="BIST")
    is_bist30 = Column(Boolean, default=False)
    is_bist100 = Column(Boolean, default=False)
    current_price = Column(Float, default=0.0)
    intrinsic_value_dcf = Column(Float, nullable=True)
    intrinsic_value_graham = Column(Float, nullable=True)
    fair_price_multiples = Column(Float, nullable=True)
    intrinsic_value_optimistic = Column(Float, nullable=True)
    intrinsic_value_pessimistic = Column(Float, nullable=True)
    intrinsic_value_avg = Column(Float, nullable=True)
    margin_of_safety = Column(Float, default=0.0)
    value_score = Column(Integer, default=50)
    momentum_score = Column(Integer, default=50)
    intelligence_score = Column(Integer, default=50)
    valuation_label = Column(String, default="TUT")
    intelligence_score_aggressive = Column(Integer, default=50)
    valuation_label_aggressive = Column(String, default="TUT")
    rationale_aggressive = Column(Text, nullable=True)
    rsi = Column(Float, nullable=True)
    sma_50 = Column(Float, nullable=True)
    sma_200 = Column(Float, nullable=True)
    volume_change = Column(Float, nullable=True)
    momentum_label = Column(String, default="Yatay")
    rationale = Column(Text, nullable=True)
    data_source = Column(String, default="yfinance")
    last_updated = Column(DateTime)

class SystemState(Base):
    __tablename__ = 'system_state'
    key = Column(String, primary_key=True)
    value = Column(String, nullable=True)
    last_updated = Column(DateTime)

# Create tables
Base.metadata.create_all(bind=engine)

# Standard index components for fast lookup
BIST30_LIST = {
    "AKBNK", "ALARK", "ASELS", "ASTOR", "BIMAS", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", 
    "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "ODAS", "OYAKC", "PGSUS", "PETKM", "SAHOL", 
    "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "VAKBN", "YKBNK", "BRSAN", "DOAS"
}

BIST100_LIST = BIST30_LIST.union({
    "AEFES", "AGHOL", "AKSA", "ALBRK", "ALFAS", "ARCLK", "BAGFS", "BERA", "CANTE", "CCOLA", 
    "CIMSA", "ECILC", "EGEEN", "ENJSA", "GENIL", "GESAN", "GSDHO", "GUBRF", "GWIND", "HALKB", 
    "IPEKE", "ISGYO", "ISMEN", "KARSN", "KCAER", "KMPUR", "KOZAA", "KRDMD", "LOGO", "MAVI", 
    "MIATK", "NETAS", "NTHOL", "OTKAR", "PENTA", "QUAGR", "REEDER", "SDTTR", "SKBNK", "SOKM", 
    "SMRTG", "TABGD", "TAVHL", "TKFEN", "TMSN", "TSKB", "TTKOM", "TURSG", "ULKER", "VESBE", 
    "VESTL", "YEOTK", "AGROT", "ALFAS", "ANSGR", "ARDYZ", "AYDEM", "BOBET", "BRYAT", "BTCIM", 
    "BUCIM", "CATES", "CWENE", "ECZYT", "EUPWR", "GOLTS", "HEKTS", "IZENR", "KAYSE", "KCAER", 
    "KENT", "KLYTR", "KTSKR", "MAKTK", "MEGMT", "MHRGY", "OBAMS", "PASEU", "PEKGY", "RYGYO", 
    "SAYAS", "TARKM", "TATEN", "TSKB", "TTRAK", "TUKAS", "TUREX", "VAKFN", "YBTAS", "ZOREN"
})

SP500_LIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "LLY", "AVGO",
    "JPM", "UNH", "V", "MA", "WMT", "XOM", "PG", "COST", "HD", "JNJ",
    "ORCL", "BAC", "ABBV", "NFLX", "AMD", "ADBE", "DIS", "CVX", "PEP", "KO"
]

SECTOR_MAP = {
    # Banks
    'AKBNK': 'Banka', 'GARAN': 'Banka', 'ISCTR': 'Banka', 'YKBNK': 'Banka', 'HALKB': 'Banka', 
    'VAKBN': 'Banka', 'TSKB': 'Banka', 'ALBRK': 'Banka', 'SKBNK': 'Banka', 'QNBFB': 'Banka',
    # Aviation
    'THYAO': 'Havacılık', 'PGSUS': 'Havacılık', 'TAVHL': 'Havacılık',
    # Steel / Mining / Industry
    'EREGL': 'Demir-Çelik / Çimento', 'KRDMD': 'Demir-Çelik / Çimento', 'KOZAL': 'Madencilik', 
    'KOZAA': 'Madencilik', 'IPEKE': 'Madencilik', 'SISE': 'Cam Sanayii', 'FROTO': 'Otomotiv', 
    'TOASO': 'Otomotiv', 'DOAS': 'Otomotiv', 'ARCLK': 'Dayanıklı Tüketim', 'VESBE': 'Dayanıklı Tüketim',
    'VESTL': 'Dayanıklı Tüketim', 'OTKAR': 'Savunma / Otomotiv', 'OYAKC': 'Demir-Çelik / Çimento',
    'CIMSA': 'Demir-Çelik / Çimento',
    # Energy
    'TUPRS': 'Enerji / Rafineri', 'ASTOR': 'Enerji / Elektrik', 'KONTR': 'Enerji / Teknoloji', 
    'SMRTG': 'Enerji / Elektrik', 'ALARK': 'Holding / Enerji', 'AYDEM': 'Enerji / Elektrik',
    'ZOREN': 'Enerji / Elektrik', 'ENJSA': 'Enerji / Dağıtım', 'YEOTK': 'Enerji / Elektrik',
    # Tech / Telecom
    'MIATK': 'Teknoloji / Yazılım', 'REEDER': 'Teknoloji / Üretim', 'LOGO': 'Teknoloji / Yazılım',
    'ARDYZ': 'Teknoloji / Yazılım', 'TCELL': 'Telekomünikasyon', 'TTKOM': 'Telekomünikasyon',
    'ASELS': 'Savunma Sanayii',
    # Retail / Food / Beverage
    'BIMAS': 'Gıda Perakende', 'SOKM': 'Gıda Perakende', 'AEFES': 'Gıda / İçecek', 
    'CCOLA': 'Gıda / İçecek', 'ULKER': 'Gıda / İçecek', 'SOKM': 'Gıda Perakende',
    # Holding
    'KCHOL': 'Holding', 'SAHOL': 'Holding', 'DOHOL': 'Holding', 'AGHOL': 'Holding',
    # Chemistry / Defense
    'SASA': 'Petrokimya / Tekstil', 'PETKM': 'Petrokimya / Tekstil', 'HEKTS': 'Kimya / Tarım',
    # GYO (Real Estate Investment Trusts)
    'MHRGY': 'GYO', 'PEKGY': 'GYO', 'RYGYO': 'GYO', 'ISGYO': 'GYO',
    # S&P 500 US Stocks
    'AAPL': 'Teknoloji', 'MSFT': 'Teknoloji', 'NVDA': 'Teknoloji', 'AVGO': 'Teknoloji', 'ORCL': 'Teknoloji', 'AMD': 'Teknoloji', 'ADBE': 'Teknoloji',
    'AMZN': 'Perakende', 'WMT': 'Perakende', 'COST': 'Perakende', 'HD': 'Perakende',
    'META': 'İletişim / Medya', 'GOOGL': 'İletişim / Medya', 'NFLX': 'İletişim / Medya', 'DIS': 'İletişim / Medya',
    'TSLA': 'Otomotiv',
    'JPM': 'Finans / Banka', 'BAC': 'Finans / Banka', 'V': 'Finans / Banka', 'MA': 'Finans / Banka', 'BRK-B': 'Holding',
    'LLY': 'Sağlık', 'UNH': 'Sağlık', 'JNJ': 'Sağlık', 'ABBV': 'Sağlık',
    'XOM': 'Enerji', 'CVX': 'Enerji',
    'PG': 'Tüketici Ürünleri', 'PEP': 'Gıda / İçecek', 'KO': 'Gıda / İçecek'
}

# --- 3-TIER SCRAPING & RESILIENCE MODUL ---

def scrape_all_bist_tickers():
    """Scrapes all active BIST tickers from Bigpara. Falls back to static list if blocked."""
    logger.info("Scraping all active BIST tickers from Bigpara...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = "https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all("a", href=re.compile(r"/borsa/hisse-fiyatlari/([A-Z0-9]+)-detay/"))
            tickers = set()
            for link in links:
                match = re.search(r"/borsa/hisse-fiyatlari/([A-Z0-9]+)-detay/", link['href'])
                if match:
                    tickers.add(match.group(1))
            
            if len(tickers) > 100:
                logger.info(f"Successfully scraped {len(tickers)} tickers from Bigpara!")
                return sorted(list(tickers))
    except Exception as e:
        logger.error(f"Error scraping BIST tickers from Bigpara: {e}")
    
    # Fallback to compiled list of major BIST stocks if scraper fails
    logger.warning("Scraper failed. Seeding from fallback static BIST ticker set.")
    fallback_set = BIST100_LIST.union({
        "ALBRK", "BAGFS", "BERA", "CANTE", "CIMSA", "ECILC", "EGEEN", "GENIL", "GESAN", "GSDHO",
        "GUBRF", "GWIND", "IPEKE", "ISGYO", "ISMEN", "KARSN", "KMPUR", "MAVI", "NETAS", "NTHOL", 
        "PENTA", "QUAGR", "SDTTR", "SKBNK", "TABGD", "TMSN", "TURSG", "YEOTK", "CLEBI", "CEMTS",
        "KARTN", "GOZDE", "EGGUB", "TKNSA", "HEKTS", "KENT", "ZOREN", "EUPWR", "SOKM", "KRDMA"
    })
    return sorted(list(fallback_set))

def scrape_bigpara_fallback(ticker_code: str):
    """
    Tier 2 Backup: Scrapes stock price and basic valuation metrics from Bigpara.
    """
    logger.info(f"Tier 2: Scraping {ticker_code} from Bigpara...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{ticker_code}-detay/"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # 1. Price Parsing
        price = None
        price_el = soup.find(class_="sym__price")
        if price_el:
            try:
                price = float(price_el.text.strip().replace(".", "").replace(",", "."))
            except Exception as e_price:
                logger.warning(f"Error parsing sym__price for {ticker_code}: {e_price}")
                
        # 2. Extract metrics from information-list__item elements
        pe = None
        pb = None
        mcap = None
        
        items = soup.find_all(class_="information-list__item")
        for item in items:
            name_el = item.find(class_="name")
            val_el = item.find(class_="value")
            if name_el and val_el:
                name = name_el.text.strip().lower()
                val_str = val_el.text.strip().replace(".", "").replace(",", ".")
                try:
                    # Clean numerical values
                    nums = re.findall(r"-?\d+\.?\d*", val_str)
                    if nums:
                        val = float(nums[0])
                    else:
                        continue
                except Exception:
                    continue
                    
                if "f/k" in name:
                    pe = val if val > 0 else -1.0
                elif "pd/dd" in name:
                    pb = val
                elif "piyasa" in name:
                    mcap = val
                elif "son işlem" in name or "son islem" in name or "fiyat" in name:
                    if price is None:
                        price = val

        if price is None or price <= 0:
            logger.warning(f"Could not extract valid price for {ticker_code} from Bigpara.")
            return None
            
        scraped_data = {
            "price": price,
            "pe_ratio": pe,
            "pb_ratio": pb,
            "dividend_yield": None,
            "roe": None,
            "market_cap": mcap,
            "name": ticker_code,
            "sector": SECTOR_MAP.get(ticker_code, "Diğer")
        }
        
        logger.info(f"Tier 2 successfully scraped {ticker_code} from Bigpara: {scraped_data}")
        return scraped_data
        
    except Exception as e:
        logger.error(f"Error scraping Bigpara for {ticker_code}: {e}")
        return None

# --- CORE VALUATION ENGINE ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def update_all_fundamentals(db):
    """
    3-Tier Data Sync. Loops through all seeded Assets, 
    obtains financials from yfinance, scrapes if blocked, or falls back to Cache.
    """
    logger.info("Starting 3-Tier Stock Fundamentals Sync...")
    assets = db.query(Asset).all()
    
    # Track Scheduler Info
    state = db.query(SystemState).filter(SystemState.key == "last_sync_status").first()
    if not state:
        state = SystemState(key="last_sync_status")
        db.add(state)
    
    state.value = "Running"
    state.last_updated = datetime.now()
    db.commit()
    
    success_count = 0
    fail_count = 0
    
    for asset in assets:
        ticker_code = asset.code
        if asset.market == "SP500":
            ticker_yf = ticker_code
        else:
            ticker_yf = f"{ticker_code}.IS"
        
        stock_fund = db.query(StockFundamental).filter(StockFundamental.ticker == ticker_yf).first()
        if not stock_fund:
            stock_fund = StockFundamental(ticker=ticker_yf, name=ticker_code, market=asset.market, sector=SECTOR_MAP.get(ticker_code, "Diğer"))
            db.add(stock_fund)
            
        # 1. Tier 1: Try yfinance
        try:
            logger.info(f"Tier 1: Fetching yfinance for {ticker_yf}...")
            ticker = yf.Ticker(ticker_yf)
            info = ticker.info
            
            if not info or len(info) <= 5 or (info.get('marketCap') is None and info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
                raise Exception("yfinance returned empty or incomplete info dictionary")
                
            stock_fund.name = info.get('longName') or ticker_code
            stock_fund.sector = SECTOR_MAP.get(ticker_code, info.get('sector') or "Diğer")
            stock_fund.market_cap = info.get('marketCap')
            
            pe = info.get('trailingPE')
            if not pe or pe <= 0:
                pe = info.get('forwardPE')
            stock_fund.pe_ratio = pe if pe and pe > 0 else -1.0 # loss maker
            
            stock_fund.pb_ratio = info.get('priceToBook')
            # Custom adjustments for high outliers
            if stock_fund.pb_ratio and stock_fund.pb_ratio > 10 and ticker_code == 'THYAO':
                stock_fund.pb_ratio = 1.2
            elif stock_fund.pb_ratio and stock_fund.pb_ratio > 60:
                stock_fund.pb_ratio = 2.0
                
            stock_fund.ev_ebitda = info.get('enterpriseToEbitda')
            stock_fund.dividend_yield = info.get('dividendYield')
            stock_fund.roe = info.get('returnOnEquity')
            stock_fund.beta = info.get('beta', 1.0)
            stock_fund.trailing_eps = info.get('trailingEps') or info.get('forwardEps')
            stock_fund.debt_to_equity = info.get('debtToEquity')
            if stock_fund.debt_to_equity and stock_fund.debt_to_equity > 10:
                stock_fund.debt_to_equity = stock_fund.debt_to_equity / 100.0 # format to decimal
                
            # ROE based Sustainable growth rate estimate
            roe_val = stock_fund.roe or 0.25
            if asset.market == "SP500":
                # US stocks have more stable, lower growth
                stock_fund.eps_growth_5y = max(5.0, min(30.0, roe_val * 100 * 0.6))
            else:
                stock_fund.eps_growth_5y = max(15.0, min(100.0, roe_val * 100 * 0.8))
            
            stock_fund.data_source = "yfinance"
            stock_fund.last_updated = datetime.now()
            success_count += 1
            
        except Exception as e1:
            logger.warning(f"Tier 1 failed for {ticker_yf}: {e1}.")
            
            scraped = None
            if asset.market != "SP500":
                logger.warning("Initiating Tier 2 Scraper Fallback...")
                scraped = scrape_bigpara_fallback(ticker_code)
                
            if scraped:
                stock_fund.pe_ratio = scraped["pe_ratio"] or stock_fund.pe_ratio
                stock_fund.pb_ratio = scraped["pb_ratio"] or stock_fund.pb_ratio
                stock_fund.market_cap = scraped["market_cap"] or stock_fund.market_cap
                stock_fund.data_source = "scraping"
                stock_fund.last_updated = datetime.now()
                success_count += 1
                logger.info(f"Tier 2 successfully loaded data for {ticker_code}")
            else:
                # 3. Tier 3: Local Cache Fallback
                logger.error(f"Tier 2/Fallback failed for {ticker_code}. Fallback to Tier 3 Local Cache...")
                stock_fund.data_source = "cached"
                stock_fund.last_updated = stock_fund.last_updated or datetime.now()
                fail_count += 1
                
        db.commit()
        # Sleep to avoid hitting rate limits too fast
        time_to_sleep = 0.5 if stock_fund.data_source == "yfinance" else 0.1
        time.sleep(time_to_sleep)

    # Log System Sync Complete State
    state = db.query(SystemState).filter(SystemState.key == "last_sync_status").first()
    if state:
        state.value = f"Success ({success_count} OK, {fail_count} Cached)"
        state.last_updated = datetime.now()
    db.commit()

def calculate_all_valuations(db):
    """
    Main valuation scanner. Takes SQLite stock fundamentals, 
    downloads historical data to get momentum, and runs the valuation formulas.
    """
    logger.info("Starting Valuation Computations for BIST & SP500 stocks...")
    stocks = db.query(StockFundamental).all()
    
    for stock in stocks:
        ticker_code = stock.ticker.replace(".IS", "")
        asset = db.query(Asset).filter(Asset.code == ticker_code).first()
        market = asset.market if asset else "BIST"
        curr = "$" if market == "SP500" else "TL"
        
        # Get Current Price
        cur_price = 0.0
        rsi, sma50, sma200, v_change = 50.0, 0.0, 0.0, 0.0
        trend = "Yatay"
        tech_source = "yfinance"
        
        # Technical indicators download
        try:
            hist = yf.download(stock.ticker, period="1y", interval="1d", progress=False)
            if not hist.empty:
                close = hist['Close']
                vol = hist['Volume']
                
                # Check if pandas returns a DataFrame or a Series
                if isinstance(close, pd.DataFrame):
                    cur_price = float(close.iloc[-1].iloc[0])
                    close_series = close.iloc[:, 0]
                    vol_series = vol.iloc[:, 0]
                else:
                    cur_price = float(close.iloc[-1])
                    close_series = close
                    vol_series = vol
                
                # Technical calculations
                if len(close_series) > 200:
                    sma50 = float(close_series.rolling(window=50).mean().iloc[-1])
                    sma200 = float(close_series.rolling(window=200).mean().iloc[-1])
                    rsi = float(calculate_rsi(close_series).iloc[-1])
                    
                    v_current = vol_series.iloc[-5:].mean()
                    v_prev = vol_series.iloc[-25:-5].mean()
                    v_change = (v_current / v_prev - 1) * 100 if v_prev > 0 else 0
                else:
                    sma50 = float(close_series.rolling(window=min(len(close_series), 50)).mean().iloc[-1])
                    sma200 = float(close_series.rolling(window=min(len(close_series), 200)).mean().iloc[-1])
                    rsi = 50.0
                    v_change = 0
            else:
                raise Exception("Empty historical dataframe")
        except Exception as e_tech:
            logger.warning(f"Failed to fetch technical trends for {stock.ticker} via yfinance: {e_tech}.")
            tech_source = "scraping"
            
            scraped = None
            if market != "SP500":
                logger.warning("Trying scraper pricing...")
                scraped = scrape_bigpara_fallback(ticker_code)
                
            if scraped:
                cur_price = scraped["price"]
            else:
                # Cache recovery
                cached_val = db.query(ValuationResult).filter(ValuationResult.ticker == stock.ticker).first()
                if cached_val:
                    cur_price = cached_val.current_price
                    rsi, sma50, sma200, v_change = cached_val.rsi, cached_val.sma_50, cached_val.sma_200, cached_val.volume_change
                    trend = cached_val.momentum_label
                    tech_source = "cached"
                    logger.info(f"Used cached technical values for {ticker_code}")
                else:
                    continue # Nothing we can do, skip this stock
        
        # VALUATION MODELS
        sector = stock.sector or "Diğer"
        pe = stock.pe_ratio or -1.0
        pb = stock.pb_ratio or 1.0
        roe = stock.roe or 0.25
        eps_g = stock.eps_growth_5y or 25.0
        
        # Calculate EPS
        if stock.trailing_eps and stock.trailing_eps > 0:
            eps = stock.trailing_eps
        elif pe > 0:
            eps = cur_price / pe
        else:
            eps = cur_price * (roe * 0.4) / (pb or 1.0) # Estimated synthetic EPS
            
        if eps <= 0:
            eps = cur_price * 0.05 # safe fallback floor
            
        # GYO and Holding EPS adjustment (ROE momentum scaling)
        if sector in ['GYO', 'Holding', 'Holding / Enerji']:
            roe_factor = max(0.05, min(1.0, (roe or 0.0) * 2.0))
            eps *= roe_factor
            
        bvps = cur_price / (pb if pb > 0 else 1.0)
        is_loss_making = (pe < 0) or (roe < 0)
        
        rationale_parts = []
        is_bank_holding = (sector == 'Banka') or (market == 'SP500' and sector == 'Finans / Banka' and ticker_code in ['JPM', 'BAC'])
        
        # A. DUPONT / NIM BANKACILIK MODÜLÜ
        if is_bank_holding:
            roe_val = roe
            if market == "SP500":
                nim_val = 0.02 + 0.01 * roe_val
                npl_val = max(0.002, 0.015 - 0.005 * roe_val)
                syr_val = 0.12 + 0.02 * (stock.beta or 1.0)
                
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
                syr_val = 0.16 + 0.04 * (stock.beta or 1.0)
                
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
                
            debt_to_equity = stock.debt_to_equity or 0.0
            debt_penalty = delta_weight * max(0.0, np.log(1.0 + debt_to_equity))
            cost_of_equity = rf_star + (stock.beta or 1.0) * erp_rate + debt_penalty
            
            if market == "SP500":
                if debt_to_equity > 4.0:
                    eps_g *= 0.8
                    rationale_parts.append(f"DİKKAT: Yüksek borç yükü (D/E: {debt_to_equity:.2f}) sebebiyle büyüme beklentisi ve DCF değeri cezalandırıldı.")
            else:
                if debt_to_equity > 2.0:
                    eps_g *= 0.6
                    rationale_parts.append(f"DİKKAT: Yüksek borç yükü (D/E: {debt_to_equity:.2f}) sebebiyle büyüme beklentisi ve DCF değeri cezalandırıldı.")
                
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
        if tech_source != "cached":
            momentum_score = 50
            if cur_price > sma50 > sma200:
                momentum_score += 25
                trend = "Güçlü Boğa / Yükseliş"
            elif cur_price > sma50:
                momentum_score += 15
                trend = "Yükseliş Trendi"
            elif cur_price < sma50 < sma200:
                momentum_score -= 25
                trend = "Güçlü Ayı / Düşüş"
            elif cur_price < sma50:
                momentum_score -= 15
                trend = "Düşüş Trendi"
                
            if rsi < 30:
                momentum_score += 10
                rationale_parts.append("Teknik aşırı satım bölgesinde (rsi < 30) - Toplama fırsatı olabilir.")
            elif rsi > 70:
                momentum_score -= 10
                rationale_parts.append("Teknik aşırı alım bölgesinde (rsi > 70) - Kar realizasyonu düşünülebilir.")
                
            if v_change > 30:
                momentum_score += 5
                rationale_parts.append("Hisse hacminde ani artış var - Kurumsal ilgi yükseliyor.")
        else:
            momentum_score = 50
            
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
            debt_penalty_agg = delta_weight_agg * max(0.0, np.log(1.0 + debt_to_equity))
            cost_of_equity_agg = rf_star_agg + (stock.beta or 1.0) * erp_agg + debt_penalty_agg
            
            eps_g_agg = stock.eps_growth_5y or 25.0
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
                # Re-calculate composite aggressive score with penalized value_score_agg
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
            if debt_to_equity > 3.0:
                rationale_agg_parts.append(f"Kaldıraç oranı ({debt_to_equity:.1f}) yüksek olsa da turn-around ralli beklentisi ön plandadır.")
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
            
        # Write back to DB
        res = db.query(ValuationResult).filter(ValuationResult.ticker == stock.ticker).first()
        if not res:
            res = ValuationResult(ticker=stock.ticker)
            db.add(res)
            
        res.name = stock.name
        res.sector = sector
        res.market = market
        res.is_bist30 = asset.is_bist30 if asset else False
        res.is_bist100 = asset.is_bist100 if asset else False
        res.current_price = float(cur_price)
        res.intrinsic_value_dcf = float(dcf_val)
        res.intrinsic_value_graham = float(fair_graham)
        res.fair_price_multiples = float(fair_multiples)
        res.intrinsic_value_optimistic = float(opt_val)
        res.intrinsic_value_pessimistic = float(pes_val)
        res.intrinsic_value_avg = float(intrinsic_avg)
        res.margin_of_safety = float(mos)
        res.value_score = int(value_score)
        res.momentum_score = int(momentum_score)
        res.intelligence_score = int(score)
        res.valuation_label = label
        res.intelligence_score_aggressive = int(score_agg)
        res.valuation_label_aggressive = label_agg
        res.rationale_aggressive = rationale_agg
        res.rsi = float(rsi)
        res.sma_50 = float(sma50)
        res.sma_200 = float(sma200)
        res.volume_change = float(v_change)
        res.momentum_label = trend
        res.rationale = " ".join(rationale_parts)
        res.data_source = stock.data_source
        res.last_updated = datetime.now()
        
        db.commit()
    logger.info("Valuation computations successfully completed!")


# --- SEEDING METHOD ON STARTUP ---

def init_seed_db():
    """Seeds stock list on startup by scraping BIST active codes and seeding SP500 assets."""
    db = SessionLocal()
    try:
        count = db.query(Asset).count()
        if count == 0:
            logger.info("Database is empty. Initiating initial BIST Ticker Seeding...")
            tickers = scrape_all_bist_tickers()
            for code in tickers:
                is_b30 = code in BIST30_LIST
                is_b100 = code in BIST100_LIST
                asset = Asset(
                    code=code, 
                    name=code, 
                    market="BIST",
                    is_bist30=is_b30, 
                    is_bist100=is_b100,
                    last_updated=datetime.now()
                )
                db.add(asset)
            
            logger.info("Seeding SP500 assets...")
            for code in SP500_LIST:
                asset = Asset(
                    code=code,
                    name=code,
                    market="SP500",
                    is_bist30=False,
                    is_bist100=False,
                    last_updated=datetime.now()
                )
                db.add(asset)
                
            db.commit()
            logger.info(f"Seeded BIST ({len(tickers)}) and SP500 ({len(SP500_LIST)}) assets inside SQLite.")
            
            # Initial run of calculations in background
            logger.info("Triggering initial sync in background...")
            asyncio.create_task(initial_run_task())
    except Exception as e:
        logger.error(f"Error during db seeding: {e}")
    finally:
        db.close()


async def initial_run_task():
    db = SessionLocal()
    try:
        await asyncio.to_thread(update_all_fundamentals, db)
        await asyncio.to_thread(calculate_all_valuations, db)
    finally:
        db.close()

# --- BACKING AUTOMATIC SCHEDULER TASK ---

async def background_scheduler():
    """Runs a background loop that updates all financials automatically every 24 hours."""
    await asyncio.sleep(10) # wait for uvicorn to settle
    logger.info("Background Auto-Scheduler started!")
    
    while True:
        try:
            logger.info("Auto-Scheduler triggered! Refreshing all BIST stocks...")
            db = SessionLocal()
            await asyncio.to_thread(update_all_fundamentals, db)
            await asyncio.to_thread(calculate_all_valuations, db)
            db.close()
            logger.info("Auto-Scheduler successfully completed cycle. Sleeping for 24 hours...")
        except Exception as e:
            logger.error(f"Error in Auto-Scheduler cycle: {e}")
            
        await asyncio.sleep(24 * 3600) # sleep 24 hours

# --- FASTAPI SERVER ENDPOINTS ---

app = FastAPI(
    title="BIST Fırsat Radarı API",
    description="Tüm BIST Şirketlerinin Otomatik Değerleme ve Fırsat Analiz Portalı",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_seed_db()
    asyncio.create_task(background_scheduler())

# Ziyaretçi API Rotaları (Halka Açık)

@app.get("/api/stocks")
def get_stocks():
    """Returns all calculated valuations, sorted by Intelligence Score descending (best buys first)."""
    db = SessionLocal()
    try:
        results = db.query(ValuationResult).order_by(ValuationResult.intelligence_score.desc()).all()
        
        # Serialize to dict list
        data = []
        for r in results:
            data.append({
                "ticker": r.ticker.replace(".IS", ""),
                "name": r.name,
                "sector": r.sector,
                "market": r.market,
                "is_bist30": r.is_bist30,
                "is_bist100": r.is_bist100,
                "current_price": r.current_price,
                "dcf_value": r.intrinsic_value_dcf,
                "graham_value": r.intrinsic_value_graham,
                "multiples_value": r.fair_price_multiples,
                "opt_value": r.intrinsic_value_optimistic,
                "pes_value": r.intrinsic_value_pessimistic,
                "avg_value": r.intrinsic_value_avg,
                "margin_of_safety": r.margin_of_safety,
                "value_score": r.value_score,
                "momentum_score": r.momentum_score,
                "intelligence_score": r.intelligence_score,
                "valuation_label": r.valuation_label,
                "intelligence_score_aggressive": r.intelligence_score_aggressive,
                "valuation_label_aggressive": r.valuation_label_aggressive,
                "rationale_aggressive": r.rationale_aggressive,
                "rsi": r.rsi,
                "sma_50": r.sma_50,
                "sma_200": r.sma_200,
                "volume_change": r.volume_change,
                "momentum_label": r.momentum_label,
                "rationale": r.rationale,
                "data_source": r.data_source,
                "last_updated": r.last_updated.strftime("%Y-%m-%d %H:%M:%S") if r.last_updated else None
            })
        return data

    finally:
        db.close()

@app.get("/api/system-status")
def get_system_status():
    """Returns scheduler run state and DB counts."""
    db = SessionLocal()
    try:
        state = db.query(SystemState).filter(SystemState.key == "last_sync_status").first()
        total_assets = db.query(Asset).count()
        total_valuations = db.query(ValuationResult).count()
        
        status_val = state.value if state else "Bilinmiyor"
        updated_val = state.last_updated.strftime("%Y-%m-%d %H:%M:%S") if state and state.last_updated else "Hiçbir zaman"
        
        return {
            "sync_status": status_val,
            "last_sync_time": updated_val,
            "total_assets": total_assets,
            "total_valuations": total_valuations
        }
    finally:
        db.close()

# İndekslenmeyen Gizli Yönetim API Rotaları (/admin subdomain'i için)

@app.post("/api/admin/refresh")
def force_refresh(background_tasks: BackgroundTasks):
    """Triggers complete 3-Tier refresh and calculations in a background task."""
    db = SessionLocal()
    try:
        state = db.query(SystemState).filter(SystemState.key == "last_sync_status").first()
        if state and state.value == "Running":
            return {"status": "error", "message": "Senkronizasyon işlemi zaten devam ediyor."}
        
        background_tasks.add_task(initial_run_task)
        return {"status": "success", "message": "Güncelleme görevi arka planda başlatıldı."}
    finally:
        db.close()

@app.post("/api/admin/add-ticker")
def add_ticker(payload: dict):
    """Allows admin to manually inject a new stock ticker to scan list."""
    ticker_code = payload.get("ticker", "").strip().upper()
    if not ticker_code:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")
        
    db = SessionLocal()
    try:
        exists = db.query(Asset).filter(Asset.code == ticker_code).first()
        if exists:
            return {"status": "info", "message": f"{ticker_code} listede zaten mevcut."}
            
        is_b30 = ticker_code in BIST30_LIST
        is_b100 = ticker_code in BIST100_LIST
        
        asset = Asset(
            code=ticker_code, 
            name=ticker_code, 
            is_bist30=is_b30, 
            is_bist100=is_b100,
            last_updated=datetime.now()
        )
        db.add(asset)
        db.commit()
        logger.info(f"Admin manually added ticker: {ticker_code}")
        
        # Add basic skeleton to fundamentals
        stock_fund = StockFundamental(
            ticker=f"{ticker_code}.IS", 
            name=ticker_code, 
            sector=SECTOR_MAP.get(ticker_code, "Diğer"),
            last_updated=datetime.now()
        )
        db.add(stock_fund)
        db.commit()
        
        return {"status": "success", "message": f"{ticker_code} başarıyla eklendi. İlk güncellemede işlenecek."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/api/admin/remove-ticker/{ticker}")
def remove_ticker(ticker: str):
    """Allows admin to manually delete a stock ticker from scan list."""
    ticker_code = ticker.strip().upper()
    db = SessionLocal()
    try:
        asset = db.query(Asset).filter(Asset.code == ticker_code).first()
        if not asset:
            raise HTTPException(status_code=404, detail="Ticker not found in database")
            
        db.delete(asset)
        
        # clean fundamentals and results
        db.query(StockFundamental).filter(StockFundamental.ticker == f"{ticker_code}.IS").delete()
        db.query(ValuationResult).filter(ValuationResult.ticker == f"{ticker_code}.IS").delete()
        
        db.commit()
        logger.info(f"Admin manually removed ticker: {ticker_code}")
        return {"status": "success", "message": f"{ticker_code} listeden başarıyla kaldırıldı."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/backtest")
def get_backtest_results(market: str = "BIST"):
    """Reads and returns the historical backtest simulation results for a specific market."""
    import json
    if market.upper() == "SP500":
        file_path = "./data/backtest_results_sp500.json"
    else:
        file_path = "./data/backtest_results.json"
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Backtest results for {market} not found. Please run backtest_simulation.py first.")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount Static Files
app.mount("/", StaticFiles(directory="./static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
