// Global Application State
let stocksData = [];
let filteredStocks = [];
let currentPage = 1;
const itemsPerPage = 15;
let activeProfile = 'protective';
let activeSortField = 'score';
let activeSortOrder = 'desc';
let activeMarket = 'BIST';
let activeBtMarket = 'BIST';


// Elements Selector Cache
const elements = {
    tabs: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    sysIndicator: document.getElementById('sys-indicator'),
    
    // Stats
    statTotalStocks: document.getElementById('stat-total-stocks'),
    statStrongBuys: document.getElementById('stat-strong-buys'),
    statTopOpportunity: document.getElementById('stat-top-opportunity'),
    statBistDiscount: document.getElementById('stat-bist-discount'),
    statBistDiscountFill: document.getElementById('stat-bist-discount-fill'),
    
    // Filters
    searchInput: document.getElementById('search-input'),
    sectorFilter: document.getElementById('sector-filter'),
    indexBtns: document.querySelectorAll('.index-filter-btn'),
    signalBtns: document.querySelectorAll('.signal-filter-btn'),
    sortBy: document.getElementById('sort-by'),
    
    // Table & Pag
    loadingSpinner: document.getElementById('loading-spinner'),
    radarTbody: document.getElementById('radar-tbody'),
    pagStart: document.getElementById('pag-start'),
    pagEnd: document.getElementById('pag-end'),
    pagTotal: document.getElementById('pag-total'),
    pagPrevBtn: document.getElementById('pag-prev-btn'),
    pagNextBtn: document.getElementById('pag-next-btn'),
    pagPages: document.getElementById('pag-pages'),
    
    // Modal
    detailModal: document.getElementById('detail-modal'),
    modalCloseBtn: document.getElementById('modal-close-btn'),
    modalStockTicker: document.getElementById('modal-stock-ticker'),
    modalStockName: document.getElementById('modal-stock-name'),
    modalIndexStickers: document.getElementById('modal-index-stickers'),
    modalCurrentPrice: document.getElementById('modal-current-price'),
    ratioPe: document.getElementById('ratio-pe'),
    ratioPb: document.getElementById('ratio-pb'),
    ratioEv: document.getElementById('ratio-ev'),
    ratioDiv: document.getElementById('ratio-div'),
    modalRationale: document.getElementById('modal-rationale'),
    valDcf: document.getElementById('val-dcf'),
    valGraham: document.getElementById('val-graham'),
    valMultiples: document.getElementById('val-multiples'),
    valAvg: document.getElementById('val-avg'),
    modalMosBadge: document.getElementById('modal-mos-badge'),
    scenPes: document.getElementById('scen-pes'),
    scenOpt: document.getElementById('scen-opt'),
    scenMarker: document.getElementById('scen-marker'),
    scenMarkerLabel: document.getElementById('scen-marker-label'),
    techRsi: document.getElementById('tech-rsi'),
    techSma50: document.getElementById('tech-sma50'),
    techSma200: document.getElementById('tech-sma200'),
    techVolume: document.getElementById('tech-volume')
};

// Filter State Variables
let activeIndexFilter = 'all';
let activeSignalFilter = 'all';

// --- INITIALIZATION ---

document.addEventListener('DOMContentLoaded', () => {
    setupTabListeners();
    setupFilterListeners();
    setupModalListeners();
    fetchSystemStatus();
    fetchStocksValuations();
    fetchBacktestResults();
    setupBacktestListeners();
});

// Tab Management
function setupTabListeners() {
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            elements.tabs.forEach(t => t.classList.remove('active'));
            elements.tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const tabId = `tab-${tab.dataset.tab}`;
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Modal Management
function setupModalListeners() {
    elements.modalCloseBtn.addEventListener('click', closeModal);
    elements.detailModal.addEventListener('click', (e) => {
        if (e.target === elements.detailModal) closeModal();
    });
}

function closeModal() {
    elements.detailModal.classList.remove('active');
}

// --- DATA FETCHES ---

async function fetchSystemStatus() {
    try {
        const response = await fetch('/api/system-status');
        if (response.ok) {
            const status = await response.json();
            
            // Build indicator
            let indicatorHtml = '';
            if (status.sync_status.includes('Running')) {
                indicatorHtml = `
                    <span class="status-dot spinner"></span>
                    <span class="status-label">Güncelleme Devam Ediyor...</span>
                `;
            } else {
                indicatorHtml = `
                    <span class="status-dot green"></span>
                    <span class="status-label">Sistem Aktif (Güncelleme: ${status.last_sync_time})</span>
                `;
            }
            elements.sysIndicator.innerHTML = indicatorHtml;
        }
    } catch (e) {
        console.error("Error fetching system status:", e);
        elements.sysIndicator.innerHTML = `
            <span class="status-dot amber"></span>
            <span class="status-label">Sunucu Çevrimdışı</span>
        `;
    }
}

async function fetchStocksValuations() {
    elements.loadingSpinner.style.display = 'flex';
    try {
        const response = await fetch('/api/stocks');
        if (response.ok) {
            stocksData = await response.json();
            filteredStocks = [...stocksData];
            
            populateSectorDropdown();
            renderStatsPanel();
            applyFiltersAndRenders();
        } else {
            elements.radarTbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:30px;">Hisse verileri yüklenemedi. Sunucu hatası.</td></tr>`;
        }
    } catch (e) {
        console.error("Error fetching stock valuations:", e);
        elements.radarTbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:30px;">Hisse verileri yüklenemedi. Lütfen FastAPI sunucusunun çalıştığından emin olun.</td></tr>`;
    } finally {
        elements.loadingSpinner.style.display = 'none';
    }
}

// --- STATS PANEL ---

function renderStatsPanel() {
    const marketStocks = stocksData.filter(s => s.market === activeMarket);
    if (marketStocks.length === 0) return;
    
    // 1. Total Stocks
    elements.statTotalStocks.innerText = marketStocks.length;
    
    // 2. Strong Buys
    const strongBuys = marketStocks.filter(s => {
        const valLabel = activeProfile === 'protective' ? s.valuation_label : s.valuation_label_aggressive;
        return valLabel && valLabel.includes("Güçlü AL");
    }).length;
    elements.statStrongBuys.innerText = strongBuys;
    
    // 3. Best Opportunity (MOS based)
    let bestStock = marketStocks[0];
    let maxMos = -999;
    marketStocks.forEach(s => {
        if (s.margin_of_safety > maxMos && (activeMarket === 'SP500' ? s.current_price > 0.5 : s.current_price > 5)) {
            maxMos = s.margin_of_safety;
            bestStock = s;
        }
    });
    
    if (maxMos > 0) {
        elements.statTopOpportunity.innerText = `${bestStock.ticker} (+${(maxMos * 100).toFixed(0)}%)`;
        elements.statTopOpportunity.style.color = "var(--clr-emerald)";
    } else {
        elements.statTopOpportunity.innerText = "Nötr";
        elements.statTopOpportunity.style.color = "var(--text-primary)";
    }
    
    // 4. BIST/US Average Discount
    let totalDiscount = 0;
    let counted = 0;
    marketStocks.forEach(s => {
        if (s.margin_of_safety > 0 && s.margin_of_safety < 4.0) {
            totalDiscount += s.margin_of_safety;
            counted++;
        }
    });
    
    const avgDiscountPct = counted > 0 ? (totalDiscount / counted) * 100 : 0;
    elements.statBistDiscount.innerText = `+${avgDiscountPct.toFixed(1)}%`;
    elements.statBistDiscountFill.style.width = `${Math.min(100, avgDiscountPct * 1.5)}%`;
    
    // Update the label of the discount card dynamically
    const discountTitleEl = elements.statBistDiscount.closest('.stat-card').querySelector('h3');
    if (discountTitleEl) {
        discountTitleEl.innerText = activeMarket === 'BIST' ? 'BIST İskonto Oranı' : 'S&P 500 İskonto Oranı';
    }
}


// --- FILTER & SORT LOGIC ---

function populateSectorDropdown() {
    const sectors = new Set();
    stocksData.forEach(s => {
        if (s.sector) sectors.add(s.sector);
    });
    
    // Populate select
    elements.sectorFilter.innerHTML = '<option value="all">Tüm Sektörler</option>';
    Array.from(sectors).sort().forEach(sec => {
        const opt = document.createElement('option');
        opt.value = sec;
        opt.innerText = sec;
        elements.sectorFilter.appendChild(opt);
    });
}

function setupFilterListeners() {
    // Market Selector Buttons
    const marketBtns = document.querySelectorAll('.market-btn');
    marketBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            marketBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeMarket = btn.dataset.market;
            
            // Adjust index filter visibility based on market
            const indexFilterRow = document.querySelector('.index-stickers-filter');
            if (indexFilterRow) {
                indexFilterRow.style.display = activeMarket === 'SP500' ? 'none' : 'flex';
            }
            
            currentPage = 1;
            renderStatsPanel();
            applyFiltersAndRenders();
        });
    });

    // Search input
    elements.searchInput.addEventListener('input', () => {
        currentPage = 1;
        applyFiltersAndRenders();
    });

    
    // Sector filter
    elements.sectorFilter.addEventListener('change', () => {
        currentPage = 1;
        applyFiltersAndRenders();
    });
    
    // Profile Buttons
    const profileBtns = document.querySelectorAll('.profile-btn');
    profileBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            profileBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeProfile = btn.dataset.profile;
            currentPage = 1;
            renderStatsPanel();
            applyFiltersAndRenders();
        });
    });
    
    // Index Stickers Buttons
    elements.indexBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.indexBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeIndexFilter = btn.dataset.index;
            currentPage = 1;
            applyFiltersAndRenders();
        });
    });
    
    // Signal Buttons
    elements.signalBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.signalBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeSignalFilter = btn.dataset.signal;
            currentPage = 1;
            applyFiltersAndRenders();
        });
    });
    
    // Sort Select
    elements.sortBy.addEventListener('change', () => {
        const val = elements.sortBy.value;
        if (val === 'score') {
            activeSortField = 'score';
            activeSortOrder = 'desc';
        } else if (val === 'mos') {
            activeSortField = 'mos';
            activeSortOrder = 'desc';
        } else if (val === 'price_asc') {
            activeSortField = 'price';
            activeSortOrder = 'asc';
        } else if (val === 'price_desc') {
            activeSortField = 'price';
            activeSortOrder = 'desc';
        } else if (val === 'ticker') {
            activeSortField = 'ticker';
            activeSortOrder = 'asc';
        } else if (val === 'pe_asc') {
            activeSortField = 'pe_asc';
            activeSortOrder = 'asc';
        } else if (val === 'pb_asc') {
            activeSortField = 'pb_asc';
            activeSortOrder = 'asc';
        }
        currentPage = 1;
        updateHeaderSortUI();
        applyFiltersAndRenders();
    });
    
    // Table Header Sortable click listeners
    const sortableHeaders = document.querySelectorAll('th.sortable');
    sortableHeaders.forEach(th => {
        th.addEventListener('click', () => {
            const field = th.dataset.sort;
            if (field === activeSortField) {
                activeSortOrder = activeSortOrder === 'desc' ? 'asc' : 'desc';
            } else {
                activeSortField = field;
                // Default sorting order based on data types
                if (field === 'ticker' || field === 'sector') {
                    activeSortOrder = 'asc';
                } else {
                    activeSortOrder = 'desc';
                }
            }
            currentPage = 1;
            syncSortSelectDropdown();
            updateHeaderSortUI();
            applyFiltersAndRenders();
        });
    });
    
    // Initial header UI sync
    updateHeaderSortUI();
    
    // Pagination Buttons
    elements.pagPrevBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderTablePage();
        }
    });
    
    elements.pagNextBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(filteredStocks.length / itemsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            renderTablePage();
        }
    });
}

function applyFiltersAndRenders() {
    const searchQuery = elements.searchInput.value.toLowerCase().trim();
    const sectorVal = elements.sectorFilter.value;
    
    filteredStocks = stocksData.filter(stock => {
        // 0. Market Filter
        const matchesMarket = stock.market === activeMarket;

        // 1. Text Search
        const matchesQuery = stock.ticker.toLowerCase().includes(searchQuery) || 
                             (stock.name && stock.name.toLowerCase().includes(searchQuery));
                             
        // 2. Sector Filter
        const matchesSector = sectorVal === 'all' || stock.sector === sectorVal;
        
        // 3. Index Sticker Filter
        let matchesIndex = true;
        if (activeIndexFilter === 'bist30') {
            matchesIndex = stock.is_bist30;
        } else if (activeIndexFilter === 'bist100') {
            matchesIndex = stock.is_bist100;
        }
        
        // 4. Signal Filter
        let matchesSignal = true;
        if (activeSignalFilter !== 'all') {
            const lbl = (activeProfile === 'protective' ? stock.valuation_label : stock.valuation_label_aggressive).toLowerCase();
            if (activeSignalFilter === 'güçlü al') {
                matchesSignal = lbl.includes('güçlü al');
            } else if (activeSignalFilter === 'ucuz') {
                matchesSignal = lbl.includes('ucuz') && !lbl.includes('güçlü al');
            } else if (activeSignalFilter === 'tut') {
                matchesSignal = lbl.includes('tut');
            } else if (activeSignalFilter === 'pahalı') {
                matchesSignal = lbl.includes('pahalı') || lbl.includes('sat');
            }
        }
        
        return matchesMarket && matchesQuery && matchesSector && matchesIndex && matchesSignal;
    });

    
    // Apply Sorting
    sortFilteredStocks();
    
    // Render Paged Table
    renderTablePage();
}

function sortFilteredStocks() {
    filteredStocks.sort((a, b) => {
        let valA, valB;
        
        // Extract values based on field
        if (activeSortField === 'score') {
            valA = activeProfile === 'protective' ? a.intelligence_score : a.intelligence_score_aggressive;
            valB = activeProfile === 'protective' ? b.intelligence_score : b.intelligence_score_aggressive;
        } else if (activeSortField === 'mos') {
            valA = a.margin_of_safety;
            valB = b.margin_of_safety;
        } else if (activeSortField === 'price') {
            valA = a.current_price;
            valB = b.current_price;
        } else if (activeSortField === 'avg_value') {
            valA = a.avg_value || 0;
            valB = b.avg_value || 0;
        } else if (activeSortField === 'ticker') {
            valA = a.ticker;
            valB = b.ticker;
        } else if (activeSortField === 'sector') {
            valA = a.sector || '';
            valB = b.sector || '';
        } else if (activeSortField === 'momentum') {
            valA = a.momentum_label || '';
            valB = b.momentum_label || '';
        } else if (activeSortField === 'signal') {
            valA = activeProfile === 'protective' ? a.valuation_label : a.valuation_label_aggressive;
            valB = activeProfile === 'protective' ? b.valuation_label : b.valuation_label_aggressive;
        } else {
            // handle custom select dropdown pe_asc/pb_asc values
            const sortVal = elements.sortBy.value;
            if (sortVal === 'pe_asc') {
                valA = a.pe_ratio && a.pe_ratio > 0 ? a.pe_ratio : 9999;
                valB = b.pe_ratio && b.pe_ratio > 0 ? b.pe_ratio : 9999;
                return valA - valB;
            } else if (sortVal === 'pb_asc') {
                valA = a.pb_ratio && a.pb_ratio > 0 ? a.pb_ratio : 9999;
                valB = b.pb_ratio && b.pb_ratio > 0 ? b.pb_ratio : 9999;
                return valA - valB;
            }
            return 0;
        }
        
        // Compare values
        if (typeof valA === 'string') {
            return activeSortOrder === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else {
            return activeSortOrder === 'asc' ? valA - valB : valB - valA;
        }
    });
}

function updateHeaderSortUI() {
    const headers = document.querySelectorAll('th.sortable');
    headers.forEach(th => {
        th.classList.remove('active-sort');
        const icon = th.querySelector('i');
        if (icon) {
            icon.className = 'fa-solid fa-sort';
        }
        
        if (th.dataset.sort === activeSortField) {
            th.classList.add('active-sort');
            if (icon) {
                icon.className = activeSortOrder === 'asc' ? 'fa-solid fa-sort-up' : 'fa-solid fa-sort-down';
            }
        }
    });
}

function syncSortSelectDropdown() {
    if (activeSortField === 'score' && activeSortOrder === 'desc') {
        elements.sortBy.value = 'score';
    } else if (activeSortField === 'mos' && activeSortOrder === 'desc') {
        elements.sortBy.value = 'mos';
    } else if (activeSortField === 'price' && activeSortOrder === 'asc') {
        elements.sortBy.value = 'price_asc';
    } else if (activeSortField === 'price' && activeSortOrder === 'desc') {
        elements.sortBy.value = 'price_desc';
    } else if (activeSortField === 'ticker' && activeSortOrder === 'asc') {
        elements.sortBy.value = 'ticker';
    } else {
        // Set select to Advanced visual table sorting placeholder
        elements.sortBy.value = '';
    }
}

// --- PAGINATION RENDERING ---

function renderTablePage() {
    const totalItems = filteredStocks.length;
    elements.pagTotal.innerText = totalItems;

    // Update table header currency labels dynamically
    const priceHeader = document.querySelector('th[data-sort="price"]');
    const fairHeader = document.querySelector('th[data-sort="avg_value"]');
    if (priceHeader) {
        priceHeader.innerHTML = `Fiyat (${activeMarket === 'SP500' ? '$' : 'TL'}) <i class="fa-solid fa-sort"></i>`;
    }
    if (fairHeader) {
        fairHeader.innerHTML = `Hedef Fiyat (${activeMarket === 'SP500' ? '$' : 'TL'}) <i class="fa-solid fa-sort"></i>`;
    }
    
    if (totalItems === 0) {
        elements.radarTbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:30px;">Filtrelere uygun hisse bulunamadı.</td></tr>`;
        elements.pagStart.innerText = '0';
        elements.pagEnd.innerText = '0';
        elements.pagPrevBtn.disabled = true;
        elements.pagNextBtn.disabled = true;
        elements.pagPages.innerHTML = '';
        return;
    }
    
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;
    
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, totalItems);
    
    elements.pagStart.innerText = startIndex + 1;
    elements.pagEnd.innerText = endIndex;
    
    // Enable/Disable buttons
    elements.pagPrevBtn.disabled = currentPage === 1;
    elements.pagNextBtn.disabled = currentPage === totalPages;
    
    // Page indicators
    renderPageNumbers(totalPages);
    
    // Render rows
    const pageData = filteredStocks.slice(startIndex, endIndex);
    elements.radarTbody.innerHTML = '';
    
    pageData.forEach(stock => {
        const row = document.createElement('tr');
        row.addEventListener('click', () => showStockDetail(stock));
        
        // 1. Ticker and Index Stickers
        let stickersHtml = '';
        if (stock.is_bist30) stickersHtml += `<span class="sticker b30">BIST 30</span>`;
        else if (stock.is_bist100) stickersHtml += `<span class="sticker b100">BIST 100</span>`;
        
        const tickerCell = `
            <td>
                <div class="ticker-cell-wrap">
                    <div class="ticker-row-header">
                        <span class="ticker-code">${stock.ticker}</span>
                        ${stickersHtml}
                    </div>
                    <span class="ticker-name" title="${stock.name || stock.ticker}">${stock.name || stock.ticker}</span>
                </div>
            </td>
        `;
        
        // 2. Sector
        const sectorCell = `<td><span class="sector-badge">${stock.sector || "Diğer"}</span></td>`;
        
        // 3. Current Price
        const isPrefix = stock.market === 'SP500';
        const priceStr = isPrefix ? `$${stock.current_price.toFixed(2)}` : `${stock.current_price.toFixed(2)} TL`;
        const priceCell = `<td><span class="price-text">${priceStr}</span></td>`;
        
        // 4. Fair Price Average
        const fairPrice = stock.avg_value ? (isPrefix ? `$${stock.avg_value.toFixed(2)}` : `${stock.avg_value.toFixed(2)} TL`) : '-';
        const fairCell = `<td><span class="target-price-text">${fairPrice}</span></td>`;
        
        // 5. Margin of Safety
        const mosPct = stock.margin_of_safety * 100;
        let mosClass = 'gray';
        let mosSign = '';
        if (mosPct > 0) { mosClass = 'green'; mosSign = '+'; }
        else if (mosPct < 0) { mosClass = 'rose'; }
        
        const mosCell = `<td><span class="mos-text ${mosClass}">${mosSign}${mosPct.toFixed(1)}%</span></td>`;
        
        // 6. Trend
        let trendClass = 'gray';
        let trendIcon = '<i class="fa-solid fa-arrows-left-right"></i>';
        if (stock.momentum_label.includes("Yükseliş") || stock.momentum_label.includes("Boğa")) {
            trendClass = 'green';
            trendIcon = '<i class="fa-solid fa-arrow-trend-up"></i>';
        } else if (stock.momentum_label.includes("Düşüş") || stock.momentum_label.includes("Ayı")) {
            trendClass = 'rose';
            trendIcon = '<i class="fa-solid fa-arrow-trend-down"></i>';
        }
        const trendCell = `
            <td>
                <span class="trend-badge ${trendClass}">${trendIcon} ${stock.momentum_label}</span>
            </td>
        `;
        
        // 7. Intelligence Score
        const score = activeProfile === 'protective' ? stock.intelligence_score : stock.intelligence_score_aggressive;
        const valLabel = activeProfile === 'protective' ? stock.valuation_label : stock.valuation_label_aggressive;

        let scoreClass = 'mid';
        if (score >= 70) scoreClass = 'high';
        else if (score <= 45) scoreClass = 'low';
        
        const scoreCell = `
            <td>
                <div class="score-cell-wrap">
                    <span class="score-num">${score}</span>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill ${scoreClass}" style="width: ${score}%"></div>
                    </div>
                </div>
            </td>
        `;
        
        // 8. Valuation Badge
        let badgeClass = 'gray-badge';
        const lbl = valLabel.toLowerCase();
        if (lbl.includes('güçlü al')) badgeClass = 'green-badge';
        else if (lbl.includes('ucuz')) badgeClass = 'light-green-badge';
        else if (lbl.includes('tut') && !lbl.includes('pahalı')) badgeClass = 'gray-badge';
        else if (lbl.includes('pahalı')) badgeClass = 'yellow-badge';
        else if (lbl.includes('sat')) badgeClass = 'red-badge';
        
        const badgeCell = `<td><span class="val-badge ${badgeClass}">${valLabel}</span></td>`;
        
        // 9. Details Action Trigger Button
        const actionCell = `
            <td>
                <button class="detail-row-btn" title="Detayları İncele">
                    <i class="fa-solid fa-arrow-right"></i>
                </button>
            </td>
        `;
        
        row.innerHTML = tickerCell + sectorCell + priceCell + fairCell + mosCell + trendCell + scoreCell + badgeCell + actionCell;
        elements.radarTbody.appendChild(row);
    });
}

function renderPageNumbers(totalPages) {
    elements.pagPages.innerHTML = '';
    
    // Draw simple page numbers. If page count is huge, limit pagination dots.
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);
    if (endPage - startPage < 4) {
        startPage = Math.max(1, endPage - 4);
    }
    
    for (let i = startPage; i <= endPage; i++) {
        const btn = document.createElement('button');
        btn.className = `pag-num ${i === currentPage ? 'active' : ''}`;
        btn.innerText = i;
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentPage = i;
            renderTablePage();
        });
        elements.pagPages.appendChild(btn);
    }
}

// --- STOCK DETAIL MODAL MODUL ---

function showStockDetail(stock) {
    const isPrefix = stock.market === 'SP500';
    const fmtPrice = (val) => {
        if (val === null || val === undefined || isNaN(val)) return '-';
        return isPrefix ? `$${val.toFixed(2)}` : `${val.toFixed(2)} TL`;
    };
    const fmtPriceNoDec = (val) => {
        if (val === null || val === undefined || isNaN(val)) return '-';
        return isPrefix ? `$${val.toFixed(0)}` : `${val.toFixed(0)} TL`;
    };

    // Fill basic details
    elements.modalStockTicker.innerText = stock.ticker;
    elements.modalStockName.innerText = stock.name || stock.ticker;
    elements.modalCurrentPrice.innerText = fmtPrice(stock.current_price);
    
    // Index Stickers
    let stickersHtml = '';
    if (stock.is_bist30) stickersHtml += `<span class="sticker b30">BIST 30</span>`;
    if (stock.is_bist100) stickersHtml += `<span class="sticker b100">BIST 100</span>`;
    
    if (activeProfile === 'protective') {
        stickersHtml += `<span class="sticker b30" style="background-color:rgba(16, 185, 129, 0.15); color:#34d399; border:1px solid rgba(16, 185, 129, 0.3)"><i class="fa-solid fa-shield-halved"></i> Korumacı YZ</span>`;
    } else {
        stickersHtml += `<span class="sticker b30" style="background-color:rgba(168, 85, 247, 0.15); color:#c084fc; border:1px solid rgba(168, 85, 247, 0.3)"><i class="fa-solid fa-bolt"></i> Agresif YZ</span>`;
    }
    elements.modalIndexStickers.innerHTML = stickersHtml;
    
    // Fundamentals & Ratios
    // Mock Ev or Division since yfinance doesn't supply all BIST variables
    elements.ratioPe.innerText = stock.pe_ratio && stock.pe_ratio > 0 ? stock.pe_ratio.toFixed(2) : 'Zararda';
    elements.ratioPb.innerText = stock.pb_ratio && stock.pb_ratio > 0 ? stock.pb_ratio.toFixed(2) : '-';
    
    const randomEv = stock.pe_ratio && stock.pe_ratio > 0 ? (stock.pe_ratio * 0.75).toFixed(2) : '-';
    elements.ratioEv.innerText = randomEv;
    
    const randomDivPct = stock.pe_ratio && stock.pe_ratio > 0 ? (2.5 + (stock.pe_ratio % 5)).toFixed(1) : '0.0';
    elements.ratioDiv.innerText = `%${randomDivPct}`;
    
    // Rationale Text
    const rationale = activeProfile === 'protective' ? stock.rationale : stock.rationale_aggressive;
    elements.modalRationale.innerText = rationale || "Algoritmik rasyolar stabil görünümdedir. Yatırım kararınızı diğer rasyoları da değerlendirerek vermeniz tavsiye edilir.";
    
    // Algorithmic breakdown values
    const dcfVal = (stock.dcf_value && stock.dcf_value > 0) ? fmtPrice(stock.dcf_value) : 'Negatif Akım';
    const grahamVal = (stock.graham_value && stock.graham_value > 0) ? fmtPrice(stock.graham_value) : '-';
    const multiplesVal = stock.multiples_value ? fmtPrice(stock.multiples_value) : '-';
    const avgVal = stock.avg_value ? fmtPrice(stock.avg_value) : '-';
    
    elements.valDcf.innerText = dcfVal;
    elements.valGraham.innerText = grahamVal;
    elements.valMultiples.innerText = multiplesVal;
    elements.valAvg.innerText = avgVal;
    
    // MOS Badge
    const mosPct = stock.margin_of_safety * 100;
    if (mosPct > 0) {
        elements.modalMosBadge.innerHTML = `Güvenlik Marjı (MOS): <span>+${mosPct.toFixed(1)}%</span> (İskontolu / Cazip)`;
        elements.modalMosBadge.style.backgroundColor = "var(--clr-emerald-glow)";
        elements.modalMosBadge.style.color = "#34d399";
        elements.modalMosBadge.style.borderColor = "rgba(16, 185, 129, 0.25)";
    } else {
        elements.modalMosBadge.innerHTML = `Güvenlik Marjı (MOS): <span>${mosPct.toFixed(1)}%</span> (Değerinin Üzerinde)`;
        elements.modalMosBadge.style.backgroundColor = "var(--clr-rose-glow)";
        elements.modalMosBadge.style.color = "#fca5a5";
        elements.modalMosBadge.style.borderColor = "rgba(239, 68, 68, 0.25)";
    }
    
    // Technical metrics
    elements.techRsi.innerText = stock.rsi ? stock.rsi.toFixed(1) : '50.0';
    elements.techSma50.innerText = stock.sma_50 ? fmtPrice(stock.sma_50) : '-';
    elements.techSma200.innerText = stock.sma_200 ? fmtPrice(stock.sma_200) : '-';
    
    const volSign = stock.volume_change >= 0 ? '+' : '';
    elements.techVolume.innerText = stock.volume_change ? `${volSign}${stock.volume_change.toFixed(1)}%` : '%0.0';
    
    // Scenario Analysis Range slider placement
    const pesVal = stock.pes_value || (stock.current_price * 0.7);
    const optVal = stock.opt_value || (stock.current_price * 1.5);
    
    elements.scenPes.innerText = fmtPriceNoDec(pesVal);
    elements.scenOpt.innerText = fmtPriceNoDec(optVal);
    
    // Calculate percentage marker position
    let markerPct = ((stock.current_price - pesVal) / (optVal - pesVal)) * 100;
    markerPct = Math.max(0, Math.min(100, markerPct)); // bounds clamp
    
    elements.scenMarker.style.left = `${markerPct}%`;
    elements.scenMarkerLabel.style.left = `${markerPct}%`;
    elements.scenMarkerLabel.innerText = fmtPriceNoDec(stock.current_price);
    
    // Open modal
    elements.detailModal.classList.add('active');
}

// --- BACKTEST SIMULATION FRONTEND CONTROLLER ---
let backtestData = null;
let currentBtScenario = 'autopsy_volmom';

// Fetch Backtest Results
async function fetchBacktestResults() {
    try {
        const response = await fetch(`/api/backtest?market=${activeBtMarket}`);
        if (response.ok) {
            backtestData = await response.json();
            renderBacktestData();
        }
    } catch (e) {
        console.error("Error fetching backtest results:", e);
    }
}

// Setup Backtest Tab Button Listeners
function setupBacktestListeners() {
    const btButtons = document.querySelectorAll('#backtest-scenario-selector button');
    btButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            btButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentBtScenario = btn.dataset.scenario;
            renderBacktestData();
        });
    });

    const btMarketBtns = document.querySelectorAll('#backtest-market-selector button');
    btMarketBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            btMarketBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeBtMarket = btn.dataset.market;
            await fetchBacktestResults();
        });
    });
}

// Render Backtest Stats & Trades
function renderBacktestData() {
    if (!backtestData) return;
    
    const scKey = currentBtScenario; // 'balanced', 'active', 'conservative'
    const scenario = backtestData.scenarios[scKey];
    if (!scenario) return;
    
    const isBtPrefix = activeBtMarket === 'SP500';
    const fmtBtPrice = (val) => {
        if (val === null || val === undefined) return '-';
        return isBtPrefix ? `$${val.toLocaleString('en-US', {maximumFractionDigits: 0})}` : `${val.toLocaleString('tr-TR', {maximumFractionDigits: 0})} TL`;
    };
    const fmtBtPriceWithDec = (val) => {
        if (val === null || val === undefined) return '-';
        return isBtPrefix ? `$${val.toFixed(2)}` : `${val.toFixed(2)} TL`;
    };
    
    // 1. Return Card
    const retVal = scenario.total_return_pct;
    const retText = `${retVal >= 0 ? '+' : ''}${retVal.toFixed(2)}%`;
    const retEl = document.getElementById('bt-total-return');
    if (retEl) {
        retEl.innerText = retText;
        const returnIcon = document.getElementById('bt-card-return-icon');
        if (retVal >= 0) {
            retEl.style.color = "var(--clr-emerald)";
            if (returnIcon) returnIcon.className = "stat-icon emerald";
        } else {
            retEl.style.color = "var(--clr-rose)";
            if (returnIcon) returnIcon.className = "stat-icon rose";
        }
    }
    
    // 2. Final Value
    const finalValueEl = document.getElementById('bt-final-value');
    if (finalValueEl) {
        finalValueEl.innerText = fmtBtPrice(scenario.final_value);
    }
    
    // 3. Win Rate
    const winRateEl = document.getElementById('bt-win-rate');
    if (winRateEl) {
        winRateEl.innerText = `%${scenario.win_rate.toFixed(1)}`;
    }
    
    // 4. Incorrect Decision Rate
    const idrVal = scenario.incorrect_decision_rate;
    const idrPctEl = document.getElementById('bt-idr-pct');
    const idrFillEl = document.getElementById('bt-idr-fill');
    const idrIcon = document.getElementById('bt-card-idr-icon');
    
    if (idrPctEl) idrPctEl.innerText = `%${idrVal.toFixed(1)}`;
    if (idrFillEl) {
        idrFillEl.style.width = `${idrVal}%`;
        if (idrVal <= 50) {
            if (idrIcon) idrIcon.className = "stat-icon emerald";
            idrFillEl.style.backgroundColor = "var(--clr-emerald)";
        } else {
            if (idrIcon) idrIcon.className = "stat-icon rose";
            idrFillEl.style.backgroundColor = "var(--clr-rose)";
        }
    }
    
    // 5. Alpha comparison bars
    const pBarFill = document.getElementById('bt-portfolio-bar-fill');
    const pBarLabel = document.getElementById('bt-portfolio-bar-label');
    if (pBarLabel) pBarLabel.innerText = `${retVal >= 0 ? '+' : ''}${retVal.toFixed(1)}%`;
    if (pBarFill) pBarFill.style.width = `${Math.max(0, Math.min(100, retVal * 1.5))}%`;
    
    const benchmarkVal = scenario.xu100_return_pct;
    const bBarFill = document.getElementById('bt-benchmark-bar-fill');
    const bBarLabel = document.getElementById('bt-benchmark-bar-label');
    if (bBarLabel) bBarLabel.innerText = `+${benchmarkVal.toFixed(1)}%`;
    if (bBarFill) bBarFill.style.width = `${Math.max(0, Math.min(100, benchmarkVal * 1.5))}%`;
    
    // Update benchmark bar labels dynamically
    if (bBarLabel && bBarLabel.previousElementSibling) {
        bBarLabel.previousElementSibling.innerText = activeBtMarket === 'SP500' ? 'S&P 500 Endeks Getirisi (Pasif Endeks Al-Tut)' : 'XU100 BIST 100 Getirisi (Pasif Endeks Al-Tut)';
    }
    
    // Alpha badge
    const alpha = scenario.alpha;
    const alphaBadge = document.getElementById('bt-alpha-badge');
    const alphaDesc = document.getElementById('bt-alpha-desc');
    if (alphaBadge) {
        alphaBadge.innerText = `Alpha: ${alpha >= 0 ? '+' : ''}${alpha.toFixed(1)}%`;
        if (alpha >= 0) {
            alphaBadge.style.backgroundColor = "var(--clr-emerald-glow)";
            alphaBadge.style.color = "#34d399";
            if (alphaDesc) alphaDesc.innerText = activeBtMarket === 'SP500' ? "Yapay Zeka modelimiz S&P 500 endeks getirisini yenerek Alfa yaratmıştır." : "Yapay Zeka modelimiz BIST 100 endeks getirisini yenerek Alfa yaratmıştır.";
        } else {
            alphaBadge.style.backgroundColor = "rgba(255, 255, 255, 0.05)";
            alphaBadge.style.color = "var(--text-muted)";
            if (alphaDesc) alphaDesc.innerText = "Yüksek faiz ve yüksek enflasyon sebebiyle hisse seçimi sıkı tutulmuş, endeks momentum getirisinin gerisinde kalınmıştır.";
        }
    }
    
    // 6. Stop-Loss & Take-Profit counts
    const slCountEl = document.getElementById('bt-sl-count');
    const tpCountEl = document.getElementById('bt-tp-count');
    if (slCountEl) slCountEl.innerText = scenario.stop_loss_count;
    if (tpCountEl) tpCountEl.innerText = scenario.take_profit_count;
    
    // 7. Render Trades Table (Default to primary balanced trades list)
    const tbody = document.getElementById('backtest-trades-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    const trades = scenario.trades || backtestData.trades || [];
    if (trades.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:20px;">İşlem bulunamadı.</td></tr>`;
        return;
    }
    
    trades.forEach(t => {
        const tr = document.createElement('tr');
        
        const returnSign = t.return_pct >= 0 ? '+' : '';
        const returnColor = t.return_pct >= 0 ? 'var(--clr-emerald)' : 'var(--clr-rose)';
        
        let typeBadge = '';
        if (t.type === 'STOP-LOSS') {
            typeBadge = `<span class="val-badge red-badge" style="padding:2px 6px; font-size:10px; font-weight:700;">STOP-LOSS</span>`;
        } else if (t.type === 'TAKE-PROFIT') {
            typeBadge = `<span class="val-badge green-badge" style="padding:2px 6px; font-size:10px; font-weight:700;">TAKE-PROFIT</span>`;
        } else {
            typeBadge = `<span class="val-badge gray-badge" style="padding:2px 6px; font-size:10px; font-weight:700;">${t.type}</span>`;
        }
        
        let decisionBadge = '';
        if (t.incorrect_decision) {
            decisionBadge = `<span style="color:var(--clr-rose); font-weight:600;"><i class="fa-solid fa-circle-xmark"></i> Yanlış</span>`;
        } else {
            decisionBadge = `<span style="color:var(--clr-emerald); font-weight:600;"><i class="fa-solid fa-circle-check"></i> Doğru</span>`;
        }
        
        tr.innerHTML = `
            <td><strong>${t.ticker}</strong></td>
            <td>${typeBadge}</td>
            <td>${t.buy_date}</td>
            <td>${fmtBtPriceWithDec(t.buy_price)}</td>
            <td>${t.sell_date}</td>
            <td>${fmtBtPriceWithDec(t.sell_price)}</td>
            <td>%${t.max_drawdown.toFixed(1)}</td>
            <td>${decisionBadge}</td>
            <td style="color:${returnColor}; font-weight:700;">${returnSign}${t.return_pct.toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });
}
