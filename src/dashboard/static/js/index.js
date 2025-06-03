
$(document).ready(function() {
    // 페이지 로드 시 데이터 로드
    loadDashboardData();

    // 1분마다 자동 새로고침
    setInterval(loadDashboardData, 60000);
});

function loadDashboardData() {
    // 트레이딩 통계 로드
    $.getJSON('/api/trading-stats', function(data) {
        updateTradingStats(data);
    });

    // 차트 로드
    loadCharts();
}

function refreshDashboard() {
    // 새로고침 버튼 클릭 시 데이터 다시 로드
    $.getJSON('/api/trading-stats?refresh=true', function(data) {
        updateTradingStats(data);
    });

    loadCharts();
}

// 트레이딩 통계 
function updateTradingStats(data) {
    if (!data || $.isEmptyObject(data)) {
        $('#portfolio-value').text('데이터 없음');
        $('#portfolio-change').html('변화: <span>-</span>');
        $('#today-trades').html('거래 횟수: <span>-</span>');
        $('#total-return').text('데이터 없음');
        $('#total-duration').html('기간: <span>-</span>');
        $('#positions-count').text('데이터 없음');
        $('#positions-value').html('가치: <span>-</span>');
        $('#recent-trades').html('<tr><td colspan="7" class="text-center">데이터가 없습니다.</td></tr>');
        return;
    }

    // 트레이딩 통계 데이터 추출(대시보드 화면에 표시 필요한 데이터)
    const stats = data.trading_stats;  // "trading_stats"라는 리스트에 있는 값 받아서 stats에 저장

    if (stats) {
        const currentBalance = stats.portfolio_value || 0;
        const totalReturn = stats.total_pnl || 0;
        const initialBalance = currentBalance - totalReturn;
        const returnPercent = initialBalance > 0 ? (totalReturn / initialBalance * 100) : 0;

        const pnl = stats.total_pnl || 0;
        const dailyPnl = stats.daily_pnl || 0;
        const pnlPercent = initialBalance > 0 ? (pnl / initialBalance * 100) : 0;

        $('#portfolio-value').text(`$${currentBalance.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
        $('#portfolio-change').html(`변화: <span class="${pnl >= 0 ? 'text-success' : 'text-danger'}">${pnl >= 0 ? '+' : ''}$${pnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })} (${pnlPercent.toFixed(2)}%)</span>`);
        $('#total-return').text(`${returnPercent.toFixed(2)}%`);
        $('#daily-pnl').text(`$${dailyPnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
    }


    // 오늘 거래 횟수 계산 (trades 기반)
    if (data.trades && data.trades.length > 0) {
        const today = new Date().toISOString().split('T')[0];
        const todayTrades = data.trades.filter(t => t.timestamp && t.timestamp.startsWith(today));
        $('#daily-trades').html(`거래 횟수: <span>${todayTrades.length}</span>`);
    }

    // 기간 계산
    if (stats && stats.timestamp) {
        const startTime = new Date(stats.timestamp);
        const now = new Date();
        const diffDays = Math.floor((now - startTime) / (1000 * 60 * 60 * 24));
        $('#total-duration').html(`기간: <span>${diffDays}일</span>`);
    }

    // 포지션 데이터
    if (data.positions) {
        const positionsCount = Object.keys(data.positions).length;
        let positionsValue = 0;
        for (const symbol in data.positions) {
            positionsValue += data.positions[symbol].market_value || 0;
        }
        $('#positions-count').text(positionsCount);
        $('#positions-value').html(`가치: <span>$${positionsValue.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}</span>`);
    }

    // 최근 거래 내역
    if (data.trades && data.trades.length > 0) {
        const recentTrades = data.trades.slice(-10).reverse();
        let tradesHtml = '';
        recentTrades.forEach(trade => {
            const timestamp = trade.timestamp || '';
            const symbol = trade.symbol || '';
            const side = (trade.side || '').toUpperCase();
            const quantity = (trade.quantity || 0).toLocaleString('ko-KR', { maximumFractionDigits: 6 });
            const price = (trade.price || 0).toLocaleString('ko-KR', { maximumFractionDigits: 2 });
            const amount = ((trade.quantity || 0) * (trade.price || 0)).toLocaleString('ko-KR', { maximumFractionDigits: 2 });
            const status = trade.status || '';
            const sideClass = side === 'BUY' ? 'text-success' : 'text-danger';
            const statusClass = status === 'success' ? 'text-success' : 'text-danger';
            tradesHtml += `
                <tr>
                    <td>${timestamp}</td>
                    <td>${symbol}</td>
                    <td class="${sideClass}">${side}</td>
                    <td>${quantity}</td>
                    <td>$${price}</td>
                    <td>$${amount}</td>
                    <td class="${statusClass}">${status}</td>
                </tr>`;
        });
        $('#recent-trades').html(tradesHtml);
    } else {
        $('#recent-trades').html('<tr><td colspan="7" class="text-center">거래 내역이 없습니다.</td></tr>');
    }
}
       
    
function loadCharts() {
    // 포트폴리오 가치 차트
    $.getJSON('/api/charts/portfolio', function(data) {
        if (data && !data.error) {
            const finalLayout = { ...data.layout };
            Plotly.newPlot('portfolio-chart', data.data, finalLayout);
        } else {
            $('#portfolio-chart').html('<div class="text-center py-5">포트폴리오 데이터가 없습니다.</div>');
        }
    });

    // 수익률 차트
    $.getJSON('/api/charts/returns', function(data) {
        if (data && !data.error) {
            const finalLayout = { ...data.layout };
            Plotly.newPlot('returns-chart', data.data, finalLayout);
        } else {
            $('#returns-chart').html('<div class="text-center py-5">수익률 데이터가 없습니다.</div>');
        }
    });

    // 낙폭 차트
    $.getJSON('/api/charts/drawdown', function(data) {
        if (data && !data.error) {
            const finalLayout = { ...data.layout };
            Plotly.newPlot('drawdown-chart', data.data, finalLayout);
        } else {
            $('#drawdown-chart').html('<div class="text-center py-5">낙폭 데이터가 없습니다.</div>');
        }
    });

    // 거래 분포 차트
    $.getJSON('/api/charts/trade-distribution', function(data) {
        if (data && !data.error) {
            const finalLayout = { ...data.layout };
            Plotly.newPlot('trade-distribution-chart', data.data, finalLayout);
        } else {
            $('#trade-distribution-chart').html('<div class="text-center py-5">거래 분포 데이터가 없습니다.</div>');
        }
    });
}
