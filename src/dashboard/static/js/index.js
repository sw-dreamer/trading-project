
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

function updateTradingStats(data) {
    if (!data || $.isEmptyObject(data)) {
        $('#portfolio-value').text('데이터 없음');
        $('#portfolio-change').html('변화: <span>-</span>');
        $('#today-pnl').text('데이터 없음');
        $('#today-trades').html('거래 횟수: <span>-</span>');
        $('#total-return').text('데이터 없음');
        $('#total-duration').html('기간: <span>-</span>');
        $('#positions-count').text('데이터 없음');
        $('#positions-value').html('가치: <span>-</span>');
        $('#recent-trades').html('<tr><td colspan="7" class="text-center">데이터가 없습니다.</td></tr>');
        return;
    }

    const stats = data.trading_stats;

    if (stats) {
        const currentBalance = stats.portfolio_value || 0;
        const initialBalance = stats.initial_balance || 0;
        const pnl = stats.daily_pnl || 0;
        const pnlPercent = initialBalance > 0 ? (pnl / initialBalance * 100) : 0;

        $('#portfolio-value').text(`$${currentBalance.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
        const changeClass = pnl >= 0 ? 'text-success' : 'text-danger';
        const changeSign = pnl >= 0 ? '+' : '';
        $('#portfolio-change').html(`변화: <span class="${changeClass}">${changeSign}$${pnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })} (${changeSign}${pnlPercent.toFixed(2)}%)</span>`);
    }

    if (stats && stats.trades) {
        const today = new Date().toISOString().split('T')[0];
        const todayTrades = stats.trades.filter(t => t.timestamp && t.timestamp.startsWith(today));
        const todayPnl = todayTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);

        $('#today-pnl').text(`$${todayPnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
        $('#today-trades').html(`거래 횟수: <span>${todayTrades.length}</span>`);
    }

    if (stats && stats.start_time) {
        const totalReturn = stats.pnl || 0;
        const initialBalance = stats.initial_balance || 0;
        const returnPercent = initialBalance > 0 ? (totalReturn / initialBalance * 100) : 0;

        $('#total-return').text(`${returnPercent.toFixed(2)}%`);
        const startTime = new Date(stats.start_time);
        const now = new Date();
        const diffDays = Math.floor((now - startTime) / (1000 * 60 * 60 * 24));
        $('#total-duration').html(`기간: <span>${diffDays}일</span>`);
    }

    if (data.positions) {
        const positionsCount = Object.keys(data.positions).length;
        let positionsValue = 0;
        for (const symbol in data.positions) {
            positionsValue += data.positions[symbol].market_value || 0;
        }
        $('#positions-count').text(positionsCount);
        $('#positions-value').html(`가치: <span>$${positionsValue.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}</span>`);
    }

    if (stats && stats.trades && stats.trades.length > 0) {
        const recentTrades = stats.trades.slice(-10).reverse();
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
