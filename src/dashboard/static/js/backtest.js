

  function loadModels() {
    fetch('/api/backtest-results')
      .then(response => response.json())
      .then(data => {
        const modelSelect = document.getElementById('modelSelect');
        for (const modelId in data) {
          const option = document.createElement('option');
          option.value = modelId;
          option.textContent = modelId;
          modelSelect.appendChild(option);
        }
      });
  }

  function loadBacktestResult(modelId) {
    if (!modelId) return;
    fetch(`/api/backtest-results?model_id=${modelId}`)
      .then(response => response.json())
      .then(data => {
        if (!data || !data[modelId]) return;
        const result = data[modelId];
        displayMetrics(result.metrics);
        loadCharts(modelId);
        displayTrades(result.trades);
      });
  }

  function displayMetrics(metrics) {
    document.getElementById('totalReturn').textContent = (metrics.total_return).toFixed(2) + '%';
    document.getElementById('sharpeRatio').textContent = metrics.sharpe_ratio.toFixed(2);
    document.getElementById('maxDrawdown').textContent = (metrics.max_drawdown * 100).toFixed(2) + '%';
    document.getElementById('winRate').textContent = (metrics.win_rate * 100).toFixed(2) + '%';
  }

  function loadCharts(modelId) {
    fetch(`/api/charts/portfolio?model_id=${modelId}`)
      .then(res => res.json())
      .then(data => Plotly.newPlot('portfolioChart', data.data, data.layout));

    fetch(`/api/charts/drawdown?model_id=${modelId}`)
      .then(res => res.json())
      .then(data => Plotly.newPlot('drawdownChart', data.data, data.layout));

    fetch(`/api/charts/trade-distribution?model_id=${modelId}`)
      .then(res => res.json())
      .then(data => Plotly.newPlot('tradeDistributionChart', data.data, data.layout));

    fetch(`/api/charts/model-comparison`)
      .then(res => res.json())
      .then(data => Plotly.newPlot('modelComparisonChart', data.data, data.layout));
  }

  function displayTrades(trades) {
    const tbody = document.querySelector('#tradesTable tbody');
    tbody.innerHTML = '';
    if (!trades || trades.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="text-center">거래 내역이 없습니다.</td></tr>';
      return;
    }
    trades.forEach(trade => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${trade.timestamp}</td>
        <td>${trade.action > 0 ? '매수' : '매도'}</td>
        <td>${trade.price.toFixed(2)}</td>
        <td>${trade.shares}</td>
        <td>${trade.cost ? trade.cost.toFixed(2) : '-'}</td>
        <td>${trade.portfolio_value.toFixed(2)}</td>
      `;
      tbody.appendChild(row);
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    loadModels();
    document.getElementById('modelSelect').addEventListener('change', function () {
      loadBacktestResult(this.value);
    });
  });
