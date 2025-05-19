"""
대시보드 시각화 모듈
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class Visualizer:
    """
    대시보드 차트 및 시각화를 담당하는 클래스
    """
    
    def __init__(self):
        """
        Visualizer 클래스 초기화
        """
        # 기본 차트 색상
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'buy': '#2ca02c',
            'sell': '#d62728',
            'hold': '#7f7f7f',
            'profit': '#2ca02c',
            'loss': '#d62728',
            'background': '#ffffff',
            'grid': '#e6e6e6'
        }
        
        # 기본 차트 레이아웃
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif'},
            'xaxis': {'gridcolor': self.colors['grid']},
            'yaxis': {'gridcolor': self.colors['grid']},
            'paper_bgcolor': self.colors['background'],
            'plot_bgcolor': self.colors['background'],
            'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
        }
    
    def create_portfolio_value_chart(
        self, 
        portfolio_values: List[float], 
        timestamps: List[str],
        benchmark_values: Optional[List[float]] = None,
        title: str = "포트폴리오 가치 변화"
    ) -> go.Figure:
        """
        포트폴리오 가치 변화 차트 생성
        
        Args:
            portfolio_values: 포트폴리오 가치 목록
            timestamps: 타임스탬프 목록
            benchmark_values: 벤치마크 가치 목록 (옵션)
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 포트폴리오 가치 라인 추가
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=portfolio_values,
            mode='lines',
            name='포트폴리오',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # 벤치마크 추가 (있는 경우)
        if benchmark_values is not None and len(benchmark_values) == len(timestamps):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=benchmark_values,
                mode='lines',
                name='벤치마크',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
        
        # 시작점과 종료점 강조
        if len(portfolio_values) > 0:
            fig.add_trace(go.Scatter(
                x=[timestamps[0]],
                y=[portfolio_values[0]],
                mode='markers',
                name='시작',
                marker=dict(color=self.colors['primary'], size=10),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[timestamps[-1]],
                y=[portfolio_values[-1]],
                mode='markers',
                name='종료',
                marker=dict(color=self.colors['primary'], size=10),
                showlegend=False
            ))
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': '날짜',
            'yaxis_title': '가치 (원)',
            'hovermode': 'x unified'
        })
        
        fig.update_layout(layout)
        
        return fig
    
    def create_returns_chart(
        self, 
        returns: List[float], 
        timestamps: List[str],
        benchmark_returns: Optional[List[float]] = None,
        title: str = "수익률 변화"
    ) -> go.Figure:
        """
        수익률 변화 차트 생성
        
        Args:
            returns: 수익률 목록
            timestamps: 타임스탬프 목록
            benchmark_returns: 벤치마크 수익률 목록 (옵션)
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 수익률 라인 추가
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[r * 100 for r in returns],  # 퍼센트로 변환
            mode='lines',
            name='전략 수익률',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # 벤치마크 추가 (있는 경우)
        if benchmark_returns is not None and len(benchmark_returns) == len(timestamps):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[r * 100 for r in benchmark_returns],  # 퍼센트로 변환
                mode='lines',
                name='벤치마크 수익률',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
        
        # 0% 라인 추가
        fig.add_trace(go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[0, 0],
            mode='lines',
            name='기준선',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': '날짜',
            'yaxis_title': '수익률 (%)',
            'hovermode': 'x unified'
        })
        
        fig.update_layout(layout)
        
        return fig
    
    def create_drawdown_chart(
        self, 
        drawdowns: List[float], 
        timestamps: List[str],
        title: str = "낙폭(Drawdown) 변화"
    ) -> go.Figure:
        """
        낙폭(Drawdown) 차트 생성
        
        Args:
            drawdowns: 낙폭 목록
            timestamps: 타임스탬프 목록
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 낙폭 라인 추가
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[d * 100 for d in drawdowns],  # 퍼센트로 변환
            mode='lines',
            name='낙폭',
            line=dict(color=self.colors['danger'], width=2),
            fill='tozeroy',
            fillcolor=f'rgba({int(self.colors["danger"][1:3], 16)}, {int(self.colors["danger"][3:5], 16)}, {int(self.colors["danger"][5:7], 16)}, 0.2)'
        ))
        
        # 최대 낙폭 표시
        max_drawdown_idx = np.argmin(drawdowns)
        max_drawdown = drawdowns[max_drawdown_idx]
        
        fig.add_trace(go.Scatter(
            x=[timestamps[max_drawdown_idx]],
            y=[max_drawdown * 100],
            mode='markers',
            name=f'최대 낙폭: {max_drawdown * 100:.2f}%',
            marker=dict(color=self.colors['danger'], size=10)
        ))
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': '날짜',
            'yaxis_title': '낙폭 (%)',
            'hovermode': 'x unified'
        })
        
        fig.update_layout(layout)
        
        return fig
    
    def create_trade_chart(
        self, 
        prices: List[float], 
        timestamps: List[str],
        buy_signals: List[Tuple[str, float]],
        sell_signals: List[Tuple[str, float]],
        title: str = "거래 신호"
    ) -> go.Figure:
        """
        거래 신호 차트 생성
        
        Args:
            prices: 가격 목록
            timestamps: 타임스탬프 목록
            buy_signals: 매수 신호 목록 (타임스탬프, 가격)
            sell_signals: 매도 신호 목록 (타임스탬프, 가격)
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 가격 라인 추가
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines',
            name='가격',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # 매수 신호 추가
        if buy_signals:
            buy_timestamps, buy_prices = zip(*buy_signals)
            fig.add_trace(go.Scatter(
                x=buy_timestamps,
                y=buy_prices,
                mode='markers',
                name='매수 신호',
                marker=dict(color=self.colors['buy'], size=10, symbol='triangle-up')
            ))
        
        # 매도 신호 추가
        if sell_signals:
            sell_timestamps, sell_prices = zip(*sell_signals)
            fig.add_trace(go.Scatter(
                x=sell_timestamps,
                y=sell_prices,
                mode='markers',
                name='매도 신호',
                marker=dict(color=self.colors['sell'], size=10, symbol='triangle-down')
            ))
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': '날짜',
            'yaxis_title': '가격',
            'hovermode': 'closest'
        })
        
        fig.update_layout(layout)
        
        return fig
    
    def create_performance_comparison_chart(
        self, 
        metrics: Dict[str, Dict[str, float]],
        metric_name: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        모델 성능 비교 차트 생성
        
        Args:
            metrics: 모델별 성능 지표 딕셔너리
            metric_name: 비교할 지표 이름
            title: 차트 제목 (옵션)
            
        Returns:
            Plotly Figure 객체
        """
        models = []
        values = []
        
        for model_id, model_metrics in metrics.items():
            if metric_name in model_metrics:
                models.append(model_id)
                values.append(model_metrics[metric_name])
        
        if not models:
            # 데이터가 없는 경우 빈 차트 반환
            fig = go.Figure()
            fig.update_layout(
                title=f"데이터 없음: {metric_name}",
                xaxis_title="모델",
                yaxis_title=metric_name
            )
            return fig
        
        # 값에 따라 정렬
        sorted_indices = np.argsort(values)[::-1]  # 내림차순
        sorted_models = [models[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # 차트 색상 결정
        colors = [self.colors['primary'] for _ in range(len(sorted_models))]
        
        # 막대 차트 생성
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_models,
                y=sorted_values,
                marker_color=colors
            )
        ])
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title if title else f"{metric_name} 비교",
            'xaxis_title': "모델",
            'yaxis_title': metric_name
        })
        
        fig.update_layout(layout)
        
        return fig
    
    def create_trade_distribution_chart(
        self, 
        trades: List[Dict[str, Any]],
        title: str = "거래 분포"
    ) -> go.Figure:
        """
        거래 분포 차트 생성
        
        Args:
            trades: 거래 목록
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        # 거래별 손익 계산
        pnls = []
        for trade in trades:
            if 'profit' in trade and trade['profit'] is not None:
                pnls.append(trade['profit'])
            elif 'pnl' in trade and trade['pnl'] is not None:
                pnls.append(trade['pnl'])
        
        if not pnls:
            # 데이터가 없는 경우 빈 차트 반환
            fig = go.Figure()
            fig.update_layout(
                title="데이터 없음: 거래 분포",
                xaxis_title="손익",
                yaxis_title="거래 수"
            )
            return fig
        
        # 히스토그램 생성
        fig = go.Figure(data=[
            go.Histogram(
                x=pnls,
                marker_color=self.colors['primary'],
                opacity=0.7,
                name="거래 분포"
            )
        ])
        
        # 0 기준선 추가
        fig.add_vline(
            x=0, 
            line_width=1, 
            line_dash="dash", 
            line_color="gray"
        )
        
        # 이익 거래와 손실 거래 수 계산
        profit_trades = sum(1 for p in pnls if p > 0)
        loss_trades = sum(1 for p in pnls if p < 0)
        
        # 승률 계산
        win_rate = profit_trades / len(pnls) if pnls else 0
        
        # 주석 추가
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"총 거래: {len(pnls)}건<br>이익 거래: {profit_trades}건<br>손실 거래: {loss_trades}건<br>승률: {win_rate:.2%}",
            showarrow=False,
            font=dict(size=12),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': "손익",
            'yaxis_title': "거래 수",
            'bargap': 0.1
        })
        
        fig.update_layout(layout)
        
        return fig
    
    def create_model_comparison_radar_chart(
        self, 
        metrics: Dict[str, Dict[str, float]],
        metric_names: List[str],
        model_ids: Optional[List[str]] = None,
        title: str = "모델 성능 비교"
    ) -> go.Figure:
        """
        모델 성능 비교 레이더 차트 생성
        
        Args:
            metrics: 모델별 성능 지표 딕셔너리
            metric_names: 비교할 지표 이름 목록
            model_ids: 비교할 모델 ID 목록 (옵션, 없으면 모든 모델 비교)
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        if model_ids is None:
            model_ids = list(metrics.keys())
        
        # 최대 5개 모델만 비교
        if len(model_ids) > 5:
            model_ids = model_ids[:5]
        
        # 레이더 차트 생성
        fig = go.Figure()
        
        # 각 모델별 데이터 추가
        for i, model_id in enumerate(model_ids):
            if model_id not in metrics:
                continue
                
            model_metrics = metrics[model_id]
            
            # 선택한 지표만 사용
            values = []
            for metric in metric_names:
                values.append(model_metrics.get(metric, 0))
            
            # 첫 값을 마지막에 다시 추가하여 레이더 차트 닫기
            values.append(values[0])
            labels = metric_names + [metric_names[0]]
            
            # 색상 선택
            color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=model_id,
                line_color=color
            ))
        
        # 레이아웃 설정
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]  # 정규화된 값 사용
                )
            ),
            title=title,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_live_trading_dashboard(
        self, 
        trading_stats: Dict[str, Any],
        price_data: pd.DataFrame,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, go.Figure]:
        """
        실시간 트레이딩 대시보드 차트 생성
        
        Args:
            trading_stats: 트레이딩 통계 정보
            price_data: 가격 데이터
            trades: 거래 내역
            
        Returns:
            차트 목록 딕셔너리
        """
        charts = {}
        
        # 포트폴리오 가치 차트
        if 'portfolio_values' in trading_stats and 'timestamps' in trading_stats:
            charts['portfolio'] = self.create_portfolio_value_chart(
                portfolio_values=trading_stats['portfolio_values'],
                timestamps=trading_stats['timestamps'],
                title="실시간 포트폴리오 가치"
            )
        
        # 수익률 차트
        if 'returns' in trading_stats and 'timestamps' in trading_stats:
            charts['returns'] = self.create_returns_chart(
                returns=trading_stats['returns'],
                timestamps=trading_stats['timestamps'],
                title="실시간 수익률 변화"
            )
        
        # 가격 및 거래 차트
        if not price_data.empty and 'date' in price_data.columns and 'close' in price_data.columns:
            # 매수/매도 신호 추출
            buy_signals = []
            sell_signals = []
            
            for trade in trades:
                if 'timestamp' in trade and 'price' in trade:
                    if trade.get('side') == 'buy':
                        buy_signals.append((trade['timestamp'], trade['price']))
                    elif trade.get('side') == 'sell':
                        sell_signals.append((trade['timestamp'], trade['price']))
            
            charts['price'] = self.create_trade_chart(
                prices=price_data['close'].tolist(),
                timestamps=price_data['date'].astype(str).tolist(),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                title="가격 차트 및 거래 신호"
            )
        
        return charts