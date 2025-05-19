import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment
from src.utils.logger import Logger
from src.config.config import Config


class Backtester:
    """
    백테스팅 모듈: 학습된 SAC 에이전트를 사용하여 과거 데이터에서 성능을 테스트합니다.
    """
    
    """
    Fixed Backtester.__init__ method with default values for missing configuration parameters
    """

    def __init__(self, agent, test_data, config, logger, initial_balance=10000.0, transaction_fee_percent=0.0025):
        """
        Backtester 클래스 초기화
        
        Args:
            agent: 학습된 SAC 에이전트
            test_data: 테스트할 과거 데이터
            config: 설정 객체
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
            initial_balance: 초기 자본금
            transaction_fee_percent: 거래 수수료율 (0.0025 = 0.25%)
        """
        self.agent = agent
        self.test_data = test_data
        self.config = config
        self.logger = logger
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Remove reward_scaling parameter or check if TradingEnvironment accepts it
        self.env = TradingEnvironment(
            data=test_data,
            raw_data=test_data,
            window_size=getattr(config, 'window_size', 20),
            initial_balance=initial_balance,
            transaction_fee_percent=transaction_fee_percent
            # reward_scaling=getattr(config, 'reward_scaling', 1.0)
        )
        
    def run_backtest(self, verbose: bool = True) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            verbose: 진행 상황을 표시할지 여부
        
        Returns:
            백테스트 결과를 담은 딕셔너리
        """
        # 환경 초기화
        state = self.env.reset()
        done = False
        total_reward = 0
        self.results = {
        "portfolio_values": [],
        "returns": [],
        "actions": [],
        "positions": [],
        "rewards": [],
        "timestamps": [],
        "trades": [],
        "metrics": {}
         }   
        # 백테스트 진행 상황 표시 설정
        iterator = tqdm(range(len(self.test_data) - self.config.window_size)) if verbose else range(len(self.test_data) - self.config.window_size)
        
        # 에피소드 진행
        for _ in iterator:
            if done:
                break
                
            # 에이전트로부터 행동 선택
            action = self.agent.select_action(state, evaluate=True)
            
            # 환경에서 한 스텝 진행
            next_state, reward, done, info = self.env.step(action)
            if verbose:
                print(f"[Step {info['step']}]")
                for k, v in info.items():
                    print(f"  {k}: {v}")
            # 결과 저장
            self.results["portfolio_values"].append(info["portfolio_value"])
            self.results["returns"].append(info.get("return", 0))  # 기본값 0으로 안전하게 접근
            self.results["actions"].append(action)
            self.results["positions"].append(info["position"])
            self.results["rewards"].append(reward)
            self.results["timestamps"].append(info["timestamps"])
            
            # 거래 기록 저장
            if info["trade_executed"]:
                self.results["trades"].append({
                    "timestamp": info["timestamp"],
                    "action": action,
                    "price": info["current_price"],
                    "shares": info["trade_shares"],
                    "cost": info["trade_cost"],
                    "position": info["position"],
                    "portfolio_value": info["portfolio_value"]
                })
            
            # 상태 및 보상 업데이트
            state = next_state
            total_reward += reward
            
        # 데이터프레임으로 변환
        self.results["portfolio_values"] = np.array(self.results["portfolio_values"])
        self.results["returns"] = np.array(self.results["returns"])
        self.results["actions"] = np.array(self.results["actions"])
        self.results["positions"] = np.array(self.results["positions"])
        self.results["rewards"] = np.array(self.results["rewards"])
        # 성능 지표 계산
        self.calculate_metrics()
        
        # 로깅
        if self.logger:
            self.logger.info(f"Backtest completed with final portfolio value: {self.results['portfolio_values'][-1]:.2f}")
            self.logger.info(f"Total reward: {total_reward:.2f}")
            self.logger.info(f"Performance metrics: {self.results['metrics']}")
        
        return self.results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        백테스트 결과에서 성능 지표 계산
        
        Returns:
            성능 지표를 담은 딕셔너리
        """
        # 일별 수익률 계산
        daily_returns = pd.Series(self.results["returns"])
        
        # 누적 수익률
        cumulative_return = (self.results["portfolio_values"][-1] / self.initial_balance) - 1
        
        # 연간화된 수익률 (252 트레이딩 데이)
        n_days = len(daily_returns)
        annual_return = ((1 + cumulative_return) ** (252 / n_days)) - 1
        
        # 일별 표준편차
        daily_std = daily_returns.std()
        
        # 연간화된 변동성
        annual_volatility = daily_std * np.sqrt(252)
        
        # 샤프 비율 (무위험 이자율 0% 가정)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # 최대 낙폭 (Maximum Drawdown)
        portfolio_values = pd.Series(self.results["portfolio_values"])
        cumulative_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # 승률
        if len(self.results["trades"]) > 0:
            profitable_trades = sum(1 for trade in self.results["trades"] if 
                                   (trade["action"] > 0 and self.results["returns"][self.results["timestamps"].index(trade["timestamp"])] > 0) or
                                   (trade["action"] < 0 and self.results["returns"][self.results["timestamps"].index(trade["timestamp"])] < 0))
            win_rate = profitable_trades / len(self.results["trades"])
        else:
            win_rate = 0
        
        # 총 거래 횟수
        total_trades = len(self.results["trades"])
        
        # 지표 저장
        metrics = {
            "cumulative_return": cumulative_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
        }
        
        self.results["metrics"] = metrics
        return metrics
    
    def save_results(self, filepath: str) -> None:
        """
        백테스트 결과를 JSON 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        # 결과 저장을 위한 딕셔너리 생성
        save_data = {
            "initial_balance": self.initial_balance,
            "final_portfolio_value": float(self.results["portfolio_values"][-1]),
            "total_trades": len(self.results["trades"]),
            "metrics": self.results["metrics"],
            "trades": self.results["trades"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # JSON 파일로 저장
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
            
        if self.logger:
            self.logger.info(f"Backtest results saved to {filepath}")
    
    def plot_portfolio_performance(self, save_path: Optional[str] = None) -> None:
        """
        포트폴리오 성능 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        plt.figure(figsize=(14, 7))
        
        # 스타일 설정
        sns.set_style("whitegrid")
        
        # 포트폴리오 가치 플롯
        plt.plot(self.results["portfolio_values"], label="Portfolio Value", color="royalblue", linewidth=2)
        
        # 매수/매도 지점 표시
        for trade in self.results["trades"]:
            idx = self.results["timestamps"].index(trade["timestamp"])
            
            if trade["action"] > 0:  # 매수
                plt.scatter(idx, self.results["portfolio_values"][idx], color="green", marker="^", s=100)
            elif trade["action"] < 0:  # 매도
                plt.scatter(idx, self.results["portfolio_values"][idx], color="red", marker="v", s=100)
        
        # 성능 지표 텍스트
        metrics_text = f"""
        Cumulative Return: {self.results['metrics']['cumulative_return']:.2%}
        Annual Return: {self.results['metrics']['annual_return']:.2%}
        Sharpe Ratio: {self.results['metrics']['sharpe_ratio']:.2f}
        Max Drawdown: {self.results['metrics']['max_drawdown']:.2%}
        Win Rate: {self.results['metrics']['win_rate']:.2%}
        Total Trades: {self.results['metrics']['total_trades']}
        """
        
        plt.text(0.02, 0.97, metrics_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.3))
        
        # 그래프 레이블 및 제목
        plt.title("Portfolio Performance during Backtest", fontsize=16)
        plt.xlabel("Trading Days", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.logger:
                self.logger.info(f"Portfolio performance plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_drawdown(self, save_path: Optional[str] = None) -> None:
        """
        낙폭(Drawdown) 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        plt.figure(figsize=(14, 7))
        
        # 스타일 설정
        sns.set_style("whitegrid")
        
        # 포트폴리오 가치
        portfolio_values = pd.Series(self.results["portfolio_values"])
        
        # 누적 최대값
        cumulative_max = portfolio_values.cummax()
        
        # 낙폭 계산
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        
        # 낙폭 플롯
        plt.fill_between(range(len(drawdown)), 0, drawdown, color='crimson', alpha=0.3, label='Drawdown')
        plt.plot(drawdown, color='crimson', linestyle='-', linewidth=1)
        
        # 최대 낙폭 표시
        max_dd = drawdown.min()
        max_dd_idx = drawdown.argmin()
        plt.scatter(max_dd_idx, max_dd, color='darkred', marker='o', s=100, label=f'Max Drawdown: {max_dd:.2%}')
        
        # 그래프 레이블 및 제목
        plt.title("Portfolio Drawdown", fontsize=16)
        plt.xlabel("Trading Days", fontsize=12)
        plt.ylabel("Drawdown", fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.logger:
                self.logger.info(f"Drawdown plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_returns_distribution(self, save_path: Optional[str] = None) -> None:
        """
        수익률 분포 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        plt.figure(figsize=(14, 7))
        
        # 스타일 설정
        sns.set_style("whitegrid")
        
        # 일별 수익률
        daily_returns = pd.Series(self.results["returns"])
        
        # 수익률 분포 플롯
        sns.histplot(daily_returns, kde=True, color="royalblue")
        
        # 0선 표시
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # 평균 수익률 표시
        mean_return = daily_returns.mean()
        plt.axvline(x=mean_return, color='green', linestyle='-', alpha=0.7)
        
        # 통계 정보
        stats_text = f"""
        Mean Return: {mean_return:.4%}
        Std Deviation: {daily_returns.std():.4%}
        Min Return: {daily_returns.min():.4%}
        Max Return: {daily_returns.max():.4%}
        Skewness: {daily_returns.skew():.4f}
        Kurtosis: {daily_returns.kurtosis():.4f}
        """
        
        plt.text(0.02, 0.97, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.3))
        
        # 그래프 레이블 및 제목
        plt.title("Distribution of Daily Returns", fontsize=16)
        plt.xlabel("Daily Return", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.logger:
                self.logger.info(f"Returns distribution plot saved to {save_path}")
        else:
            plt.show()
            
    def visualize_results(self, save_dir: Optional[str] = None) -> None:
        """
        백테스트 결과를 시각화하고 저장
        
        Args:
            save_dir: 시각화 결과를 저장할 디렉토리 경로 (옵션)
        """
        # 저장 디렉토리 설정
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 포트폴리오 성능 차트
            portfolio_path = os.path.join(save_dir, f"portfolio_performance_{timestamp}.png")
            self.plot_portfolio_performance(portfolio_path)
            
            # 낙폭 차트
            drawdown_path = os.path.join(save_dir, f"drawdown_{timestamp}.png")
            self.plot_drawdown(drawdown_path)
            
            # 수익률 분포 차트
            returns_path = os.path.join(save_dir, f"returns_distribution_{timestamp}.png")
            self.plot_returns_distribution(returns_path)
            
            # 결과 데이터 저장
            results_path = os.path.join(save_dir, f"backtest_results_{timestamp}.json")
            self.save_results(results_path)
        else:
            # 시각화만 수행
            self.plot_portfolio_performance()
            self.plot_drawdown()
            self.plot_returns_distribution() 