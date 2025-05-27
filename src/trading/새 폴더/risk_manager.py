import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.utils.logger import Logger


class RiskManager:
    """
    리스크 관리를 담당하는 매니저 클래스
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # 계정 자본금 대비 최대 포지션 크기 (10%)
        max_drawdown: float = 0.05,      # 허용 가능한 최대 낙폭 (5%)
        max_trade_amount: float = 1000.0, # 단일 거래 최대 금액
        max_daily_loss: float = 0.02,    # 일일 최대 손실 허용치 (계정 자본금 대비, 2%)
        position_sizing_method: str = "fixed_percent",  # 포지션 크기 결정 방법
        logger: Optional[Logger] = None
    ):
        """
        RiskManager 클래스 초기화
        
        Args:
            max_position_size: 계정 자본금 대비 최대 포지션 크기 (비율)
            max_drawdown: 허용 가능한 최대 낙폭 (비율)
            max_trade_amount: 단일 거래 최대 금액
            max_daily_loss: 일일 최대 손실 허용치 (계정 자본금 대비 비율)
            position_sizing_method: 포지션 크기 결정 방법
                                  ("fixed_percent", "kelly", "volatility_adjusted")
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_trade_amount = max_trade_amount
        self.max_daily_loss = max_daily_loss
        self.position_sizing_method = position_sizing_method
        self.logger = logger
        
        # 일일 거래 및 손익 추적
        self.daily_trades = {}
        self.daily_pnl = {}
        self.initial_balance = None
        
        # 최대 낙폭 추적
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        
        if self.logger:
            self.logger.info(f"RiskManager 초기화 완료")
            self.logger.info(f"최대 포지션 크기: {self.max_position_size * 100}%")
            self.logger.info(f"최대 낙폭: {self.max_drawdown * 100}%")
            self.logger.info(f"단일 거래 최대 금액: ${self.max_trade_amount}")
            self.logger.info(f"일일 최대 손실: {self.max_daily_loss * 100}%")
            self.logger.info(f"포지션 크기 결정 방법: {self.position_sizing_method}")
    
    def calculate_position_size(
        self,
        symbol: str,
        side: str,
        available_balance: float,
        current_price: float,
        position_ratio: float = 1.0,
        current_position: float = 0.0,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None
    ) -> float:
        """
        해당 심볼의 적절한 거래 수량 계산
        
        Args:
            symbol: 심볼/티커
            side: 매수/매도 ('buy' 또는 'sell')
            available_balance: 사용 가능한 자본금
            current_price: 현재 가격
            position_ratio: 포지션 크기 비율 (0.0 ~ 1.0)
            current_position: 현재 보유 수량
            volatility: 가격 변동성 (옵션, 변동성 조정 방식에 사용)
            win_rate: 승률 (옵션, 켈리 공식에 사용)
            
        Returns:
            거래 수량
        """
        try:
            # 1. 일일 손실 한도 확인
            today = datetime.now().strftime("%Y-%m-%d")
            daily_loss_limit = available_balance * self.max_daily_loss
            
            if today in self.daily_pnl and self.daily_pnl[today] < -daily_loss_limit:
                if self.logger:
                    self.logger.warning(f"일일 손실 한도 초과: ${self.daily_pnl[today]:.2f}, 거래 중단")
                return 0.0
            
            # 2. 현재 낙폭 확인
            if self.current_drawdown <= -self.max_drawdown:
                if self.logger:
                    self.logger.warning(f"최대 낙폭 초과: {self.current_drawdown * 100:.2f}%, 거래 제한")
                return 0.0
            
            # 3. 포지션 크기 계산
            quantity = 0.0
            
            if self.position_sizing_method == "fixed_percent":
                # 고정 비율 방식
                position_amount = available_balance * self.max_position_size * position_ratio
                position_amount = min(position_amount, self.max_trade_amount)
                quantity = position_amount / current_price
                
            elif self.position_sizing_method == "kelly" and win_rate is not None:
                # 켈리 공식 방식
                # f* = (p(b + 1) - 1) / b, p: 승률, b: 보상/위험 비율 (여기서는 1로 가정)
                reward_risk_ratio = 1.0  # 단순화를 위해 1로 가정
                kelly_fraction = max(0, (win_rate * (reward_risk_ratio + 1) - 1) / reward_risk_ratio)
                kelly_fraction = min(kelly_fraction, self.max_position_size)  # 최대 포지션 크기로 제한
                
                position_amount = available_balance * kelly_fraction * position_ratio
                position_amount = min(position_amount, self.max_trade_amount)
                quantity = position_amount / current_price
                
            elif self.position_sizing_method == "volatility_adjusted" and volatility is not None:
                # 변동성 조정 방식
                # 변동성이 높을수록 포지션 크기를 줄임
                volatility_factor = 1.0 / (1.0 + volatility)  # 0 ~ 1 사이로 정규화
                adjusted_position_size = self.max_position_size * volatility_factor
                
                position_amount = available_balance * adjusted_position_size * position_ratio
                position_amount = min(position_amount, self.max_trade_amount)
                quantity = position_amount / current_price
                
            else:
                # 기본 방식 (고정 비율)
                position_amount = available_balance * self.max_position_size * position_ratio
                position_amount = min(position_amount, self.max_trade_amount)
                quantity = position_amount / current_price
            
            # 4. 매수/매도에 따른 수량 조정
            if side == "buy":
                # 매수량 계산
                if current_position < 0:  # 현재 공매도 포지션이 있으면
                    quantity = max(quantity, abs(current_position))  # 최소한 현재 포지션을 청산하는 수량
                    
            elif side == "sell":
                # 매도량 계산
                if current_position > 0:  # 현재 매수 포지션이 있으면
                    quantity = max(quantity, current_position)  # 최소한 현재 포지션을 청산하는 수량
                    
            # 5. 수량 소수점 처리 (거래소마다 다름, 기본값은 소수점 4자리)
            quantity = round(quantity, 4)
            
            if self.logger:
                position_amount = quantity * current_price
                self.logger.info(f"{symbol} {side} 주문 수량 계산: {quantity}, 금액: ${position_amount:.2f}")
                
            return quantity
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"포지션 크기 계산 중 오류 발생: {e}")
            return 0.0
    
    def update_balance(self, current_balance: float) -> None:
        """
        계정 자본금 업데이트 및 낙폭 계산
        
        Args:
            current_balance: 현재 계정 자본금
        """
        try:
            # 초기 자본금 설정 (처음 호출 시)
            if self.initial_balance is None:
                self.initial_balance = current_balance
                self.peak_balance = current_balance
            
            # 최대 자본금 업데이트
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            # 현재 낙폭 계산
            if self.peak_balance > 0:
                self.current_drawdown = (current_balance - self.peak_balance) / self.peak_balance
            
            if self.logger and self.current_drawdown < 0:
                self.logger.info(f"현재 낙폭: {self.current_drawdown * 100:.2f}%")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"계정 자본금 업데이트 중 오류 발생: {e}")
    
    def record_trade(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: float, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        거래 기록
        
        Args:
            symbol: 심볼/티커
            side: 매수/매도 ('buy' 또는 'sell')
            quantity: 수량
            price: 가격
            timestamp: 거래 시간 (기본값: 현재 시간)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            date_str = timestamp.strftime("%Y-%m-%d")
            
            # 일일 거래 기록 초기화
            if date_str not in self.daily_trades:
                self.daily_trades[date_str] = []
                
            # 거래 기록 추가
            trade = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "amount": quantity * price,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.daily_trades[date_str].append(trade)
            
            if self.logger:
                self.logger.info(f"거래 기록 추가: {symbol} {side} {quantity} @ {price}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"거래 기록 중 오류 발생: {e}")
    
    def update_pnl(self, symbol: str, realized_pnl: float, timestamp: Optional[datetime] = None) -> None:
        """
        손익 업데이트
        
        Args:
            symbol: 심볼/티커
            realized_pnl: 실현 손익
            timestamp: 거래 시간 (기본값: 현재 시간)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            date_str = timestamp.strftime("%Y-%m-%d")
            
            # 일일 손익 기록 초기화
            if date_str not in self.daily_pnl:
                self.daily_pnl[date_str] = 0.0
                
            # 손익 업데이트
            self.daily_pnl[date_str] += realized_pnl
            
            if self.logger:
                if realized_pnl > 0:
                    self.logger.info(f"{symbol} 거래 이익: ${realized_pnl:.2f}, 일일 총 손익: ${self.daily_pnl[date_str]:.2f}")
                else:
                    self.logger.info(f"{symbol} 거래 손실: ${realized_pnl:.2f}, 일일 총 손익: ${self.daily_pnl[date_str]:.2f}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"손익 업데이트 중 오류 발생: {e}")
    
    def check_risk_limits(self, current_balance: float) -> Dict[str, Any]:
        """
        리스크 한도 확인
        
        Args:
            current_balance: 현재 계정 자본금
            
        Returns:
            리스크 상태 정보
        """
        try:
            # 낙폭 업데이트
            self.update_balance(current_balance)
            
            # 오늘 날짜 정보
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 일일 손실 계산
            daily_loss = self.daily_pnl.get(today, 0.0)
            daily_loss_limit = self.initial_balance * self.max_daily_loss if self.initial_balance else float('inf')
            
            # 리스크 상태 결과
            risk_status = {
                "trade_allowed": True,
                "warnings": [],
                "daily_loss": daily_loss,
                "daily_loss_limit": daily_loss_limit,
                "daily_loss_pct": (daily_loss / self.initial_balance * 100) if self.initial_balance else 0,
                "current_drawdown": self.current_drawdown * 100,  # 퍼센트로 표시
                "max_drawdown_limit": self.max_drawdown * 100,  # 퍼센트로 표시
            }
            
            # 일일 손실 한도 확인
            if daily_loss < -daily_loss_limit:
                risk_status["trade_allowed"] = False
                risk_status["warnings"].append(f"일일 손실 한도 초과: ${daily_loss:.2f} (한도: ${-daily_loss_limit:.2f})")
            
            # 최대 낙폭 확인
            if self.current_drawdown <= -self.max_drawdown:
                risk_status["trade_allowed"] = False
                risk_status["warnings"].append(f"최대 낙폭 초과: {self.current_drawdown * 100:.2f}% (한도: {self.max_drawdown * 100:.2f}%)")
            
            return risk_status
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"리스크 한도 확인 중 오류 발생: {e}")
            
            # 오류 발생 시 거래 중단
            return {
                "trade_allowed": False,
                "warnings": [f"리스크 확인 중 오류 발생: {e}"],
                "error": str(e)
            }
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        특정 날짜의 일일 거래 요약 조회
        
        Args:
            date: 날짜 문자열 (YYYY-MM-DD 형식, 기본값: 오늘)
            
        Returns:
            일일 거래 요약 정보
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
                
            # 해당 날짜의 거래 및 손익 정보
            trades = self.daily_trades.get(date, [])
            pnl = self.daily_pnl.get(date, 0.0)
            
            # 거래 횟수 및 거래량 계산
            total_trades = len(trades)
            total_buy_amount = sum(t["amount"] for t in trades if t["side"] == "buy")
            total_sell_amount = sum(t["amount"] for t in trades if t["side"] == "sell")
            
            # 심볼별 거래량 계산
            symbol_volumes = {}
            for trade in trades:
                symbol = trade["symbol"]
                if symbol not in symbol_volumes:
                    symbol_volumes[symbol] = {"buy": 0.0, "sell": 0.0}
                
                symbol_volumes[symbol][trade["side"]] += trade["amount"]
            
            return {
                "date": date,
                "total_trades": total_trades,
                "total_buy_amount": total_buy_amount,
                "total_sell_amount": total_sell_amount,
                "pnl": pnl,
                "pnl_pct": (pnl / self.initial_balance * 100) if self.initial_balance else 0,
                "symbol_volumes": symbol_volumes,
                "trades": trades
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"일일 거래 요약 조회 중 오류 발생: {e}")
            return {"error": str(e)} 