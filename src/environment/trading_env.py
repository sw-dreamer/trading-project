"""
강화학습을 위한 트레이딩 환경 모듈
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import gym
from gym import spaces
import sys
import os
from src.data_collection.data_collector import DataCollector
from decimal import Decimal, ROUND_DOWN, getcontext

# 저장 경로 폴더 생성
save_dir = './results'
os.makedirs(save_dir, exist_ok=True)

from src.config.config import (
    INITIAL_BALANCE,
    MAX_TRADING_UNITS,
    TRANSACTION_FEE_PERCENT,
    WINDOW_SIZE,
    LOGGER
)

class TradingEnvironment:
    """
    강화학습을 위한 트레이딩 환경 클래스
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        raw_data: pd.DataFrame = None,  # 추가: 원본 데이터 (정규화되지 않은 가격)
        window_size: int = WINDOW_SIZE,
        initial_balance: float = INITIAL_BALANCE,
        max_trading_units: int = MAX_TRADING_UNITS,
        transaction_fee_percent: float = TRANSACTION_FEE_PERCENT,
        symbol: str = None,
        train_data: bool = True,  # 추가: 학습용 데이터 여부
    ):
        """
        TradingEnvironment 클래스 초기화
        
        Args:
            data: 학습에 사용할 주식 데이터 (정규화된 데이터)
            raw_data: 정규화되지 않은 원본 가격 데이터 (None인 경우 data 사용)
            window_size: 관측 윈도우 크기
            initial_balance: 초기 자본금
            max_trading_units: 최대 거래 단위
            transaction_fee_percent: 거래 수수료 비율
            symbol: 주식 심볼 (로깅용)
            train_data: 학습용 데이터 여부
            X: 입력 특성 데이터
            y: 타겟 데이터
        """
        
        # # 원본 데이터가 제공되지 않은 경우 data를 사용
        # self.raw_data = raw_data if raw_data is not None else data
        # if 'timestamp' in self.data.columns:
        #     self.timestamps = self.data.index.values
        # else:
        #     self.timestamps = np.array([None] * len(self.data))  # 또는 적절한 기본값
        
        self.data = data
        
        # raw_data 처리: None인 경우 data에서 가격 정보 추출 시도
        if raw_data is not None:
            self.raw_data = raw_data
        else:
            # raw_data가 없는 경우 정규화된 데이터에서 가격 정보 추출 시도
            # 주의: 정규화된 데이터이므로 실제 거래에는 부적합할 수 있음
            LOGGER.warning("raw_data가 제공되지 않았습니다. 정규화된 데이터를 사용합니다.")
            print("⚠️  raw_data가 제공되지 않았습니다. 정규화된 데이터를 사용합니다.")
            self.raw_data = data
        
        print(f'정규화된 데이터 형태: {data.shape}')
        print(f'원본 데이터 형태: {self.raw_data.shape}')
        
        # 타임스탬프 초기화 수정
        if hasattr(self.raw_data.index, 'values'):
            self.timestamps = self.raw_data.index.values
        else:
            self.timestamps = np.array([None] * len(self.raw_data))

        # # 추가: X, y 데이터 저장
        # self.X = X
        # self.y = y
        # self.train_data = train_data

        # print(f'data 형태: {data.shape}')
        # if raw_data is not None:
        #     print(f'raw_data 형태: {raw_data.shape}')
        # else:
        #     print('raw_data가 제공되지 않아 data를 사용합니다.')
        
        self.window_size = window_size
        self.initial_balance = initial_balance
        print(f'initial_balance : {initial_balance}')
        self.max_trading_units = max_trading_units
        self.transaction_fee_percent = transaction_fee_percent
        self.symbol = symbol if symbol else "UNKNOWN"
        self.train_data=train_data
        
        # 데이터 관련 변수
        self.feature_dim = data.shape[1]
        self.data_length = len(data)
        
        # 환경 상태 변수
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_commission = 0
        self.position = "홀드"
        self.trade_executed = False 
        self.previous_shares_held = 0
        self.invalid_sell_penalty = False
        
        # 에피소드 히스토리
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []
        
        # 행동 공간: [-1.0, 1.0] 범위의 연속적인 값
        # -1.0은 최대 매도, 0.0은 홀드, 1.0은 최대 매수
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 관측 공간: 가격 데이터 + 포트폴리오 상태
        # 가격 데이터: window_size x feature_dim
        # 포트폴리오 상태: [보유 현금 비율, 보유 주식 비율]
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.window_size, self.feature_dim), dtype=np.float32
            ),
            'portfolio_state': spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float32
            )
        })
        
        # # 원본 데이터가 제공되었는지 확인하고 주요 컬럼이 존재하는지 검증
        # if raw_data is not None:
        #     if 'close' not in raw_data.columns:
        #         LOGGER.warning("raw_data에 'close' 컬럼이 없습니다. 마지막 컬럼을 종가로 사용합니다.")
        #         print("raw_data에 'close' 컬럼이 없습니다. 마지막 컬럼을 종가로 사용합니다.")
        
        # LOGGER.info(f"{self.symbol} 트레이딩 환경 초기화 완료: 데이터 길이 {self.data_length}")
        
        # 원본 데이터 검증
        if 'close' not in self.raw_data.columns:
            LOGGER.warning("raw_data에 'close' 컬럼이 없습니다. 마지막 컬럼을 종가로 사용합니다.")
            print("⚠️  raw_data에 'close' 컬럼이 없습니다. 마지막 컬럼을 종가로 사용합니다.")
        
        LOGGER.info(f"{self.symbol} 트레이딩 환경 초기화 완료: 데이터 길이 {self.data_length}")
    
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        환경 초기화
        
        Returns:
            초기 관측값
        """
        # 상태 초기화
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_commission = 0
        self.trade_executed = False
        self.position = "홀드"
        self.previous_shares_held=0
        self.invalid_sell_penalty = False


        # 히스토리 초기화
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []
        
        # 초기 관측값 반환
        return self._get_observation()
    
    def step(self, action: float) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        환경에서 한 스텝 진행
        
        Args:
            action: 에이전트의 행동 (-1.0 ~ 1.0 범위의 값)
            
        Returns:
            (관측값, 보상, 종료 여부, 추가 정보) 튜플
        """
        self.invalid_sell_penalty = False
        
        # 행동 기록
        self.actions_history.append(action)
        
        # 이전 포트폴리오 가치 계산
        prev_portfolio_value = self._get_portfolio_value()
        current_price_before = self._get_current_price()  # 행동 실행 전 가격 저장
        
        # 행동 실행
        self._execute_trade_action(action)
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 현재 포트폴리오 가치 계산
        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_values_history.append(current_portfolio_value)
        
        # 보상 계산
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)
        self.rewards_history.append(reward)
        
        # 종료 여부 확인
        done = self.current_step >= self.data_length - 1
        # self.previous_shares_held = self.shares_held
        
        # 관측값 및 추가 정보 반환
        observation = self._get_observation()
        info = self._get_info()
        
        # 디버깅 정보 추가
        current_price_after = self._get_current_price()
        LOGGER.info('='*50)
        LOGGER.info(f"step {self.current_step}: position = {self.position}, trade_executed = {self.trade_executed}")
        LOGGER.info(f"가격 변화: {current_price_before:.2f} -> {current_price_after:.2f}")
        LOGGER.info(f"포트폴리오 변화: {prev_portfolio_value:.2f} -> {current_portfolio_value:.2f}")
        LOGGER.info(f"보상: {reward:.6f}")
        LOGGER.info(f'주식 보유량 : {self.shares_held}')
        LOGGER.info('='*50)
        
        return observation, reward, done, info
    
    def _execute_trade_action(self, action: float) -> None:
        """
        거래 행동 실행
        
        Args:
            action: 에이전트의 행동 (-1.0 ~ 1.0 범위의 값)
        """
        current_price = self._get_current_price()
        
        # 현재 가격이 유효한지 확인
        if current_price <= 0:
            LOGGER.warning(f"현재 가격이 0 이하입니다: {current_price}")
            return
        
        # 행동 값을 거래 단위로 변환
        action_value = action[0] if isinstance(action, np.ndarray) else action
        
        # 행동 값 크기가 너무 작으면 증폭시킴 (최소 거래량 보장)
        min_action_threshold = 0.1  # 최소 행동 임계값
        if 0 < abs(action_value) < min_action_threshold:
            action_value = min_action_threshold * (1 if action_value > 0 else -1)
        
        # 기본값으로 설정
        self.trade_executed = False
        self.position = "홀드"
        self.invalid_sell_penalty = False  # 페널티 플래그 초기화
        
        getcontext().prec = 10
        
        if action_value > 0:  # 매수
            # 매수할 수 있는 최대 주식 수 계산
            max_affordable = self.balance / (current_price * (1 + self.transaction_fee_percent))
            # 행동 값에 따라 매수할 주식 수 결정 (0 ~ max_trading_units 범위)
            shares_to_buy = min(
                max_affordable,
                self.max_trading_units * action_value
            )
            # if shares_to_buy > 0:
                
            # 최소 1주 이상 매수하도록 조정 (0.5 이상이면 올림)
            # if 0 < shares_to_buy < 1:
            #     shares_to_buy = 1
            # else:
            #     shares_to_buy = max(1, int(shares_to_buy))
            
            if shares_to_buy > 0:
                shares_to_buy = Decimal(str(shares_to_buy)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN) # 소수점 거래
                shares_to_buy = float(shares_to_buy)
                # 매수 비용 계산
                buy_cost = shares_to_buy * current_price
                # 수수료 계산
                commission = buy_cost * self.transaction_fee_percent
                # 총 비용
                total_cost = buy_cost + commission
                
                # 잔고가 충분한지 확인
                if self.balance >= total_cost:
                    # 매수 실행
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
                    self.total_shares_purchased += shares_to_buy
                    self.total_commission += commission
                    
                    # 거래 실행 상태 및 포지션 업데이트
                    self.trade_executed = True
                    self.position = "매수"
                    
                    # 평균 매수 단가 업데이트
                    if self.shares_held > 0:
                        self.cost_basis = ((self.cost_basis * (self.shares_held - shares_to_buy)) + buy_cost) / self.shares_held
                    
                LOGGER.debug(f"매수: {shares_to_buy}주 @ {current_price:.2f}, 비용: {total_cost:.2f}, 수수료: {commission:.2f}")
                print(f"매수: {shares_to_buy}주 @ {current_price:.2f}, 비용: {total_cost:.2f}")
        
        elif action_value < 0:  # 매도
            if self.shares_held <= 0:
                # 보유 주식 없는데 매도 시도 → 강한 페널티 부여
                self.trade_executed = False
                self.position = "잘못된 매도"
                self.invalid_sell_penalty = True
                print("❌ 잘못된 매도 시도 (보유 주식 없음) - 큰 페널티 적용!")
                return  # 매도 실행하지 않음

            shares_to_sell = min(
                self.shares_held,
                self.max_trading_units * abs(action_value)
            )
            # if shares_to_sell > 0:
            #   shares_to_sell = round(shares_to_sell, 6)
            # if 0 < shares_to_sell < 1:
            #     shares_to_sell = 1
            # else:
            #     shares_to_sell = max(1, int(shares_to_sell))
            
            if shares_to_sell > 0:
                shares_to_sell = float(Decimal(str(shares_to_sell)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
                # 매도 수익 계산
                sell_value = shares_to_sell * current_price
                # 수수료 계산
                commission = sell_value * self.transaction_fee_percent
                # 순 수익
                net_value = sell_value - commission
                
                # 매도 실행
                self.balance += net_value
                self.shares_held -= shares_to_sell
                self.total_shares_sold += shares_to_sell
                self.total_sales_value += sell_value
                self.total_commission += commission
                # 거래 실행 상태 및 포지션 업데이트
                self.trade_executed = True
                self.position = "매도"
                
                LOGGER.debug(f"매도: {shares_to_sell}주 @ {current_price:.2f}, 수익: {net_value:.2f}, 수수료: {commission:.2f}")
                print(f"매도: {shares_to_sell}주 @ {current_price:.2f}, 수익: {net_value:.2f}")
            
            # elif shares_to_sell <= 0:
            #     # 매도량이 너무 적어서 실행 불가 (하지만 잘못된 매도는 아님!)
            #     self.trade_executed = False
            #     self.position = "홀드"
            #     return


    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        현재 관측값 반환
        
        Returns:
            관측값 딕셔너리
        """
        # 윈도우 크기만큼의 정규화된 가격 데이터 가져오기
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # 데이터가 충분하지 않은 경우 패딩 처리
        if start_idx == 0 and end_idx - start_idx < self.window_size:
            padding_size = self.window_size - (end_idx - start_idx)
            market_data = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
            actual_data = self.data.iloc[start_idx:end_idx].values
            market_data[-len(actual_data):] = actual_data
        else:
            # 충분한 데이터가 있는 경우
            market_data = self.data.iloc[start_idx:end_idx].values
            
            # 데이터 길이가 윈도우 크기보다 작은 경우 패딩
            if len(market_data) < self.window_size:
                padding = np.zeros((self.window_size - len(market_data), self.feature_dim), dtype=np.float32)
                market_data = np.vstack([padding, market_data])
        
        # 포트폴리오 상태 계산
        portfolio_value = self._get_portfolio_value()
        stock_value = self.shares_held * self._get_current_price()
        
        # 포트폴리오 가치가 0이 아닌지 확인
        if portfolio_value <= 0:
            portfolio_value = max(self.balance, 1.0)  # 최소 포트폴리오 가치 보장
        
        portfolio_state = np.array([
            self.balance / portfolio_value,  # 현금 비율
            stock_value / portfolio_value  # 주식 비율
        ], dtype=np.float32)
        
        # 관측값 딕셔너리 생성
        observation = {
            'market_data': market_data.astype(np.float32),
            'portfolio_state': portfolio_state
        }
        
        # 상태 기록
        self.states_history.append(observation)
        return observation    
                
    def _get_current_price(self) -> float:
        """
        현재 주가 반환
        
        Returns:
            현재 종가
        """
        # 인덱스가 유효한지 확인
        if self.current_step >= len(self.raw_data):
            LOGGER.error(f"유효하지 않은 인덱스: {self.current_step}, 데이터 길이: {len(self.raw_data)}")
            return 0.0
            
        # 원본 데이터에서 가격 정보를 가져옴
        if 'close' in self.raw_data.columns:
            price = float(self.raw_data.iloc[self.current_step]['close'])
        else:
            # 'close' 열이 없는 경우 마지막 열을 종가로 가정
            price = float(self.raw_data.iloc[self.current_step][-1])
            
        # 가격이 0 이하인 경우 처리
        if price <= 0:
            LOGGER.warning(f"현재 가격이 0 이하입니다: {price}, 최소값으로 조정")
            price = 0.01
            
        return price
    
    def _get_portfolio_value(self) -> float:
        """
        현재 포트폴리오 가치 계산
        
        Returns:
            포트폴리오 총 가치
        """
        current_price = self._get_current_price()
        stock_value = self.shares_held * current_price
        total_value = self.balance + stock_value
        
        # 포트폴리오 가치가 너무 작으면 최소 가치 보장
        if total_value <= 0:
            return max(self.balance, 1.0)
            
        return total_value
    
    def _calculate_reward(self, prev_portfolio_value: float, current_portfolio_value: float) -> float:
        """
        보상 계산
        
        Args:
            prev_portfolio_value: 이전 포트폴리오 가치
            current_portfolio_value: 현재 포트폴리오 가치
            
        Returns:
            보상값
        """
        # 포트폴리오 수익률 기반 보상
        if prev_portfolio_value <= 0:
            prev_portfolio_value = max(self.balance, 1.0)  # 최소값 보장
        
        # 수익률 계산
        return_rate = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        ##### 200배 증폭이 너무 크고 -10~10으로 클리핑하면 대부분의 보상이 극값에 몰림 
        ##### 해결: 증폭 계수를 10-50 정도로 줄이고 클리핑 범위 확대해야함
        # 수익률에 기반한 보상에 가중치 부여
        # reward = np.clip(return_rate * 200, -10, 10)  # 수익률을 200배하여 보상 크기 조정
        reward = np.clip(return_rate * 50, -5, 5)
        
        
        # 디버깅을 위한 출력
        print(f"포트폴리오 변화율: {return_rate:.6f}, 기본 보상: {reward:.6f}")
        
        # 보상이 너무 작으면 최소 보상 적용
        if abs(reward) < 0.001 and return_rate != 0:
            reward = 0.001 * (1 if return_rate > 0 else -1)
        
        # 잘못된 매도 시도가 있었는지 확인하고 별도의 큰 페널티 적용
        # (기존 보상과 별개로 추가)
        # if hasattr(self, 'invalid_sell_penalty') and self.invalid_sell_penalty:
        #     penalty = -10.0 
        #     print(f"잘못된 매도 시도에 대한 강한 페널티 적용: {penalty}")
        #     reward += penalty  # 기존 계산된 보상에 페널티를 더함
        #     print(f"최종 보상 (패널티 적용 후): {reward:.6f}")
        
        ##### -10 페널티가 너무 커서 에이전트가 매도를 기피하게 됨
        ##### 페널티를 -1.0 정도로 줄이거나 점진적 페널티 적용
        # 잘못된 매도 패널티
        if self.invalid_sell_penalty:
            penalty = -2.5
            reward += penalty
            print(f"잘못된 매도 페널티: {penalty}, 최종 보상: {reward:.6f}")
        
        print(f"포트폴리오 변화율: {return_rate:.6f}, 기본 보상: {reward:.6f}")
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """
        추가 정보 반환
        
        Returns:
            추가 정보 딕셔너리
        """
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        total_return = ((portfolio_value - self.initial_balance) / self.initial_balance) if self.initial_balance > 0 else 0
        
        # # 수익률 계산
        # if self.initial_balance > 0:
        #     total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        # else:
        #     total_return = 0
            
        # position = "홀드"  # 기본값을 "홀드"로 설정
        # previous_shares_held = self.previous_shares_held
        # if self.trade_executed:
        #     print(self.trade_executed)
        #     position='홀드'
        # else:
        #     if self.shares_held < previous_shares_held:
        #         position = "매도"  # 주식 보유량이 줄어들었으면 "매도"
        #     elif self.shares_held > previous_shares_held:
        #         position = "매수"  # 주식 보유량이 늘어났으면 "매수"
        #     else:
        #         position='홀드'
        # self.previous_shares_held = self.shares_held
        
        # 포지션 결정
        if self.trade_executed:
            position = self.position
        else:
            if self.shares_held < self.previous_shares_held:
                position = "매도"
            elif self.shares_held > self.previous_shares_held:
                position = "매수"
            else:
                position = "홀드"
        
        self.previous_shares_held = self.shares_held
        
        # 현재 타임스탬프 정보 가져오기
        # current_timestamp = self.raw_data.index[self.current_step] if self.current_step < len(self.timestamps) else None
        current_timestamp = None
        if self.current_step < len(self.timestamps):
            current_timestamp = self.timestamps[self.current_step]
        
        # 인덱스 정보도 안전하게 처리
        timestamps_info = None
        if self.current_step < len(self.raw_data):
            timestamps_info = self.raw_data.index[self.current_step]
        
        return {
            'step': self.current_step,
            'timestamp': current_timestamp,  # 타임스탬프 추가
            # 'timestamps': self.raw_data.index[self.current_step],
            'timestamps': timestamps_info,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'position': position, 
            "previous_shares_held": self.previous_shares_held,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'cost_basis': self.cost_basis,
            'total_shares_purchased': self.total_shares_purchased,
            'total_shares_sold': self.total_shares_sold,
            'total_sales_value': self.total_sales_value,
            'total_commission': self.total_commission,
            'trade_executed': self.trade_executed,
            # "trade_cost": self.last_trade_cost if hasattr(self, "last_trade_cost") else 0,
            # 'trade_shares': last_trade_shares
        }
    
    def render(self, mode: str = 'human') -> None:
        """
        환경 시각화
        
        Args:
            mode: 시각화 모드
        """
        info = self._get_info()
        
        print(f"Step: {info['step']}")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Shares held: {info['shares_held']}")
        print(f"Current price: ${info['current_price']:.2f}")
        print(f"Portfolio value: ${info['portfolio_value']:.2f}")
        print(f"Total return: {info['total_return'] * 100:.2f}%")
        print(f"Total commission paid: ${info['total_commission']:.2f}")
        print("-" * 50)
    
    def get_episode_data(self) -> Dict[str, List]:
        """
        에피소드 데이터 반환
        
        Returns:
            에피소드 데이터 딕셔너리
        """
        return {
            'actions': self.actions_history,
            'rewards': self.rewards_history,
            'portfolio_values': self.portfolio_values_history
        }
    
    def get_final_portfolio_value(self) -> float:
        """
        최종 포트폴리오 가치 반환
        
        Returns:
            최종 포트폴리오 가치
        """
        return self._get_portfolio_value()
    
    def get_total_reward(self) -> float:
        """
        총 보상 반환
        
        Returns:
            에피소드의 총 보상
        """
        return sum(self.rewards_history)
    
def create_environment_from_results(results: Dict[str, Dict[str, Any]], symbol: str, data_type: str = 'test', **kwargs) -> TradingEnvironment:
    """
    DataProcessor 결과로부터 환경 생성하는 헬퍼 함수
    
    Args:
        results: DataProcessor.process_all_symbols()의 결과
        symbol: 주식 심볼
        data_type: 'train', 'valid', 'test' 중 하나
        **kwargs: TradingEnvironment 추가 인자
        
    Returns:
        TradingEnvironment 인스턴스
    """
    if symbol not in results:
        raise ValueError(f"Symbol {symbol} not found in results")
    
    result = results[symbol]
    
    # 정규화된 데이터 (에이전트 관측용)
    if data_type not in result:
        raise ValueError(f"Data type {data_type} not found for symbol {symbol}")
    
    normalized_data = result[data_type]
    
    # 원본 데이터 (실제 거래용) - featured_data에서 해당 구간 추출
    featured_data = result['featured_data']
    
    if data_type == 'train':
        raw_data = featured_data.iloc[:len(normalized_data)]
    elif data_type == 'valid':
        train_len = len(result['train'])
        raw_data = featured_data.iloc[train_len:train_len + len(normalized_data)]
    else:  # test
        train_len = len(result['train'])
        valid_len = len(result['valid'])
        raw_data = featured_data.iloc[train_len + valid_len:train_len + valid_len + len(normalized_data)]
    
    # 환경 생성
    env = TradingEnvironment(
        data=normalized_data,
        raw_data=raw_data,
        symbol=symbol,
        train_data=(data_type == 'train'),
        **kwargs
    )
    
    return env    

class MultiAssetTradingEnvironment:
    """다중 자산 트레이딩 환경 클래스 (개선된 버전)"""
    
    def __init__(
        self,
        results: Dict[str, Dict[str, Any]],
        symbols: List[str],
        data_type: str = 'test',
        window_size: int = WINDOW_SIZE,
        initial_balance: float = INITIAL_BALANCE,
        max_trading_units: int = MAX_TRADING_UNITS,
        transaction_fee_percent: float = TRANSACTION_FEE_PERCENT
    ):
        """
        MultiAssetTradingEnvironment 클래스 초기화
        
        Args:
            results: DataProcessor.process_all_symbols()의 결과
            symbols: 거래할 주식 심볼 리스트
            data_type: 'train', 'valid', 'test' 중 하나
            기타: TradingEnvironment와 동일한 인자들
        """
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.data_type = data_type
        self.initial_balance = initial_balance
        
        # 개별 환경 생성
        self.envs = {}
        for symbol in symbols:
            self.envs[symbol] = create_environment_from_results(
                results=results,
                symbol=symbol,
                data_type=data_type,
                window_size=window_size,
                initial_balance=initial_balance / self.n_assets,  # 자산별 균등 배분
                max_trading_units=max_trading_units,
                transaction_fee_percent=transaction_fee_percent
            )
        
        # 행동 공간: 각 자산에 대한 연속 행동
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        
        # 관측 공간
        self.observation_space = spaces.Dict({
            symbol: env.observation_space for symbol, env in self.envs.items()
        })
        
        LOGGER.info(f"다중 자산 환경 초기화 완료: {self.n_assets}개 자산, {data_type} 데이터")

    
    def reset(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        환경 초기화
        
        Returns:
            초기 관측값
        """
        observations = {}
        for symbol, env in self.envs.items():
            observations[symbol] = env.reset()
        
        return observations
    
    def step(self, actions: Dict[str, float]) -> Tuple[Dict[str, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """
        환경에서 한 스텝 진행
        
        Args:
            actions: 심볼을 키로 하고 행동을 값으로 하는 딕셔너리
            
        Returns:
            (관측값, 보상, 종료 여부, 추가 정보) 튜플
        """
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # 각 자산에 대한 행동 실행
        for symbol, env in self.envs.items():
            action = actions.get(symbol, 0.0)  # 행동이 없는 경우 홀드
            obs, rew, done, info = env.step(action)
            
            observations[symbol] = obs
            rewards[symbol] = rew
            dones[symbol] = done
            infos[symbol] = info
        
        # 전체 보상은 각 자산의 보상 평균
        total_reward = sum(rewards.values()) / self.n_assets
        
        # 모든 자산의 에피소드가 종료되면 전체 에피소드 종료
        done = all(dones.values())
        
        # 전체 포트폴리오 가치 계산
        total_portfolio_value = sum(info['portfolio_value'] for info in infos.values())
        trade_executed_any = any(info.get('trade_executed', False) for info in infos.values())
        
        # 추가 정보에 전체 포트폴리오 가치 포함
        infos['total'] = {
            'portfolio_value': total_portfolio_value,
            'total_return': (total_portfolio_value - self.initial_balance) / self.initial_balance,
            'trade_executed': trade_executed_any
        }

        return observations, total_reward, done, infos
    
    def render(self, mode: str = 'human') -> None:
        """
        환경 시각화
        
        Args:
            mode: 시각화 모드
        """
        total_portfolio_value = 0
        
        LOGGER.info("=" * 50)
        LOGGER.info("다중 자산 트레이딩 환경 상태")
        LOGGER.info("=" * 50)
        
        for symbol, env in self.envs.items():
            info = env._get_info()
            total_portfolio_value += info['portfolio_value']
            
            LOGGER.info(f"자산: {symbol}")
            LOGGER.info(f"  가격: ${info['current_price']:.2f}")
            LOGGER.info(f"  보유량: {info['shares_held']:.3f}")
            LOGGER.info(f"  포트폴리오 가치: ${info['portfolio_value']:.2f}")
            LOGGER.info(f"  수익률: {info['total_return'] * 100:.2f}%")
            LOGGER.info("-" * 50)
        
        total_return = (total_portfolio_value - self.initial_balance) / self.initial_balance
        LOGGER.info(f"총 포트폴리오 가치: ${total_portfolio_value:.2f}")
        LOGGER.info(f"총 수익률: {total_return * 100:.2f}%")
        LOGGER.info("=" * 50)
    
    def get_final_portfolio_value(self) -> float:
        """
        최종 포트폴리오 가치 반환
        
        Returns:
            최종 포트폴리오 가치
        """
        return sum(env.get_final_portfolio_value() for env in self.envs.values())
    
    def get_total_reward(self) -> float:
        """
        총 보상 반환
        
        Returns:
            에피소드의 총 보상
        """
        return sum(env.get_total_reward() for env in self.envs.values()) / self.n_assets


if __name__ == "__main__":
    # 모듈 테스트 코드
    import matplotlib.pyplot as plt
    from src.data_collection.data_collector import DataCollector
    from src.preprocessing.data_processor import DataProcessor
    
    # 데이터 수집 및 전처리
    collector = DataCollector(symbols=['AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','TSLA'])
    data = collector.load_all_data()
    
    try:
        # 데이터 수집 및 전처리
        collector = DataCollector(symbols=['AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','TSLA'])
        data = collector.load_all_data()
        
        if data:
            processor = DataProcessor()
            results = processor.process_all_symbols(data)
            if 'AAPL' in results:
                print("=" * 60)
                print("TradingEnvironment 테스트 시작")
                print("=" * 60)
                
                # 새로운 방식으로 환경 생성
                env = create_environment_from_results(
                    results=results,
                    symbol='AAPL',
                    data_type='train',  # 'train', 'valid', 'test' 중 선택
                    initial_balance=10000.0
                )
                
                print(f"환경 생성 완료: {env.symbol}")
                print(f"데이터 길이: {env.data_length}")
                print(f"특성 차원: {env.feature_dim}")
                print(f"초기 자본금: ${env.initial_balance}")
                
                # 간단한 테스트
                obs = env.reset()
                print(f"초기 관측값 형태:")
                print(f"  - market_data: {obs['market_data'].shape}")
                print(f"  - portfolio_state: {obs['portfolio_state'].shape}")
                print(f"  - portfolio_state 값: {obs['portfolio_state']}")
                
                print("\n" + "=" * 40)
                print("5스텝 테스트 실행")
                print("=" * 40)
                
                # 몇 스텝 실행
                for i in range(5):
                    action = np.random.uniform(-1.0, 1.0)
                    obs, reward, done, info = env.step(action)
                    
                    print(f"\nStep {i+1}:")
                    print(f"  행동: {action:.3f}")
                    print(f"  보상: {reward:.6f}")
                    print(f"  포트폴리오: ${info['portfolio_value']:.2f}")
                    print(f"  현재 가격: ${info['current_price']:.2f}")
                    print(f"  보유 주식: {info['shares_held']:.4f}")
                    print(f"  현금: ${info['balance']:.2f}")
                    print(f"  포지션: {info['position']}")
                    
                    if done:
                        print("  에피소드 종료!")
                        break
                
                print("\n" + "=" * 40)
                print("테스트 완료!")
                print("=" * 40)
                
                # 최종 통계
                final_value = env.get_final_portfolio_value()
                total_return = (final_value - env.initial_balance) / env.initial_balance * 100
                print(f"최종 포트폴리오 가치: ${final_value:.2f}")
                print(f"총 수익률: {total_return:.2f}%")
                print(f"총 거래 수수료: ${env.total_commission:.2f}")
                
            else:
                print("❌ AAPL 데이터를 찾을 수 없습니다.")
        else:
            print("❌ 데이터를 로드할 수 없습니다.")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    # if data:
    #     processor = DataProcessor()
    #     results = processor.process_all_symbols(data)
        
    #     if "AAPL" in results:
    #         # 새로운 방식으로 환경 생성
    #         env = create_environment_from_results(
    #             results=results,
    #             symbol="AAPL",
    #             data_type='train',  # 'train', 'valid', 'test' 중 선택
    #             initial_balance=10000.0
    #         )
            
    #         print(f"환경 생성 완료: {env.symbol}")
    #         print(f"데이터 길이: {env.data_length}")
    #         print(f"특성 차원: {env.feature_dim}")
            
    #         # 간단한 테스트
    #         obs = env.reset()
    #         print(f"초기 관측값 형태: market_data {obs['market_data'].shape}, portfolio_state {obs['portfolio_state'].shape}")
            
    #         # 몇 스텝 실행
    #         for i in range(5):
    #             action = np.random.uniform(-1.0, 1.0)
    #             obs, reward, done, info = env.step(action)
    #             print(f"Step {i+1}: 행동={action:.3f}, 보상={reward:.6f}, 포트폴리오=${info['portfolio_value']:.2f}")
                
    #             if done:
    #                 break
            
    #         print("테스트 완료!")
    
    
    
    
    # if not data:
    #     LOGGER.info("저장된 데이터가 없어 데이터를 수집합니다.")
    #     data = collector.collect_and_save()
    
    # if data:
    #     # 데이터 전처리
    #     processor = DataProcessor()
    #     results = processor.process_all_symbols(data)


    #     LOGGER.info('='*100)
      
    #     # 환경 생성 및 테스트
    #     # 환경 테스트 코드 수정
    #     if "AAPL" in results:
    #         # 정규화된 데이터와 원본 데이터 모두 사용
    #         # LOGGER.info(results["AAPL"])
    #         normalized_data = results["AAPL"]["normalized_data"]
    #         original_data = results["AAPL"]["featured_data"]  # 원본 데이터
            
    #         # 환경 생성 시 원본 데이터도 전달
    #         env = TradingEnvironment(
    #             data=normalized_data,
    #             raw_data=original_data,  # 원본 데이터 전달
    #             symbol="AAPL"
    #         )
                
    #         # 환경 테스트
    #         obs = env.reset()
    #         done = False
    #         total_reward = 0
            
    #         # 랜덤 행동으로 테스트
    #         while not done:
    #             action = np.random.uniform(-1.0, 1.0)
    #             obs, reward, done, info = env.step(action)
    #             total_reward += reward
                
    #             if env.current_step % 100 == 0:
    #                 env.render()
            
            # # 최종 결과 출력
            # print("\n최종 결과:")
            # print(f"총 보상: {total_reward:.2f}")
            # print(f"최종 포트폴리오 가치: ${env.get_final_portfolio_value():.2f}")
            # print(f"총 수익률: {(env.get_final_portfolio_value() - env.initial_balance) / env.initial_balance * 100:.2f}%")
            
            # # 포트폴리오 가치 변화 시각화
            # episode_data = env.get_episode_data()
            # plt.figure(figsize=(12, 6))
            # plt.plot(episode_data['portfolio_values'])
            # plt.title('포트폴리오 가치 변화')
            # plt.xlabel('스텝')
            # plt.ylabel('포트폴리오 가치 ($)')
            # plt.grid(True, alpha=0.3)
            # plt.savefig('./results/portfolio_value_test.png')
            # plt.close() 