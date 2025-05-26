import numpy as np
import pandas as pd
import time
import threading
import queue
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable

from src.models.sac_agent import SACAgent
from src.preprocessing.data_processor import DataProcessor
from src.utils.logger import Logger
from src.config.config import Config
from src.trading.api_connector import APIConnector
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager

class LiveTrader:
    """
    실시간 트레이딩 모듈: 학습된 SAC 에이전트를 사용하여 실시간 트레이딩을 수행합니다.
    백테스팅과 동일한 데이터 전처리 파이프라인을 사용합니다.
    """
    
    def __init__(
        self,
        agent: SACAgent, 
        api_connector: APIConnector,
        config: Config,
        logger: Optional[Logger] = None,
        data_processor: Optional[DataProcessor] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        LiveTrader 클래스 초기화
        
        Args:
            agent: 학습된 SAC 에이전트
            api_connector: API 커넥터 인스턴스
            config: 설정 객체
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
            data_processor: 데이터 전처리 인스턴스 (백테스팅과 동일한 것 사용)
            risk_manager: 리스크 관리자 인스턴스 (옵션)
        """
        self.agent = agent
        self.api = api_connector
        self.config = config
        self.logger = logger
        
        # 백테스팅과 동일한 데이터 전처리 파이프라인 사용
        self.data_processor = data_processor if data_processor else DataProcessor(
            window_size=config.window_size
        )
        
        # API 연결 확인
        if not self.api.connected:
            self.api.connect()
            
        # 계정 정보 조회
        self.account_info = self.api.get_account_info()
        
        # 주문 관리자 및 포지션 관리자 초기화
        self.order_manager = OrderManager(self.api, logger=self.logger)
        self.position_manager = PositionManager(self.api, logger=self.logger)
        
        # 리스크 관리자 설정
        if risk_manager is None:
            self.risk_manager = RiskManager(
                max_position_size=config.max_position_size,
                max_drawdown=config.max_drawdown,
                max_trade_amount=config.max_trade_amount,
                max_daily_loss=config.max_daily_loss,
                logger=self.logger
            )
        else:
            self.risk_manager = risk_manager
            
        # 실시간 데이터 스트림 설정
        self.data_queue = queue.Queue()
        self.state_dict = {}
        self.running = False
        self.data_thread = None
        self.trading_thread = None
        
        # 트레이딩 설정
        self.trading_symbols = config.trading_symbols
        self.trading_interval = config.trading_interval
        self.window_size = config.window_size
        
        # 각 심볼별 원본 데이터 저장 (특성 추출용)
        self.raw_data_buffer = {symbol: pd.DataFrame() for symbol in self.trading_symbols}
        
        # 트레이딩 통계
        self.trading_stats = {
            "start_time": None,
            "trades": [],
            "successful_trades": 0,
            "failed_trades": 0,
            "initial_balance": 0,
            "current_balance": 0,
            "pnl": 0
        }
        
        if self.logger:
            self.logger.info("LiveTrader 초기화 완료")
    
    def start(self) -> bool:
        """실시간 트레이딩 시작"""
        if self.running:
            if self.logger:
                self.logger.warning("이미 실시간 트레이딩이 실행 중입니다.")
            return False
            
        try:
            # API 연결 확인
            if not self.api.connected:
                self.api.connect()
                
            if not self.api.connected:
                if self.logger:
                    self.logger.error("API 서버에 연결할 수 없어 트레이딩을 시작할 수 없습니다.")
                return False
                
            # 초기 계정 정보 업데이트
            self.account_info = self.api.get_account_info()
            
            # 포지션 및 주문 정보 로드
            self.position_manager.update_all_positions()
            self.order_manager.update_open_orders()
            
            # 초기 데이터 로드
            self._load_initial_data()
            
            # 트레이딩 통계 초기화
            self.trading_stats["start_time"] = datetime.now()
            self.trading_stats["initial_balance"] = float(self.account_info.get("cash", 0))
            self.trading_stats["current_balance"] = self.trading_stats["initial_balance"]
            
            # 스레드 시작
            self.running = True
            self.data_thread = threading.Thread(target=self._data_stream_worker)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            self.trading_thread = threading.Thread(target=self._trading_worker)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            if self.logger:
                self.logger.info("실시간 트레이딩을 시작합니다.")
                self.logger.info(f"트레이딩 심볼: {', '.join(self.trading_symbols)}")
                self.logger.info(f"트레이딩 간격: {self.trading_interval}초")
                
            return True
            
        except Exception as e:
            self.running = False
            if self.logger:
                self.logger.error(f"실시간 트레이딩 시작 중 오류 발생: {e}")
            return False
    
    def stop(self) -> bool:
        """실시간 트레이딩 중지"""
        if not self.running:
            if self.logger:
                self.logger.warning("실시간 트레이딩이 실행 중이 아닙니다.")
            return False
            
        try:
            # 종료 플래그 설정
            self.running = False
            
            # 스레드 종료 대기
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=5.0)
                
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5.0)
            
            # 최종 계정 정보 업데이트
            self.account_info = self.api.get_account_info()
            
            # 트레이딩 통계 업데이트
            self.trading_stats["current_balance"] = float(self.account_info.get("cash", 0))
            self.trading_stats["pnl"] = self.trading_stats["current_balance"] - self.trading_stats["initial_balance"]
            
            if self.logger:
                self.logger.info("실시간 트레이딩을 중지합니다.")
                self.logger.info(f"수익률: {((self.trading_stats['pnl'] / self.trading_stats['initial_balance']) * 100):.2f}%")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"실시간 트레이딩 중지 중 오류 발생: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """현재 트레이딩 상태 조회"""
        # 계정 정보 업데이트
        self.account_info = self.api.get_account_info()
        
        # 포지션 업데이트
        positions = self.position_manager.update_all_positions()
        
        # 주문 업데이트
        open_orders = self.order_manager.update_open_orders()
        
        # 트레이딩 통계 업데이트
        self.trading_stats["current_balance"] = float(self.account_info.get("cash", 0))
        self.trading_stats["pnl"] = self.trading_stats["current_balance"] - self.trading_stats["initial_balance"]
        
        return {
            "running": self.running,
            "account": self.account_info,
            "positions": positions,
            "open_orders": open_orders,
            "trading_stats": self.trading_stats,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def execute_trade(self, symbol: str, action: float) -> Dict[str, Any]:
        """트레이딩 행동 실행"""
        try:
            # 계정 및 포지션 정보 업데이트
            self.account_info = self.api.get_account_info()
            current_position = self.position_manager.get_position(symbol)
            
            # 행동 값에 따른 거래 방향 및 크기 결정
            side = "buy" if action > 0 else "sell"
            position_size = abs(action)
            
            # 사용 가능한 현금 확인
            available_cash = float(self.account_info.get("cash", 0))
            current_price = self._get_current_price(symbol)
            
            if current_price <= 0:
                if self.logger:
                    self.logger.error(f"{symbol}의 현재 가격을 얻을 수 없습니다.")
                return {"success": False, "error": "현재 가격을 얻을 수 없습니다."}
            
            # 리스크 관리자를 통한 주문 수량 계산
            quantity = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side=side,
                available_balance=available_cash,
                current_price=current_price,
                position_ratio=position_size,
                current_position=current_position.get("qty", 0)
            )
            
            # 거래 수량이 0이면 거래 실행하지 않음
            if quantity <= 0:
                if self.logger:
                    self.logger.info(f"{symbol} {side} 거래 건너뜀: 수량이 0 이하입니다.")
                return {"success": True, "action": "no_trade", "reason": "수량이 0 이하입니다."}
            
            # 시장가 주문 실행
            order_result = self.api.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            
            # 거래 실행 결과 처리
            if order_result.get("success", True):
                # 성공한 거래
                self.trading_stats["successful_trades"] += 1
                self.trading_stats["trades"].append({
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_id": order_result.get("id", ""),
                    "status": "success"
                })
                
                if self.logger:
                    self.logger.info("=" * 60)
                    self.logger.info(f"📌 트레이드 실행 결과")
                    self.logger.info(f"📈 종목: {symbol}")
                    self.logger.info(f"🧭 방향: {'🟢 매수(BUY)' if side == 'buy' else '🔴 매도(SELL)'}")
                    self.logger.info(f"🔢 수량: {quantity:.4f} 주")
                    self.logger.info(f"💵 체결가: ${current_price:.2f}")
                    self.logger.info(f"🕒 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    self.logger.info("=" * 60)
                    
                # 포지션 정보 업데이트
                self.position_manager.update_position(symbol)
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "order_id": order_result.get("id", "")
                }
            else:
                # 실패한 거래
                self.trading_stats["failed_trades"] += 1
                if self.logger:
                    self.logger.error("=" * 60)
                    self.logger.error(f"❌ 트레이드 실패")
                    self.logger.error(f"📈 종목: {symbol}")
                    self.logger.error(f"🧭 방향: {side.upper()}")
                    self.logger.error(f"🔢 수량: {quantity:.4f} 주")
                    self.logger.error(f"💵 가격: ${current_price:.2f}")
                    self.logger.error(f"🚨 오류: {order_result.get('error', 'Unknown error')}")
                    self.logger.error("=" * 60)
                        
                return {
                    "success": False,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "error": order_result.get("error", "Unknown error")
                }
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 거래 실행 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    # def execute_trade(self, symbol: str, action: float) -> Dict[str, Any]:
    #     """트레이딩 행동 실행"""
    # try:
    #     self.account_info = self.api.get_account_info()
    #     current_position = self.position_manager.get_position(symbol)

    #     side = "buy" if action > 0 else "sell"
    #     position_size = abs(action)

    #     available_cash = float(self.account_info.get("cash", 0))
    #     current_price = self._get_current_price(symbol)

    #     if current_price <= 0:
    #         if self.logger:
    #             self.logger.error(f"{symbol}의 현재 가격을 얻을 수 없습니다.")
    #         return {"success": False, "error": "현재 가격을 얻을 수 없습니다."}

    #     quantity = self.risk_manager.calculate_position_size(
    #         symbol=symbol,
    #         side=side,
    #         available_balance=available_cash,
    #         current_price=current_price,
    #         position_ratio=position_size,
    #         current_position=current_position.get("qty", 0)
    #     )

    #     if quantity <= 0:
    #         if self.logger:
    #             self.logger.info(f"{symbol} {side} 거래 건너뜀: 수량이 0 이하입니다.")
    #         return {"success": True, "action": "no_trade", "reason": "수량이 0 이하입니다."}

    #     order_result = self.api.place_market_order(
    #         symbol=symbol,
    #         side=side,
    #         quantity=quantity
    #     )

    #     if order_result.get("success", True):
    #         self.trading_stats["successful_trades"] += 1
    #         self.trading_stats["trades"].append({
    #             "symbol": symbol,
    #             "side": side,
    #             "quantity": quantity,
    #             "price": current_price,
    #             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #             "order_id": order_result.get("id", ""),
    #             "status": "success"
    #         })

    #         if self.logger:
    #             self.logger.info("=" * 60)
    #             self.logger.info(f"📌 트레이드 실행 결과")
    #             self.logger.info(f"📈 종목: {symbol}")
    #             self.logger.info(f"🧭 방향: {'🟢 매수(BUY)' if side == 'buy' else '🔴 매도(SELL)'}")
    #             self.logger.info(f"🔢 수량: {quantity:.4f} 주")
    #             self.logger.info(f"💵 체결가: ${current_price:.2f}")
    #             self.logger.info(f"🕒 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #             self.logger.info("=" * 60)

    #         self.position_manager.update_position(symbol)

    #         return {
    #             "success": True,
    #             "symbol": symbol,
    #             "side": side,
    #             "quantity": quantity,
    #             "price": current_price,
    #             "order_id": order_result.get("id", "")
    #         }

    #     else:
    #         self.trading_stats["failed_trades"] += 1
    #         if self.logger:
    #             self.logger.error("=" * 60)
    #             self.logger.error(f"❌ 트레이드 실패")
    #             self.logger.error(f"📈 종목: {symbol}")
    #             self.logger.error(f"🧭 방향: {side.upper()}")
    #             self.logger.error(f"🔢 수량: {quantity:.4f} 주")
    #             self.logger.error(f"💵 가격: ${current_price:.2f}")
    #             self.logger.error(f"🚨 오류: {order_result.get('error', 'Unknown error')}")
    #             self.logger.error("=" * 60)

    #         return {
    #             "success": False,
    #             "symbol": symbol,
    #             "side": side,
    #             "quantity": quantity,
    #             "price": current_price,
    #             "error": order_result.get("error", "Unknown error")
    #         }

    # except Exception as e:
    #     if self.logger:
    #         self.logger.error(f"{symbol} 거래 실행 중 오류 발생: {e}")
    #     return {"success": False, "error": str(e)}

    
    
    def save_trading_stats(self, filepath: str) -> bool:
        """트레이딩 통계 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 통계 업데이트
            self.account_info = self.api.get_account_info()
            self.trading_stats["current_balance"] = float(self.account_info.get("cash", 0))
            self.trading_stats["pnl"] = self.trading_stats["current_balance"] - self.trading_stats["initial_balance"]
            
            # 저장할 데이터 준비
            save_data = {
                "trading_stats": self.trading_stats,
                "account_info": self.account_info,
                "positions": self.position_manager.get_all_positions(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # JSON 파일로 저장
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=4, default=str)
                
            if self.logger:
                self.logger.info(f"트레이딩 통계를 {filepath}에 저장했습니다.")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"트레이딩 통계 저장 중 오류 발생: {e}")
            return False
    
    def _data_stream_worker(self) -> None:
        """실시간 데이터 스트림 처리 워커 스레드"""
        while self.running:
            try:
                # 각 심볼별 최신 데이터 가져오기
                for symbol in self.trading_symbols:
                    # TradingView에서 최신 데이터 가져오기
                    market_data = self.api.get_market_data(
                        symbol=symbol,
                        limit=1  # 최신 데이터 1개만
                    )
                    
                    if market_data.empty:
                        if self.logger:
                            self.logger.warning(f"{symbol} 데이터를 가져올 수 없습니다.")
                        continue
                    
                    # 원본 데이터 버퍼 업데이트
                    if self.raw_data_buffer[symbol].empty:
                        # 처음이면 window_size만큼의 이전 데이터 로드
                        historic_data = self.api.get_market_data(
                            symbol=symbol,
                            limit=self.window_size + 50  # 특성 추출을 위해 여유분 확보
                        )
                        self.raw_data_buffer[symbol] = historic_data
                    else:
                        # 기존 데이터에 새 데이터 추가
                        self.raw_data_buffer[symbol] = pd.concat([
                            self.raw_data_buffer[symbol], 
                            market_data
                        ]).drop_duplicates().sort_index()
                        
                        # 메모리 절약을 위해 오래된 데이터 제거 (window_size + 200개만 유지)
                        if len(self.raw_data_buffer[symbol]) > self.window_size + 200:
                            self.raw_data_buffer[symbol] = self.raw_data_buffer[symbol].iloc[-(self.window_size + 200):]
                    
                    # 백테스팅과 동일한 전처리 파이프라인 적용
                    try:
                        # 1. 데이터 전처리 (결측치 처리, 이상치 제거)
                        processed_data = self.data_processor.preprocess_data(self.raw_data_buffer[symbol])
                        
                        # 2. 특성 추출 (기술적 지표 계산)
                        featured_data = self.data_processor.extract_features(processed_data)
                        
                        # 3. 특성 정규화 (학습 시와 동일한 스케일러 사용)
                        normalized_data = self.data_processor.normalize_features(
                            featured_data, 
                            symbol, 
                            is_training=False  # 기존 스케일러 사용
                        )
                        
                        # 4. 백테스팅과 동일한 형태의 상태 생성
                        state = self._create_trading_state(normalized_data, symbol)
                        
                        # 상태 업데이트
                        self.state_dict[symbol] = state
                        
                        # 데이터 큐에 추가
                        self.data_queue.put({
                            "symbol": symbol,
                            "state": state,
                            "timestamp": datetime.now()
                        })
                        
                        if self.logger:
                            self.logger.debug(f"{symbol} 데이터 처리 완료")
                            
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"{symbol} 데이터 전처리 중 오류 발생: {e}")
                        continue
                
                # 다음 데이터 수집까지 대기
                time.sleep(60)  # 1분 간격으로 데이터 수집
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"데이터 스트림 처리 중 오류 발생: {e}")
                time.sleep(5)  # 오류 발생 시 5초 대기 후 재시도
    
    def _trading_worker(self) -> None:
        """트레이딩 로직 실행 워커 스레드"""
        last_trade_time = {symbol: datetime.min for symbol in self.trading_symbols}
        
        while self.running:
            try:
                # 데이터 큐에서 메시지 가져오기
                try:
                    message = self.data_queue.get(timeout=1.0)
                    symbol = message["symbol"]
                    state = message["state"]
                    timestamp = message["timestamp"]
                except queue.Empty:
                    continue
                
                # 트레이딩 간격 확인
                current_time = datetime.now()
                time_since_last_trade = current_time - last_trade_time[symbol]
                
                if time_since_last_trade.total_seconds() < self.trading_interval:
                    continue  # 트레이딩 간격이 지나지 않았으면 건너뜀
                
                # 리스크 관리 체크
                current_balance = float(self.account_info.get("cash", 0))
                risk_check = self.risk_manager.check_risk_limits(current_balance)
                
                if not risk_check.get("trade_allowed", True):
                    if self.logger:
                        self.logger.warning(f"리스크 한도 초과로 거래 중단: {risk_check.get('warnings', [])}")
                    continue
                
                # 현재 포지션 조회
                position = self.position_manager.get_position(symbol)
                position_size = position.get("qty", 0)
                
                # 에이전트로부터 행동 선택 (백테스팅과 동일한 형태의 상태 사용)
                action = self.agent.select_action(state, evaluate=True)
                
                if self.logger:
                    self.logger.debug(f"{symbol} 에이전트 행동: {action:.4f}, 현재 포지션: {position_size}")
                
                # 행동 임계값 확인 (너무 작은 행동은 무시)
                if abs(action) < 0.1:
                    continue
                
                # 거래 실행
                trade_result = self.execute_trade(symbol, action)
                
                # 마지막 거래 시간 업데이트
                if trade_result.get("success", False):
                    last_trade_time[symbol] = current_time
                    
                    # 리스크 관리자에 거래 기록
                    self.risk_manager.record_trade(
                        symbol=symbol,
                        side=trade_result["side"],
                        quantity=trade_result["quantity"],
                        price=trade_result["price"]
                    )
                        
                # 트레이딩 로직 주기 대기
                time.sleep(1)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"트레이딩 로직 실행 중 오류 발생: {e}")
                time.sleep(5)  # 오류 발생 시 5초 대기 후 재시도
    
    def _load_initial_data(self) -> None:
        """초기 과거 데이터 로드"""
        try:
            for symbol in self.trading_symbols:
                # window_size + 여유분만큼의 과거 데이터 로드
                historic_data = self.api.get_market_data(
                    symbol=symbol,
                    limit=self.window_size + 50  # 특성 추출을 위해 여유분 확보
                )
                
                if historic_data.empty:
                    if self.logger:
                        self.logger.warning(f"{symbol} 초기 데이터를 로드할 수 없습니다.")
                    continue
                
                # 원본 데이터 버퍼 초기화
                self.raw_data_buffer[symbol] = historic_data
                
                # 백테스팅과 동일한 전처리 파이프라인 적용
                try:
                    processed_data = self.data_processor.preprocess_data(historic_data)
                    featured_data = self.data_processor.extract_features(processed_data)
                    normalized_data = self.data_processor.normalize_features(
                        featured_data, 
                        symbol, 
                        is_training=False
                    )
                    
                    # 상태 생성
                    state = self._create_trading_state(normalized_data, symbol)
                    self.state_dict[symbol] = state
                    
                    if self.logger:
                        self.logger.info(f"{symbol} 초기 데이터 로드 및 전처리 완료: {len(historic_data)} 행")
                        
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"{symbol} 초기 데이터 전처리 중 오류: {e}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"초기 데이터 로드 중 오류 발생: {e}")
    
    def _create_trading_state(self, normalized_data: pd.DataFrame, symbol: str) -> Dict[str, np.ndarray]:
        """
        백테스팅과 동일한 형태의 상태 생성
        
        Args:
            normalized_data: 정규화된 특성 데이터
            symbol: 심볼
            
        Returns:
            TradingEnvironment과 동일한 형태의 상태
        """
        try:
            # 최근 window_size 개의 데이터만 사용
            if len(normalized_data) < self.window_size:
                # 데이터가 부족한 경우 패딩
                padding_size = self.window_size - len(normalized_data)
                padding = pd.DataFrame(
                    np.zeros((padding_size, normalized_data.shape[1])),
                    columns=normalized_data.columns
                )
                market_data = pd.concat([padding, normalized_data], ignore_index=True)
            else:
                market_data = normalized_data.iloc[-self.window_size:]
            
            # market_data를 numpy 배열로 변환
            market_data_array = market_data.values.astype(np.float32)
            
            # 포트폴리오 상태 계산
            current_position = self.position_manager.get_position(symbol)
            account_info = self.api.get_account_info()
            
            cash = float(account_info.get("cash", 0))
            portfolio_value = float(account_info.get("portfolio_value", cash))
            stock_value = abs(float(current_position.get("market_value", 0)))
            
            # 포트폴리오 가치가 0이 아닌지 확인
            if portfolio_value <= 0:
                portfolio_value = max(cash, 1.0)
            
            portfolio_state = np.array([
                cash / portfolio_value,        # 현금 비율
                stock_value / portfolio_value  # 주식 비율
            ], dtype=np.float32)
            
            # 백테스팅과 동일한 형태의 상태 반환
            return {
                'market_data': market_data_array,
                'portfolio_state': portfolio_state
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 상태 생성 중 오류: {e}")
            
            # 오류 발생 시 기본 상태 반환
            return {
                'market_data': np.zeros((self.window_size, 40), dtype=np.float32),  # 40은 대략적인 특성 수
                'portfolio_state': np.array([1.0, 0.0], dtype=np.float32)
            }
    
    def _get_current_price(self, symbol: str) -> float:
        """현재 가격 조회"""
        try:
            # 최신 시장 데이터 조회
            market_data = self.api.get_market_data(
                symbol=symbol,
                limit=1
            )
            
            if market_data.empty:
                return 0.0
            
            # 종가 반환
            return float(market_data.iloc[-1]["close"])
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 현재 가격 조회 중 오류 발생: {e}")
            return 0.0