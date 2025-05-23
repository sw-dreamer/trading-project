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
from src.preprocessing.feature_engineer import FeatureEngineer
from src.utils.logger import Logger
from src.config.config import Config
from src.trading.api_connector import APIConnector
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager


class LiveTrader:
    """
    실시간 트레이딩 모듈: 학습된 SAC 에이전트를 사용하여 실시간 트레이딩을 수행합니다.
    """
    
    def __init__(
        self,
        agent: SACAgent, 
        api_connector: APIConnector,
        config: Config,
        logger: Optional[Logger] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        LiveTrader 클래스 초기화
        
        Args:
            agent: 학습된 SAC 에이전트
            api_connector: API 커넥터 인스턴스
            config: 설정 객체
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
            feature_engineer: 특성 엔지니어링 인스턴스 (옵션)
            risk_manager: 리스크 관리자 인스턴스 (옵션)
        """
        self.agent = agent
        self.api = api_connector
        self.config = config
        self.logger = logger
        self.feature_engineer = feature_engineer
        
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
        self.trading_interval = config.trading_interval  # 초 단위
        self.window_size = config.window_size
        self.lookback_data = {symbol: None for symbol in self.trading_symbols}
        
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
        
        # 성공적인 초기화 로깅
        if self.logger:
            self.logger.info("LiveTrader 초기화 완료")
        
    def start(self) -> bool:
        """
        실시간 트레이딩 시작
        
        Returns:
            시작 성공 여부
        """
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
            
            # 초기 데이터 로드 (각 심볼별 과거 데이터)
            self._load_initial_data()
            
            # 트레이딩 통계 초기화
            self.trading_stats["start_time"] = datetime.now()
            self.trading_stats["initial_balance"] = float(self.account_info.get("balance", 0))
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
        """
        실시간 트레이딩 중지
        
        Returns:
            중지 성공 여부
        """
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
            self.trading_stats["current_balance"] = float(self.account_info.get("balance", 0))
            self.trading_stats["pnl"] = self.trading_stats["current_balance"] - self.trading_stats["initial_balance"]
            
            # 결과 로깅
            if self.logger:
                self.logger.info("실시간 트레이딩을 중지합니다.")
                self.logger.info(f"트레이딩 기간: {datetime.now() - self.trading_stats['start_time']}")
                self.logger.info(f"총 거래 횟수: {len(self.trading_stats['trades'])}")
                self.logger.info(f"성공한 거래: {self.trading_stats['successful_trades']}")
                self.logger.info(f"실패한 거래: {self.trading_stats['failed_trades']}")
                self.logger.info(f"초기 자본금: ${self.trading_stats['initial_balance']:.2f}")
                self.logger.info(f"최종 자본금: ${self.trading_stats['current_balance']:.2f}")
                self.logger.info(f"수익률: {((self.trading_stats['pnl'] / self.trading_stats['initial_balance']) * 100):.2f}%")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"실시간 트레이딩 중지 중 오류 발생: {e}")
            return False
    
    def pause(self) -> bool:
        """
        실시간 트레이딩 일시 중지
        
        Returns:
            일시 중지 성공 여부
        """
        if not self.running:
            if self.logger:
                self.logger.warning("실시간 트레이딩이 실행 중이 아닙니다.")
            return False
            
        try:
            # 일시 중지 플래그 설정
            self.running = False
            
            if self.logger:
                self.logger.info("실시간 트레이딩이 일시 중지되었습니다.")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"실시간 트레이딩 일시 중지 중 오류 발생: {e}")
            return False
    
    def resume(self) -> bool:
        """
        일시 중지된 실시간 트레이딩 재개
        
        Returns:
            재개 성공 여부
        """
        if self.running:
            if self.logger:
                self.logger.warning("이미 실시간 트레이딩이 실행 중입니다.")
            return False
            
        try:
            # 재개 플래그 설정
            self.running = True
            
            # 스레드 재시작
            if not self.data_thread or not self.data_thread.is_alive():
                self.data_thread = threading.Thread(target=self._data_stream_worker)
                self.data_thread.daemon = True
                self.data_thread.start()
                
            if not self.trading_thread or not self.trading_thread.is_alive():
                self.trading_thread = threading.Thread(target=self._trading_worker)
                self.trading_thread.daemon = True
                self.trading_thread.start()
            
            if self.logger:
                self.logger.info("실시간 트레이딩이 재개되었습니다.")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"실시간 트레이딩 재개 중 오류 발생: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """
        현재 트레이딩 상태 조회
        
        Returns:
            현재 상태 정보가 담긴 딕셔너리
        """
        # 계정 정보 업데이트
        self.account_info = self.api.get_account_info()
        
        # 포지션 업데이트
        positions = self.position_manager.update_all_positions()
        
        # 주문 업데이트
        open_orders = self.order_manager.update_open_orders()
        
        # 트레이딩 통계 업데이트
        self.trading_stats["current_balance"] = float(self.account_info.get("balance", 0))
        self.trading_stats["pnl"] = self.trading_stats["current_balance"] - self.trading_stats["initial_balance"]
        
        # 현재 상태 반환
        return {
            "running": self.running,
            "account": self.account_info,
            "positions": positions,
            "open_orders": open_orders,
            "trading_stats": self.trading_stats,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def execute_trade(self, symbol: str, action: float) -> Dict[str, Any]:
        """
        트레이딩 행동 실행
        
        Args:
            symbol: 거래할 심볼/티커
            action: 에이전트의 행동 값 (-1 ~ 1 사이의 값, 음수는 매도, 양수는 매수)
            
        Returns:
            거래 실행 결과가 담긴 딕셔너리
        """
        try:
            # 계정 및 포지션 정보 업데이트
            self.account_info = self.api.get_account_info()
            current_position = self.position_manager.get_position(symbol)
            
            # 행동 값에 따른 거래 방향 및 크기 결정
            side = "buy" if action > 0 else "sell"
            position_size = abs(action)  # 행동의 절대값이 포지션 크기 비율을 결정
            
            # 리스크 관리자를 통한 주문 수량 계산
            available_balance = float(self.account_info.get("available_balance", 0))
            current_price = self._get_current_price(symbol)
            
            if current_price <= 0:
                if self.logger:
                    self.logger.error(f"{symbol}의 현재 가격을 얻을 수 없습니다.")
                return {"success": False, "error": "현재 가격을 얻을 수 없습니다."}
            
            # 리스크 관리자를 통한 주문 수량 계산
            quantity = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side=side,
                available_balance=available_balance,
                current_price=current_price,
                position_ratio=position_size,
                current_position=current_position.get("quantity", 0)
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
            
            # 거래 실행 결과 로깅 및 통계 업데이트
            if order_result.get("success", True):
                # 성공한 거래
                self.trading_stats["successful_trades"] += 1
                self.trading_stats["trades"].append({
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_id": order_result.get("order_id", ""),
                    "status": "success"
                })
                
                if self.logger:
                    self.logger.info(f"{symbol} {side.upper()} 주문 성공: {quantity} 수량, 가격: {current_price}")
                    
                # 포지션 및 주문 정보 업데이트
                self.position_manager.update_position(symbol)
                self.order_manager.update_order(order_result.get("order_id", ""))
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "order_id": order_result.get("order_id", "")
                }
            else:
                # 실패한 거래
                self.trading_stats["failed_trades"] += 1
                self.trading_stats["trades"].append({
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "failed",
                    "error": order_result.get("error", "Unknown error")
                })
                
                if self.logger:
                    self.logger.error(f"{symbol} {side.upper()} 주문 실패: {order_result.get('error', 'Unknown error')}")
                    
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
    
    def save_trading_stats(self, filepath: str) -> bool:
        """
        트레이딩 통계 저장
        
        Args:
            filepath: 저장할 파일 경로
            
        Returns:
            저장 성공 여부
        """
        try:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 통계 업데이트
            self.account_info = self.api.get_account_info()
            self.trading_stats["current_balance"] = float(self.account_info.get("balance", 0))
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
                json.dump(save_data, f, indent=4)
                
            if self.logger:
                self.logger.info(f"트레이딩 통계를 {filepath}에 저장했습니다.")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"트레이딩 통계 저장 중 오류 발생: {e}")
            return False
            
    def _data_stream_worker(self) -> None:
        """
        실시간 데이터 스트림 처리 워커 스레드
        """
        while self.running:
            try:
                # 각 심볼별 최신 데이터 가져오기
                for symbol in self.trading_symbols:
                    market_data = self.api.get_market_data(
                        symbol=symbol,
                        interval='1m',  # 1분 간격의 데이터
                        limit=1  # 최신 데이터 1개만
                    )
                    
                    if market_data.empty:
                        if self.logger:
                            self.logger.warning(f"{symbol} 데이터를 가져올 수 없습니다.")
                        continue
                    
                    # lookback_data 업데이트
                    if self.lookback_data[symbol] is None:
                        # 처음이면 window_size만큼의 이전 데이터 로드
                        historic_data = self.api.get_market_data(
                            symbol=symbol,
                            interval='1m',
                            limit=self.window_size
                        )
                        self.lookback_data[symbol] = historic_data
                    else:
                        # 기존 데이터에 새 데이터 추가
                        self.lookback_data[symbol] = pd.concat([self.lookback_data[symbol], market_data]).drop_duplicates()
                        
                        # window_size 유지
                        if len(self.lookback_data[symbol]) > self.window_size:
                            self.lookback_data[symbol] = self.lookback_data[symbol].iloc[-self.window_size:]
                    
                    # 특성 엔지니어링
                    if self.feature_engineer:
                        features = self.feature_engineer.transform(self.lookback_data[symbol])
                    else:
                        # 간단한 기본 특성 추출
                        features = self._extract_basic_features(self.lookback_data[symbol])
                    
                    # 상태 업데이트
                    self.state_dict[symbol] = features
                    
                    # 데이터 큐에 추가
                    self.data_queue.put({
                        "symbol": symbol,
                        "data": market_data,
                        "features": features,
                        "timestamp": datetime.now()
                    })
                
                # 다음 데이터 수집까지 대기
                time.sleep(60)  # 1분 간격으로 데이터 수집
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"데이터 스트림 처리 중 오류 발생: {e}")
                time.sleep(5)  # 오류 발생 시 5초 대기 후 재시도
    
    def _trading_worker(self) -> None:
        """
        트레이딩 로직 실행 워커 스레드
        """
        last_trade_time = {symbol: datetime.min for symbol in self.trading_symbols}
        
        while self.running:
            try:
                # 데이터 큐에서 메시지 가져오기
                try:
                    message = self.data_queue.get(timeout=1.0)
                    symbol = message["symbol"]
                    features = message["features"]
                    timestamp = message["timestamp"]
                except queue.Empty:
                    continue
                
                # 트레이딩 간격 확인
                current_time = datetime.now()
                time_since_last_trade = current_time - last_trade_time[symbol]
                
                if time_since_last_trade.total_seconds() < self.trading_interval:
                    continue  # 트레이딩 간격이 지나지 않았으면 건너뜀
                
                # 현재 포지션 조회
                position = self.position_manager.get_position(symbol)
                position_size = position.get("quantity", 0)
                
                # 에이전트로부터 행동 선택
                state = np.array(features).reshape(1, -1)
                action = self.agent.select_action(state, evaluate=True)
                
                # 현재 행동이 포지션과 반대 방향이거나 포지션이 없는 경우에만 거래 실행
                if (action > 0 and position_size <= 0) or (action < 0 and position_size >= 0):
                    # 거래 실행
                    trade_result = self.execute_trade(symbol, action)
                    
                    # 마지막 거래 시간 업데이트
                    if trade_result["success"]:
                        last_trade_time[symbol] = current_time
                        
                # 트레이딩 로직 주기 대기
                time.sleep(1)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"트레이딩 로직 실행 중 오류 발생: {e}")
                time.sleep(5)  # 오류 발생 시 5초 대기 후 재시도
    
    def _load_initial_data(self) -> None:
        """
        초기 과거 데이터 로드
        """
        try:
            for symbol in self.trading_symbols:
                # window_size만큼의 과거 데이터 로드
                historic_data = self.api.get_market_data(
                    symbol=symbol,
                    interval='1m',
                    limit=self.window_size
                )
                
                if historic_data.empty:
                    if self.logger:
                        self.logger.warning(f"{symbol} 초기 데이터를 로드할 수 없습니다.")
                    continue
                
                # lookback_data 초기화
                self.lookback_data[symbol] = historic_data
                
                # 특성 엔지니어링
                if self.feature_engineer:
                    features = self.feature_engineer.transform(historic_data)
                else:
                    # 간단한 기본 특성 추출
                    features = self._extract_basic_features(historic_data)
                
                # 상태 초기화
                self.state_dict[symbol] = features
                
                if self.logger:
                    self.logger.info(f"{symbol} 초기 데이터 로드 완료: {len(historic_data)} 행")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"초기 데이터 로드 중 오류 발생: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        """
        현재 가격 조회
        
        Args:
            symbol: 심볼/티커
            
        Returns:
            현재 가격
        """
        try:
            # 최신 시장 데이터 조회
            market_data = self.api.get_market_data(
                symbol=symbol,
                interval='1m',
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
    
    def _extract_basic_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        기본 특성 추출
        
        Args:
            data: 시장 데이터 DataFrame
            
        Returns:
            특성 배열
        """
        try:
            # 간단한 특성 추출
            # 종가, 거래량, 고가/저가 비율 등
            features = []
            
            # 정규화된 종가
            closes = data["close"].values
            normalized_closes = closes / closes[0] if len(closes) > 0 else [1.0]
            features.extend(normalized_closes)
            
            # 정규화된 거래량
            if "volume" in data.columns:
                volumes = data["volume"].values
                max_volume = np.max(volumes) if len(volumes) > 0 else 1.0
                normalized_volumes = volumes / max_volume if max_volume > 0 else volumes
                features.extend(normalized_volumes)
            
            # 고가/저가 비율
            if "high" in data.columns and "low" in data.columns:
                highs = data["high"].values
                lows = data["low"].values
                if len(highs) > 0 and len(lows) > 0:
                    hl_ratios = (highs - lows) / lows
                    features.extend(hl_ratios)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"기본 특성 추출 중 오류 발생: {e}")
            # 오류 발생 시 0으로 채워진 배열 반환
            return np.zeros(self.window_size * 3, dtype=np.float32) 