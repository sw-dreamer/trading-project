import requests
import pandas as pd
import time
import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

from src.utils.logger import Logger


class APIConnector:
    """
    외부 거래소/브로커 API와의 연결을 관리하는 커넥터 클래스
    """
    
    def __init__(
        self, 
        api_key: str, 
        api_secret: str, 
        base_url: str,
        logger: Optional[Logger] = None
    ):
        """
        APIConnector 클래스 초기화
        
        Args:
            api_key: API 키
            api_secret: API 시크릿 키
            base_url: API 베이스 URL
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.logger = logger
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.connected = False
        
    def connect(self) -> bool:
        """
        API 서버에 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            # API 상태 체크
            response = self.session.get(f"{self.base_url}/status")
            if response.status_code == 200:
                self.connected = True
                if self.logger:
                    self.logger.info("API 서버에 성공적으로 연결되었습니다.")
                return True
            else:
                if self.logger:
                    self.logger.error(f"API 서버 연결 실패: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"API 서버 연결 중 오류 발생: {e}")
            return False
            
    def disconnect(self) -> bool:
        """
        API 서버 연결 종료
        
        Returns:
            연결 종료 성공 여부
        """
        try:
            self.session.close()
            self.connected = False
            if self.logger:
                self.logger.info("API 서버 연결이 종료되었습니다.")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"API 서버 연결 종료 중 오류 발생: {e}")
            return False
            
    def get_account_info(self) -> Dict[str, Any]:
        """
        계정 정보 조회
        
        Returns:
            계정 정보가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            response = self.session.get(f"{self.base_url}/account")
            if response.status_code == 200:
                account_data = response.json()
                if self.logger:
                    self.logger.info("계정 정보 조회 성공")
                return account_data
            else:
                if self.logger:
                    self.logger.error(f"계정 정보 조회 실패: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            if self.logger:
                self.logger.error(f"계정 정보 조회 중 오류 발생: {e}")
            return {}
            
    def get_market_data(
        self, 
        symbol: str, 
        interval: str = '1d', 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            symbol: 심볼/티커 (예: AAPL, BTC/USD)
            interval: 시간 간격 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: 가져올 데이터 개수
            
        Returns:
            시장 데이터가 담긴 DataFrame
        """
        if not self.connected:
            self._check_connection()
            
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            response = self.session.get(f"{self.base_url}/market_data", params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # 데이터 형식 변환 및 정리
                df = pd.DataFrame(data)
                
                # 타임스탬프를 날짜/시간으로 변환
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                if self.logger:
                    self.logger.info(f"{symbol} 시장 데이터 조회 성공: {len(df)} 행")
                    
                return df
            else:
                if self.logger:
                    self.logger.error(f"시장 데이터 조회 실패: {response.status_code} - {response.text}")
                return pd.DataFrame()
        except Exception as e:
            if self.logger:
                self.logger.error(f"시장 데이터 조회 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def place_market_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float
    ) -> Dict[str, Any]:
        """
        시장가 주문 실행
        
        Args:
            symbol: 심볼/티커 (예: AAPL, BTC/USD)
            side: 매수/매도 ('buy' 또는 'sell')
            quantity: 수량
            
        Returns:
            주문 정보가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            order_data = {
                "symbol": symbol,
                "side": side.lower(),
                "type": "MARKET",
                "quantity": quantity
            }
            
            response = self.session.post(f"{self.base_url}/order", json=order_data)
            
            if response.status_code == 200:
                order_response = response.json()
                
                if self.logger:
                    self.logger.info(f"{side.upper()} 시장가 주문 성공: {symbol}, 수량: {quantity}")
                    
                return order_response
            else:
                if self.logger:
                    self.logger.error(f"시장가 주문 실패: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"시장가 주문 실패: {response.status_code} - {response.text}"
                }
        except Exception as e:
            if self.logger:
                self.logger.error(f"시장가 주문 중 오류 발생: {e}")
            return {
                "success": False,
                "error": f"시장가 주문 중 오류 발생: {e}"
            }
            
    def place_limit_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: float
    ) -> Dict[str, Any]:
        """
        지정가 주문 실행
        
        Args:
            symbol: 심볼/티커 (예: AAPL, BTC/USD)
            side: 매수/매도 ('buy' 또는 'sell')
            quantity: 수량
            price: 가격
            
        Returns:
            주문 정보가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            order_data = {
                "symbol": symbol,
                "side": side.lower(),
                "type": "LIMIT",
                "quantity": quantity,
                "price": price,
                "timeInForce": "GTC"  # Good Till Cancel
            }
            
            response = self.session.post(f"{self.base_url}/order", json=order_data)
            
            if response.status_code == 200:
                order_response = response.json()
                
                if self.logger:
                    self.logger.info(f"{side.upper()} 지정가 주문 성공: {symbol}, 수량: {quantity}, 가격: {price}")
                    
                return order_response
            else:
                if self.logger:
                    self.logger.error(f"지정가 주문 실패: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"지정가 주문 실패: {response.status_code} - {response.text}"
                }
        except Exception as e:
            if self.logger:
                self.logger.error(f"지정가 주문 중 오류 발생: {e}")
            return {
                "success": False,
                "error": f"지정가 주문 중 오류 발생: {e}"
            }
            
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        주문 취소
        
        Args:
            order_id: 주문 ID
            
        Returns:
            취소 결과가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            response = self.session.delete(f"{self.base_url}/order/{order_id}")
            
            if response.status_code == 200:
                cancel_response = response.json()
                
                if self.logger:
                    self.logger.info(f"주문 취소 성공: {order_id}")
                    
                return cancel_response
            else:
                if self.logger:
                    self.logger.error(f"주문 취소 실패: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"주문 취소 실패: {response.status_code} - {response.text}"
                }
        except Exception as e:
            if self.logger:
                self.logger.error(f"주문 취소 중 오류 발생: {e}")
            return {
                "success": False,
                "error": f"주문 취소 중 오류 발생: {e}"
            }
            
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        주문 상태 조회
        
        Args:
            order_id: 주문 ID
            
        Returns:
            주문 상태 정보가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            response = self.session.get(f"{self.base_url}/order/{order_id}")
            
            if response.status_code == 200:
                order_status = response.json()
                
                if self.logger:
                    self.logger.info(f"주문 상태 조회 성공: {order_id}")
                    
                return order_status
            else:
                if self.logger:
                    self.logger.error(f"주문 상태 조회 실패: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"주문 상태 조회 실패: {response.status_code} - {response.text}"
                }
        except Exception as e:
            if self.logger:
                self.logger.error(f"주문 상태 조회 중 오류 발생: {e}")
            return {
                "success": False,
                "error": f"주문 상태 조회 중 오류 발생: {e}"
            }
            
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        미체결 주문 조회
        
        Args:
            symbol: 심볼/티커 (옵션, 특정 심볼만 조회)
            
        Returns:
            미체결 주문 목록
        """
        if not self.connected:
            self._check_connection()
            
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
                
            response = self.session.get(f"{self.base_url}/open_orders", params=params)
            
            if response.status_code == 200:
                open_orders = response.json()
                
                if self.logger:
                    symbol_text = f" ({symbol})" if symbol else ""
                    self.logger.info(f"미체결 주문 조회 성공{symbol_text}: {len(open_orders)}개")
                    
                return open_orders
            else:
                if self.logger:
                    self.logger.error(f"미체결 주문 조회 실패: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            if self.logger:
                self.logger.error(f"미체결 주문 조회 중 오류 발생: {e}")
            return []
            
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        특정 심볼의 포지션 조회
        
        Args:
            symbol: 심볼/티커
            
        Returns:
            포지션 정보가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            response = self.session.get(f"{self.base_url}/position/{symbol}")
            
            if response.status_code == 200:
                position = response.json()
                
                if self.logger:
                    self.logger.info(f"{symbol} 포지션 조회 성공")
                    
                return position
            else:
                if self.logger:
                    self.logger.error(f"포지션 조회 실패: {response.status_code} - {response.text}")
                return {
                    "symbol": symbol,
                    "quantity": 0,
                    "entry_price": 0,
                    "unrealized_pnl": 0
                }
        except Exception as e:
            if self.logger:
                self.logger.error(f"포지션 조회 중 오류 발생: {e}")
            return {
                "symbol": symbol,
                "quantity": 0,
                "entry_price": 0,
                "unrealized_pnl": 0
            }
            
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        모든 포지션 조회
        
        Returns:
            포지션 목록
        """
        if not self.connected:
            self._check_connection()
            
        try:
            response = self.session.get(f"{self.base_url}/positions")
            
            if response.status_code == 200:
                positions = response.json()
                
                if self.logger:
                    self.logger.info(f"모든 포지션 조회 성공: {len(positions)}개")
                    
                return positions
            else:
                if self.logger:
                    self.logger.error(f"모든 포지션 조회 실패: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            if self.logger:
                self.logger.error(f"모든 포지션 조회 중 오류 발생: {e}")
            return []
            
    def get_trade_history(
        self, 
        symbol: Optional[str] = None, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        거래 내역 조회
        
        Args:
            symbol: 심볼/티커 (옵션, 특정 심볼만 조회)
            start_time: 시작 시간 (옵션)
            end_time: 종료 시간 (옵션)
            limit: 조회할 최대 건수
            
        Returns:
            거래 내역 목록
        """
        if not self.connected:
            self._check_connection()
            
        try:
            params = {"limit": limit}
            
            if symbol:
                params["symbol"] = symbol
                
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
                
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
                
            response = self.session.get(f"{self.base_url}/trades", params=params)
            
            if response.status_code == 200:
                trades = response.json()
                
                if self.logger:
                    symbol_text = f" ({symbol})" if symbol else ""
                    self.logger.info(f"거래 내역 조회 성공{symbol_text}: {len(trades)}건")
                    
                return trades
            else:
                if self.logger:
                    self.logger.error(f"거래 내역 조회 실패: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            if self.logger:
                self.logger.error(f"거래 내역 조회 중 오류 발생: {e}")
            return []
            
    def _check_connection(self) -> None:
        """
        API 연결 상태 확인 및 재연결
        """
        if not self.connected:
            self.connect()
            if not self.connected:
                raise ConnectionError("API 서버에 연결할 수 없습니다.") 