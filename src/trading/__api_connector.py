# API 연결 + 주문/포지션 관리 통합
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.entity import Position
from datetime import datetime, timedelta
import pandas as pd
import requests
import time
from tvDatafeed import TvDatafeed, Interval

from src.utils.logger import Logger

from src.config.config import (
    API_KEY,
    API_SECRET,
    BASE_URL,
    DATA_FEED,
    MAX_RETRIES
)

class APIConnector:
    """
    Alpaca Trading API와의 연결을 관리하는 클라이언트 클래스
    실시간 데이터 스트리밍, 주문 실행, 포트폴리오 관리 기능 제공
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        secret_key: str = None, 
        base_url: str = None, 
        data_feed: str = None,
        logger: Optional[Logger] = None
    ):
        """
        APIConnector 클래스 초기화
        
        Args:
            api_key: Alpaca API 키 (기본값: config의 API_KEY)
            secret_key: Alpaca 시크릿 키 (기본값: config의 API_SECRET)
            base_url: API 베이스 URL (기본값: config의 BASE_URL)
            data_feed: 데이터 피드 타입 (기본값: config의 DATA_FEED)
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        # config에서 기본값 가져오기
        self.api_key = api_key or API_KEY
        self.secret_key = secret_key or API_SECRET
        self.base_url = base_url or BASE_URL
        self.data_feed = data_feed or DATA_FEED
        self.max_retries = MAX_RETRIES
        
        self.paper = 'paper' in self.base_url.lower()
        self.logger = logger or self._setup_default_logger()
        
        # API 키 검증
        if not self.api_key or not self.secret_key:
            raise ValueError("API_KEY와 API_SECRET이 설정되어야 합니다. config 파일을 확인하세요.")
        
        # REST API 클라이언트 초기화
        self.rest = REST(self.api_key, self.secret_key, self.base_url)
        
        # WebSocket 스트림 클라이언트 초기화
        self.stream = Stream(
            self.api_key, 
            self.secret_key, 
            self.base_url, 
            data_feed=self.data_feed,
            raw_data=True
        )
        
        self._handlers_registered = False
        self.connected = False
        
    def _setup_default_logger(self) -> logging.Logger:
        """기본 로거 설정"""
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logger

    def connect(self) -> bool:
        """
        Alpaca API 서버에 연결 (재시도 로직 포함)
        
        Returns:
            연결 성공 여부
        """
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # API 계정 상태 체크
                account = self.rest.get_account()
                if account and account.status == 'ACTIVE':
                    self.connected = True
                    self.logger.info(f"Alpaca API에 성공적으로 연결되었습니다. 계정: {account.id}")
                    self.logger.info(f"환경: {'Paper Trading' if self.paper else 'Live Trading'}")
                    self.logger.info(f"데이터 피드: {self.data_feed}")
                    return True
                else:
                    self.logger.error(f"계정이 활성화되지 않았습니다. 상태: {account.status if account else 'Unknown'}")
                    return False
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Alpaca API 연결 시도 {retry_count}/{self.max_retries} 실패: {e}")
                
                if retry_count < self.max_retries:
                    wait_time = min(2 ** retry_count, 30)  # 지수 백오프, 최대 30초
                    self.logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("최대 재시도 횟수 초과. 연결에 실패했습니다.")
                    return False
            
    def disconnect(self) -> bool:
        """
        API 연결 종료
        
        Returns:
            연결 종료 성공 여부
        """
        try:
            # WebSocket 연결이 있다면 종료
            if hasattr(self.stream, '_ws') and self.stream._ws:
                asyncio.create_task(self.stream.stop_ws())
            
            self.connected = False
            self.logger.info("Alpaca API 연결이 종료되었습니다.")
            return True
        except Exception as e:
            self.logger.error(f"API 연결 종료 중 오류 발생: {e}")
            return False

    async def start_websocket(self, symbols: List[str] = None) -> None:
        """
        WebSocket 실시간 데이터 스트림 시작
        
        Args:
            symbols: 구독할 심볼 리스트 (기본값: ["SPY"])
        """
        if not self.connected:
            self._check_connection()
            
        if symbols is None:
            symbols = ["SPY"]  # 기본값으로 S&P 500 ETF
        
        self.logger.info(f"WebSocket 연결 시작 - 데이터 피드: {self.data_feed}")
        
        if not self._handlers_registered:
            self.logger.info("이벤트 핸들러 등록 중...")
            
            async def on_trade(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    price = data.price if hasattr(data, 'price') else 0.0
                    size = data.size if hasattr(data, 'size') else 0
                    timestamp = data.timestamp if hasattr(data, 'timestamp') else datetime.now()
                    self.logger.info(f"[거래] {symbol}: 가격=${price:.2f}, 수량={size}, 시간={timestamp}")
                except Exception as e:
                    self.logger.error(f"거래 데이터 처리 오류: {e}")
            
            async def on_quote(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    bid = data.bid_price if hasattr(data, 'bid_price') else 0.0
                    ask = data.ask_price if hasattr(data, 'ask_price') else 0.0
                    self.logger.info(f"[호가] {symbol}: 매수=${bid:.2f}, 매도=${ask:.2f}, 스프레드=${ask-bid:.2f}")
                except Exception as e:
                    self.logger.error(f"호가 데이터 처리 오류: {e}")
            
            async def on_bar(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    close = data.close if hasattr(data, 'close') else 0.0
                    volume = data.volume if hasattr(data, 'volume') else 0
                    self.logger.info(f"[바] {symbol}: 종가=${close:.2f}, 거래량={volume}")
                except Exception as e:
                    self.logger.error(f"바 데이터 처리 오류: {e}")
            
            @self.stream.on_status("*")
            async def on_status(status):
                self.logger.info(f"[상태] WebSocket 상태 변경: {status}")
                if status.get('status') == 'connected' or status.get('status') == 'auth_success':
                    self.logger.info(f"[연결 성공] WebSocket 연결이 설정되었습니다: {status}")
            
            self.logger.info(f"심볼 구독 등록 중: {', '.join(symbols)}")
            for symbol in symbols:
                self.stream.subscribe_trades(on_trade, symbol)
                self.logger.info(f"- {symbol} 거래 데이터 구독")
                
                self.stream.subscribe_quotes(on_quote, symbol)
                self.logger.info(f"- {symbol} 호가 데이터 구독")
                
                # 1분봉 구독
                @self.stream.on_bar(symbol)
                async def _(bar):
                    await on_bar(bar)
                self.logger.info(f"- {symbol} 바 데이터 구독")
            
            self._handlers_registered = True
            self.logger.info("모든 이벤트 핸들러 등록 완료")
        
        try:
            self.logger.info("WebSocket 데이터 스트림 시작...")
            await self.stream._run_forever()
        except Exception as e:
            self.logger.error(f"WebSocket 오류: {e}")
            raise
    

    def get_market_data(
        self,
        symbol: str,
        exchange="NASDAQ",
        interval=Interval.in_1_minute,  # 문자열이 아니라 Interval 객체로 지정  ( Interval.in_1_minute, Interval.in_5_minute,Interval.in_15_minute,Interval.in_30_minute,Interval.in_60_minute,Interval.in_1_hour,Interval.in_1_day)
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:

        """
        TradingView 데이터 가져오기 (tvdatafeed 사용)
        
        Args:
            symbol: 심볼/티커 (예: AAPL)
            exchange: 거래소 (예: NASDAQ)
            interval: 데이터 간격 (1Min, 5Min, 1H, 1D)
            n_bars: 가져올 데이터 개수
        
        Returns:
            시장 데이터가 담긴 DataFrame
        """
        try:
            tv = TvDatafeed()
            df = tv.get_hist(symbol, exchange, interval=interval, n_bars=limit)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"TradingView 데이터 로드 실패: {e}")
            return pd.DataFrame()

    # 시장가 주문
    def place_market_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float
    ) -> Dict[str, Any]:
        """
        시장가 주문 실행
        
        Args:
            symbol: 심볼/티커 (예: AAPL)
            side: 매수/매도 ('buy' 또는 'sell')
            quantity: 수량
            
        Returns:
            주문 정보가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            order = self.rest.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side.lower(),
                type="market",
                time_in_force="day"
            )
            
            result = {
                "success": True,
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.order_type,
                "status": order.status,
                "created_at": order.created_at
            }
            
            self.logger.info(f"{side.upper()} 시장가 주문 성공: {symbol}, 수량: {quantity}")
            return result
            
        except Exception as e:
            self.logger.error(f"시장가 주문 실행 실패: {e}")
            return {
                "success": False,
                "error": f"시장가 주문 실행 실패: {e}"
            }

    # 지정가 주문
    def place_limit_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: float,
        time_in_force: str = "day"
    ) -> Dict[str, Any]:
        """
        지정가 주문 실행
        
        Args:
            symbol: 심볼/티커 (예: AAPL)
            side: 매수/매도 ('buy' 또는 'sell')
            quantity: 수량
            price: 가격
            time_in_force: 주문 유효 시간 ('day', 'gtc', 'ioc', 'fok')
            
        Returns:
            주문 정보가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            order = self.rest.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side.lower(),
                type="limit",
                time_in_force=time_in_force,
                limit_price=str(price)
            )
            
            result = {
                "success": True,
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.order_type,
                "status": order.status,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "created_at": order.created_at
            }
            
            self.logger.info(f"{side.upper()} 지정가 주문 성공: {symbol}, 수량: {quantity}, 가격: {price}")
            return result
            
        except Exception as e:
            self.logger.error(f"지정가 주문 실행 실패: {e}")
            return {
                "success": False,
                "error": f"지정가 주문 실행 실패: {e}"
            }

    # 다용도 주문 처리 함수
    def place_order(
        self, 
        symbol: str, 
        qty: float, 
        side: str, 
        order_type: str = "market", 
        time_in_force: str = "day",
        limit_price: float = None, 
        stop_price: float = None
    ) -> Dict[str, Any]:
        """
        주문 실행 (기존 호환성 유지)
        
        Args:
            symbol: 심볼/티커
            qty: 수량
            side: 매수/매도 ('buy' 또는 'sell')
            order_type: 주문 타입 ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: 주문 유효 시간
            limit_price: 지정가 (limit/stop_limit 주문시 필요)
            stop_price: 스탑 가격 (stop/stop_limit 주문시 필요)
            
        Returns:
            주문 정보가 담긴 딕셔너리
        """
        if order_type == "market":
            return self.place_market_order(symbol, side, qty)
        elif order_type == "limit" and limit_price is not None:
            return self.place_limit_order(symbol, side, qty, limit_price, time_in_force)
        else:
            # 기존 복잡한 주문 로직 유지
            if not self.connected:
                self._check_connection()
                
            try:
                params = {
                    "symbol": symbol,
                    "qty": qty,
                    "side": side,
                    "type": order_type,
                    "time_in_force": time_in_force
                }
                
                if limit_price is not None and order_type in ["limit", "stop_limit"]:
                    params["limit_price"] = str(limit_price)
                
                if stop_price is not None and order_type in ["stop", "stop_limit"]:
                    params["stop_price"] = str(stop_price)
                
                order = self.rest.submit_order(**params)
                
                return {
                    "success": True,
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side,
                    "type": order.order_type,
                    "status": order.status,
                    "created_at": order.created_at
                }
            except Exception as e:
                self.logger.error(f"주문 실행 실패: {e}")
                return {
                    "success": False,
                    "error": f"주문 실행 실패: {e}"
                }

    # 주문 취소
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
            self.rest.cancel_order(order_id)
            
            self.logger.info(f"주문 취소 성공: {order_id}")
            return {
                "success": True,
                "order_id": order_id,
                "message": "주문이 성공적으로 취소되었습니다."
            }
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {e}")
            return {
                "success": False,
                "error": f"주문 취소 실패: {e}"
            }

    # 주문 상태 조회
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
            order = self.rest.get_order(order_id)
            
            result = {
                "success": True,
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "filled_qty": float(order.filled_qty),
                "side": order.side,
                "type": order.order_type,
                "status": order.status,
                "created_at": order.created_at,
                "updated_at": order.updated_at
            }
            
            self.logger.info(f"주문 상태 조회 성공: {order_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"주문 상태 조회 실패: {e}")
            return {
                "success": False,
                "error": f"주문 상태 조회 실패: {e}"
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
            orders = self.rest.list_orders(status='open', symbols=symbol)
            
            result = []
            for order in orders:
                order_data = {
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "filled_qty": float(order.filled_qty),
                    "side": order.side,
                    "type": order.order_type,
                    "status": order.status,
                    "created_at": order.created_at
                }
                result.append(order_data)
            
            symbol_text = f" ({symbol})" if symbol else ""
            self.logger.info(f"미체결 주문 조회 성공{symbol_text}: {len(result)}개")
            return result
            
        except Exception as e:
            self.logger.error(f"미체결 주문 조회 실패: {e}")
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
            position = self.rest.get_position(symbol)
            
            result = {
                "symbol": position.symbol,
                "qty": float(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "market_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "change_today": float(position.change_today)
            }
            
            self.logger.info(f"{symbol} 포지션 조회 성공")
            return result
            
        except Exception as e:
            self.logger.error(f"포지션 조회 실패: {e}")
            return {
                "symbol": symbol,
                "qty": 0,
                "avg_entry_price": 0,
                "current_price": 0,
                "market_value": 0,
                "unrealized_pl": 0,
                "unrealized_plpc": 0,
                "change_today": 0
            }

    def get_portfolio(self) -> List[Dict[str, Any]]:
        """
        포트폴리오 조회 (모든 포지션)
        
        Returns:
            포지션 목록
        """
        return self.get_all_positions()

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        모든 포지션 조회
        
        Returns:
            포지션 목록
        """
        if not self.connected:
            self._check_connection()
            
        try:
            positions = self.rest.list_positions()
            
            result = []
            for p in positions:
                position_data = {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "change_today": float(p.change_today)
                }
                result.append(position_data)
                
                self.logger.info(f"{p.symbol}: {p.qty} 주 @ ${p.current_price} "
                               f"(평균가: ${p.avg_entry_price}, 손익: ${p.unrealized_pl})")
            
            self.logger.info(f"모든 포지션 조회 성공: {len(result)}개")
            return result
            
        except Exception as e:
            self.logger.error(f"포트폴리오 조회 실패: {e}")
            return []

    def get_account_info(self) -> Dict[str, Any]:
        """
        계정 정보 조회
        
        Returns:
            계정 정보가 담긴 딕셔너리
        """
        return self.get_account()

    def get_account(self) -> Dict[str, Any]:
        """
        계정 정보 조회
        
        Returns:
            계정 정보가 담긴 딕셔너리
        """
        if not self.connected:
            self._check_connection()
            
        try:
            account = self.rest.get_account()
            
            result = {
                "id": account.id,  # 계좌의 고유 ID
                "cash": float(account.cash),  # 계좌의 현금 잔액 (사용 가능한 현금)
                "portfolio_value": float(account.portfolio_value),  # 전체 포트폴리오 가치 (보유 주식 + 현금 포함)
                "equity": float(account.equity),  # 현재 계좌의 순자산 가치 (실시간으로 변동 가능)
                "buying_power": float(account.buying_power),  # 매수 가능 금액 (현금 + 증거금 기준 계산)
                "status": account.status,  # 계좌 상태 (예: 'ACTIVE', 'INACTIVE' 등)
                "day_trade_count": int(getattr(account, "day_trade_count", 0)),  # 없는 경우 기본값 0   # 오늘 실행한 데이 트레이딩(동일 종목의 당일 매수/매도) 횟수 
                # 같은 종목을 하루 안에 '매수 → 매도' 또는 '매도 → 매수'한 경우만 계산
                "pattern_day_trader": account.pattern_day_trader  # 패턴 데이 트레이더로 간주되는지 여부 (True/False)
            }
            print(f'result\n{result}')
            
            self.logger.info("계정 정보 조회 성공")
            return result
            
        except Exception as e:
            self.logger.error(f"계정 정보 조회 실패: {e}")
            return {"error": str(e)}

    # 거래 완료된 기록 조회
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
            params = {
                'status': 'filled',
                'limit': limit
            }
            
            if start_time:
                params['after'] = start_time.strftime('%Y-%m-%d')
            if end_time:
                params['until'] = end_time.strftime('%Y-%m-%d')
            if symbol:
                params['symbols'] = symbol
                
            orders = self.rest.list_orders(**params)
            
            result = []
            for order in orders:
                if order.status == 'filled':
                    trade_data = {
                        "id": order.id,
                        "symbol": order.symbol,
                        "qty": float(order.qty),
                        "side": order.side,
                        "price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                        "filled_at": order.filled_at,
                        "created_at": order.created_at
                    }
                    result.append(trade_data)
            
            symbol_text = f" ({symbol})" if symbol else ""
            self.logger.info(f"거래 내역 조회 성공{symbol_text}: {len(result)}건")
            return result
            
        except Exception as e:
            self.logger.error(f"거래 내역 조회 실패: {e}")
            return []

    def _check_connection(self) -> None:
        """
        API 연결 상태 확인 및 재연결
        """
        if not self.connected:
            if not self.connect():
                raise ConnectionError("Alpaca API 서버에 연결할 수 없습니다.")


# 팩토리 함수 - config를 사용한 기본 클라이언트 생성
def create_alpaca_client(logger: Optional[Logger] = None) -> APIConnector:
    """
    config 설정을 사용하여 APIConnector 인스턴스를 생성하는 팩토리 함수
    
    Args:
        logger: 로깅을 위한 Logger 인스턴스 (옵션)
        
    Returns:
        설정된 APIConnector 인스턴스
    """
    return APIConnector(logger=logger)



if __name__ == "__main__":
    # config 설정을 사용한 클라이언트 생성
    client = create_alpaca_client()
    
    # 또는 직접 생성 (모든 값이 config에서 자동으로 가져와짐)
    # client = APIConnector()
    
    # 연결
    if client.connect():
        # 계정 정보 조회
        account = client.get_account_info()
        print(f"계정 현금: ${account.get('cash', 0)}")
        
        # 포트폴리오 조회
        positions = client.get_all_positions()
        print(f"보유 포지션: {len(positions)}개")
        
        # 시장 데이터 조회
        market_data = client.get_market_data("AAPL", limit=10)
        print(f"AAPL 데이터: {len(market_data)} 행")
        
        # 연결 종료
        client.disconnect()
    else:
        print("Alpaca API 연결에 실패했습니다. config 파일의 API 키를 확인하세요.")