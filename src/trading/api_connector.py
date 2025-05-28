import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.entity import Position
from datetime import datetime, timedelta
import pandas as pd
import requests
import time
from tvDatafeed import TvDatafeed, Interval
import pytz
import argparse
import websocket
import json
import threading
from src.utils.logger import Logger

from src.config.config import (
    API_KEY,
    API_SECRET,
    BASE_URL,
    DATA_FEED,
    MAX_RETRIES
)

class TradingViewWebSocketClient:
    """TradingView WebSocket 클라이언트"""
    
    def __init__(self, symbol: str, interval: str = "1", logger=None):
        self.symbol = symbol
        self.interval = interval
        self.session_id = None
        self.series_id = None
        self.ws_app = None
        self.ws_thread = None
        self.lock = threading.Lock()
        self.is_connected = False
        self.logger = logger or logging.getLogger(__name__)
        self.data_callback = None
        self.data_buffer = []
        self.max_buffer_size = 1000
        
    def generate_ids(self):
        timestamp = int(time.time() * 1000)
        self.session_id = f'cs_{self.symbol.replace(":", "_")}_{timestamp}'
        self.series_id = f'series_{self.symbol.replace(":", "_")}_{timestamp}'

    def encode_msg(self, msg):
        s = json.dumps(msg)
        return f"~m~{len(s)}~m~{s}"

    def on_message(self, ws, message):
        try:
            payloads = message.split("~m~")
            for payload in payloads:
                if not payload.strip() or payload.strip().isdigit():
                    continue

                msg = json.loads(payload)
                if isinstance(msg, dict) and "m" in msg:
                    method = msg["m"]
                    if method == "timescale_update":
                        bars = msg["p"][1].get(self.series_id, {}).get("s", [])
                        for bar in bars:
                            bar_data = {
                                'timestamp': datetime.utcfromtimestamp(bar["v"][0]),
                                'open': bar["v"][1],
                                'high': bar["v"][2],
                                'low': bar["v"][3],
                                'close': bar["v"][4],
                                'volume': bar["v"][5]
                            }
                            
                            # 데이터 버퍼에 추가
                            self.data_buffer.append(bar_data)
                            if len(self.data_buffer) > self.max_buffer_size:
                                self.data_buffer = self.data_buffer[-self.max_buffer_size:]
                            
                            # 콜백 함수가 있으면 호출
                            if self.data_callback:
                                self.data_callback(bar_data)
                                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[{self.symbol}] 메시지 처리 오류: {e}")

    def on_open(self, ws):
        if self.logger:
            self.logger.info(f"[{self.symbol}] TradingView WebSocket 연결됨")
        
        self.is_connected = True
        self.generate_ids()
        
        try:
            messages = [
                {"m": "set_auth_token", "p": ["unauthorized_user_token"]},
                {"m": "chart_create_session", "p": [self.session_id, ""]},
                {"m": "resolve_symbol", "p": [self.session_id, "symbol_1", self.symbol]},
                {"m": "create_series", "p": [self.session_id, self.series_id, "s1", "symbol_1", self.interval, 100]},
            ]

            for msg in messages:
                ws.send(self.encode_msg(msg))

        except Exception as e:
            if self.logger:
                self.logger.error(f"[{self.symbol}] WebSocket 초기화 오류: {e}")

    def on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False
        if self.logger:
            self.logger.info(f"[{self.symbol}] TradingView WebSocket 연결 종료")

    def on_error(self, ws, error):
        if self.logger:
            self.logger.error(f"[{self.symbol}] TradingView WebSocket 오류: {error}")

    def start(self, data_callback: Callable = None):
        """WebSocket 연결 시작"""
        with self.lock:
            if self.ws_app:
                self.ws_app.close()

            self.data_callback = data_callback
            self.ws_app = websocket.WebSocketApp(
                "wss://data.tradingview.com/socket.io/websocket",
                on_open=self.on_open,
                on_message=self.on_message,
                on_close=self.on_close,
                on_error=self.on_error,
            )

            self.ws_thread = threading.Thread(target=self.ws_app.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()

    def stop(self):
        """WebSocket 연결 중지"""
        with self.lock:
            if self.ws_app:
                self.ws_app.close()
                self.ws_app = None
            self.is_connected = False

    def get_buffered_data(self) -> pd.DataFrame:
        """버퍼된 데이터를 DataFrame으로 반환"""
        if not self.data_buffer:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data_buffer)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

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
        logger: Optional[Logger] = None,
        market_hours_only: bool = True
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
        self.market_hours_only = market_hours_only
        
        # TradingView WebSocket 클라이언트들을 관리하기 위한 딕셔너리
        self.tv_clients = {}
        
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
    
    def get_market_calendar(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """
        시장 달력 조회 (휴장일, 개장/폐장 시간 등)
        
        Args:
            start_date: 조회 시작일 (기본값: 오늘)
            end_date: 조회 종료일 (기본값: 오늘)
            
        Returns:
            시장 달력 정보 리스트
        """
        try:
            if not start_date:
                start_date = datetime.now().date()
            if not end_date:
                end_date = start_date
                
            calendar = self.rest.get_calendar(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            result = []
            for day in calendar:
                result.append({
                    'date': day.date,
                    'open': day.open,
                    'close': day.close,
                    'session_open': day.session_open,
                    'session_close': day.session_close
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"시장 달력 조회 실패: {e}")
            return []
    
    def is_market_open(self) -> Dict[str, Any]:
        """
        현재 시장이 열려있는지 확인
        
        Returns:
            시장 상태 정보가 담긴 딕셔너리
        """
        try:
            clock = self.rest.get_clock()
            
            result = {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'timestamp': clock.timestamp
            }
            
            if clock.is_open:
                self.logger.info("✅ 시장이 현재 열려있습니다")
            else:
                # 다음 개장 시간을 한국 시간으로도 표시
                next_open_kr = clock.next_open.astimezone(pytz.timezone('Asia/Seoul'))
                self.logger.info(f"🔒 시장이 현재 닫혀있습니다. 다음 개장: {clock.next_open} (한국시간: {next_open_kr})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"시장 상태 확인 실패: {e}")
            return {
                'is_open': False,
                'error': str(e)
            }
    
    def is_trading_day(self, date: datetime = None) -> bool:
        """
        특정 날짜가 거래일인지 확인
        
        Args:
            date: 확인할 날짜 (기본값: 오늘)
            
        Returns:
            거래일 여부 (True/False)
        """
        if not date:
            date = datetime.now()
            
        try:
            calendar = self.get_market_calendar(date.date(), date.date())
            return len(calendar) > 0
            
        except Exception as e:
            self.logger.error(f"거래일 확인 실패: {e}")
            return False

    def get_next_market_open(self) -> datetime:
        """
        다음 시장 개장 시간 조회
        
        Returns:
            다음 개장 시간
        """
        try:
            clock = self.rest.get_clock()
            return clock.next_open
        except Exception as e:
            self.logger.error(f"다음 개장 시간 조회 실패: {e}")
            return None

    def wait_for_market_open(self, check_interval: int = 300) -> bool:
        """
        시장이 열릴 때까지 대기
        
        Args:
            check_interval: 확인 주기 (초, 기본값: 5분)
            
        Returns:
            시장 개장 여부
        """
        self.logger.info("시장 개장을 기다리는 중...")
        
        while True:
            market_status = self.is_market_open()
            
            if market_status.get('is_open', False):
                self.logger.info("✅ 시장이 열렸습니다!")
                return True
            
            next_open = market_status.get('next_open')
            if next_open:
                # 다음 개장까지 남은 시간 계산
                now = datetime.now(pytz.UTC)
                time_until_open = next_open - now
                
                if time_until_open.total_seconds() > 0:
                    hours, remainder = divmod(int(time_until_open.total_seconds()), 3600)
                    minutes, _ = divmod(remainder, 60)
                    
                    self.logger.info(f"⏰ 시장 개장까지 {hours}시간 {minutes}분 남았습니다. {check_interval}초 후 다시 확인...")
                    time.sleep(check_interval)
                else:
                    self.logger.info("시장 개장 시간이 지났지만 아직 열리지 않았습니다. 잠시 후 다시 확인...")
                    time.sleep(30)  # 30초 후 재확인
            else:
                self.logger.warning("시장 정보를 가져올 수 없습니다. 30초 후 재시도...")
                time.sleep(30)

    def _setup_default_logger(self) -> logging.Logger:
        """기본 로거 설정"""
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logger

    def connect(self, force_connect: bool = False) -> bool:
        """
        Alpaca API 서버에 연결 (시장 시간 체크 포함)
        
        Args:
            force_connect: 시장 시간과 관계없이 강제 연결 (기본값: False)
            
        Returns:
            연결 성공 여부
        """
        # 시장 시간 체크
        if self.market_hours_only and not force_connect:
            market_status = self.is_market_open()
            
            if not market_status.get('is_open', False):
                if 'error' in market_status:
                    self.logger.warning(f"시장 상태 확인 중 오류 발생: {market_status['error']}")
                    self.logger.info("오류가 발생했지만 연결을 시도합니다...")
                else:
                    self.logger.warning("❌ 시장이 닫혀있어 연결하지 않습니다.")
                    self.logger.info("💡 시장 시간과 관계없이 연결하려면 force_connect=True를 사용하세요.")
                    return False
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # API 계정 상태 체크
                account = self.rest.get_account()
                if account and account.status == 'ACTIVE':
                    self.connected = True
                    
                    # 연결 성공 로그
                    self.logger.info(f"🎉 Alpaca API에 성공적으로 연결되었습니다!")
                    self.logger.info(f"📊 계정: {account.id}")
                    self.logger.info(f"🏷️  환경: {'Paper Trading' if self.paper else 'Live Trading'}")
                    self.logger.info(f"📡 데이터 피드: {self.data_feed}")
                    
                    # 시장 상태 정보 추가 출력
                    if not force_connect:
                        market_status = self.is_market_open()
                        if market_status.get('is_open'):
                            self.logger.info("🟢 시장 상태: 개장 중")
                        else:
                            next_open = market_status.get('next_open')
                            if next_open:
                                next_open_kr = next_open.astimezone(pytz.timezone('Asia/Seoul'))
                                self.logger.info(f"🔴 시장 상태: 폐장 중 (다음 개장: {next_open_kr.strftime('%Y-%m-%d %H:%M KST')})")
                    
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
                    self.logger.error("❌ 최대 재시도 횟수 초과. 연결에 실패했습니다.")
                    return False
 
    def connect_with_market_wait(self, check_interval: int = 300) -> bool:
        """
        시장이 열릴 때까지 기다린 후 연결
        
        Args:
            check_interval: 시장 상태 확인 주기 (초, 기본값: 5분)
            
        Returns:
            연결 성공 여부
        """
        # 먼저 시장이 열려있는지 확인
        market_status = self.is_market_open()
        
        if not market_status.get('is_open', False):
            # 시장이 닫혀있으면 대기
            if not self.wait_for_market_open(check_interval):
                return False
        
        # 시장이 열렸으면 연결
        return self.connect(force_connect=True)

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
            
            # TradingView WebSocket 클라이언트들 정리
            for symbol, client in self.tv_clients.items():
                client.stop()
            self.tv_clients.clear()
            
            self.connected = False
            self.logger.info("👋 Alpaca API 연결이 종료되었습니다.")
            return True
        except Exception as e:
            self.logger.error(f"API 연결 종료 중 오류 발생: {e}")
            return False

    async def start_websocket(self, symbols: List[str] = None) -> None:
        """
        WebSocket 실시간 데이터 스트림 시작 (시장 시간 체크 포함)
        
        Args:
            symbols: 구독할 심볼 리스트 (기본값: ["SPY"])
        """
        if not self.connected:
            self._check_connection()
        
        # 시장 시간 체크 (강제 연결이 아닌 경우)
        if self.market_hours_only:
            market_status = self.is_market_open()
            if not market_status.get('is_open', False):
                self.logger.warning("❌ 시장이 닫혀있어 WebSocket 연결을 시작하지 않습니다.")
                return
            
        if symbols is None:
            symbols = ["SPY"]  # 기본값으로 S&P 500 ETF
        
        self.logger.info(f"📡 WebSocket 연결 시작 - 데이터 피드: {self.data_feed}")
        
        if not self._handlers_registered:
            self.logger.info("🔧 이벤트 핸들러 등록 중...")
            
            async def on_trade(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    price = data.price if hasattr(data, 'price') else 0.0
                    size = data.size if hasattr(data, 'size') else 0
                    timestamp = data.timestamp if hasattr(data, 'timestamp') else datetime.now()
                    self.logger.info(f"💰 [거래] {symbol}: 가격=${price:.2f}, 수량={size}, 시간={timestamp}")
                except Exception as e:
                    self.logger.error(f"거래 데이터 처리 오류: {e}")
            
            async def on_quote(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    bid = data.bid_price if hasattr(data, 'bid_price') else 0.0
                    ask = data.ask_price if hasattr(data, 'ask_price') else 0.0
                    self.logger.info(f"📊 [호가] {symbol}: 매수=${bid:.2f}, 매도=${ask:.2f}, 스프레드=${ask-bid:.2f}")
                except Exception as e:
                    self.logger.error(f"호가 데이터 처리 오류: {e}")
            
            async def on_bar(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    close = data.close if hasattr(data, 'close') else 0.0
                    volume = data.volume if hasattr(data, 'volume') else 0
                    self.logger.info(f"📈 [바] {symbol}: 종가=${close:.2f}, 거래량={volume}")
                except Exception as e:
                    self.logger.error(f"바 데이터 처리 오류: {e}")
            
            @self.stream.on_status("*")
            async def on_status(status):
                self.logger.info(f"🔄 [상태] WebSocket 상태 변경: {status}")
                if status.get('status') == 'connected' or status.get('status') == 'auth_success':
                    self.logger.info(f"✅ [연결 성공] WebSocket 연결이 설정되었습니다: {status}")
            
            self.logger.info(f"📋 심볼 구독 등록 중: {', '.join(symbols)}")
            for symbol in symbols:
                self.stream.subscribe_trades(on_trade, symbol)
                self.logger.info(f"  📈 {symbol} 거래 데이터 구독")
                
                self.stream.subscribe_quotes(on_quote, symbol)
                self.logger.info(f"  📊 {symbol} 호가 데이터 구독")
                
                # 1분봉 구독
                @self.stream.on_bar(symbol)
                async def _(bar):
                    await on_bar(bar)
                self.logger.info(f"  📊 {symbol} 바 데이터 구독")
            
            self._handlers_registered = True
            self.logger.info("✅ 모든 이벤트 핸들러 등록 완료")
        
        try:
            self.logger.info("🚀 WebSocket 데이터 스트림 시작...")
            await self.stream._run_forever()
        except Exception as e:
            self.logger.error(f"WebSocket 오류: {e}")
            raise

    def get_market_data(
        self,
        symbol: str,
        exchange: str = "NASDAQ",
        interval: str = "1",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
        use_websocket: bool = True,
        wait_time: int = 10
    ) -> pd.DataFrame:
        """
        TradingView 데이터 가져오기 (WebSocket 또는 tvdatafeed 사용)
        
        Args:
            symbol: 심볼/티커
            exchange: 거래소 (기본값: NASDAQ)
            interval: 시간 간격 (기본값: 1분)
            start: 시작 시간 (옵션)
            end: 종료 시간 (옵션)
            limit: 조회할 최대 건수
            use_websocket: WebSocket 사용 여부 (기본값: True)
            wait_time: WebSocket 데이터 수집 대기 시간 (초, 기본값: 10)
            
        Returns:
            시장 데이터 DataFrame
        """
        if use_websocket:
            try:
                # 심볼 형식 변환 (예: AAPL -> NASDAQ:AAPL)
                tv_symbol = f"{exchange}:{symbol}" if ":" not in symbol else symbol
                
                # 기존 클라이언트가 있으면 정리
                if tv_symbol in self.tv_clients:
                    self.tv_clients[tv_symbol].stop()
                
                # 새로운 WebSocket 클라이언트 생성
                client = TradingViewWebSocketClient(tv_symbol, interval, self.logger)
                self.tv_clients[tv_symbol] = client
                
                self.logger.info(f"📡 TradingView WebSocket 시작: {tv_symbol}")
                
                # WebSocket 연결 시작
                client.start()
                
                # 데이터 수집 대기
                self.logger.info(f"⏰ {wait_time}초 동안 데이터 수집 중...")
                time.sleep(wait_time)
                
                # 버퍼된 데이터 가져오기
                df = client.get_buffered_data()
                
                if df.empty:
                    self.logger.warning(f"WebSocket에서 데이터를 받지 못했습니다. tvdatafeed로 대체합니다.")
                    return self._get_market_data_fallback(symbol, exchange, interval, start, end, limit)
                
                # limit 적용
                if len(df) > limit:
                    df = df.tail(limit)
                
                self.logger.info(f"✅ WebSocket에서 {len(df)}개 데이터 수집 완료: {tv_symbol}")
                return df
                
            except Exception as e:
                self.logger.error(f"TradingView WebSocket 데이터 로드 실패: {e}")
                self.logger.info("tvdatafeed로 대체합니다.")
                return self._get_market_data_fallback(symbol, exchange, interval, start, end, limit)
        else:
            return self._get_market_data_fallback(symbol, exchange, interval, start, end, limit)

    def _get_market_data_fallback(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: Optional[datetime],
        end: Optional[datetime],
        limit: int
    ) -> pd.DataFrame:
        """tvdatafeed를 사용한 대체 데이터 로드"""
        try:
            # interval을 tvdatafeed 형식으로 변환
            interval_map = {
                "1": Interval.in_1_minute,
                "5": Interval.in_5_minute,
                "15": Interval.in_15_minute,
                "30": Interval.in_30_minute,
                "60": Interval.in_1_hour,
                "1H": Interval.in_1_hour,
                "4H": Interval.in_4_hour,
                "1D": Interval.in_daily,
                "1W": Interval.in_weekly,
                "1M": Interval.in_monthly
            }
            
            tv_interval = interval_map.get(interval, Interval.in_1_minute)
            tv = TvDatafeed()
            df = tv.get_hist(symbol, exchange, interval=tv_interval, n_bars=limit)
            
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                self.logger.info(f"✅ tvdatafeed에서 {len(df)}개 데이터 로드 완료: {symbol}")
                return df
            else:
                self.logger.warning(f"tvdatafeed에서도 데이터를 가져올 수 없습니다: {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"tvdatafeed 데이터 로드 실패: {e}")
            return pd.DataFrame()

    def start_market_data_stream(
        self,
        symbol: str,
        exchange: str = "NASDAQ",
        interval: str = "1",
        data_callback: Callable = None
    ) -> bool:
        """
        실시간 시장 데이터 스트림 시작
        
        Args:
            symbol: 심볼/티커
            exchange: 거래소
            interval: 시간 간격
            data_callback: 데이터 수신 시 호출될 콜백 함수
            
        Returns:
            스트림 시작 성공 여부
        """
        try:
            tv_symbol = f"{exchange}:{symbol}" if ":" not in symbol else symbol
            
            # 기존 클라이언트가 있으면 정리
            if tv_symbol in self.tv_clients:
                self.tv_clients[tv_symbol].stop()
            
            # 새로운 WebSocket 클라이언트 생성
            client = TradingViewWebSocketClient(tv_symbol, interval, self.logger)
            self.tv_clients[tv_symbol] = client
            
            # WebSocket 연결 시작
            client.start(data_callback)
            
            self.logger.info(f"📡 실시간 데이터 스트림 시작: {tv_symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"실시간 데이터 스트림 시작 실패: {e}")
            return False

    def stop_market_data_stream(self, symbol: str, exchange: str = "NASDAQ") -> bool:
        """
        실시간 시장 데이터 스트림 중지
        
        Args:
            symbol: 심볼/티커
            exchange: 거래소
            
        Returns:
            스트림 중지 성공 여부
        """
        try:
            tv_symbol = f"{exchange}:{symbol}" if ":" not in symbol else symbol
            
            if tv_symbol in self.tv_clients:
                self.tv_clients[tv_symbol].stop()
                del self.tv_clients[tv_symbol]
                self.logger.info(f"📡 실시간 데이터 스트림 중지: {tv_symbol}")
                return True
            else:
                self.logger.warning(f"활성화된 스트림이 없습니다: {tv_symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"실시간 데이터 스트림 중지 실패: {e}")
            return False

    def get_active_streams(self) -> List[str]:
        """
        현재 활성화된 데이터 스트림 목록 반환
        
        Returns:
            활성화된 스트림 심볼 목록
        """
        active_streams = []
        for symbol, client in self.tv_clients.items():
            if client.is_connected:
                active_streams.append(symbol)
        return active_streams
    
    def _check_connection(self) -> None:
        """API 연결 상태 확인 및 재연결"""
        if not self.connected:
            if not self.connect():
                raise ConnectionError("Alpaca API 서버에 연결할 수 없습니다.")
            
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
                "pattern_day_trader": account.pattern_day_trader  # 패턴 데이 트레이더로 간주되는지 여부 (True/False)
            }
            
            self.logger.info("계정 정보 조회 성공")
            return result
            
        except Exception as e:
            self.logger.error(f"계정 정보 조회 실패: {e}")
            return {"error": str(e)}
    
    
    def get_account_info(self) -> Dict[str, Any]:  # ✅ 여기에 추가
        """
        계정 정보 조회 (get_account의 별칭)
        
        Returns:
            계정 정보가 담긴 딕셔너리
        """
        return self.get_account()

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
def create_alpaca_client(market_hours_only: bool = True, logger: Optional[Logger] = None) -> APIConnector:
    """
    config 설정을 사용하여 APIConnector 인스턴스를 생성하는 팩토리 함수
    
    Args:
        market_hours_only: 시장 시간에만 연결 허용 여부 (기본값: True)
        logger: 로깅을 위한 Logger 인스턴스 (옵션)
        
    Returns:
        설정된 APIConnector 인스턴스
    """
    return APIConnector(logger=logger, market_hours_only=market_hours_only)


# 사용 예시
if __name__ == "__main__":
    # config 설정을 사용한 클라이언트 생성
    client = create_alpaca_client()
    
    print("=== 시장 상태 확인 ===")
    market_status = client.is_market_open()
    print(f"시장 열림 여부: {market_status.get('is_open', False)}")  
    
    # 연결
    if client.connect():
        # 계정 정보 조회
        account = client.get_account()
        print(f"계정 현금: ${account.get('cash', 0)}")
        
        # 포트폴리오 조회
        positions = client.get_all_positions()
        print(f"보유 포지션: {len(positions)}개")
        
        # WebSocket을 사용한 시장 데이터 조회
        print("\n=== WebSocket 시장 데이터 조회 ===")
        market_data = client.get_market_data("AAPL", exchange="NASDAQ", use_websocket=True, wait_time=5)
        print(f"AAPL WebSocket 데이터: {len(market_data)} 행")
        if not market_data.empty:
            print(market_data.tail())
        
        # 실시간 데이터 스트림 시작 (예시)
        def on_new_data(data):
            print(f"📊 실시간 데이터: {data['timestamp']} - AAPL 종가: ${data['close']}")
        
        print("\n=== 실시간 데이터 스트림 시작 ===")
        if client.start_market_data_stream("AAPL", data_callback=on_new_data):
            print("실시간 스트림이 시작되었습니다.")
            time.sleep(10)  # 10초 동안 데이터 수신
            client.stop_market_data_stream("AAPL")
            print("실시간 스트림을 중지했습니다.")
        
        # 연결 종료
        client.disconnect()
    else:
        print("❌ 연결 실패 - 시장이 닫혀있습니다.")
        
        print("\n=== 강제 연결 시도 ===")
        if client.connect(force_connect=True):
            print("✅ 강제 연결 성공!")
            
            # tvdatafeed를 사용한 데이터 조회
            print("\n=== tvdatafeed 시장 데이터 조회 ===")
            market_data = client.get_market_data("AAPL", use_websocket=False)
            print(f"AAPL tvdatafeed 데이터: {len(market_data)} 행")
            if not market_data.empty:
                print(market_data.tail())
            
            client.disconnect()
        else:
            print("❌ 강제 연결도 실패했습니다.")