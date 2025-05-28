#!/usr/bin/env python
# -*- coding: utf-8 -*-

# run_live_trading.py 기존 테이블 구조 호환 버전 - 심볼 해결 수정 + 장 마감 자동 종료

import os
import argparse
import torch
import signal
import sys
import time
import json
import pytz
from datetime import datetime, timedelta
from pathlib import Path

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.config import Config
from src.trading.api_connector import APIConnector
from src.trading.live_trader import LiveTrader
from src.trading.risk_manager import RiskManager
from src.trading.model_state_manager import create_complete_trading_system
from src.trading.data_validator import RealTimeDataValidator
from src.trading.database_manager import DatabaseManager
from typing import Dict, List


class EnhancedLiveTrader:
    """데이터베이스 연동이 포함된 향상된 실시간 트레이더 - 기존 테이블 호환 버전 + 장 마감 자동 종료"""
    
    def __init__(self, live_trader: LiveTrader, db_manager: DatabaseManager, 
                 symbol: str, model_id: str, session_id: str):
        self.live_trader = live_trader
        self.db_manager = db_manager
        self.symbol = symbol
        self.model_id = model_id
        self.session_id = session_id
        self.last_db_save_time = 0
        
        # 한국 시간대 설정
        self.korea_tz = pytz.timezone('Asia/Seoul')
        
        self.today_date = datetime.now().strftime("%Y-%m-%d")
        self.daily_start_portfolio_value = None
        self.session_start_time = datetime.now()
        
        # 초기 포트폴리오 값 설정
        try:
            state = self.live_trader.get_state()
            account = state.get('account', {})
            self.daily_start_portfolio_value = float(account.get('portfolio_value', 0))
            self.live_trader.logger.info(f"✅ 초기 포트폴리오 값 설정: ${self.daily_start_portfolio_value:,.2f}")
        except Exception as e:
            self.daily_start_portfolio_value = 0.0
            self.live_trader.logger.warning(f"⚠️ 초기 포트폴리오 값 설정 실패: {e}")
        
        # LiveTrader에 콜백 등록
        self._register_callbacks()
        
    def _register_callbacks(self):
        """LiveTrader에 거래 실행 콜백 등록"""
        # 기존 execute_trade 메서드를 래핑
        original_execute_trade = self.live_trader.execute_trade
        
        def enhanced_execute_trade(symbol, action):
            result = original_execute_trade(symbol, action)
            if result.get('success', False):
                # 거래 성공 시 DB 저장
                self.on_trade_executed(result)
            return result
        
        # 메서드 교체
        self.live_trader.execute_trade = enhanced_execute_trade
    
    def on_trade_executed(self, trade_info: Dict):
        """거래 실행 시 콜백 - 데이터베이스에 저장"""
        try:
            # 안전한 값 추출
            symbol = trade_info.get('symbol', self.symbol)
            side = trade_info.get('side', 'unknown')
            quantity = self._safe_float(trade_info.get('quantity', 0))
            price = self._safe_float(trade_info.get('price', 0))
            fee = self._safe_float(trade_info.get('fee', 0))
            
            # 거래 정보를 데이터베이스에 저장
            success = self.db_manager.save_trade(
                symbol=symbol,
                side=side,
                quantity=abs(quantity),
                price=price,
                fee=fee,
                pnl=None,  # 실시간에서는 pnl을 바로 계산하기 어려움
                model_id=self.model_id,
                session_id=self.session_id
            )
            
            if success:
                self.live_trader.logger.info(f"✅ 거래 DB 저장 성공: {symbol} {side} {quantity}@${price} (세션: {self.session_id})")
            else:
                self.live_trader.logger.warning(f"⚠️ 거래 DB 저장 실패: {symbol}")
            
            # 포지션 정보 업데이트 (약간의 지연 후)
            time.sleep(1)  # API 데이터 업데이트 대기
            self.update_position_in_db()
            
        except Exception as e:
            self.live_trader.logger.error(f"❌ 거래 DB 저장 중 오류: {e}")
    
    def update_position_in_db(self):
        """현재 포지션을 데이터베이스에 업데이트"""
        try:
            # API에서 직접 최신 포지션 정보 가져오기
            position = self.live_trader.api.get_position(self.symbol)
            
            if position:
                # 안전한 타입 변환
                qty = self._safe_float(position.get('qty', 0))
                
                # 다양한 키로 평균 진입가 시도
                avg_cost = (self._safe_float(position.get('avg_entry_price', 0)) or 
                           self._safe_float(position.get('avg_cost', 0)) or 
                           self._safe_float(position.get('average_entry_price', 0)) or
                           self._safe_float(position.get('cost_basis', 0)))
                
                current_price = self._safe_float(position.get('current_price', 0))
                market_value = self._safe_float(position.get('market_value', 0))
                
                # 다양한 키로 미실현 손익 시도
                unrealized_pl = (self._safe_float(position.get('unrealized_pnl', 0)) or 
                                self._safe_float(position.get('unrealized_pl', 0)) or
                                self._safe_float(position.get('unrealized_plpc', 0)))
                
                # 현재 가격이 0이면 시장 데이터에서 가져오기
                if current_price == 0 and abs(qty) > 0.001:
                    try:
                        market_data = self.live_trader.api.get_market_data(self.symbol, limit=1)
                        if not market_data.empty:
                            current_price = float(market_data.iloc[-1]['close'])
                    except:
                        pass
                
                # 평균 진입가가 0이고 수량이 있으면 현재 가격 사용
                if avg_cost == 0 and abs(qty) > 0.001 and current_price > 0:
                    avg_cost = current_price
                
                success = self.db_manager.save_position(
                    symbol=self.symbol,
                    quantity=qty,
                    avg_entry_price=avg_cost,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pl,
                    session_id=self.session_id
                )
                
                if success:
                    self.live_trader.logger.debug(f"✅ 포지션 DB 업데이트: {self.symbol} {qty}주 @${avg_cost:.2f}")
                else:
                    self.live_trader.logger.warning(f"⚠️ 포지션 DB 업데이트 실패: {self.symbol}")
                    
            else:
                # 포지션이 없는 경우 0으로 업데이트
                success = self.db_manager.save_position(
                    symbol=self.symbol,
                    quantity=0,
                    avg_entry_price=0,
                    current_price=0,
                    unrealized_pnl=0,
                    session_id=self.session_id
                )
                
                if success:
                    self.live_trader.logger.debug(f"✅ 포지션 0으로 DB 업데이트: {self.symbol}")
                
        except Exception as e:
            self.live_trader.logger.error(f"❌ 포지션 DB 업데이트 실패: {e}")

    def save_trading_stats_to_db(self):
        """거래 통계를 데이터베이스에 저장"""
        try:
            # 날짜 업데이트 확인
            self.update_today_date()
            
            state = self.live_trader.get_state()
            account = state.get('account', {})
            
            # 안전한 타입 변환
            portfolio_value = self._safe_float(account.get('portfolio_value', 0))
            cash_balance = self._safe_float(account.get('cash', 0))
            equity_value = self._safe_float(account.get('equity', portfolio_value))
            
            # 일일 손익 계산
            if self.daily_start_portfolio_value is None:
                self.daily_start_portfolio_value = portfolio_value
            
            daily_pnl = portfolio_value - self.daily_start_portfolio_value
            
            # 총 손익 계산
            trading_stats = state.get('trading_stats', {})
            initial_balance = self._safe_float(trading_stats.get('initial_balance', portfolio_value))
            total_pnl = portfolio_value - initial_balance
            
            success = self.db_manager.save_trading_stats(
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                equity_value=equity_value,
                daily_pnl=daily_pnl,
                total_pnl=total_pnl,
                session_id=self.session_id
            )
            
            if success:
                self.live_trader.logger.debug(f"✅ 통계 DB 저장: 포트폴리오=${portfolio_value:,.2f}, 일일손익=${daily_pnl:+.2f}")
            
        except Exception as e:
            self.live_trader.logger.error(f"❌ 통계 DB 저장 실패: {e}")
    
    def update_today_date(self):
        """날짜가 바뀌면 오늘 날짜 및 관련 값들 업데이트"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != self.today_date:
            self.today_date = current_date
            
            # 새로운 날이 시작되면 일일 시작 포트폴리오 값 재설정
            try:
                state = self.live_trader.get_state()
                account = state.get('account', {})
                self.daily_start_portfolio_value = self._safe_float(account.get('portfolio_value', 0))
                self.live_trader.logger.info(f"📅 새로운 날 시작: {self.today_date}, 시작 포트폴리오: ${self.daily_start_portfolio_value:,.2f}")
            except Exception as e:
                self.live_trader.logger.error(f"❌ 날짜 업데이트 중 오류: {e}")
    
    def _safe_float(self, value, default=0.0):
        """안전한 float 변환"""
        try:
            if value is None:
                return default
            if isinstance(value, str) and value.strip() == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_daily_summary(self):
        """일일 요약 정보 반환"""
        try:
            state = self.live_trader.get_state()
            account = state.get('account', {})
            
            current_portfolio = self._safe_float(account.get('portfolio_value', 0))
            daily_pnl = current_portfolio - (self.daily_start_portfolio_value or current_portfolio)
            daily_return_pct = (daily_pnl / (self.daily_start_portfolio_value or 1)) * 100
            
            trading_stats = state.get('trading_stats', {})
            total_trades = len(trading_stats.get('trades', []))
            successful_trades = trading_stats.get('successful_trades', 0)
            
            return {
                'date': self.today_date,
                'session_id': self.session_id,
                'start_portfolio': self.daily_start_portfolio_value,
                'current_portfolio': current_portfolio,
                'daily_pnl': daily_pnl,
                'daily_return_pct': daily_return_pct,
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'session_duration': str(datetime.now() - self.session_start_time).split('.')[0]
            }
        except Exception as e:
            self.live_trader.logger.error(f"❌ 일일 요약 생성 실패: {e}")
            return {}
    
    def start(self):
        """트레이딩 시작"""
        return self.live_trader.start()
    
    def stop(self, reason="수동 종료"):
        """트레이딩 중지"""
        return self.live_trader.stop(reason)
    
    def get_state(self):
        """상태 조회"""
        return self.live_trader.get_state()
    
    def save_trading_stats(self, path):
        """통계 파일 저장"""
        return self.live_trader.save_trading_stats(path)


def load_model_info(model_path: str) -> Dict:
    """
    모델 경로에서 상세 정보 로드 - 심볼 해결 개선
    """
    model_info = {
        'metadata': None,
        'config': None,
        'symbols': [],
        'model_type': 'MLP'
    }
    
    try:
        model_path = Path(model_path)
        
        # 1. model_metadata.json 로드
        metadata_path = model_path / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_info['metadata'] = json.load(f)
                symbols = model_info['metadata'].get('symbols', [])
                if symbols:
                    model_info['symbols'] = symbols
        
        # 2. config.pth 로드
        config_path = model_path / "config.pth"
        if config_path.exists():
            config_data = torch.load(config_path, map_location='cpu')
            model_info['config'] = config_data
            
            # 모델 타입 확인
            use_cnn = config_data.get('use_cnn', False)
            use_lstm = config_data.get('use_lstm', False)
            if use_cnn:
                model_info['model_type'] = 'CNN'
            elif use_lstm:
                model_info['model_type'] = 'LSTM'
            else:
                model_info['model_type'] = 'MLP'
                
            # 심볼 정보가 없으면 config에서 가져오기
            if not model_info['symbols']:
                symbols = config_data.get('symbols', [])
                if symbols:
                    model_info['symbols'] = symbols
        
        # 3. 심볼이 여전히 없으면 모델 경로에서 추론
        if not model_info['symbols']:
            # 모델 경로에서 심볼 추출 시도
            # 예: models/AAPL/final_sac_model_AAPL -> ['AAPL']
            path_parts = str(model_path).split(os.sep)
            
            # 일반적인 심볼들 목록
            common_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
            
            # 경로에서 심볼 찾기
            found_symbols = []
            for part in path_parts:
                if part.upper() in common_symbols:
                    found_symbols.append(part.upper())
            
            if found_symbols:
                model_info['symbols'] = list(set(found_symbols))  # 중복 제거
                print(f"   └─ 경로에서 심볼 추출: {model_info['symbols']}")
            else:
                # 마지막 수단: 기본 심볼 할당
                # 모델명에서 심볼 패턴 찾기
                model_name = model_path.name
                for symbol in common_symbols:
                    if symbol in model_name.upper():
                        model_info['symbols'] = [symbol]
                        print(f"   └─ 모델명에서 심볼 추출: {symbol}")
                        break
                
                # 여전히 없으면 config의 TRADING_SYMBOLS 사용
                if not model_info['symbols']:
                    try:
                        from src.config.config import TRADING_SYMBOLS
                        if TRADING_SYMBOLS:
                            model_info['symbols'] = TRADING_SYMBOLS[:1]  # 첫 번째 심볼만 사용
                            print(f"   └─ config에서 기본 심볼 할당: {model_info['symbols']}")
                        else:
                            # 최후의 수단: AAPL 할당
                            model_info['symbols'] = ['AAPL']
                            print(f"   └─ 기본 심볼 할당: AAPL")
                    except:
                        model_info['symbols'] = ['AAPL']
                        print(f"   └─ 기본 심볼 할당: AAPL")
        
        print(f"📊 최종 심볼 목록: {model_info['symbols']}")
        
        return model_info
        
    except Exception as e:
        print(f"⚠️ 모델 정보 로드 실패: {e}")
        # 오류 발생 시에도 기본 심볼 할당
        if not model_info['symbols']:
            try:
                from src.config.config import TRADING_SYMBOLS
                if TRADING_SYMBOLS:
                    model_info['symbols'] = TRADING_SYMBOLS[:1]
                else:
                    model_info['symbols'] = ['AAPL']
            except:
                model_info['symbols'] = ['AAPL']
        
        return model_info


def extract_symbol_from_path(model_path: str) -> str:
    """
    모델 경로에서 심볼을 추출하는 헬퍼 함수
    """
    path_str = str(model_path).upper()
    common_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    for symbol in common_symbols:
        if symbol in path_str:
            return symbol
    
    return 'AAPL'  # 기본값


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='SAC 모델 실시간 트레이딩 실행 (기존 테이블 호환 + 장 마감 자동 종료)')
    
    parser.add_argument('--model_path', type=str, required=False,
                        help='백테스팅 완료된 모델의 경로 (지정하지 않으면 config.py의 설정 사용)')
    parser.add_argument('--results_dir', type=str, default='results/live_trading',
                        help='결과 저장 디렉토리')
    parser.add_argument('--log_interval', type=int, default=300,
                        help='로깅 간격 (초, 기본값: 5분)')
    parser.add_argument('--save_interval', type=int, default=1800,
                        help='통계 저장 간격 (초, 기본값: 30분)')
    parser.add_argument('--db_save_interval', type=int, default=60,
                        help='데이터베이스 저장 간격 (초, 기본값: 1분)')
    parser.add_argument('--dry_run', action='store_true',
                        help='실제 거래 없이 시뮬레이션만 실행')
    parser.add_argument('--force_connect', action='store_true',
                    help='시장 시간과 관계없이 연결하려면 force_connect=True')
    parser.add_argument('--disable_auto_stop', action='store_true',
                        help='장 마감 자동 종료 비활성화')
    parser.add_argument('--market_close_time', type=str, default='05:00',
                        help='장 마감 시간 (한국시간, HH:MM 형식, 기본값: 05:00)')
    parser.add_argument('--db_host', type=str, default='192.168.40.199',
                        help='MySQL 서버 호스트')
    parser.add_argument('--db_name', type=str, default='trading',
                        help='데이터베이스 이름')
    parser.add_argument('--db_user', type=str, default='root',
                        help='데이터베이스 사용자')
    parser.add_argument('--db_password', type=str, default='mysecretpassword',
                        help='데이터베이스 비밀번호')
    
    return parser.parse_args()


def setup_logger(results_dir):
    """로거 설정"""
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    import logging
    logger = logging.getLogger('live_trading')
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_signal_handlers(live_traders: Dict[str, EnhancedLiveTrader], db_manager: DatabaseManager, 
                         logger, args, session_id: str, symbol_to_model_mapping: Dict[str, str], 
                         model_info_dict: Dict[str, Dict]):
    """시그널 핸들러 설정 (종료 시 정리 작업)"""
    def signal_handler(signum, frame):
        if signum in [signal.SIGINT, signal.SIGTERM]:
            print("🛑 종료 신호를 받았습니다. 트레이딩을 안전하게 중지합니다...")
            
            # 트레이딩 중지
            for symbol, live_trader in live_traders.items():
                if live_trader.stop("강제 종료"):
                    logger.info(f"✅ {symbol} 트레이딩이 성공적으로 중지되었습니다.")
                else:
                    logger.error(f"❌ {symbol} 트레이딩 중지 중 오류가 발생했습니다.")
            
            # 모델 정보 저장
            logger.info("💾 강제 종료 - 모델 정보 저장 중...")
            try:
                for symbol, live_trader in live_traders.items():
                    model_path = symbol_to_model_mapping.get(symbol, '')
                    
                    if model_path:
                        model_info = model_info_dict.get(model_path, {})
                        model_id = os.path.splitext(os.path.basename(model_path))[0]
                        final_summary = live_trader.get_daily_summary()
                        
                        success = db_manager.save_model_info_detailed(
                            model_id=f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            file_path=model_path,
                            symbols=[symbol],
                            description=f"강제 종료된 모델 - {symbol} ({model_info.get('model_type', 'Unknown')}) - {final_summary.get('total_trades', 0)}회 거래",
                            is_active=False,
                            model_metadata={
                                **model_info.get('metadata', {}),
                                'trading_results': final_summary,
                                'session_id': session_id,
                                'forced_termination': True,
                                'completion_time': datetime.now().isoformat()
                            },
                            config_info=model_info.get('config')
                        )
                        
                        if success:
                            logger.info(f"✅ {symbol} 모델 정보 저장 완료: {model_id}")
                            
            except Exception as e:
                logger.error(f"❌ 강제 종료 시 모델 정보 저장 실패: {e}")
            
            # 최종 DB 저장
            for symbol, live_trader in live_traders.items():
                try:
                    live_trader.save_trading_stats_to_db()
                    live_trader.update_position_in_db()
                except Exception as e:
                    logger.error(f"❌ {symbol} 최종 DB 저장 실패: {e}")
            
            # 세션 종료 정보 저장
            try:
                final_stats = {}
                if live_traders:
                    first_trader = next(iter(live_traders.values()))
                    summary = first_trader.get_daily_summary()
                    final_stats = {
                        'total_trades': summary.get('total_trades', 0),
                        'successful_trades': summary.get('successful_trades', 0),
                        'return_pct': summary.get('daily_return_pct', 0),
                        'forced_termination': True
                    }
                
                db_manager.update_trading_session_status(session_id, 'STOPPED', final_stats)
                logger.info(f"🏁 트레이딩 세션 종료 정보 저장 완료: {session_id}")
                
            except Exception as e:
                logger.error(f"❌ 세션 종료 정보 저장 실패: {e}")
            
            # 트레이딩 통계 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for symbol, live_trader in live_traders.items():
                results_path = os.path.join(args.results_dir, f"{symbol}_final_trading_stats_{timestamp}.json")
                
                if live_trader.save_trading_stats(results_path):
                    logger.info(f"📊 {symbol} 최종 트레이딩 통계가 저장되었습니다: {results_path}")
            
            # 데이터베이스 연결 종료
            db_manager.disconnect()
            
            logger.info("👋 프로그램을 종료합니다.")
            os._exit(0)
    
    # Ctrl+C (SIGINT) 및 SIGTERM 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_environment(config, logger):
    """실행 환경 검증"""
    logger.info("🔍 실행 환경 검증 중...")
    
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        logger.info(f"✅ GPU 사용 가능: {torch.cuda.get_device_name()}")
    else:
        logger.info("⚠️  GPU를 사용할 수 없습니다. CPU를 사용합니다.")
    
    # 설정 검증
    required_configs = ['trading_symbols', 'trading_interval', 'max_position_size']
    for config_name in required_configs:
        if not hasattr(config, config_name):
            logger.error(f"❌ 필수 설정이 누락되었습니다: {config_name}")
            return False
        value = getattr(config, config_name)
        if value is None:
            logger.error(f"❌ 필수 설정 {config_name} 값이 None입니다.")
            return False
    
    logger.info("✅ 실행 환경 검증 완료")
    return True


def main():
    """메인 함수"""
    global args
    args = parse_args()
    
    print("🚀 SAC 실시간 트레이딩 시스템 시작 (기존 테이블 호환 + 장 마감 자동 종료)")
    print("=" * 80)
    
    # 한국 시간대 설정
    korea_tz = pytz.timezone('Asia/Seoul')
    korea_now = datetime.now(korea_tz)
    
    # 결과 디렉토리 설정
    results_dir = os.path.join(args.results_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(results_dir)
    logger.info("🎯 SAC 실시간 트레이딩 시작 (기존 테이블 호환 + 장 마감 자동 종료)")
    logger.info(f"📁 결과 저장 경로: {results_dir}")
    logger.info(f"🕐 현재 한국시간: {korea_now.strftime('%Y-%m-%d %H:%M:%S')} KST")
    
    # 장 마감 자동 종료 설정
    if args.disable_auto_stop:
        logger.info("⚠️ 장 마감 자동 종료가 비활성화되었습니다.")
    else:
        logger.info(f"🕐 장 마감 자동 종료 설정: {args.market_close_time} KST")
    
    if args.dry_run:
        logger.info("🔄 DRY RUN 모드: 실제 거래는 실행되지 않습니다.")
    
    # 설정 로드
    config = Config()
    
    # 환경 검증
    if not validate_environment(config, logger):
        logger.error("❌ 환경 검증 실패. 프로그램을 종료합니다.")
        sys.exit(1)
        
    # 세션 ID 생성
    session_id = f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"🆔 트레이딩 세션 ID: {session_id}")
        
    # 데이터베이스 매니저 초기화
    logger.info("🗄️  MySQL 데이터베이스 연결 중...")
    db_manager = DatabaseManager(
        host=args.db_host,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
        logger=logger
    )
    
    if not db_manager.connect():
        logger.error("❌ 데이터베이스 연결 실패. 프로그램을 종료합니다.")
        sys.exit(1) 
        
    try:
        # 모델 경로 결정 및 심볼 매핑 개선
        if args.model_path:
            # 명령행 인자로 지정된 모델 경로 사용
            model_paths = {args.model_path: {'enabled': True}}
            logger.info(f"🤖 지정된 모델 경로 사용: {args.model_path}")
        else:
            # config.py의 ACTIVE_TRADING_SYMBOLS 사용
            model_paths = {}
            for symbol, data in config.ACTIVE_TRADING_SYMBOLS.items():
                if data.get('enabled', True):
                    model_path = data.get('model_path', '')
                    if model_path and os.path.exists(os.path.join(model_path, 'config.pth')):
                        model_paths[model_path] = {
                            'enabled': True,
                            'preferred_symbol': symbol,  # 선호하는 심볼 명시
                            **data
                        }
            
            logger.info(f"🤖 config.py에서 활성화된 {len(model_paths)}개 모델 사용")
        
        if not model_paths:
            logger.error("❌ 활성화된 모델이 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
        
        # 모델 정보 로드 및 심볼 매핑
        model_info_dict = {}
        symbol_to_model_mapping = {}  # 심볼 -> 모델 경로 매핑
        
        logger.info("📊 모델 정보 로드 중...")
        for model_path, model_config in model_paths.items():
            logger.info(f"🔍 모델 분석 중: {model_path}")
            
            # 모델 정보 로드
            model_info = load_model_info(model_path)
            model_info_dict[model_path] = model_info
            
            # 심볼 매핑 생성
            if model_info['symbols']:
                # 메타데이터에서 찾은 심볼들 사용
                for symbol in model_info['symbols']:
                    symbol_to_model_mapping[symbol] = model_path
                    logger.info(f"   └─ {symbol} → {model_path}")
            elif 'preferred_symbol' in model_config:
                # config에서 지정한 선호 심볼 사용
                preferred_symbol = model_config['preferred_symbol']
                symbol_to_model_mapping[preferred_symbol] = model_path
                model_info['symbols'] = [preferred_symbol]  # 모델 정보 업데이트
                logger.info(f"   └─ {preferred_symbol} → {model_path} (선호 심볼)")
            else:
                # 모델 경로에서 심볼 추출
                extracted_symbol = extract_symbol_from_path(model_path)
                symbol_to_model_mapping[extracted_symbol] = model_path
                model_info['symbols'] = [extracted_symbol]  # 모델 정보 업데이트
                logger.info(f"   └─ {extracted_symbol} → {model_path} (경로에서 추출)")
        
        logger.info(f"📋 최종 심볼-모델 매핑: {symbol_to_model_mapping}")
        
        # API 커넥터 초기화
        logger.info("🔌 Alpaca API 연결 중...")
        api_connector = APIConnector(logger=logger)
        if args.force_connect:
            api_connector.force_connect()
            logger.info("✅ API 서버 강제 연결 성공")

        if not api_connector.connect():
            logger.error("❌ API 서버에 연결할 수 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)

        # 계정 정보 확인
        account_info = api_connector.get_account_info()
        logger.info(f"💰 계정 정보:")
        logger.info(f"   └─ 현금: ${account_info.get('cash', 0):,.2f}")
        logger.info(f"   └─ 포트폴리오 가치: ${account_info.get('portfolio_value', 0):,.2f}")
        logger.info(f"   └─ 매수력: ${account_info.get('buying_power', 0):,.2f}")
        
        # 데이터 검증 시스템 초기화
        data_validator = RealTimeDataValidator(logger=logger)
        
        # 리스크 관리자 초기화
        risk_manager = RiskManager(
            max_position_size=config.max_position_size,
            max_drawdown=config.max_drawdown,
            max_trade_amount=config.max_trade_amount,
            max_daily_loss=config.max_daily_loss,
            logger=logger
        )
        
        # 초기 계정 자본금으로 리스크 관리자 업데이트
        initial_balance = float(account_info.get('cash', 0))
        risk_manager.update_balance(initial_balance)

        # 간단한 세션 시작 로그만 저장
        logger.info(f"🚀 트레이딩 세션 시작: {session_id}")
        logger.info(f"📊 대상 심볼: {len(symbol_to_model_mapping)}개")
        
        # 실시간 트레이더 초기화 - 심볼 기반으로 생성
        live_traders = {}
        missing_models = []

        for symbol, model_path in symbol_to_model_mapping.items():
            logger.info(f"🏗️  {symbol} 실시간 트레이더 초기화 중...")
            
            try:
                # 백테스팅 완료된 모델 시스템 생성
                agent, data_processor = create_complete_trading_system(model_path, config)
                
                # config 업데이트 - 현재 심볼을 trading_symbols에 설정
                config.trading_symbols = [symbol]
                
                # 실시간 트레이더 초기화
                live_trader = LiveTrader(
                    agent=agent,
                    api_connector=api_connector,
                    config=config,
                    logger=logger,
                    data_processor=data_processor,
                    risk_manager=risk_manager
                )
                
                # 장 마감 자동 종료 설정 적용
                if args.disable_auto_stop:
                    live_trader.auto_stop_on_market_close = False
                else:
                    live_trader.auto_stop_on_market_close = True
                    live_trader.market_close_time_kst = args.market_close_time
                
                # 데이터 검증 시스템 연동
                live_trader.data_validator = data_validator
                
                # 모델 ID 생성 (파일명 기반)
                model_id = os.path.splitext(os.path.basename(model_path))[0]
                
                # 향상된 트레이더로 래핑
                enhanced_live_trader = EnhancedLiveTrader(
                    live_trader=live_trader,
                    db_manager=db_manager,
                    symbol=symbol,
                    model_id=model_id,
                    session_id=session_id
                )
            
                live_traders[symbol] = enhanced_live_trader
                
                model_info = model_info_dict.get(model_path, {})
                logger.info(f"✅ {symbol} 실시간 트레이더 초기화 완료 (모델: {model_info.get('model_type', 'Unknown')})")
                
                # 장 마감 자동 종료 설정 로그
                if live_trader.auto_stop_on_market_close:
                    logger.info(f"   └─ 장 마감 자동 종료: {live_trader.market_close_time_kst} KST")
                else:
                    logger.info(f"   └─ 장 마감 자동 종료: 비활성화")
                
            except FileNotFoundError as e:
                logger.warning(f"⚠️  {symbol} 모델을 찾을 수 없습니다: {e}")
                missing_models.append(symbol)
            except Exception as e:
                logger.error(f"❌ {symbol} 모델 초기화 중 오류 발생: {e}")
                import traceback
                logger.error(traceback.format_exc())
                missing_models.append(symbol)
        
        if missing_models:
            logger.warning(f"⚠️  다음 심볼의 모델이 없어 거래가 제외됩니다: {', '.join(missing_models)}")

        if not live_traders:
            logger.error("❌ 초기화된 트레이딩 시스템이 없습니다. 프로그램을 종료합니다.")
            logger.error("💡 문제 해결 방법:")
            logger.error("   1. 모델 경로가 올바른지 확인하세요")
            logger.error("   2. 모델 파일(config.pth)이 존재하는지 확인하세요")
            logger.error("   3. config.py의 ACTIVE_TRADING_SYMBOLS 설정을 확인하세요")
            logger.error(f"   4. 현재 심볼-모델 매핑: {symbol_to_model_mapping}")
            sys.exit(1)
        
        logger.info(f"✅ {len(live_traders)}개 심볼에 대한 트레이딩 시스템이 초기화되었습니다: {', '.join(live_traders.keys())}")
        
        # 모델 사용 통계 출력
        logger.info("🤖 사용된 모델 정보:")
        for symbol, trader in live_traders.items():
            model_path = symbol_to_model_mapping.get(symbol, '')
            model_info = model_info_dict.get(model_path, {})
            logger.info(f"   └─ {symbol}: {model_info.get('model_type', 'Unknown')} 모델 ({model_path})")
        
        # 시그널 핸들러 설정
        setup_signal_handlers(live_traders, db_manager, logger, args, session_id, 
                            symbol_to_model_mapping, model_info_dict)
        
        # 실시간 트레이딩 시작
        logger.info("🚀 실시간 트레이딩 시작...")
        logger.info(f"📈 거래 대상: {', '.join(live_traders.keys())}")
        logger.info(f"🆔 세션 ID: {session_id}")
        
        # 장 마감 자동 종료 상태 출력
        if not args.disable_auto_stop:
            logger.info(f"🕐 장 마감 자동 종료: {args.market_close_time} KST에 자동 종료됩니다")
        else:
            logger.info("⚠️ 장 마감 자동 종료: 비활성화됨")
        
        successful_starts = 0
        for symbol, live_trader in live_traders.items():
            if live_trader.start():
                successful_starts += 1
                logger.info(f"✅ {symbol} 실시간 트레이딩 시작 성공")
            else:
                logger.error(f"❌ {symbol} 실시간 트레이딩을 시작할 수 없습니다.")
        
        if successful_starts == 0:
            logger.error("❌ 모든 트레이딩 시작에 실패했습니다.")
            sys.exit(1)
        
        logger.info(f"🎉 {successful_starts}/{len(live_traders)}개 심볼 트레이딩 시작 완료")
        
        # 주기적인 모니터링 루프 (장 마감 체크 포함)
        last_log_time = time.time()
        last_save_time = time.time()
        last_db_save_time = time.time()
        last_risk_check_time = time.time()
        last_model_info_log_time = time.time()
        last_market_status_log_time = time.time()
        
        logger.info("🔄 모니터링 루프 시작")
        
        try:
            while True:
                current_time = time.time()
                
                # 트레이딩 시스템이 모두 중지되었는지 확인 (장 마감 자동 종료 체크)
                all_stopped = True
                for symbol, live_trader in live_traders.items():
                    state = live_trader.get_state()
                    if state.get('running', False):
                        all_stopped = False
                        break
                
                if all_stopped:
                    logger.info("🏁 모든 트레이딩 시스템이 중지되었습니다. (장 마감 자동 종료)")
                    break
                
                # 시장 상태 주기적 로깅 (10분마다)
                if current_time - last_market_status_log_time >= 600:
                    korea_now = datetime.now(korea_tz)
                    market_status = api_connector.is_market_open()
                    
                    if market_status.get('is_open', False):
                        logger.info(f"🟢 시장 상태: 개장 중 ({korea_now.strftime('%Y-%m-%d %H:%M:%S')} KST)")
                    else:
                        next_open = market_status.get('next_open')
                        if next_open:
                            next_open_kr = next_open.astimezone(korea_tz)
                            logger.info(f"🔴 시장 상태: 폐장 중 ({korea_now.strftime('%Y-%m-%d %H:%M:%S')} KST)")
                            logger.info(f"   └─ 다음 개장: {next_open_kr.strftime('%Y-%m-%d %H:%M')} KST")
                    
                    # 장 마감까지 남은 시간 계산
                    if not args.disable_auto_stop:
                        try:
                            close_time_str = args.market_close_time
                            close_hour, close_minute = map(int, close_time_str.split(':'))
                            
                            # 오늘의 장 마감 시간
                            today_close = korea_now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
                            
                            # 만약 현재 시간이 장 마감 시간을 지났다면 다음 날 장 마감 시간으로 설정
                            if korea_now >= today_close:
                                today_close += timedelta(days=1)
                            
                            time_until_close = today_close - korea_now
                            hours, remainder = divmod(int(time_until_close.total_seconds()), 3600)
                            minutes, _ = divmod(remainder, 60)
                            
                            logger.info(f"⏰ 장 마감까지: {hours}시간 {minutes}분")
                            
                        except Exception as e:
                            logger.debug(f"장 마감 시간 계산 오류: {e}")
                    
                    last_market_status_log_time = current_time
                
                # 데이터베이스 저장 간격마다 통계 저장
                if current_time - last_db_save_time >= args.db_save_interval:
                    for symbol, live_trader in live_traders.items():
                        try:
                            live_trader.save_trading_stats_to_db()
                            live_trader.update_position_in_db()
                        except Exception as e:
                            logger.error(f"❌ {symbol} DB 저장 실패: {e}")
                    
                    last_db_save_time = current_time
                
                # 로깅 간격마다 상태 로깅
                if current_time - last_log_time >= args.log_interval:
                    korea_now = datetime.now(korea_tz)
                    
                    for symbol, live_trader in live_traders.items():
                        state = live_trader.get_state()
                        
                        logger.info("=" * 50)
                        logger.info(f"📊 {symbol} 현재 트레이딩 상태 (세션: {session_id})")
                        logger.info(f"🕐 한국시간: {korea_now.strftime('%Y-%m-%d %H:%M:%S')} KST")
                        logger.info("=" * 50)
                        logger.info(f"🔄 실행 상태: {'✅ 실행 중' if state['running'] else '❌ 중지됨'}")
                        
                        # 장 마감 자동 종료 상태
                        market_settings = state.get('market_close_settings', {})
                        if market_settings.get('auto_stop_enabled', False):
                            close_time = market_settings.get('market_close_time_kst', '05:00')
                            checked = market_settings.get('market_close_checked', False)
                            logger.info(f"🕐 장 마감 자동 종료: {close_time} KST {'(체크됨)' if checked else '(대기중)'}")
                        else:
                            logger.info("🕐 장 마감 자동 종료: 비활성화")
                        
                        logger.info(f"💰 계정 현금: ${state['account'].get('cash', 0):,.2f}")
                        logger.info(f"📈 포트폴리오: ${state['account'].get('portfolio_value', 0):,.2f}")
                        logger.info(f"🔢 총 거래: {len(state['trading_stats']['trades'])}회")
                        logger.info(f"✅ 성공 거래: {state['trading_stats']['successful_trades']}회")
                        logger.info(f"❌ 실패 거래: {state['trading_stats']['failed_trades']}회")
                        
                        # 수익률 계산
                        initial_balance = state['trading_stats']['initial_balance']
                        current_balance = state['trading_stats']['current_balance']
                        daily_start_balance = state['trading_stats'].get('daily_start_balance', initial_balance)
                        current_portfolio = state['account'].get('portfolio_value', 0)
                        
                        if initial_balance > 0:
                            session_return_pct = ((current_portfolio - initial_balance) / initial_balance) * 100
                            logger.info(f"📊 세션 수익률: {session_return_pct:+.2f}%")
                        
                        if daily_start_balance > 0:
                            daily_return_pct = ((current_portfolio - daily_start_balance) / daily_start_balance) * 100
                            logger.info(f"📅 일일 수익률: {daily_return_pct:+.2f}%")
                        
                        # 포지션 정보
                        positions = state.get('positions', {})
                        if positions:
                            logger.info(f"🏢 현재 포지션: {len(positions)}개")
                            for pos_symbol, position in positions.items():
                                qty = position.get('qty', 0)
                                if abs(qty) > 0.001:  # 의미있는 포지션만 표시
                                    unrealized_pl = position.get('unrealized_pl', 0)
                                    logger.info(f"   └─ {pos_symbol}: {qty:+.2f}주, 미실현손익: ${unrealized_pl:+,.2f}")
                        else:
                            logger.info("🏢 현재 포지션: 없음")
                        
                        logger.info("=" * 50)
                    
                    # 데이터베이스에서 최근 통계 조회해서 로깅
                    try:
                        recent_stats = db_manager.get_latest_trading_stats(limit=1)
                        if recent_stats:
                            stat = recent_stats[0]
                            logger.info("💾 DB 저장된 최신 통계:")
                            logger.info(f"   └─ 포트폴리오: ${stat['portfolio_value']:,.2f}")
                            logger.info(f"   └─ 현금: ${stat['cash_balance']:,.2f}")
                            logger.info(f"   └─ 주식가치: ${stat['equity_value']:,.2f}")
                            logger.info(f"   └─ 총손익: ${stat['total_pnl']:+,.2f}")
                    except Exception as e:
                        logger.debug(f"DB 통계 조회 실패: {e}")
            
                    last_log_time = current_time
                
                # 모델 정보 주기적 로깅 (30분마다)
                if current_time - last_model_info_log_time >= 1800:  # 30분
                    try:
                        logger.info("🤖 모델 사용 통계:")
                        model_stats = db_manager.get_model_usage_stats()
                        for stat in model_stats:
                            if stat.get('is_active'):
                                model_id = stat.get('model_id', 'Unknown')
                                total_trades = stat.get('total_trades', 0)
                                last_trade = stat.get('last_trade_time', 'Never')
                                logger.info(f"   └─ {model_id}: {total_trades}회 거래, 마지막: {last_trade}")
                        
                        # 활성 세션 정보
                        sessions = db_manager.get_trading_sessions(limit=5)
                        logger.info(f"📋 최근 트레이딩 세션: {len(sessions)}개")
                        for session in sessions:
                            session_id_from_db = session.get('model_id', '').replace('session_', '')
                            is_active = "✅ 활성" if session.get('is_active') else "❌ 종료"
                            logger.info(f"   └─ {session_id_from_db}: {is_active}")
                            
                    except Exception as e:
                        logger.debug(f"모델 통계 조회 실패: {e}")
                    
                    last_model_info_log_time = current_time
                
                # 저장 간격마다 통계 저장
                if current_time - last_save_time >= args.save_interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    for symbol, live_trader in live_traders.items():
                        results_path = os.path.join(results_dir, f"{symbol}_trading_stats_{timestamp}.json")
                        
                        if live_trader.save_trading_stats(results_path):
                            logger.info(f"💾 {symbol} 통계 저장 완료: {results_path}")
                    
                    last_save_time = current_time
                
                # 리스크 체크 (5분마다)
                if current_time - last_risk_check_time >= 300:
                    account_info = api_connector.get_account_info()
                    current_balance = float(account_info.get('cash', 0))
                    
                    risk_status = risk_manager.check_risk_limits(current_balance)
                    
                    if not risk_status.get('trade_allowed', True):
                        logger.warning("⚠️  리스크 한도 초과!")
                        for warning in risk_status.get('warnings', []):
                            logger.warning(f"   └─ {warning}")
                    
                    last_risk_check_time = current_time
                
                # CPU 사용률 절약
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("👤 사용자에 의해 중단되었습니다.")
            
        except Exception as e:
            logger.error(f"❌ 실행 중 예상치 못한 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        finally:
            # 정리 작업
            logger.info("🧹 정리 작업 시작...")
            
            logger.info("💾 사용된 모델 정보를 데이터베이스에 저장 중...")
            try:
                for symbol, live_trader in live_traders.items():
                    # 해당 심볼에 사용된 모델 경로 찾기
                    model_path = symbol_to_model_mapping.get(symbol, '')
                    
                    if model_path:
                        # 모델 정보 로드
                        model_info = model_info_dict.get(model_path, {})
                        model_id = os.path.splitext(os.path.basename(model_path))[0]
                        
                        # 트레이딩 통계 가져오기
                        final_summary = live_trader.get_daily_summary()
                        
                        # 완전한 모델 정보로 데이터베이스에 저장
                        success = db_manager.save_model_info_detailed(
                            model_id=f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            file_path=model_path,
                            symbols=[symbol],
                            description=f"트레이딩 완료 모델 - {symbol} ({model_info.get('model_type', 'Unknown')}) - {final_summary.get('total_trades', 0)}회 거래",
                            is_active=False,  # 종료된 모델이므로 비활성화
                            model_metadata={
                                **model_info.get('metadata', {}),
                                'trading_results': final_summary,
                                'session_id': session_id,
                                'completion_time': datetime.now().isoformat(),
                                'auto_stopped_by_market_close': all_stopped  # 장 마감 자동 종료 여부
                            },
                            config_info=model_info.get('config')
                        )
                        
                        if success:
                            logger.info(f"✅ {symbol} 모델 정보 저장 완료: {model_id}")
                            logger.info(f"   └─ 거래 횟수: {final_summary.get('total_trades', 0)}회")
                            logger.info(f"   └─ 수익률: {final_summary.get('daily_return_pct', 0):.2f}%")
                        else:
                            logger.warning(f"⚠️ {symbol} 모델 정보 저장 실패")
                            
            except Exception as e:
                logger.error(f"❌ 모델 정보 저장 중 오류: {e}")
            
            for symbol, live_trader in live_traders.items():
                if live_trader.stop("정상 종료"):
                    logger.info(f"✅ {symbol} 트레이딩 중지 완료")
                    
                # 최종 통계를 데이터베이스에 저장
                try:
                    live_trader.save_trading_stats_to_db()
                    live_trader.update_position_in_db()
                except Exception as e:
                    logger.error(f"❌ {symbol} 최종 DB 저장 실패: {e}")
                
                # 최종 통계 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_results_path = os.path.join(results_dir, f"{symbol}_final_trading_stats_{timestamp}.json")
                
                if live_trader.save_trading_stats(final_results_path):
                    logger.info(f"💾 {symbol} 최종 통계 저장: {final_results_path}")
            
            # 세션 종료 정보 저장
            try:
                # 최종 통계 계산
                final_stats = {}
                if live_traders:
                    first_trader = next(iter(live_traders.values()))
                    summary = first_trader.get_daily_summary()
                    final_stats = {
                        'total_trades': summary.get('total_trades', 0),
                        'successful_trades': summary.get('successful_trades', 0),
                        'return_pct': summary.get('daily_return_pct', 0),
                        'auto_stopped_by_market_close': all_stopped  # 장 마감 자동 종료 여부
                    }
                
                status = 'AUTO_STOPPED' if all_stopped else 'STOPPED'
                db_manager.update_trading_session_status(session_id, status, final_stats)
                logger.info(f"🏁 트레이딩 세션 종료 정보 저장 완료: {session_id} ({status})")
                
            except Exception as e:
                logger.error(f"❌ 세션 종료 정보 저장 실패: {e}")
            
            # API 연결 종료
            api_connector.disconnect()
            logger.info("🔌 API 연결 종료")
            
            db_manager.disconnect()
        
            logger.info("✅ 모든 정리 작업 완료")
            
            # 최종 종료 메시지
            if all_stopped:
                logger.info("🕐 SAC 실시간 트레이딩 시스템이 장 마감으로 자동 종료되었습니다")
            else:
                logger.info("👋 SAC 실시간 트레이딩 시스템 종료")
            
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 데이터베이스 연결 정리 및 세션 종료 처리
        try:
            db_manager.update_trading_session_status(session_id, 'ERROR', {'error': str(e)})
            db_manager.disconnect()
        except:
            pass
        
        sys.exit(1)
        
    
if __name__ == "__main__":
    main()