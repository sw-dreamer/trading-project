#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import signal
import sys
import time
from datetime import datetime

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
    """데이터베이스 연동이 포함된 향상된 실시간 트레이더"""
    
    def __init__(self, live_trader: LiveTrader, db_manager: DatabaseManager, 
                 symbol: str, model_id: str):
        self.live_trader = live_trader
        self.db_manager = db_manager
        self.symbol = symbol
        self.model_id = model_id
        self.last_db_save_time = 0
        
    def on_trade_executed(self, trade_info: Dict):
        """거래 실행 시 콜백 - 데이터베이스에 저장"""
        try:
            # 거래 정보를 데이터베이스에 저장
            self.db_manager.save_trade(
                symbol=trade_info.get('symbol', self.symbol),
                side=trade_info.get('side'),  # 'buy' or 'sell'
                quantity=abs(trade_info.get('quantity', 0)),
                price=trade_info.get('price', 0),
                fee=trade_info.get('fee', 0),
                pnl=trade_info.get('pnl'),
                model_id=self.model_id
            )
            
            # 포지션 정보 업데이트
            self.update_position_in_db()
            
        except Exception as e:
            self.live_trader.logger.error(f"❌ 거래 DB 저장 실패: {e}")
    
    def update_position_in_db(self):
        """현재 포지션을 데이터베이스에 업데이트"""
        try:
            state = self.live_trader.get_state()
            positions = state.get('positions', {})
            
            if self.symbol in positions:
                position = positions[self.symbol]
                
                self.db_manager.save_position(
                    symbol=self.symbol,
                    quantity=float(position.get('qty', 0)),
                    avg_entry_price=float(position.get('avg_cost', 0)),
                    current_price=float(position.get('market_value', 0)) / max(abs(float(position.get('qty', 1))), 0.001),
                    unrealized_pnl=float(position.get('unrealized_pl', 0))
                )
            else:
                # 포지션이 없는 경우 0으로 업데이트
                self.db_manager.save_position(
                    symbol=self.symbol,
                    quantity=0,
                    avg_entry_price=0,
                    current_price=0,
                    unrealized_pnl=0
                )
                
        except Exception as e:
            self.live_trader.logger.error(f"❌ 포지션 DB 업데이트 실패: {e}")
    
    def save_trading_stats_to_db(self):
        """거래 통계를 데이터베이스에 저장"""
        try:
            state = self.live_trader.get_state()
            account = state.get('account', {})
            
            portfolio_value = float(account.get('portfolio_value', 0))
            cash_balance = float(account.get('cash', 0))
            equity_value = portfolio_value - cash_balance
            
            # 일일 손익 계산 (간단히 전체 손익으로 대체)
            trading_stats = state.get('trading_stats', {})
            initial_balance = trading_stats.get('initial_balance', cash_balance)
            total_pnl = portfolio_value - initial_balance
            
            self.db_manager.save_trading_stats(
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                equity_value=equity_value,
                daily_pnl=0,  # 일일 손익은 별도 계산 로직 필요
                total_pnl=total_pnl
            )
            
        except Exception as e:
            self.live_trader.logger.error(f"❌ 통계 DB 저장 실패: {e}")
    
    def start(self):
        """트레이딩 시작"""
        return self.live_trader.start()
    
    def stop(self):
        """트레이딩 중지"""
        return self.live_trader.stop()
    
    def get_state(self):
        """상태 조회"""
        return self.live_trader.get_state()
    
    def save_trading_stats(self, path):
        """통계 파일 저장"""
        return self.live_trader.save_trading_stats(path)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='SAC 모델 실시간 트레이딩 실행 (DB 연동 포함)')
    
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


def setup_signal_handlers(live_traders: Dict[str, EnhancedLiveTrader], db_manager: DatabaseManager, logger, args):
    """시그널 핸들러 설정 (종료 시 정리 작업)"""
    def signal_handler(signum, frame):
        if signum in [signal.SIGINT, signal.SIGTERM]:
            print("🛑 종료 신호를 받았습니다. 트레이딩을 안전하게 중지합니다...")
            
            # 트레이딩 중지
            for symbol, live_trader in live_traders.items():
                if live_trader.stop():
                    logger.info(f"✅ {symbol} 트레이딩이 성공적으로 중지되었습니다.")
                else:
                    logger.error(f"❌ {symbol} 트레이딩 중지 중 오류가 발생했습니다.")
            
            for symbol, live_trader in live_traders.items():
                try:
                    live_trader.save_trading_stats_to_db()
                    live_trader.update_position_in_db()
                except Exception as e:
                    logger.error(f"❌ {symbol} 최종 DB 저장 실패: {e}")
            
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

def resolve_model_path(path):
    """단일 pth 파일 또는 폴더 내 최신 pth 파일 찾기"""
    if os.path.isfile(path) and path.endswith('.pth'):
        return path
    elif os.path.isdir(path):
        # 폴더 내 .pth 파일 중 최신 파일 찾기
        pth_files = [f for f in os.listdir(path) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"❌ 폴더에 .pth 파일이 없습니다: {path}")
        # 최신 수정 시간 기준으로 정렬
        pth_files.sort(key=lambda f: os.path.getmtime(os.path.join(path, f)), reverse=True)
        latest_pth = os.path.join(path, pth_files[0])
        print(f"📂 폴더에서 최신 모델 선택됨: {latest_pth}")
        return latest_pth
    else:
        raise FileNotFoundError(f"❌ 잘못된 경로입니다: {path}")



def main():
    """메인 함수"""
    global args
    args = parse_args()
    
    print("🚀 SAC 실시간 트레이딩 시스템 시작")
    print("=" * 60)
    
    # 결과 디렉토리 설정
    results_dir = os.path.join(args.results_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(results_dir)
    logger.info("🎯 SAC 실시간 트레이딩 시작")
    logger.info(f"📁 결과 저장 경로: {results_dir}")
    
    if args.dry_run:
        logger.info("🔄 DRY RUN 모드: 실제 거래는 실행되지 않습니다.")
    
    # 설정 로드
    config = Config()
    
    # 환경 검증
    if not validate_environment(config, logger):
        logger.error("❌ 환경 검증 실패. 프로그램을 종료합니다.")
        sys.exit(1)
        
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
        # 모델 경로 결정
        if args.model_path:
            # 명령행 인자로 지정된 모델 경로 사용
            model_paths = {args.model_path: {'enabled': True}}
            logger.info(f"🤖 지정된 모델 경로 사용: {args.model_path}")
        else:
            # config.py의 ACTIVE_TRADING_SYMBOLS 사용
            model_paths = {symbol: data for symbol, data in config.ACTIVE_TRADING_SYMBOLS.items() if data['enabled']}
            logger.info(f"🤖 config.py에서 활성화된 {len(model_paths)}개 심볼의 모델 사용")
        
        if not model_paths:
            logger.error("❌ 활성화된 모델이 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
        
        # 2. API 커넥터 초기화
        logger.info("🔌 Alpaca API 연결 중...")
        api_connector = APIConnector(logger=logger)
        if args.force_connect:
            api_connector.force_connect()
            logger.info("✅ API 서버 강제 연결 성공")

        if not api_connector.connect():
            logger.error("❌ API 서버에 연결할 수 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)

        # 3. 계정 정보 확인
        account_info = api_connector.get_account_info()
        logger.info(f"💰 계정 정보:")
        logger.info(f"   └─ 현금: ${account_info.get('cash', 0):,.2f}")
        logger.info(f"   └─ 포트폴리오 가치: ${account_info.get('portfolio_value', 0):,.2f}")
        logger.info(f"   └─ 매수력: ${account_info.get('buying_power', 0):,.2f}")
        
        # 4. 데이터 검증 시스템 초기화
        data_validator = RealTimeDataValidator(logger=logger)
        
        # 5. 리스크 관리자 초기화
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
        
        # 6. 실시간 트레이더 초기화
        live_traders = {}
        missing_models = []

        for symbol, model_data in model_paths.items():
            if not model_data['enabled']:
                continue
                
            logger.info(f"🏗️  {symbol} 실시간 트레이더 초기화 중...")
            
            try:
                # 백테스팅 완료된 모델 시스템 생성
                agent, data_processor = create_complete_trading_system(model_data['model_path'], config)
                
                # 실시간 트레이더 초기화
                live_trader = LiveTrader(
                    agent=agent,
                    api_connector=api_connector,
                    config=config,
                    logger=logger,
                    data_processor=data_processor,
                    risk_manager=risk_manager
                )
                
                # 데이터 검증 시스템 연동
                live_trader.data_validator = data_validator
                
                # 모델 ID 생성 (파일명 기반)
                model_id = os.path.splitext(os.path.basename(model_data['model_path']))[0]
                
                # 모델 정보를 데이터베이스에 저장
                db_manager.save_model_info(
                    model_id=model_id,
                    file_path=model_data['model_path'],
                    description=f"실시간 트레이딩 모델 for {symbol}",
                    is_active=True
                )
                
                # 향상된 트레이더로 래핑
                enhanced_live_trader = EnhancedLiveTrader(
                    live_trader=live_trader,
                    db_manager=db_manager,
                    symbol=symbol,
                    model_id=model_id
                )
            
                live_traders[symbol] = enhanced_live_trader
                logger.info(f"✅ {symbol} 실시간 트레이더 초기화 완료")
            except FileNotFoundError as e:
                logger.warning(f"⚠️  {symbol} 모델을 찾을 수 없습니다: {e}")
                missing_models.append(symbol)
            except Exception as e:
                logger.error(f"❌ {symbol} 모델 초기화 중 오류 발생: {e}")
                missing_models.append(symbol)
        
        if missing_models:
            logger.warning(f"⚠️  다음 심볼의 모델이 없어 거래가 제외됩니다: {', '.join(missing_models)}")

        if not live_traders:
            logger.error("❌ 초기화된 트레이딩 시스템이 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
        
        logger.info(f"✅ {len(live_traders)}개 심볼에 대한 트레이딩 시스템이 초기화되었습니다: {', '.join(live_traders.keys())}")
        
        # 시그널 핸들러 설정
        setup_signal_handlers(live_traders, db_manager, logger, args)
        
        # 실시간 트레이딩 시작
        logger.info("🚀 실시간 트레이딩 시작...")
        logger.info(f"📈 거래 대상: {', '.join(live_traders.keys())}")
        
        for symbol, live_trader in live_traders.items():
            if not live_trader.start():
                logger.error(f"❌ {symbol} 실시간 트레이딩을 시작할 수 없습니다.")
                continue
        
        # 주기적인 모니터링 루프
        last_log_time = time.time()
        last_save_time = time.time()
        last_db_save_time = time.time()
        last_risk_check_time = time.time()
        
        logger.info("🔄 모니터링 루프 시작")
        
        try:
            while True:
                current_time = time.time()
                
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
                    for symbol, live_trader in live_traders.items():
                        state = live_trader.get_state()
                        
                        logger.info("=" * 50)
                        logger.info(f"📊 {symbol} 현재 트레이딩 상태")
                        logger.info("=" * 50)
                        logger.info(f"🔄 실행 상태: {'✅ 실행 중' if state['running'] else '❌ 중지됨'}")
                        logger.info(f"💰 계정 현금: ${state['account'].get('cash', 0):,.2f}")
                        logger.info(f"📈 포트폴리오: ${state['account'].get('portfolio_value', 0):,.2f}")
                        logger.info(f"🔢 총 거래: {len(state['trading_stats']['trades'])}회")
                        logger.info(f"✅ 성공 거래: {state['trading_stats']['successful_trades']}회")
                        logger.info(f"❌ 실패 거래: {state['trading_stats']['failed_trades']}회")
                        
                        # 수익률 계산
                        initial_balance = state['trading_stats']['initial_balance']
                        current_balance = state['trading_stats']['current_balance']
                        if initial_balance > 0:
                            return_pct = ((current_balance - initial_balance) / initial_balance) * 100
                            logger.info(f"📊 수익률: {return_pct:+.2f}%")
                        
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
            
            for symbol, live_trader in live_traders.items():
                if live_trader.stop():
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
            
            # API 연결 종료
            api_connector.disconnect()
            logger.info("🔌 API 연결 종료")
            
            db_manager.disconnect()
        
            logger.info("✅ 모든 정리 작업 완료")
            logger.info("👋 SAC 실시간 트레이딩 시스템 종료")
            
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 데이터베이스 연결 정리
        try:
            db_manager.disconnect()
        except:
            pass
        
        sys.exit(1)
        
    
if __name__ == "__main__":
    main()