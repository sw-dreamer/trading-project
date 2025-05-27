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


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='SAC 모델 실시간 트레이딩 실행 (개선된 버전)')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='백테스팅 완료된 모델의 경로')
    parser.add_argument('--results_dir', type=str, default='results/live_trading',
                        help='결과 저장 디렉토리')
    parser.add_argument('--log_interval', type=int, default=300,
                        help='로깅 간격 (초, 기본값: 5분)')
    parser.add_argument('--save_interval', type=int, default=1800,
                        help='통계 저장 간격 (초, 기본값: 30분)')
    parser.add_argument('--dry_run', action='store_true',
                        help='실제 거래 없이 시뮬레이션만 실행')
    parser.add_argument('--force_connect', action='store_true',
                        help='시장 시간과 관계없이 연결하려면 force_connect=True')
    
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


def setup_signal_handlers(live_trader, logger):
    """시그널 핸들러 설정 (종료 시 정리 작업)"""
    def signal_handler(signum, frame):
        if signum in [signal.SIGINT, signal.SIGTERM]:
            logger.info("🛑 종료 신호를 받았습니다. 트레이딩을 안전하게 중지합니다...")
            
            # 트레이딩 중지
            if live_trader.stop():
                logger.info("✅ 트레이딩이 성공적으로 중지되었습니다.")
            else:
                logger.error("❌ 트레이딩 중지 중 오류가 발생했습니다.")
            
            # 트레이딩 통계 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(args.results_dir, f"final_trading_stats_{timestamp}.json")
            
            if live_trader.save_trading_stats(results_path):
                logger.info(f"📊 최종 트레이딩 통계가 저장되었습니다: {results_path}")
            
            logger.info("👋 프로그램을 종료합니다.")
            sys.exit(0)
    
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
    
    logger.info("✅ 실행 환경 검증 완료")
    return True


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
    logger.info(f"🤖 모델 경로: {args.model_path}")
    
    if args.dry_run:
        logger.info("🔄 DRY RUN 모드: 실제 거래는 실행되지 않습니다.")
    
    # 설정 로드
    config = Config()
    
    # 환경 검증
    if not validate_environment(config, logger):
        logger.error("❌ 환경 검증 실패. 프로그램을 종료합니다.")
        sys.exit(1)
    
    try:
        # 1. 백테스팅 완료된 모델 시스템 생성
        logger.info("🤖 백테스팅 완료된 모델 로드 중...")
        agent, data_processor = create_complete_trading_system(args.model_path, config)
        logger.info("✅ 모델 시스템 생성 완료")
        
        # 2. API 커넥터 초기화
        logger.info("🔌 Alpaca API 연결 중...")
        api_connector = APIConnector(logger=logger)
        
        if not api_connector.connect():
            logger.error("❌ API 서버에 연결할 수 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
        
        logger.info("✅ API 서버 연결 성공")
        
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
        logger.info("🏗️  실시간 트레이더 초기화 중...")
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
        
        logger.info("✅ 실시간 트레이더 초기화 완료")
        
        # 7. 시그널 핸들러 설정
        setup_signal_handlers(live_trader, logger)
        
        # 8. 실시간 트레이딩 시작
        logger.info("🚀 실시간 트레이딩 시작...")
        logger.info(f"📈 거래 대상: {', '.join(config.trading_symbols)}")
        logger.info(f"⏰ 거래 간격: {config.trading_interval}초")
        logger.info(f"💼 최대 포지션 크기: {config.max_position_size * 100:.1f}%")
        logger.info(f"📉 최대 낙폭 한도: {config.max_drawdown * 100:.1f}%")
        
        if not live_trader.start():
            logger.error("❌ 실시간 트레이딩을 시작할 수 없습니다.")
            sys.exit(1)
        
        # 9. 주기적인 모니터링 루프
        last_log_time = time.time()
        last_save_time = time.time()
        last_risk_check_time = time.time()
        
        logger.info("🔄 모니터링 루프 시작")
        
        try:
            while True:
                current_time = time.time()
                
                # 로깅 간격마다 상태 로깅
                if current_time - last_log_time >= args.log_interval:
                    state = live_trader.get_state()
                    
                    logger.info("=" * 50)
                    logger.info("📊 현재 트레이딩 상태")
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
                        for symbol, position in positions.items():
                            qty = position.get('qty', 0)
                            if abs(qty) > 0.001:  # 의미있는 포지션만 표시
                                unrealized_pl = position.get('unrealized_pl', 0)
                                logger.info(f"   └─ {symbol}: {qty:+.2f}주, 미실현손익: ${unrealized_pl:+,.2f}")
                    else:
                        logger.info("🏢 현재 포지션: 없음")
                    
                    logger.info("=" * 50)
                    last_log_time = current_time
                
                # 저장 간격마다 통계 저장
                if current_time - last_save_time >= args.save_interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_path = os.path.join(results_dir, f"trading_stats_{timestamp}.json")
                    
                    if live_trader.save_trading_stats(results_path):
                        logger.info(f"💾 통계 저장 완료: {results_path}")
                    
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
            
            if live_trader.stop():
                logger.info("✅ 트레이딩 중지 완료")
            
            # 최종 통계 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_results_path = os.path.join(results_dir, f"final_trading_stats_{timestamp}.json")
            
            if live_trader.save_trading_stats(final_results_path):
                logger.info(f"📊 최종 통계 저장: {final_results_path}")
            
            # API 연결 종료
            api_connector.disconnect()
            logger.info("🔌 API 연결 종료")
            
            logger.info("✅ 모든 정리 작업 완료")
            logger.info("👋 SAC 실시간 트레이딩 시스템 종료")
            
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()