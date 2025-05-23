#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import json
import signal
import sys
import time
from datetime import datetime

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sac_agent import SACAgent
from src.utils.logger import Logger
from src.config.config import Config
from src.trading.api_connector import APIConnector
from src.trading.live_trader import LiveTrader
from src.trading.risk_manager import RiskManager
from src.preprocessing.feature_engineer import FeatureEngineer


def parse_args():
    """
    명령행 인자 파싱
    """
    parser = argparse.ArgumentParser(description='SAC 모델 실시간 트레이딩 실행')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델의 경로')
    parser.add_argument('--config_path', type=str, required=True,
                        help='설정 파일 경로')
    parser.add_argument('--api_config_path', type=str, required=True,
                        help='API 설정 파일 경로 (api_key, api_secret, base_url 포함)')
    parser.add_argument('--results_dir', type=str, default='results/live_trading',
                        help='결과 저장 디렉토리')
    parser.add_argument('--log_interval', type=int, default=60,
                        help='로깅 간격 (초)')
    parser.add_argument('--save_interval', type=int, default=3600,
                        help='통계 저장 간격 (초)')
    
    return parser.parse_args()


def load_model(model_path, config, device):
    """
    학습된 모델 로드
    
    Args:
        model_path: 모델 경로
        config: 설정 객체
        device: 연산 장치
        
    Returns:
        로드된 SAC 에이전트
    """
    try:
        # SAC 에이전트 초기화
        agent = SACAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            actor_learning_rate=config.actor_lr,
            critic_learning_rate=config.critic_lr,
            alpha_learning_rate=config.alpha_lr,
            gamma=config.gamma,
            tau=config.tau,
            batch_size=config.batch_size,
            device=device,
            initial_alpha=config.initial_alpha,
            target_entropy=config.target_entropy
        )
        
        # 저장된 모델 로드
        agent.load_model(model_path)
        print(f"모델을 성공적으로 로드했습니다: {model_path}")
        return agent
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        sys.exit(1)


def load_api_config(api_config_path):
    """
    API 설정 로드
    
    Args:
        api_config_path: API 설정 파일 경로
        
    Returns:
        API 설정 딕셔너리
    """
    try:
        with open(api_config_path, 'r') as f:
            api_config = json.load(f)
            
        required_keys = ['api_key', 'api_secret', 'base_url']
        for key in required_keys:
            if key not in api_config:
                print(f"API 설정에 필수 키가 없습니다: {key}")
                sys.exit(1)
                
        return api_config
    except Exception as e:
        print(f"API 설정 로드 중 오류 발생: {e}")
        sys.exit(1)


def setup_logger(results_dir):
    """
    로거 설정
    
    Args:
        results_dir: 결과 저장 디렉토리
        
    Returns:
        로거 객체
    """
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    return Logger(log_file)


def setup_signal_handlers(live_trader, logger):
    """
    시그널 핸들러 설정 (종료 시 정리 작업)
    
    Args:
        live_trader: LiveTrader 인스턴스
        logger: 로거 객체
    """
    def signal_handler(signum, frame):
        if signum in [signal.SIGINT, signal.SIGTERM]:
            if logger:
                logger.info("종료 신호를 받았습니다. 트레이딩을 중지합니다...")
            
            # 트레이딩 중지
            live_trader.stop()
            
            # 트레이딩 통계 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(args.results_dir, f"trading_stats_{timestamp}.json")
            live_trader.save_trading_stats(results_path)
            
            if logger:
                logger.info(f"트레이딩 통계를 {results_path}에 저장했습니다.")
                logger.info("프로그램을 종료합니다.")
                
            sys.exit(0)
    
    # Ctrl+C (SIGINT) 및 SIGTERM 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """
    메인 함수
    """
    global args
    args = parse_args()
    
    # 결과 디렉토리 설정
    results_dir = os.path.join(args.results_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(results_dir)
    logger.info("실시간 트레이딩 시작")
    logger.info(f"설정: {args}")
    
    # 설정 로드
    try:
        config = Config.load_config(args.config_path)
        logger.info("설정을 성공적으로 로드했습니다.")
    except Exception as e:
        logger.error(f"설정 로드 중 오류 발생: {e}")
        sys.exit(1)
    
    # API 설정 로드
    api_config = load_api_config(args.api_config_path)
    logger.info("API 설정을 성공적으로 로드했습니다.")
    
    # 장치 설정 (GPU 사용 가능 여부 확인)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용할 장치: {device}")
    
    # 모델 로드
    agent = load_model(args.model_path, config, device)
    
    # API 커넥터 초기화
    api_connector = APIConnector(
        api_key=api_config['api_key'],
        api_secret=api_config['api_secret'],
        base_url=api_config['base_url'],
        logger=logger
    )
    
    # API 연결
    if not api_connector.connect():
        logger.error("API 서버에 연결할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)
    
    logger.info("API 서버에 성공적으로 연결되었습니다.")
    
    # 특성 엔지니어링 모듈 초기화 (옵션)
    feature_engineer = None
    if hasattr(config, 'feature_engineering') and config.feature_engineering:
        try:
            feature_engineer = FeatureEngineer(config.feature_config)
            logger.info("특성 엔지니어링 모듈 초기화 완료")
        except Exception as e:
            logger.warning(f"특성 엔지니어링 모듈 초기화 실패: {e}, 기본 특성 추출 사용")
    
    # 리스크 관리자 초기화
    risk_manager = RiskManager(
        max_position_size=config.max_position_size,
        max_drawdown=config.max_drawdown,
        max_trade_amount=config.max_trade_amount,
        logger=logger
    )
    
    # 실시간 트레이더 초기화
    live_trader = LiveTrader(
        agent=agent,
        api_connector=api_connector,
        config=config,
        logger=logger,
        feature_engineer=feature_engineer,
        risk_manager=risk_manager
    )
    
    # 시그널 핸들러 설정
    setup_signal_handlers(live_trader, logger)
    
    # 실시간 트레이딩 시작
    if not live_trader.start():
        logger.error("실시간 트레이딩을 시작할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)
    
    # 주기적인 상태 확인 및 로깅
    last_log_time = time.time()
    last_save_time = time.time()
    
    try:
        while True:
            # 현재 시간
            current_time = time.time()
            
            # 로깅 간격마다 상태 로깅
            if current_time - last_log_time >= args.log_interval:
                # 현재 상태 조회
                state = live_trader.get_state()
                
                # 상태 로깅
                logger.info(f"현재 상태: 실행 중 = {state['running']}")
                logger.info(f"계정 잔고: ${state['account'].get('balance', 0):.2f}")
                logger.info(f"총 거래 횟수: {len(state['trading_stats']['trades'])}")
                logger.info(f"성공한 거래: {state['trading_stats']['successful_trades']}")
                logger.info(f"실패한 거래: {state['trading_stats']['failed_trades']}")
                
                # 포지션 로깅
                if state['positions']:
                    logger.info(f"현재 포지션: {len(state['positions'])}개")
                    for symbol, position in state['positions'].items():
                        logger.info(f"- {symbol}: {position.get('quantity', 0)} 수량, 미실현 손익: ${position.get('unrealized_pnl', 0):.2f}")
                
                # 로깅 시간 업데이트
                last_log_time = current_time
            
            # 저장 간격마다 통계 저장
            if current_time - last_save_time >= args.save_interval:
                # 통계 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_path = os.path.join(results_dir, f"trading_stats_{timestamp}.json")
                live_trader.save_trading_stats(results_path)
                
                # 저장 시간 업데이트
                last_save_time = current_time
                
                logger.info(f"트레이딩 통계를 {results_path}에 저장했습니다.")
            
            # CPU 사용량 줄이기 위한 대기
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Ctrl+C로 종료 시
        logger.info("사용자에 의해 중단되었습니다. 트레이딩을 중지합니다...")
        
    except Exception as e:
        # 예외 발생 시
        logger.error(f"실행 중 오류 발생: {e}")
        
    finally:
        # 종료 처리
        live_trader.stop()
        
        # 최종 트레이딩 통계 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(results_dir, f"trading_stats_final_{timestamp}.json")
        live_trader.save_trading_stats(results_path)
        
        logger.info(f"최종 트레이딩 통계를 {results_path}에 저장했습니다.")
        logger.info("프로그램을 종료합니다.")
        

if __name__ == "__main__":
    main() 