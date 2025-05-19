"""
SAC 모델 평가 실행 스크립트
"""
import os
import argparse
import torch
import numpy as np
from pathlib import Path

from src.config.config import (
    DEVICE,
    TARGET_SYMBOLS,
    LOGGER
)
from src.data_collection.data_collector import DataCollector
from src.preprocessing.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment
from src.models.sac_agent import SACAgent
from src.evaluation.evaluator import Evaluator
from src.utils.utils import create_directory, get_timestamp

def parse_args():
    """
    명령줄 인자 파싱
    
    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(description='SAC 모델 평가 스크립트')
    
    # 데이터 관련 인자
    parser.add_argument('--symbols', nargs='+', default=None, help='평가에 사용할 주식 심볼 목록')
    parser.add_argument('--collect_data', action='store_true', help='데이터 수집 여부')
    
    # 환경 관련 인자
    parser.add_argument('--window_size', type=int, default=30, help='관측 윈도우 크기')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='초기 자본금')
    parser.add_argument('--multi_asset', action='store_true', help='다중 자산 환경 사용 여부')
    
    # 모델 관련 인자
    parser.add_argument('--model_path', type=str, required=True, help='로드할 모델 경로')
    parser.add_argument('--use_cnn', action='store_true', help='CNN 모델 사용 여부')
    
    # 평가 관련 인자
    parser.add_argument('--num_episodes', type=int, default=1, help='평가할 에피소드 수')
    parser.add_argument('--render', action='store_true', help='환경 렌더링 여부')
    parser.add_argument('--result_prefix', type=str, default='', help='결과 파일 접두사')
    
    return parser.parse_args()

def main():
    """
    메인 함수
    """
    # 인자 파싱
    args = parse_args()
    
    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS
    
    LOGGER.info(f"평가 시작: 대상 심볼 {symbols}")
    
    # 데이터 수집
    collector = DataCollector(symbols=symbols)
    
    if args.collect_data:
        LOGGER.info("데이터 수집 중...")
        data = collector.collect_and_save()
    else:
        LOGGER.info("저장된 데이터 로드 중...")
        data = collector.load_all_data()
        
        if not data:
            LOGGER.warning("저장된 데이터가 없어 데이터를 수집합니다.")
            data = collector.collect_and_save()
    
    if not data:
        LOGGER.error("데이터 수집 실패")
        return
    
    # 데이터 전처리
    LOGGER.info("데이터 전처리 중...")
    processor = DataProcessor(window_size=args.window_size)
    results = processor.process_all_symbols(data)
    
    if not results:
        LOGGER.error("데이터 전처리 실패")
        return
    
    # 환경 생성
    if args.multi_asset:
        # 다중 자산 환경 생성
        LOGGER.info("다중 자산 트레이딩 환경 생성 중...")
        normalized_data_dict = {symbol: result['normalized_data'] for symbol, result in results.items()}
        env = MultiAssetTradingEnvironment(
            data_dict=normalized_data_dict,
            window_size=args.window_size,
            initial_balance=args.initial_balance
        )
        action_dim = len(symbols)
    else:
        # 단일 자산 환경 생성 (첫 번째 심볼 사용)
        symbol = symbols[0]
        LOGGER.info(f"단일 자산 트레이딩 환경 생성 중: {symbol}")
        
        if symbol not in results:
            LOGGER.error(f"{symbol} 데이터 처리 결과가 없습니다.")
            return
        
        normalized_data = results[symbol]['normalized_data']
        original_data = results[symbol]['featured_data']
        env = TradingEnvironment(
            data=normalized_data,
            raw_data=original_data,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            symbol=symbol
        )
        action_dim = 1
    
    # 에이전트 생성
    LOGGER.info("SAC 에이전트 생성 중...")
    model_path = Path(args.model_path)
    config = torch.load(model_path / "config.pth", map_location=DEVICE)
    
    # 환경에서 실제 상태 차원 계산
    LOGGER.info("환경에서 실제 상태 차원 계산 중...")
    if isinstance(env, TradingEnvironment):
        # 단일 자산 환경인 경우
        market_shape = env.observation_space['market_data'].shape
        portfolio_shape = env.observation_space['portfolio_state'].shape
        actual_state_dim = market_shape[0] * market_shape[1] + portfolio_shape[0]
        LOGGER.info(f"계산된 실제 상태 차원: {actual_state_dim} (마켓 데이터: {market_shape}, 포트폴리오: {portfolio_shape})")
    else:
        # 다중 자산 환경인 경우
        first_env = list(env.envs.values())[0]
        market_shape = first_env.observation_space['market_data'].shape
        portfolio_shape = first_env.observation_space['portfolio_state'].shape
        actual_state_dim = market_shape[0] * market_shape[1] + portfolio_shape[0]
        LOGGER.info(f"계산된 실제 상태 차원: {actual_state_dim} (마켓 데이터: {market_shape}, 포트폴리오: {portfolio_shape})")
    
    # 모델 구성이 정확히 같도록 수정
    if args.use_cnn:
        # CNN 모델 사용
        input_shape = config.get('input_shape', (args.window_size, env.feature_dim))
        agent = SACAgent(
            input_shape=input_shape,
            use_cnn=True,
            action_dim=config['action_dim'],
            hidden_dim=config.get('hidden_dim', 256)
        )
    else:
        # 일반 모델 사용 - 실제 환경의 상태 차원 사용
        saved_state_dim = config.get('state_dim')
        LOGGER.info(f'저장된 모델의 상태 차원: {saved_state_dim}')
        LOGGER.info(f'현재 환경의 상태 차원: {actual_state_dim}')
        
        if saved_state_dim != actual_state_dim:
            LOGGER.warning(f"상태 차원 불일치: 저장된 모델 ({saved_state_dim}) vs 현재 환경 ({actual_state_dim})")
            LOGGER.warning("동일한 환경 설정으로 모델을 학습하고 평가해야 합니다.")
            LOGGER.info("모델을 현재 환경에 맞게 재구성합니다.")
        
        # 현재 환경의 상태 차원을 사용하여 모델 생성
        agent = SACAgent(
            state_dim=actual_state_dim,
            action_dim=config['action_dim'],
            hidden_dim=config.get('hidden_dim', 256)
        )
    
    # 모델 로드
    LOGGER.info(f"모델 로드 중: {args.model_path}")
    try:
        agent.load_model(args.model_path)
    except Exception as e:
        LOGGER.error(f"표준 모델 로드 실패: {e}")
        LOGGER.info("크기 조정 방식으로 모델 로드 시도...")
        try:
            agent.load_model_with_resize(args.model_path)
        except Exception as e:
            LOGGER.error(f"크기 조정 모델 로드 실패: {e}")
            LOGGER.error("모델 로드 실패. 환경 설정과 모델 설정이 일치하는지 확인하세요.")
            return
    
    # 평가기 생성
    LOGGER.info("평가기 생성 중...")
    evaluator = Evaluator(agent=agent, env=env)
    
    # 평가 실행
    LOGGER.info(f"평가 시작: {args.num_episodes}개 에피소드")
    results = evaluator.evaluate(num_episodes=args.num_episodes, render=args.render)
    
    # 결과 저장
    result_dir = evaluator.save_results(results, prefix=args.result_prefix)
    
    LOGGER.info(f"평가 완료: 결과 저장 경로 {result_dir}")
    LOGGER.info(f"총 수익률: {results['total_return']:.2f}%")
    LOGGER.info(f"샤프 비율: {results['sharpe_ratio']:.2f}")
    LOGGER.info(f"최대 낙폭: {results['max_drawdown']:.2f}%")

if __name__ == "__main__":
    main()