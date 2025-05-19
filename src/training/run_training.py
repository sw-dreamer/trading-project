"""
SAC 모델 학습 실행 스크립트
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import os
import argparse
import torch
import numpy as np
from pathlib import Path

from src.config.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPISODES,
    EVALUATE_INTERVAL,
    SAVE_MODEL_INTERVAL,
    MAX_STEPS_PER_EPISODE,
    TARGET_SYMBOLS,
    LOGGER
)
from src.data_collection.data_collector import DataCollector
from src.preprocessing.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment
from src.models.sac_agent import SACAgent
from src.training.trainer import Trainer
from src.utils.utils import create_directory, get_timestamp

def parse_args():
    """
    명령줄 인자 파싱
    
    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(description='SAC 모델 학습 스크립트')
    
    # 데이터 관련 인자
    parser.add_argument('--symbols', nargs='+', default=None, help='학습에 사용할 주식 심볼 목록')
    parser.add_argument('--collect_data', action='store_true', help='데이터 수집 여부')
    
    # 환경 관련 인자
    parser.add_argument('--window_size', type=int, default=30, help='관측 윈도우 크기')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='초기 자본금')
    parser.add_argument('--multi_asset', action='store_true', help='다중 자산 환경 사용 여부')
    
    # 모델 관련 인자
    parser.add_argument('--hidden_dim', type=int, default=256, help='은닉층 차원')
    parser.add_argument('--use_cnn', action='store_true', help='CNN 모델 사용 여부')
    parser.add_argument('--load_model', type=str, default=None, help='로드할 모델 경로')
    
    # 학습 관련 인자
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='배치 크기')
    parser.add_argument('--num_episodes', type=int, default=NUM_EPISODES, help='학습할 총 에피소드 수')
    parser.add_argument('--evaluate_interval', type=int, default=EVALUATE_INTERVAL, help='평가 간격')
    parser.add_argument('--save_interval', type=int, default=SAVE_MODEL_INTERVAL, help='모델 저장 간격')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_PER_EPISODE, help='에피소드당 최대 스텝 수')
    
    return parser.parse_args()

def main():
    """
    메인 함수
    """
    print('='*50)
    LOGGER.info('run_training 시작')
    # 인자 파싱
    args = parse_args()
    
    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS
    
    LOGGER.info(f"학습 시작: 대상 심볼 {symbols}")
    
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
        original_data = results["AAPL"]["featured_data"]
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
    if args.use_cnn:
        # CNN 모델 사용
        input_shape = (args.window_size, env.feature_dim)
        agent = SACAgent(
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            input_shape=input_shape,
            use_cnn=True
        )
    else:
        # 일반 모델 사용
        if isinstance(env, TradingEnvironment):
            state_dim = env.observation_space['market_data'].shape[0] * env.observation_space['market_data'].shape[1] + env.observation_space['portfolio_state'].shape[0]
        else:
            # 다중 자산 환경의 경우 첫 번째 환경의 상태 차원 사용
            first_env = list(env.envs.values())[0]
            state_dim = first_env.observation_space['market_data'].shape[0] * first_env.observation_space['market_data'].shape[1] + first_env.observation_space['portfolio_state'].shape[0]
        
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim
        )
    
    # 모델 로드 (지정된 경우)
    if args.load_model:
        LOGGER.info(f"모델 로드 중: {args.load_model}")
        agent.load_model(args.load_model)
    
    # 트레이너 생성
    LOGGER.info("트레이너 생성 중...")
    trainer = Trainer(
        agent=agent,
        env=env,
        batch_size=args.batch_size,
        num_episodes=args.num_episodes,
        evaluate_interval=args.evaluate_interval,
        save_interval=args.save_interval,
        max_steps=args.max_steps
    )
    
    # 학습 실행
    LOGGER.info("학습 시작...")
    training_stats = trainer.train()
    
    LOGGER.info(f"학습 완료: 최종 평가 보상 {trainer.eval_rewards[-1] if trainer.eval_rewards else 'N/A'}")

if __name__ == "__main__":
    main() 