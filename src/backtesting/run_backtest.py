#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment
from src.utils.logger import Logger
from src.config.config import Config
from src.backtesting.backtester import Backtester
from src.backtesting.visualizer import Visualizer
from src.data_collection.data_collector import DataCollector  



def parse_args():
    """
    명령행 인자 파싱
    """
    parser = argparse.ArgumentParser(description='SAC 모델 백테스트 실행')
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델의 경로')
    parser.add_argument('--config_path', type=str, 
                        help='설정 파일 경로')
    parser.add_argument('--data_path', type=str,
                        help='테스트 데이터 경로')
    parser.add_argument('--use_db', action='store_true', 
                        help='데이터베이스에서 데이터 로드 (data_path 대신)')
    parser.add_argument('--db_user', type=str, default='postgres',
                        help='데이터베이스 사용자명')
    parser.add_argument('--db_password', type=str, default='mysecretpassword',
                        help='데이터베이스 비밀번호')
    parser.add_argument('--db_host', type=str, default='192.168.40.193',
                        help='데이터베이스 호스트')
    parser.add_argument('--db_port', type=int, default=5432,
                        help='데이터베이스 포트')
    parser.add_argument('--db_name', type=str, default='mydb',
                        help='데이터베이스 이름')
    parser.add_argument('--symbol', type=str, default=None,
                        help='테스트할 종목 심볼 (DB 사용 시)')
    parser.add_argument('--results_dir', type=str, default='results/backtest',
                        help='결과 저장 디렉토리')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='초기 자본금')
    parser.add_argument('--transaction_fee_percent', type=float, default=0.0025,
                        help='거래 수수료율 (0.0025 = 0.25%)')
    parser.add_argument('--benchmark_data_path', type=str, default=None,
                        help='벤치마크 데이터 경로 (옵션)')
    parser.add_argument('--window_size', type=int, default=None,
                        help='관측 창 크기 (설정에서 가져오지 않을 경우)')
    parser.add_argument('--override_window_size', action='store_true',
                        help='환경의 window_size를 모델의 state_dim에 맞게 강제 조정')
    parser.add_argument('--feature_selection', type=str, default=None,
                        help='사용할 특성들의 이름 (콤마로 구분)')
    
    return parser.parse_args()


def create_temp_env(test_data, config):
    """
    임시 환경을 생성하여 상태 차원을 계산합니다.
    
    Args:
        test_data: 테스트 데이터
        config: 설정 객체
        
    Returns:
        state_dim: 상태 차원
        temp_env: 생성된 임시 환경
    """
    window_size = getattr(config, 'window_size', 20)
    temp_env = TradingEnvironment(
        data=test_data,
        raw_data=test_data,
        window_size=window_size,
        initial_balance=10000.0
    )
    
    # 상태 차원 계산을 위해 초기 상태 얻기
    state = temp_env.reset()
    
    # 상태가 딕셔너리인 경우 (CNN 모델용)
    if isinstance(state, dict):
        market_shape = state['market_data'].shape
        portfolio_shape = state['portfolio_state'].shape
        
        # 환경이 구조화된 상태를 반환하는 경우
        return (market_shape, portfolio_shape), temp_env
    else:
        # 상태가 평탄화된 벡터인 경우 (일반 MLP 모델용)
        state_dim = len(state) if isinstance(state, np.ndarray) else state.shape[0]
        return state_dim, temp_env


def load_model(model_path, config, device, test_data=None):
    """
    학습된 모델 로드 - 상태 차원 불일치 문제를 해결하기 위해 개선됨
    
    Args:
        model_path: 모델 경로
        config: 설정 객체
        device: 연산 장치
        test_data: 테스트 데이터 (state_dim 계산에 필요할 경우)
        
    Returns:
        로드된 SAC 에이전트, 모델의 상태 차원
    """
    try:
        # 먼저 테스트 환경에서 실제 상태 차원을 계산
        actual_state_dim, temp_env = create_temp_env(test_data, config)
        print(f"환경에서 계산된 실제 상태 차원: {actual_state_dim}")
        print(f'actual_state_dim : {actual_state_dim}')
        print(f'actual_state_dim type : {type(actual_state_dim)}')
        # 환경에서 실제 사용하는 상태가 딕셔너리인지 확인 (CNN 모델용)
        use_cnn = isinstance(actual_state_dim, tuple)
        
        # 모델 설정 로드 시도
        model_config_path = os.path.join(model_path, "config.pth")
        if os.path.exists(model_config_path):
            loaded_config = torch.load(model_config_path, map_location=device)
            model_state_dim = loaded_config.get('state_dim')
            # 모델 설정에서 action_dim 가져오기, 없으면 기본값 3 사용
            action_dim = loaded_config.get('action_dim', 1)
            # 모델 설정에서 hidden_dim 가져오기, 없으면 config에서 찾거나 기본값 256 사용
            hidden_dim = loaded_config.get('hidden_dim', getattr(config, 'hidden_dim', 256))
            print(f"모델 설정에서 로드: state_dim={model_state_dim}, action_dim={action_dim}")
        else:
            # config에서 action_dim 가져오기, 없으면 기본값 1 사용
            action_dim = getattr(config, 'action_dim', 1)
            # config에서 hidden_dim 가져오기, 없으면 기본값 256 사용
            hidden_dim = getattr(config, 'hidden_dim', 256)
            model_state_dim = None
        
        print(f'model_state_dim : {model_state_dim}')
        print(f'action_dim : {action_dim}')
        print(f'hidden_dim : {hidden_dim}')
        # SAC 에이전트 초기화
        if use_cnn:  # CNN 모델인 경우
            # 튜플을 그대로 전달 - input_shape로 사용
            market_shape, portfolio_shape = actual_state_dim if isinstance(actual_state_dim, tuple) else (None, None)
            
            print(f'market_shape : {market_shape}')
            # dummy_market = torch.randn(1, *market_shape)
            # market_input = dummy_market.permute(0, 2, 1).unsqueeze(-1)
            # print(f'market_input : {market_input}')
            # print(f'market_input shape : {type(market_input)}')
            print(f'portfolio_shape : {portfolio_shape}')
            print(f"hidden_dim: {hidden_dim}, type: {type(hidden_dim)}")
            input_shape = market_shape 
            agent = SACAgent(
                # 중요: state_dim은 사용하지 않고 input_shape만 사용
                state_dim=None,  # CNN 모델은 state_dim 대신 input_shape 사용
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device,
                use_cnn=True,
                input_shape=input_shape  # 튜플 형태의 input_shape 전달
            )
        else:  # 일반 MLP 모델인 경우
            # 모델의 상태 차원을 우선적으로 사용 (차원 불일치 문제 방지)
            agent = SACAgent(
                state_dim=model_state_dim if model_state_dim is not None else actual_state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device
            )

        # 저장된 모델 로드 시도
        try:
            # 정상적인 모델 로드 시도
            agent.load_model(model_path)
            print(f"모델을 성공적으로 로드했습니다: {model_path}")
            
            # 상태 차원 불일치가 있으면 경고
            state_dim_mismatch = False
            if not use_cnn and model_state_dim is not None and isinstance(actual_state_dim, int):
                state_dim_mismatch = model_state_dim != actual_state_dim
                if state_dim_mismatch:
                    print("주의: 모델의 상태 차원이 환경과 일치하지 않습니다.")
                    print("환경의 window_size나 feature 설정이 모델 훈련 시와 다른 것 같습니다.")
                    print("네트워크가 자동으로 차원을 조정하여 처리합니다.")
                    print("더 정확한 결과를 위해 --override_window_size 옵션을 사용하여 환경 설정을 모델에 맞게 조정할 수 있습니다.")
            
            return agent, model_state_dim
        except Exception as e:
            print(f"기본 모델 로드 실패: {e}")
            
            # 모든 시도가 실패한 경우
            raise Exception("모델 로드 실패: 기존 모델과 환경의 상태 차원이 호환되지 않습니다.")
    
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        sys.exit(1)


def load_data(data_path):
    """
    테스트 데이터 로드
    
    Args:
        data_path: 데이터 파일 경로
        
    Returns:
        로드된 데이터 DataFrame
    """
    try:
        # 파일 확장자 확인
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        elif data_path.endswith('.h5'):
            data = pd.read_hdf(data_path)
        else:
            print(f"지원되지 않는 파일 형식: {data_path}")
            sys.exit(1)
            
        print(f"테스트 데이터 로드 완료: {len(data)} 행")
        return data
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        sys.exit(1)


def load_benchmark_data(benchmark_path):
    """
    벤치마크 데이터 로드
    
    Args:
        benchmark_path: 벤치마크 데이터 파일 경로
        
    Returns:
        로드된 벤치마크 수익률 배열
    """
    try:
        if benchmark_path is None:
            return None
            
        # 파일 확장자 확인
        if benchmark_path.endswith('.csv'):
            data = pd.read_csv(benchmark_path)
        elif benchmark_path.endswith('.parquet'):
            data = pd.read_parquet(benchmark_path)
        elif benchmark_path.endswith('.h5'):
            data = pd.read_hdf(benchmark_path)
        else:
            print(f"지원되지 않는 벤치마크 파일 형식: {benchmark_path}")
            return None
            
        # 수익률 컬럼 확인 및 추출
        if 'return' in data.columns:
            returns = data['return'].values
        elif 'returns' in data.columns:
            returns = data['returns'].values
        elif 'daily_return' in data.columns:
            returns = data['daily_return'].values
        else:
            print("벤치마크 데이터에서 수익률 컬럼을 찾을 수 없습니다.")
            return None
            
        print(f"벤치마크 데이터 로드 완료: {len(returns)} 행")
        return returns
    except Exception as e:
        print(f"벤치마크 데이터 로드 중 오류 발생: {e}")
        return None


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
    log_file = os.path.join(log_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    return Logger(log_file)


def adjust_data_for_model(data, feature_selection):
    """
    모델에 맞게 데이터의 특성을 조정합니다.
    
    Args:
        data: 원본 데이터
        feature_selection: 사용할 특성 목록 (콤마로 구분된 문자열)
        
    Returns:
        조정된 데이터
    """
    if feature_selection is None:
        return data
    
    features = [f.strip() for f in feature_selection.split(',')]
    
    # 필수 특성이 있는지 확인
    for feature in features:
        if feature not in data.columns:
            print(f"경고: 특성 '{feature}'가 데이터에 없습니다.")
            features.remove(feature)
    
    if not features:
        print("유효한 특성이 없습니다. 모든 특성을 사용합니다.")
        return data
    
    # 지정된 특성만 선택
    return data[features]


def calculate_required_window_size(model_state_dim, num_features, extra_state_dim=2):
    """
    모델의 상태 차원을 기반으로 필요한 window_size를 계산합니다.
    
    Args:
        model_state_dim: 모델의 상태 차원
        num_features: 데이터의 특성 수
        extra_state_dim: 포트폴리오 상태 등 추가 상태 차원
        
    Returns:
        필요한 window_size
    """
    # state_dim = window_size * num_features + extra_state_dim
    # window_size = (state_dim - extra_state_dim) / num_features
    window_size = (model_state_dim - extra_state_dim) // num_features
    return max(1, window_size)  # window_size는 최소 1 이상이어야 함


def main():
    """
    메인 함수
    """
    # 명령행 인자 파싱
    args = parse_args()
    
    # 결과 디렉토리 설정
    results_dir = os.path.join(args.results_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(results_dir)
    logger.info("백테스트 시작")
    logger.info(f"설정: {args}")
    
    # 설정 로드
    try:
        # config = Config.load_config(args.config_path)
        config = Config()
        logger.info("설정을 성공적으로 로드했습니다.")
        
        # 명령행에서 window_size가 지정된 경우 설정 업데이트
        if args.window_size is not None:
            config.window_size = args.window_size
            logger.info(f"window_size를 명령행 인자로 업데이트: {config.window_size}")
    except Exception as e:
        logger.error(f"설정 로드 중 오류 발생: {e}")
        sys.exit(1)
    
    # 장치 설정 (GPU 사용 가능 여부 확인)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용할 장치: {device}")
    
    # 테스트 데이터 로드 - 모델 로드 전에 데이터 로드
    if args.use_db:
        logger.info("데이터베이스에서 데이터 로드 중...")
        if not args.symbol:
            logger.error("데이터베이스 사용 시 --symbol 인자를 제공해야 합니다.")
            sys.exit(1)
            
        try:
            # DataCollector를 사용하여 DB에서 데이터 로드
            collector = DataCollector(
                user=args.db_user,
                password=args.db_password,
                host=args.db_host,
                port=args.db_port,
                db_name=args.db_name,
                symbols=[args.symbol]
            )
            data_dict = collector.load_all_data()
            
            if not data_dict or args.symbol not in data_dict:
                logger.error(f"데이터베이스에서 {args.symbol} 데이터를 찾을 수 없습니다.")
                sys.exit(1)
                
            test_data = data_dict[args.symbol]
            logger.info(f"데이터베이스에서 {args.symbol} 데이터 로드 완료: {len(test_data)} 행")
            
            # 데이터 전처리나 필요한 변환 수행
            if not test_data.index.name == 'timestamp':
                if 'timestamp' in test_data.columns:
                    test_data.set_index('timestamp', inplace=True)
                    
            # 인덱스 재설정 (필요한 경우)
            test_data = test_data.sort_index()
            
        except Exception as e:
            logger.error(f"데이터베이스에서 데이터 로드 중 오류 발생: {str(e)}")
            sys.exit(1)
    elif args.data_path:
        test_data = load_data(args.data_path)
    else:
        logger.error("테스트 데이터 소스가 지정되지 않았습니다. --data_path 또는 --use_db 인자를 제공하세요.")
        sys.exit(1)
    
    # 특성 선택이 지정된 경우 데이터 조정
    if args.feature_selection:
        logger.info(f"특성 선택 적용: {args.feature_selection}")
        test_data = adjust_data_for_model(test_data, args.feature_selection)
    
    # 모델 로드 - 이제 test_data가 정의된 후에 호출
    agent, model_state_dim = load_model(args.model_path, config, device, test_data)
    
    # window_size 조정이 필요한지 확인
    if args.override_window_size and model_state_dim is not None:
        # 필요한 window_size 계산 (포트폴리오 상태 2개 제외)
        num_features = len(test_data.columns)
        required_window_size = calculate_required_window_size(model_state_dim, num_features)
        
        logger.info(f"모델 상태 차원({model_state_dim})에 맞게 window_size 조정: {required_window_size}")
        config.window_size = required_window_size
    
    # 벤치마크 데이터 로드 (있는 경우)
    benchmark_returns = load_benchmark_data(args.benchmark_data_path)
    
    # 백테스터 초기화
    backtester = Backtester(
        agent=agent,
        test_data=test_data,
        config=config,
        logger=logger,
        initial_balance=args.initial_balance,
        transaction_fee_percent=args.transaction_fee_percent
    )
    
    # 백테스트 실행
    logger.info("백테스트 실행 중...")
    try:
        results = backtester.run_backtest(verbose=True)
        
        # 결과 저장
        results_file = os.path.join(results_dir, "backtest_results.json")
        backtester.save_results(results_file)
        logger.info(f"백테스트 결과를 {results_file}에 저장했습니다.")
        
        # 시각화
        visualizer = Visualizer(results, save_dir=os.path.join(results_dir, "plots"))
        
        try:
            logger.info("백테스트 결과 시각화 중...")
            visualizer.visualize_all(benchmark_returns=benchmark_returns)
            logger.info("시각화 완료")
        except Exception as e:
            logger.error(f"시각화 중 오류 발생: {e}")
        
        # 최종 성능 지표 로깅
        logger.info("백테스트 성능 지표:")
        logger.info(f"초기 자본금: ${args.initial_balance:.2f}")
        logger.info(f"최종 포트폴리오 가치: ${results['portfolio_values'][-1]:.2f}")
        logger.info(f"누적 수익률: {results['metrics']['cumulative_return']:.2%}")
        logger.info(f"연간 수익률: {results['metrics']['annual_return']:.2%}")
        logger.info(f"샤프 비율: {results['metrics']['sharpe_ratio']:.2f}")
        logger.info(f"최대 낙폭: {results['metrics']['max_drawdown']:.2%}")
        logger.info(f"총 거래 횟수: {results['metrics']['total_trades']}")
        logger.info(f"승률: {results['metrics']['win_rate']:.2%}")
        
        logger.info("백테스트가 성공적으로 완료되었습니다.")
        
        return results
    
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류 발생: {e}")
        # 상세 오류 정보 출력
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()