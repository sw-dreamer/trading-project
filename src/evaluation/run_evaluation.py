"""
SAC 모델 평가 실행 스크립트 (수정된 버전)
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

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
from src.environment.trading_env import TradingEnvironment, create_environment_from_results
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
    parser.add_argument('--num_episodes', type=int, default=5, help='평가할 에피소드 수')
    parser.add_argument('--render', action='store_true', help='환경 렌더링 여부')
    parser.add_argument('--result_prefix', type=str, default='', help='결과 파일 접두사')
    parser.add_argument('--data_type', type=str, default='valid', choices=['train', 'valid', 'test'], 
                        help='평가에 사용할 데이터 유형 (기본값: valid)')
    
    return parser.parse_args()

def load_model(model_path, env, args):
    """
    모델 로드
    
    Args:
        model_path: 모델 경로
        env: 환경 객체
        args: 명령줄 인자
        
    Returns:
        로드된 SAC 에이전트
    """
    LOGGER.info(f"🤖 모델 로드 중: {model_path}")
    
    try:
        # 모델 설정 로드
        config_path = os.path.join(model_path, "config.pth")
        if not os.path.exists(config_path):
            LOGGER.error(f"❌ 모델 설정 파일이 없습니다: {config_path}")
            return None
            
        config = torch.load(config_path, map_location=DEVICE)
        LOGGER.info("✅ 모델 설정 로드 성공")
        
        # 환경에서 실제 상태 차원 계산
        market_shape = env.observation_space['market_data'].shape
        portfolio_shape = env.observation_space['portfolio_state'].shape
        actual_state_dim = market_shape[0] * market_shape[1] + portfolio_shape[0]
        
        LOGGER.info(f"📏 계산된 실제 상태 차원: {actual_state_dim}")
        LOGGER.info(f"   └─ 마켓 데이터: {market_shape}")
        LOGGER.info(f"   └─ 포트폴리오: {portfolio_shape}")
        
        # 저장된 모델 설정 확인
        saved_state_dim = config.get('state_dim')
        saved_action_dim = config.get('action_dim', 1)
        saved_hidden_dim = config.get('hidden_dim', 256)
        saved_use_cnn = config.get('use_cnn', False)
        
        LOGGER.info(f"💾 저장된 모델 설정:")
        LOGGER.info(f"   └─ 상태 차원: {saved_state_dim}")
        LOGGER.info(f"   └─ 행동 차원: {saved_action_dim}")
        LOGGER.info(f"   └─ 은닉층 차원: {saved_hidden_dim}")
        LOGGER.info(f"   └─ CNN 사용: {saved_use_cnn}")
        
        # CNN 모델 여부 결정
        use_cnn = args.use_cnn or saved_use_cnn
        
        if use_cnn:
            LOGGER.info("🔧 CNN 모델 생성 중...")
            input_shape = config.get('input_shape', (args.window_size, env.feature_dim))
            agent = SACAgent(
                state_dim=None,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                input_shape=input_shape,
                use_cnn=True
            )
        else:
            LOGGER.info("🔧 MLP 모델 생성 중...")
            
            # 상태 차원 불일치 확인
            if saved_state_dim != actual_state_dim:
                LOGGER.warning(f"⚠️  상태 차원 불일치:")
                LOGGER.warning(f"   └─ 저장된 모델: {saved_state_dim}")
                LOGGER.warning(f"   └─ 현재 환경: {actual_state_dim}")
                LOGGER.warning("   └─ 저장된 모델의 차원을 사용합니다.")
            
            # 저장된 모델의 상태 차원을 우선 사용
            agent = SACAgent(
                state_dim=saved_state_dim if saved_state_dim is not None else actual_state_dim,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                use_cnn=False
            )
        
        # 모델 로드 시도
        try:
            agent.load_model(model_path)
            LOGGER.info("✅ 표준 방식으로 모델 로드 성공")
        except Exception as e:
            LOGGER.warning(f"⚠️  표준 모델 로드 실패: {e}")
            LOGGER.info("🔄 크기 조정 방식으로 모델 로드 시도...")
            
            try:
                if hasattr(agent, 'load_model_with_resize'):
                    agent.load_model_with_resize(model_path)
                    LOGGER.info("✅ 크기 조정 방식으로 모델 로드 성공")
                else:
                    LOGGER.error("❌ load_model_with_resize 메서드가 없습니다.")
                    return None
            except Exception as e:
                LOGGER.error(f"❌ 크기 조정 모델 로드도 실패: {e}")
                return None
        
        return agent
        
    except Exception as e:
        LOGGER.error(f"❌ 모델 로드 중 오류 발생: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return None

def main():
    """
    메인 함수
    """
    print('=' * 80)
    LOGGER.info('🎯 SAC 모델 평가 시작 (검증 데이터 사용 권장)')
    
    # 인자 파싱
    args = parse_args()
    
    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS
    
    LOGGER.info(f"📈 평가 설정:")
    LOGGER.info(f"   └─ 대상 심볼: {symbols}")
    LOGGER.info(f"   └─ 데이터 유형: {args.data_type}")
    LOGGER.info(f"   └─ 에피소드 수: {args.num_episodes}")
    LOGGER.info(f"   └─ 윈도우 크기: {args.window_size}")
    LOGGER.info(f"   └─ 초기 자본금: ${args.initial_balance:,.2f}")
    LOGGER.info(f"   └─ 다중 자산: {'예' if args.multi_asset else '아니오'}")
    LOGGER.info(f"   └─ CNN 사용: {'예' if args.use_cnn else '아니오'}")
    
    # 다중 자산 환경 체크
    if args.multi_asset:
        LOGGER.error("❌ 다중 자산 환경은 현재 지원되지 않습니다.")
        LOGGER.info("💡 단일 자산 환경을 사용해주세요.")
        return
    
    # 데이터 유형에 따른 안내 메시지
    if args.data_type == 'test':
        LOGGER.warning("⚠️  테스트 데이터를 사용합니다. 이는 최종 성능 측정용으로만 사용하세요!")
        LOGGER.info("💡 일반적인 모델 평가에는 --data_type valid를 권장합니다.")
    elif args.data_type == 'train':
        LOGGER.warning("⚠️  훈련 데이터를 사용합니다. 이는 과적합 확인용입니다.")
        LOGGER.info("💡 실제 성능 평가에는 --data_type valid를 권장합니다.")
    else:
        LOGGER.info("✅ 검증 데이터를 사용합니다. (권장 설정)")
    
    # 데이터 수집
    LOGGER.info("📊 데이터 수집 중...")
    collector = DataCollector(symbols=symbols)
    
    if args.collect_data:
        LOGGER.info("🔄 새로운 데이터 수집 중...")
        data = collector.load_and_save()
    else:
        LOGGER.info("💾 저장된 데이터 로드 중...")
        data = collector.load_all_data()
        
        if not data:
            LOGGER.warning("⚠️  저장된 데이터가 없어 새로 수집합니다.")
            data = collector.load_and_save()
    
    if not data:
        LOGGER.error("❌ 데이터 수집 실패")
        return
    
    LOGGER.info(f"✅ 데이터 수집 완료: {len(data)}개 심볼")
    
    # 데이터 전처리
    LOGGER.info("⚙️  데이터 전처리 중...")
    processor = DataProcessor(window_size=args.window_size)
    results = processor.process_all_symbols(data)
    
    if not results:
        LOGGER.error("❌ 데이터 전처리 실패")
        return
    
    LOGGER.info(f"✅ 데이터 전처리 완료: {len(results)}개 심볼")
    
    # 데이터 분할 정보 로깅
    symbol = symbols[0]  # 첫 번째 심볼 사용
    if symbol in results:
        LOGGER.info(f"📊 {symbol} 데이터 분할 정보:")
        LOGGER.info(f"   └─ 전체: {len(results[symbol]['normalized_data'])} 행")
        LOGGER.info(f"   └─ 훈련: {len(results[symbol]['train'])} 행")
        LOGGER.info(f"   └─ 검증: {len(results[symbol]['valid'])} 행")
        LOGGER.info(f"   └─ 테스트: {len(results[symbol]['test'])} 행")
    
    # 평가 환경 생성 - create_environment_from_results 함수 사용
    LOGGER.info(f"🏗️  {args.data_type} 환경 생성 중...")
    try:
        env = create_environment_from_results(
            results=results,
            symbol=symbol,
            data_type=args.data_type,
            window_size=args.window_size,
            initial_balance=args.initial_balance
        )
        
        if env is None:
            LOGGER.error("❌ 평가 환경 생성 실패")
            return
            
        LOGGER.info(f"✅ {args.data_type} 환경 생성 완료")
        LOGGER.info(f"   └─ 심볼: {env.symbol}")
        LOGGER.info(f"   └─ 데이터 길이: {env.data_length}")
        LOGGER.info(f"   └─ 특성 차원: {env.feature_dim}")
        LOGGER.info(f"   └─ 윈도우 크기: {env.window_size}")
        
    except Exception as e:
        LOGGER.error(f"❌ 환경 생성 실패: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return
    
    # 모델 로드
    agent = load_model(args.model_path, env, args)
    
    if agent is None:
        LOGGER.error("❌ 모델 로드 실패")
        return
    
    # 평가기 생성
    LOGGER.info("🎯 평가기 생성 중...")
    try:
        # Evaluator 생성 시 data_type 인자 전달 시도
        try:
            evaluator = Evaluator(agent=agent, env=env, data_type=args.data_type)
            LOGGER.info("✅ 평가기 생성 완료")
        except TypeError:
            # data_type 인자를 받지 않는 경우
            evaluator = Evaluator(agent=agent, env=env)
            LOGGER.info("✅ 평가기 생성 완료 (data_type 인자 없이)")
    except Exception as e:
        LOGGER.error(f"❌ 평가기 생성 실패: {str(e)}")
        return
    
    # 평가 실행
    LOGGER.info(f"🚀 {args.data_type} 데이터로 평가 시작: {args.num_episodes}개 에피소드")
    try:
        results_eval = evaluator.evaluate(num_episodes=args.num_episodes, render=args.render)
        LOGGER.info("✅ 평가 완료!")
        
        # 결과 저장
        prefix = f"{args.result_prefix}_" if args.result_prefix else ""
        prefix += f"{args.data_type}_"  # 데이터 유형을 prefix에 추가
        result_dir = evaluator.save_results(results_eval, prefix=prefix)
        
        # 결과 출력
        LOGGER.info("=" * 80)
        LOGGER.info(f"🎉 평가 결과 ({args.data_type} 데이터)")
        LOGGER.info("=" * 80)
        LOGGER.info(f"📁 결과 저장 경로: {result_dir}")
        LOGGER.info(f"💰 총 수익률: {results_eval['total_return']:.2f}%")
        LOGGER.info(f"📊 샤프 비율: {results_eval['sharpe_ratio']:.2f}")
        LOGGER.info(f"📉 최대 낙폭: {results_eval['max_drawdown']:.2f}%")
        
        # 추가 성능 지표 (있는 경우)
        if 'annualized_return' in results_eval:
            LOGGER.info(f"📈 연간 수익률: {results_eval['annualized_return']:.2f}%")
        if 'volatility' in results_eval:
            LOGGER.info(f"📊 변동성: {results_eval['volatility']:.2f}%")
        if 'total_trades' in results_eval:
            LOGGER.info(f"🔄 총 거래 횟수: {results_eval['total_trades']}")
        if 'win_rate' in results_eval:
            LOGGER.info(f"🎯 승률: {results_eval['win_rate']:.2f}%")
        
        LOGGER.info("=" * 80)
        LOGGER.info("💡 팁: 다른 데이터셋으로 평가하려면 --data_type 옵션을 사용하세요")
        LOGGER.info("   └─ --data_type train  : 훈련 데이터로 평가 (과적합 확인용)")
        LOGGER.info("   └─ --data_type valid : 검증 데이터로 평가 (기본값, 권장)")
        LOGGER.info("   └─ --data_type test  : 테스트 데이터로 평가 (최종 성능 측정용)")
        
    except Exception as e:
        LOGGER.error(f"❌ 평가 실행 중 오류 발생: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())

if __name__ == "__main__":
    main()