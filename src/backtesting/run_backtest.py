#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAC 모델 백테스트 실행 스크립트 (프로젝트 구조에 맞게 수정된 버전)
"""
import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment, create_environment_from_results
from src.data_collection.data_collector import DataCollector
from src.preprocessing.data_processor import DataProcessor
# Import 경로 수정: 실제 디렉토리 구조에 맞게
from src.backtesting.backtest_database_manager import BacktestDatabaseManager
from src.config.config import (
    DEVICE,
    TARGET_SYMBOLS,
    LOGGER,
    WINDOW_SIZE,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    # MySQL 설정 추가
    MYSQL_HOST,
    MYSQL_DATABASE,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    SAVE_TO_DATABASE,
    SKIP_DB_ON_ERROR
)
from src.utils.utils import create_directory


def parse_args():
    """
    명령행 인자 파싱
    """
    parser = argparse.ArgumentParser(description='SAC 모델 백테스트 실행')

    # 모델 관련 인자
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델의 경로')

    # 데이터베이스 관련 인자 수정 - 기본값을 True로 변경
    parser.add_argument('--save_to_db', action='store_true', default=True,
                        help='데이터베이스에 결과 저장 (기본값: 저장함)')
    parser.add_argument('--no_db', action='store_true',
                        help='데이터베이스 저장 건너뛰기')
    parser.add_argument('--skip_db_on_error', action='store_true', default=SKIP_DB_ON_ERROR,
                        help='DB 저장 실패 시 무시하고 계속 진행')

    # 데이터 관련 인자
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='백테스트할 주식 심볼 목록')
    parser.add_argument('--collect_data', action='store_true',
                        help='새로운 데이터 수집 여부')
    parser.add_argument('--data_type', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='백테스트에 사용할 데이터 유형')

    # 환경 관련 인자
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE,
                        help='관측 윈도우 크기')
    parser.add_argument('--initial_balance', type=float, default=INITIAL_BALANCE,
                        help='초기 자본금')
    parser.add_argument('--transaction_fee_percent', type=float, default=TRANSACTION_FEE_PERCENT,
                        help='거래 수수료율')

    # 결과 저장 관련
    parser.add_argument('--results_dir', type=str, default='results/backtest',
                        help='결과 저장 디렉토리')
    parser.add_argument('--save_plots', action='store_true',
                        help='그래프 저장 여부')
    parser.add_argument('--render', action='store_true',
                        help='실행 과정 렌더링 여부')

    # 모델 관련 설정
    parser.add_argument('--use_cnn', action='store_true',
                        help='CNN 모델 사용 여부')
    parser.add_argument('--use_lstm', action='store_true',
                    help='LSTM 모델 사용 여부')
    parser.add_argument('--override_window_size', action='store_true',
                        help='모델에 맞게 window_size 강제 조정')

    return parser.parse_args()


def load_model_with_compatibility(model_path, env, args):
    """
    호환성을 고려하여 모델 로드

    Args:
        model_path: 모델 경로
        env: 환경 객체
        args: 명령줄 인자

    Returns:
        로드된 SAC 에이전트
    """
    LOGGER.info(f"🤖 모델 로드 중: {model_path}")

    try:
        # 모델 설정 파일 경로
        config_path = os.path.join(model_path, "config.pth")

        if os.path.exists(config_path):
            LOGGER.info("📁 모델 설정 파일 로드 중...")
            config = torch.load(config_path, map_location=DEVICE)

            # 저장된 모델 설정
            saved_state_dim = config.get('state_dim')
            saved_action_dim = config.get('action_dim', 1)
            saved_hidden_dim = config.get('hidden_dim', 256)
            saved_use_cnn = config.get('use_cnn', False)
            saved_use_lstm = config.get('use_lstm', False)
            saved_input_shape = config.get('input_shape')

            LOGGER.info(f"💾 저장된 모델 설정:")
            LOGGER.info(f"   └─ 상태 차원: {saved_state_dim}")
            LOGGER.info(f"   └─ 행동 차원: {saved_action_dim}")
            LOGGER.info(f"   └─ 은닉층 차원: {saved_hidden_dim}")
            LOGGER.info(f"   └─ CNN 사용: {saved_use_cnn}")
            LOGGER.info(f"   └─ LSTM 사용: {saved_use_lstm}")
            if saved_input_shape:
                LOGGER.info(f"   └─ 입력 형태: {saved_input_shape}")

        else:
            LOGGER.warning("⚠️  모델 설정 파일이 없습니다. 기본값을 사용합니다.")
            saved_state_dim = None
            saved_action_dim = 1
            saved_hidden_dim = 256
            saved_use_cnn = False
            saved_input_shape = None

        # 현재 환경에서 상태 차원 계산
        obs = env.reset()
        if isinstance(obs, dict):
            # CNN 모델용 구조화된 상태
            market_shape = obs['market_data'].shape
            portfolio_shape = obs['portfolio_state'].shape
            actual_state_dim = market_shape[0] * market_shape[1] + portfolio_shape[0]
            use_cnn = True
            input_shape = market_shape

            LOGGER.info(f"📏 환경 상태 정보 (CNN 구조):")
            LOGGER.info(f"   └─ 마켓 데이터 형태: {market_shape}")
            LOGGER.info(f"   └─ 포트폴리오 형태: {portfolio_shape}")
            LOGGER.info(f"   └─ 계산된 상태 차원: {actual_state_dim}")

        else:
            # MLP 모델용 평탄화된 상태
            actual_state_dim = len(obs) if hasattr(obs, '__len__') else obs.shape[0]
            use_cnn = False
            input_shape = None

            LOGGER.info(f"📏 환경 상태 정보 (MLP 구조):")
            LOGGER.info(f"   └─ 상태 차원: {actual_state_dim}")

        # CNN / LSTM 충돌 방지
        if args.use_cnn and args.use_lstm:
            LOGGER.error("❌ CNN과 LSTM은 동시에 사용할 수 없습니다. 하나만 선택하세요.")
            return None

        # 우선순위: LSTM > CNN > MLP
        final_use_lstm = args.use_lstm or saved_use_lstm
        final_use_cnn = args.use_cnn or saved_use_cnn

        if final_use_lstm:
            LOGGER.info("🔧 LSTM 모델 생성 중...")
            final_input_shape = saved_input_shape if saved_input_shape else input_shape
            agent = SACAgent(
                state_dim=None,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                input_shape=final_input_shape,
                use_lstm=True,
                device=DEVICE
            )
        elif final_use_cnn:
            LOGGER.info("🔧 CNN 모델 생성 중...")
            final_input_shape = saved_input_shape if saved_input_shape else input_shape
            agent = SACAgent(
                state_dim=None,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                input_shape=final_input_shape,
                use_cnn=True,
                device=DEVICE
            )
        else:
            LOGGER.info("🔧 MLP 모델 생성 중...")
            final_state_dim = saved_state_dim if saved_state_dim else actual_state_dim

            if saved_state_dim and saved_state_dim != actual_state_dim:
                LOGGER.warning(f"⚠️  상태 차원 불일치:")
                LOGGER.warning(f"   └─ 저장된 모델: {saved_state_dim}")
                LOGGER.warning(f"   └─ 현재 환경: {actual_state_dim}")
                LOGGER.warning("   └─ 저장된 모델의 차원을 사용합니다.")

            agent = SACAgent(
                state_dim=final_state_dim,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                device=DEVICE
            )

        # 모델 로드 시도
        try:
            agent.load_model(model_path)
            LOGGER.info("✅ 모델 로드 성공")
            return agent

        except Exception as e:
            LOGGER.error(f"❌ 모델 로드 실패: {str(e)}")

            # 크기 조정 방식으로 재시도 (있는 경우)
            if hasattr(agent, 'load_model_with_resize'):
                LOGGER.info("🔄 크기 조정 방식으로 재시도...")
                try:
                    agent.load_model_with_resize(model_path)
                    LOGGER.info("✅ 크기 조정 방식으로 모델 로드 성공")
                    return agent
                except Exception as e2:
                    LOGGER.error(f"❌ 크기 조정 방식도 실패: {str(e2)}")

            return None

    except Exception as e:
        LOGGER.error(f"❌ 모델 로드 중 예상치 못한 오류: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return None


def run_backtest_episode(agent, env, render=False):
    """
    단일 백테스트 에피소드 실행

    Args:
        agent: SAC 에이전트
        env: 트레이딩 환경
        render: 렌더링 여부

    Returns:
        에피소드 결과 딕셔너리
    """
    state = env.reset()
    done = False
    step = 0

    # 결과 저장용 리스트
    actions = []
    rewards = []
    portfolio_values = []
    prices = []
    positions = []
    balances = []
    shares = []

    while not done and step < env.data_length - 1:
        # 행동 선택 (평가 모드)
        action = agent.select_action(state, evaluate=True)

        # 환경에서 스텝 실행
        next_state, reward, done, info = env.step(action)

        # 결과 기록
        actions.append(action)
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])
        prices.append(info['current_price'])
        positions.append(info['position'])
        balances.append(info['balance'])
        shares.append(info['shares_held'])

        # 상태 업데이트
        state = next_state
        step += 1

        # 렌더링 (요청된 경우)
        if render and step % 100 == 0:
            LOGGER.info(f"Step {step}: 포트폴리오=${info['portfolio_value']:.2f}, "
                        f"가격=${info['current_price']:.2f}, 포지션={info['position']}")

    # 최종 결과 계산
    initial_value = portfolio_values[0] if portfolio_values else env.initial_balance
    final_value = portfolio_values[-1] if portfolio_values else env.initial_balance
    total_return = (final_value - initial_value) / initial_value * 100

    return {
        'actions': actions,
        'rewards': rewards,
        'portfolio_values': portfolio_values,
        'prices': prices,
        'positions': positions,
        'balances': balances,
        'shares': shares,
        'total_return': total_return,
        'final_portfolio_value': final_value,
        'initial_portfolio_value': initial_value,
        'total_steps': step
    }


def calculate_metrics(results):
    """
    백테스트 성능 지표 계산

    Args:
        results: 백테스트 결과

    Returns:
        성능 지표 딕셔너리
    """
    portfolio_values = np.array(results['portfolio_values'])

    if len(portfolio_values) < 2:
        return {}

    # 일일 수익률 계산
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # 기본 지표
    total_return = results['total_return']
    cumulative_return = (results['final_portfolio_value'] / results['initial_portfolio_value'] - 1) * 100

    # 연간 수익률 (252 거래일 가정)
    num_days = len(portfolio_values)
    annual_return = ((results['final_portfolio_value'] / results['initial_portfolio_value']) ** (252 / num_days) - 1) * 100

    # 변동성 (연간)
    volatility = np.std(daily_returns) * np.sqrt(252) * 100

    # 샤프 비율 (무위험 수익률 0 가정)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0

    # 최대 낙폭 계산
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak * 100
    max_drawdown = np.min(drawdown)

    # 거래 통계
    positions = results['positions']
    total_trades = sum(1 for pos in positions if pos in ['매수', '매도'])
    buy_trades = sum(1 for pos in positions if pos == '매수')
    sell_trades = sum(1 for pos in positions if pos == '매도')

    # 승리 거래 계산 (매도 시점에서 수익 확인)
    winning_trades = 0
    for i, pos in enumerate(positions):
        if pos == '매도' and i > 0:
            # 이전 매수 시점 찾기
            for j in range(i - 1, -1, -1):
                if positions[j] == '매수':
                    if portfolio_values[i] > portfolio_values[j]:
                        winning_trades += 1
                    break

    win_rate = (winning_trades / sell_trades * 100) if sell_trades > 0 else 0

    return {
        'total_return': total_return,
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_steps': results['total_steps']
    }


def save_to_database(backtest_results, metrics, symbol, model_path, data_type, args, 
                    backtest_start_time=None, backtest_end_time=None):
    """
    백테스트 결과를 MySQL 데이터베이스에 저장

    Args:
        backtest_results: 백테스트 결과
        metrics: 성능 지표
        symbol: 주식 심볼
        model_path: 모델 경로
        data_type: 데이터 타입
        args: 명령행 인자
        backtest_start_time: 백테스트 시작 시간
        backtest_end_time: 백테스트 종료 시간

    Returns:
        bool: 저장 성공 여부
    """
    # --no_db 플래그가 있으면 DB 저장 건너뛰기
    if args.no_db:
        LOGGER.info("💾 데이터베이스 저장 건너뜀 (--no_db 플래그 활성화)")
        return True

    try:
        LOGGER.info("💾 데이터베이스 저장 시작...")

        # 데이터베이스 매니저 생성
        with BacktestDatabaseManager(
            host=MYSQL_HOST,
            database=MYSQL_DATABASE,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            port=MYSQL_PORT
        ) as db_manager:

            # 테이블 생성 확인
            if not db_manager.create_tables_if_not_exists():
                raise Exception("테이블 생성 실패")

            # 백테스트 결과를 DB 형식으로 변환
            db_data = db_manager.convert_backtest_results_to_db_format(
                results=backtest_results,
                metrics=metrics,
                symbol=symbol,
                model_path=model_path,
                data_type=data_type,
                backtest_start_time=backtest_start_time,
                backtest_end_time=backtest_end_time
            )

            # 데이터베이스에 저장
            backtest_id = db_manager.insert_backtest_result(
                backtest_data=db_data,
                trade_details=backtest_results  # 상세 거래 내역도 저장
            )

            if backtest_id:
                # 백테스팅 소요 시간 계산
                if backtest_start_time and backtest_end_time:
                    duration = backtest_end_time - backtest_start_time
                    duration_str = str(duration).split('.')[0]  # 마이크로초 제거
                    LOGGER.info(f"⏱️  백테스팅 소요 시간: {duration_str}")
                
                LOGGER.info(f"✅ 데이터베이스 저장 완료!")
                LOGGER.info(f"   └─ Backtest ID: {backtest_id}")
                LOGGER.info(f"   └─ Symbol: {symbol}")
                LOGGER.info(f"   └─ Model: {os.path.basename(model_path)}")

                # 요약 정보 출력
                db_manager.display_summary(db_data)
                return True
            else:
                raise Exception("데이터베이스 저장 실패")

    except Exception as e:
        error_msg = f"❌ 데이터베이스 저장 중 오류: {str(e)}"

        if args.skip_db_on_error:
            LOGGER.warning(error_msg)
            LOGGER.warning("⚠️  DB 저장 오류 무시하고 계속 진행...")
            return False
        else:
            LOGGER.error(error_msg)
            raise e


def save_results(results, metrics, results_dir, symbol):
    """
    백테스트 결과 저장

    Args:
        results: 백테스트 결과
        metrics: 성능 지표
        results_dir: 저장 디렉토리
        symbol: 주식 심볼
    """
    # JSON 결과 저장
    json_results = {
        'symbol': symbol,
        'metrics': metrics,
        'summary': {
            'initial_balance': results['initial_portfolio_value'],
            'final_balance': results['final_portfolio_value'],
            'total_return_pct': results['total_return'],
            'total_trades': metrics.get('total_trades', 0),
            'win_rate_pct': metrics.get('win_rate', 0)
        }
    }

    json_file = os.path.join(results_dir, f"backtest_results_{symbol}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    # CSV 상세 결과 저장
    detailed_df = pd.DataFrame({
        'step': range(len(results['portfolio_values'])),
        'portfolio_value': results['portfolio_values'],
        'price': results['prices'],
        'position': results['positions'],
        'balance': results['balances'],
        'shares': results['shares'],
        'reward': results['rewards'],
        'action': results['actions']
    })

    csv_file = os.path.join(results_dir, f"backtest_detailed_{symbol}.csv")
    detailed_df.to_csv(csv_file, index=False)

    LOGGER.info(f"📁 결과 저장 완료:")
    LOGGER.info(f"   └─ JSON: {json_file}")
    LOGGER.info(f"   └─ CSV: {csv_file}")


def create_plots(results, metrics, results_dir, symbol):
    """
    백테스트 결과 시각화

    Args:
        results: 백테스트 결과
        metrics: 성능 지표
        results_dir: 저장 디렉토리
        symbol: 주식 심볼
    """
    try:
        plt.style.use('default')

        # 4개 서브플롯 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{symbol} 백테스트 결과', fontsize=16, fontweight='bold')

        steps = range(len(results['portfolio_values']))

        # 1. 포트폴리오 가치 변화
        ax1.plot(steps, results['portfolio_values'], 'b-', linewidth=2, label='포트폴리오 가치')
        ax1.axhline(y=results['initial_portfolio_value'], color='r', linestyle='--', alpha=0.7, label='초기 가치')
        ax1.set_title('포트폴리오 가치 변화')
        ax1.set_xlabel('스텝')
        ax1.set_ylabel('가치 ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 주가 변화
        ax2.plot(steps, results['prices'], 'g-', linewidth=2, label='주가')
        ax2.set_title('주가 변화')
        ax2.set_xlabel('스텝')
        ax2.set_ylabel('가격 ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 보상 변화
        ax3.plot(steps[1:], results['rewards'], 'orange', linewidth=1, alpha=0.7, label='보상')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_title('보상 변화')
        ax3.set_xlabel('스텝')
        ax3.set_ylabel('보상')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 현금 vs 주식 보유량
        ax4_twin = ax4.twinx()
        ax4.plot(steps, results['balances'], 'purple', linewidth=2, label='현금', alpha=0.7)
        ax4_twin.plot(steps, results['shares'], 'brown', linewidth=2, label='주식 보유량', alpha=0.7)
        ax4.set_title('현금 vs 주식 보유량')
        ax4.set_xlabel('스텝')
        ax4.set_ylabel('현금 ($)', color='purple')
        ax4_twin.set_ylabel('주식 보유량', color='brown')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 그래프 저장
        plot_file = os.path.join(results_dir, f"backtest_plots_{symbol}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 성능 요약 텍스트 플롯
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        summary_text = f"""
백테스트 성능 요약 - {symbol}

초기 자본금: ${results['initial_portfolio_value']:,.2f}
최종 포트폴리오: ${results['final_portfolio_value']:,.2f}
총 수익률: {metrics.get('total_return', 0):.2f}%
연간 수익률: {metrics.get('annual_return', 0):.2f}%
변동성: {metrics.get('volatility', 0):.2f}%
샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}
최대 낙폭: {metrics.get('max_drawdown', 0):.2f}%

거래 통계:
총 거래 횟수: {metrics.get('total_trades', 0)}
매수 거래: {metrics.get('buy_trades', 0)}
매도 거래: {metrics.get('sell_trades', 0)}
승률: {metrics.get('win_rate', 0):.2f}%
총 스텝: {metrics.get('total_steps', 0)}
        """

        ax.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

        plt.title(f'{symbol} 백테스트 성능 요약', fontsize=14, fontweight='bold')

        summary_file = os.path.join(results_dir, f"backtest_summary_{symbol}.png")
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()

        LOGGER.info(f"📊 그래프 저장 완료:")
        LOGGER.info(f"   └─ 상세 플롯: {plot_file}")
        LOGGER.info(f"   └─ 요약 플롯: {summary_file}")

    except Exception as e:
        LOGGER.error(f"❌ 그래프 생성 중 오류: {str(e)}")


def main():
    """
    메인 함수
    """
    print('=' * 80)
    LOGGER.info('📈 SAC 모델 백테스트 시작')

    # 인자 파싱
    args = parse_args()

    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS
    if isinstance(symbols, list) and len(symbols) > 1:
        LOGGER.warning("⚠️  현재 버전은 단일 심볼만 지원합니다. 첫 번째 심볼을 사용합니다.")
        symbol = symbols[0]
    else:
        symbol = symbols[0] if isinstance(symbols, list) else symbols

    LOGGER.info(f"🎯 백테스트 설정:")
    LOGGER.info(f"   └─ 심볼: {symbol}")
    LOGGER.info(f"   └─ 데이터 유형: {args.data_type}")
    LOGGER.info(f"   └─ 모델 경로: {args.model_path}")
    LOGGER.info(f"   └─ 초기 자본금: ${args.initial_balance:,.2f}")
    LOGGER.info(f"   └─ 거래 수수료: {args.transaction_fee_percent:.4f}")
    LOGGER.info(f"   └─ 윈도우 크기: {args.window_size}")
    LOGGER.info(f"   └─ DB 저장: {'아니오 (--no_db)' if args.no_db else '예 (기본값)'}")
    LOGGER.info(f"   └─ 그래프 저장: {'예' if args.save_plots else '아니오'}")

    # 결과 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / f"backtest_{symbol}_{timestamp}"
    create_directory(results_dir)

    LOGGER.info(f"📁 결과 저장 경로: {results_dir}")

    # 데이터 수집
    LOGGER.info("📊 데이터 수집 중...")
    collector = DataCollector(symbols=[symbol])

    if args.collect_data:
        LOGGER.info("🔄 새로운 데이터 수집 중...")
        data = collector.load_and_save()
    else:
        LOGGER.info("💾 저장된 데이터 로드 중...")
        data = collector.load_all_data()

        if not data:
            LOGGER.warning("⚠️  저장된 데이터가 없어 새로 수집합니다.")
            data = collector.load_and_save()

    if not data or symbol not in data:
        LOGGER.error(f"❌ {symbol} 데이터 수집 실패")
        return

    LOGGER.info(f"✅ 데이터 수집 완료: {len(data[symbol])} 행")

    # 데이터 전처리
    LOGGER.info("⚙️  데이터 전처리 중...")
    processor = DataProcessor(window_size=args.window_size)
    results_data = processor.process_all_symbols(data)

    if not results_data or symbol not in results_data:
        LOGGER.error(f"❌ {symbol} 데이터 전처리 실패")
        return

    LOGGER.info(f"✅ 데이터 전처리 완료")

    # 데이터 분할 정보 로깅
    symbol_data = results_data[symbol]
    LOGGER.info(f"📊 {symbol} 데이터 분할 정보:")
    LOGGER.info(f"   └─ 전체: {len(symbol_data['normalized_data'])} 행")
    LOGGER.info(f"   └─ 훈련: {len(symbol_data['train'])} 행")
    LOGGER.info(f"   └─ 검증: {len(symbol_data['valid'])} 행")
    LOGGER.info(f"   └─ 테스트: {len(symbol_data['test'])} 행")

    # 백테스트 환경 생성
    LOGGER.info(f"🏗️  {args.data_type} 환경 생성 중...")
    try:
        env = create_environment_from_results(
            results=results_data,
            symbol=symbol,
            data_type=args.data_type,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee_percent=args.transaction_fee_percent
        )

        LOGGER.info(f"✅ 백테스트 환경 생성 완료")
        LOGGER.info(f"   └─ 데이터 길이: {env.data_length}")
        LOGGER.info(f"   └─ 특성 차원: {env.feature_dim}")

    except Exception as e:
        LOGGER.error(f"❌ 환경 생성 실패: {str(e)}")
        return

    # 모델 로드
    agent = load_model_with_compatibility(args.model_path, env, args)

    if agent is None:
        LOGGER.error("❌ 모델 로드 실패")
        return

    # 백테스트 실행
    LOGGER.info("🚀 백테스트 실행 중...")
    
    # 백테스팅 시간 변수 초기화 (예외 처리를 위해)
    backtest_start_time = datetime.now()
    backtest_end_time = None
    
    LOGGER.info(f"⏰ 백테스팅 시작 시각: {backtest_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        backtest_results = run_backtest_episode(agent, env, render=args.render)
        
        # 백테스팅 종료 시간 기록
        backtest_end_time = datetime.now()
        LOGGER.info(f"⏰ 백테스팅 종료 시각: {backtest_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 소요 시간 계산
        duration = backtest_end_time - backtest_start_time
        duration_str = str(duration).split('.')[0]  # 마이크로초 제거
        
        LOGGER.info("✅ 백테스트 완료!")
        LOGGER.info(f"   └─ 소요 시간: {duration_str}")
        LOGGER.info(f"   └─ 총 스텝: {backtest_results['total_steps']}")
        LOGGER.info(f"   └─ 최종 포트폴리오: ${backtest_results['final_portfolio_value']:,.2f}")
        LOGGER.info(f"   └─ 총 수익률: {backtest_results['total_return']:.2f}%")

    except Exception as e:
        # 백테스팅 실패 시에도 종료 시간 기록
        backtest_end_time = datetime.now() if backtest_end_time is None else backtest_end_time
        
        LOGGER.error(f"❌ 백테스트 실행 중 오류: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return

    # 성능 지표 계산
    LOGGER.info("📊 성능 지표 계산 중...")
    metrics = calculate_metrics(backtest_results)

    # 결과 저장
    LOGGER.info("💾 결과 저장 중...")
    save_results(backtest_results, metrics, results_dir, symbol)

    # 데이터베이스 저장
    try:
        db_success = save_to_database(
            backtest_results=backtest_results,
            metrics=metrics,
            symbol=symbol,
            model_path=args.model_path,
            data_type=args.data_type,
            args=args,
            backtest_start_time=backtest_start_time,
            backtest_end_time=backtest_end_time
        )

        if db_success:
            LOGGER.info("💾 모든 결과 저장 완료 (JSON + DB)")
        else:
            LOGGER.warning("💾 JSON 저장 완료, DB 저장은 실패")

    except Exception as e:
        LOGGER.error(f"❌ 데이터베이스 저장 중 심각한 오류: {str(e)}")
        if not args.skip_db_on_error:
            return  # DB 저장 실패 시 종료

    # 그래프 생성 (요청된 경우)
    if args.save_plots:
        LOGGER.info("📊 그래프 생성 중...")
        create_plots(backtest_results, metrics, results_dir, symbol)

    # 최종 결과 출력
    LOGGER.info("=" * 80)
    LOGGER.info(f"🎉 백테스트 완료 - {symbol} ({args.data_type} 데이터)")
    LOGGER.info("=" * 80)
    LOGGER.info(f"📁 결과 저장 경로: {results_dir}")
    LOGGER.info("")
    LOGGER.info(f"💰 성능 요약:")
    LOGGER.info(f"   └─ 초기 자본금: ${backtest_results['initial_portfolio_value']:,.2f}")
    LOGGER.info(f"   └─ 최종 포트폴리오: ${backtest_results['final_portfolio_value']:,.2f}")
    LOGGER.info(f"   └─ 총 수익률: {backtest_results['total_return']:.2f}%")
    LOGGER.info(f"   └─ 연간 수익률: {metrics.get('annual_return', 0):.2f}%")
    LOGGER.info(f"   └─ 변동성: {metrics.get('volatility', 0):.2f}%")
    LOGGER.info(f"   └─ 샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}")
    LOGGER.info(f"   └─ 최대 낙폭: {metrics.get('max_drawdown', 0):.2f}%")
    LOGGER.info("")
    LOGGER.info(f"🔄 거래 통계:")
    LOGGER.info(f"   └─ 총 거래 횟수: {metrics.get('total_trades', 0)}")
    LOGGER.info(f"   └─ 매수 거래: {metrics.get('buy_trades', 0)}")
    LOGGER.info(f"   └─ 매도 거래: {metrics.get('sell_trades', 0)}")
    LOGGER.info(f"   └─ 승률: {metrics.get('win_rate', 0):.2f}%")
    LOGGER.info(f"   └─ 총 스텝: {metrics.get('total_steps', 0)}")
    LOGGER.info("=" * 80)

    # 간단한 벤치마크 비교 (Buy & Hold)
    initial_price = backtest_results['prices'][0]
    final_price = backtest_results['prices'][-1]
    buy_hold_return = (final_price - initial_price) / initial_price * 100

    LOGGER.info(f"📈 벤치마크 비교:")
    LOGGER.info(f"   └─ SAC 모델: {backtest_results['total_return']:.2f}%")
    LOGGER.info(f"   └─ Buy & Hold: {buy_hold_return:.2f}%")
    if backtest_results['total_return'] > buy_hold_return:
        LOGGER.info(f"   └─ ✅ SAC 모델이 {backtest_results['total_return'] - buy_hold_return:.2f}%p 더 좋음")
    else:
        LOGGER.info(f"   └─ ❌ Buy & Hold가 {buy_hold_return - backtest_results['total_return']:.2f}%p 더 좋음")
    
    LOGGER.info("=" * 80)
    LOGGER.info("🏁 백테스트 완료!")
    
    # 추가 분석 제안
    if args.data_type == 'test':
        LOGGER.info("💡 팁: 이는 최종 테스트 결과입니다.")
    elif args.data_type == 'valid':
        LOGGER.info("💡 팁: 검증 데이터 결과입니다. 최종 성능은 --data_type test로 확인하세요.")
    else:
        LOGGER.info("💡 팁: 훈련 데이터 결과입니다. 실제 성능은 --data_type valid 또는 test로 확인하세요.")
    
    if not args.save_plots:
        LOGGER.info("💡 팁: 그래프를 보려면 --save_plots 옵션을 사용하세요.")
    
    return {
        'results': backtest_results,
        'metrics': metrics,
        'symbol': symbol,
        'data_type': args.data_type,
        'results_dir': str(results_dir)
    }


if __name__ == "__main__":
    main()