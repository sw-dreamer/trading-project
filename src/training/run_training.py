"""
SAC 모델 학습 실행 스크립트 (훈련 집중 버전)
- 검증 없이 훈련에만 집중
- 학습 시간 측정 및 로깅
- 간단하고 명확한 로그 출력
"""
import sys
import os
import time
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

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
from src.utils.utils import create_directory, get_timestamp

class TrainingTimer:
    """학습 시간 측정을 위한 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.episode_start_time = None
        self.episode_times = []
        
    def start_training(self):
        """전체 학습 시작"""
        self.start_time = time.time()
        LOGGER.info(f"🚀 학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def start_episode(self):
        """에피소드 시작"""
        self.episode_start_time = time.time()
        
    def end_episode(self):
        """에피소드 종료"""
        if self.episode_start_time:
            episode_time = time.time() - self.episode_start_time
            self.episode_times.append(episode_time)
            return episode_time
        return 0
        
    def get_training_time(self):
        """전체 학습 시간 반환"""
        if self.start_time:
            return time.time() - self.start_time
        return 0
        
    def get_avg_episode_time(self):
        """평균 에피소드 시간 반환"""
        if self.episode_times:
            return np.mean(self.episode_times)
        return 0
        
    def get_eta(self, current_episode, total_episodes):
        """남은 시간 추정"""
        if len(self.episode_times) > 0:
            avg_time = self.get_avg_episode_time()
            remaining_episodes = total_episodes - current_episode
            return remaining_episodes * avg_time
        return 0
        
    def format_time(self, seconds):
        """시간을 보기 좋게 포맷"""
        return str(timedelta(seconds=int(seconds)))

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='SAC 모델 학습 스크립트 (훈련 집중 버전)')
    
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
    parser.add_argument('--log_interval', type=int, default=EVALUATE_INTERVAL, help='로그 출력 간격')
    parser.add_argument('--save_interval', type=int, default=SAVE_MODEL_INTERVAL, help='모델 저장 간격')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_PER_EPISODE, help='에피소드당 최대 스텝 수')
    
    return parser.parse_args()

def create_training_environment(results, symbols, args):
    """학습용 환경을 생성 (훈련 데이터만 사용)"""
    LOGGER.info("📚 학습용 환경 생성 중...")
    
    if args.multi_asset:
        # 다중 자산 환경 생성
        LOGGER.info("🏢 다중 자산 트레이딩 환경 생성 중...")
        
        train_data_dict = {}
        raw_data_dict = {}
        
        for symbol in symbols:
            if symbol not in results:
                LOGGER.warning(f"⚠️  {symbol} 데이터 처리 결과가 없습니다.")
                continue
            
            # DataProcessor에서 이미 분할된 훈련 데이터 사용
            if 'train' in results[symbol] and 'featured_data' in results[symbol]:
                train_data_dict[symbol] = results[symbol]['train']  # 정규화된 훈련 데이터
                raw_data_dict[symbol] = results[symbol]['featured_data']  # 원본 특성 데이터 (가격 정보용)
            else:
                LOGGER.error(f"❌ {symbol} 훈련 데이터가 없습니다.")
                continue
        
        env = MultiAssetTradingEnvironment(
            data_dict=train_data_dict,
            raw_data_dict=raw_data_dict,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            train_data=True
        )
    else:
        # 단일 자산 환경 생성
        symbol = symbols[0]
        LOGGER.info(f"🏪 단일 자산 트레이딩 환경 생성 중: {symbol}")
        
        if symbol not in results:
            LOGGER.error(f"❌ {symbol} 데이터 처리 결과가 없습니다.")
            return None
        
        # DataProcessor에서 이미 분할된 훈련 데이터 사용
        if 'train' in results[symbol] and 'featured_data' in results[symbol]:
            normalized_data = results[symbol]['train']  # 정규화된 훈련 데이터
            original_data = results[symbol]['featured_data']  # 원본 특성 데이터 (가격 정보용)
        else:
            LOGGER.error(f"❌ {symbol} 훈련 데이터가 없습니다.")
            return None
        
        env = TradingEnvironment(
            data=normalized_data,
            raw_data=original_data,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            symbol=symbol,
            train_data=True
        )
    
    LOGGER.info(f"✅ 학습 환경 생성 완료")
    return env

def create_agent(env, args):
    """SAC 에이전트 생성"""
    LOGGER.info("🤖 SAC 에이전트 생성 중...")
    
    # 행동 차원 결정
    if args.multi_asset:
        action_dim = len(env.envs)
    else:
        action_dim = 1
    
    if args.use_cnn:
        LOGGER.warning("⚠️  CNN 모드는 현재 구현 중입니다. MLP 모드로 대체합니다.")
        args.use_cnn = False
    
    # MLP 모델 사용
    agent = SACAgent(
        state_dim=None,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        input_shape=(args.window_size, env.feature_dim if not args.multi_asset else list(env.envs.values())[0].feature_dim),
        use_cnn=False
    )
    
    # 모델 로드 (지정된 경우)
    if args.load_model:
        LOGGER.info(f"📂 모델 로드 중: {args.load_model}")
        try:
            agent.load_model(args.load_model)
            LOGGER.info("✅ 모델 로드 성공")
        except Exception as e:
            LOGGER.error(f"❌ 모델 로드 실패: {e}")
            LOGGER.info("🆕 새 모델로 학습을 시작합니다.")
    
    LOGGER.info(f"✅ SAC 에이전트 생성 완료 (행동 차원: {action_dim}, 은닉층: {args.hidden_dim})")
    return agent

def train_agent(agent, train_env, args, timer):
    """에이전트 학습 (훈련에만 집중)"""
    LOGGER.info("🎯 학습 시작...")
    
    episode_rewards = []
    portfolio_values = []
    shares_history = []
    
    for episode in range(args.num_episodes):
        timer.start_episode()
        
        state = train_env.reset()
        episode_reward = 0
        steps = 0
        
        # 훈련 에피소드 실행
        while steps < args.max_steps:
            # 행동 선택
            action = agent.select_action(state, evaluate=False)
            
            # 환경에서 스텝 실행
            next_state, reward, done, info = train_env.step(action)
            
            # 경험 저장
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 네트워크 업데이트
            if len(agent.replay_buffer) >= args.batch_size:
                stats = agent.update_parameters(batch_size=args.batch_size)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # 에피소드 종료 조건 체크 및 로깅
            if done:
                LOGGER.info(f"에피소드 {episode+1} 완료: {steps} 스텝, 보상 {episode_reward:.4f}")
                break
            
            # 디버깅: 주기적으로 상태 출력 (첫 번째 에피소드만)
            if episode == 0 and steps % 100 == 0:
                LOGGER.info(f"Episode {episode+1}, Step {steps}: reward={reward:.4f}, done={done}, portfolio=${info['portfolio_value']:.2f}")
        
        # 최대 스텝에 도달해서 종료된 경우
        if steps >= args.max_steps:
            LOGGER.info(f"에피소드 {episode+1} 최대 스텝 도달로 종료: {steps} 스텝, 보상 {episode_reward:.4f}")
        
        episode_time = timer.end_episode()
        episode_rewards.append(episode_reward)
        portfolio_values.append(info['portfolio_value'])
        shares_history.append(info['shares_held'])
        
        # 주기적 로깅
        if episode % args.log_interval == 0:
            log_training_progress(
                episode, args, episode_rewards, portfolio_values, 
                info, agent, timer, shares_history
            )
        elif episode < 5:  # 처음 5개 에피소드는 항상 로깅
            LOGGER.info(f"에피소드 {episode+1}: 보상 {episode_reward:.4f}, 스텝 {steps}, 포트폴리오 ${info['portfolio_value']:.2f}")
        
        # 주기적 모델 저장
        if episode % args.save_interval == 0 and episode > 0:
            model_path = agent.save_model(prefix=f'checkpoint_episode_{episode+1}_')
            LOGGER.info(f"💾 체크포인트 모델 저장: {model_path}")
    
    LOGGER.info("🎉 학습 완료!")
    return episode_rewards, portfolio_values, shares_history

def log_training_progress(episode, args, episode_rewards, portfolio_values, 
                         info, agent, timer, shares_history):
    """학습 진행 상황 로깅 (간단 버전)"""
    
    # 훈련 성능 계산
    recent_rewards = episode_rewards[-args.log_interval:] if len(episode_rewards) >= args.log_interval else episode_rewards
    recent_portfolios = portfolio_values[-args.log_interval:] if len(portfolio_values) >= args.log_interval else portfolio_values
    recent_shares = shares_history[-args.log_interval:] if len(shares_history) >= args.log_interval else shares_history
    
    avg_reward = np.mean(recent_rewards)
    avg_portfolio = np.mean(recent_portfolios)
    avg_shares = np.mean(recent_shares)
    total_return = (avg_portfolio - args.initial_balance) / args.initial_balance * 100
    
    # 보유 주식 변화 계산
    shares_change = 0
    shares_change_percent = 0
    if len(shares_history) >= 2:
        prev_shares = shares_history[-2] if len(shares_history) >= 2 else shares_history[0]
        current_shares = shares_history[-1]
        shares_change = current_shares - prev_shares
        if prev_shares != 0:
            shares_change_percent = (shares_change / abs(prev_shares)) * 100
    
    # 시간 정보
    elapsed_time = timer.get_training_time()
    avg_episode_time = timer.get_avg_episode_time()
    eta = timer.get_eta(episode, args.num_episodes)
    
    # 진행률 계산
    progress = episode / args.num_episodes * 100
    
    LOGGER.info("=" * 80)
    LOGGER.info(f"📊 EPISODE {episode+1:,}/{args.num_episodes:,} | 진행률: {progress:.1f}%")
    LOGGER.info("=" * 80)
    
    # 시간 정보
    LOGGER.info(f"⏱️  시간 정보:")
    LOGGER.info(f"   └─ 경과 시간: {timer.format_time(elapsed_time)}")
    LOGGER.info(f"   └─ 평균 에피소드 시간: {avg_episode_time:.2f}초")
    LOGGER.info(f"   └─ 예상 남은 시간: {timer.format_time(eta)}")
    
    # 훈련 성능
    LOGGER.info(f"🏋️  훈련 성능 (최근 {len(recent_rewards)}개 에피소드):")
    LOGGER.info(f"   └─ 평균 보상: {avg_reward:.4f}")
    LOGGER.info(f"   └─ 평균 포트폴리오: ${avg_portfolio:,.2f}")
    LOGGER.info(f"   └─ 현재 현금: ${info['balance']:,.2f}")
    LOGGER.info(f"   └─ 현재 보유 주식: {info['shares_held']:.4f}")
    LOGGER.info(f"   └─ 평균 보유 주식: {avg_shares:.4f}")
    
    # 보유 주식 변화 표시
    if shares_change > 0:
        LOGGER.info(f"   └─ 주식 변화: +{shares_change:.4f} (+{shares_change_percent:.2f}%) 📈")
    elif shares_change < 0:
        LOGGER.info(f"   └─ 주식 변화: {shares_change:.4f} ({shares_change_percent:.2f}%) 📉")
    else:
        LOGGER.info(f"   └─ 주식 변화: {shares_change:.4f} (0.00%) ➡️")
    
    LOGGER.info(f"   └─ 수익률: {total_return:.2f}%")
    
    # 학습 통계
    if len(agent.actor_losses) > 0:
        LOGGER.info(f"📈 학습 통계:")
        LOGGER.info(f"   └─ Actor Loss: {agent.actor_losses[-1]:.6f}")
        LOGGER.info(f"   └─ Critic Loss: {agent.critic_losses[-1]:.6f}")
        LOGGER.info(f"   └─ Alpha: {agent.alpha.item():.6f}")
        LOGGER.info(f"   └─ 버퍼 크기: {len(agent.replay_buffer):,}")
    
    LOGGER.info("=" * 80)

def main():
    """메인 함수"""
    timer = TrainingTimer()
    timer.start_training()
    
    print('=' * 50)
    LOGGER.info('🚀 SAC 모델 학습 시작 (훈련 집중 버전)')
    
    # 인자 파싱
    args = parse_args()
    
    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS
    
    LOGGER.info(f"📈 학습 대상 심볼: {symbols}")
    LOGGER.info(f"⚙️  학습 설정:")
    LOGGER.info(f"   └─ 에피소드 수: {args.num_episodes:,}")
    LOGGER.info(f"   └─ 배치 크기: {args.batch_size}")
    LOGGER.info(f"   └─ 윈도우 크기: {args.window_size}")
    LOGGER.info(f"   └─ 초기 자본금: ${args.initial_balance:,.2f}")
    LOGGER.info(f"   └─ 로그 출력 간격: {args.log_interval} 에피소드")
    LOGGER.info(f"   └─ 모델 저장 간격: {args.save_interval} 에피소드")
    LOGGER.info(f"   └─ 다중 자산: {'예' if args.multi_asset else '아니오'}")
    
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
    
    # 환경 생성 (훈련용만)
    train_env = create_training_environment(results, symbols, args)
    
    if train_env is None:
        LOGGER.error("❌ 훈련 환경 생성 실패")
        return
    
    # 에이전트 생성
    agent = create_agent(train_env, args)
    
    # 학습 실행
    episode_rewards, portfolio_values, shares_history = train_agent(agent, train_env, args, timer)
    
    # 최종 모델 저장
    final_model_path = agent.save_model(prefix='final_')
    LOGGER.info(f"💾 최종 모델 저장: {final_model_path}")
    
    # 최종 결과 출력
    total_time = timer.get_training_time()
    final_portfolio = portfolio_values[-1] if portfolio_values else args.initial_balance
    final_return = (final_portfolio - args.initial_balance) / args.initial_balance * 100
    final_shares = shares_history[-1] if shares_history else 0
    
    LOGGER.info("=" * 80)
    LOGGER.info("🎉 학습 완료 - 최종 결과")
    LOGGER.info("=" * 80)
    LOGGER.info(f"⏱️  총 학습 시간: {timer.format_time(total_time)}")
    LOGGER.info(f"📊 평균 에피소드 시간: {timer.get_avg_episode_time():.2f}초")
    LOGGER.info(f"🎯 학습된 에피소드: {args.num_episodes:,}개")
    LOGGER.info("")
    LOGGER.info(f"📈 훈련 환경 최종 성능:")
    LOGGER.info(f"   └─ 최종 포트폴리오: ${final_portfolio:,.2f}")
    LOGGER.info(f"   └─ 총 수익률: {final_return:.2f}%")
    LOGGER.info(f"   └─ 최종 보유 주식: {final_shares:.4f}")
    LOGGER.info(f"   └─ 평균 에피소드 보상: {np.mean(episode_rewards):.4f}")
    LOGGER.info("")
    LOGGER.info(f"🤖 최종 학습 통계:")
    LOGGER.info(f"   └─ 총 학습 스텝: {agent.train_step_counter:,}")
    LOGGER.info(f"   └─ 최종 버퍼 크기: {len(agent.replay_buffer):,}")
    LOGGER.info(f"   └─ 최종 Alpha 값: {agent.alpha.item():.6f}")
    if agent.actor_losses:
        LOGGER.info(f"   └─ 최종 Actor Loss: {agent.actor_losses[-1]:.6f}")
        LOGGER.info(f"   └─ 최종 Critic Loss: {agent.critic_losses[-1]:.6f}")
    
    LOGGER.info("=" * 80)
    LOGGER.info(f"🏁 학습 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    LOGGER.info("💡 평가를 원하시면 run_evaluation.py를 사용하세요!")

if __name__ == "__main__":
    main()