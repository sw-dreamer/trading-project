"""
SAC 모델 학습을 위한 트레이너 모듈
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import time
from tqdm import tqdm

from src.config.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPISODES,
    EVALUATE_INTERVAL,
    SAVE_MODEL_INTERVAL,
    MAX_STEPS_PER_EPISODE,
    MODELS_DIR,
    RESULTS_DIR,
    LOGGER
)
from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment
from src.utils.utils import create_directory, plot_learning_curve, plot_equity_curve, get_timestamp
class Trainer:
    """
    SAC 모델 학습을 위한 트레이너 클래스
    """
    
    def __init__(
        self,
        agent: SACAgent,
        env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
        batch_size: int = BATCH_SIZE,
        num_episodes: int = NUM_EPISODES,
        evaluate_interval: int = EVALUATE_INTERVAL,
        save_interval: int = SAVE_MODEL_INTERVAL,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        models_dir: Union[str, Path] = MODELS_DIR,
        results_dir: Union[str, Path] = RESULTS_DIR
    ):
        """
        Trainer 클래스 초기화
        
        Args:
            agent: 학습할 SAC 에이전트
            env: 학습에 사용할 트레이딩 환경
            batch_size: 배치 크기
            num_episodes: 학습할 총 에피소드 수
            evaluate_interval: 평가 간격 (에피소드 단위)
            save_interval: 모델 저장 간격 (에피소드 단위)
            max_steps: 에피소드당 최대 스텝 수
            models_dir: 모델 저장 디렉토리
            results_dir: 결과 저장 디렉토리
        """
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # 디렉토리 생성
        create_directory(self.models_dir)
        create_directory(self.results_dir)
        
        # 학습 통계
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.train_losses = []
        
        LOGGER.info(f"Trainer 초기화 완료: {num_episodes}개 에피소드, 배치 크기 {batch_size}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        SAC 모델 학습 수행
        
        Returns:
            학습 통계 딕셔너리
        """
        start_time = time.time()
        timestamp = get_timestamp()
        
        LOGGER.info(f"학습 시작: {self.num_episodes}개 에피소드")
        
        for episode in range(1, self.num_episodes + 1):
            episode_start_time = time.time()
            state = self.env.reset()
            episode_reward = 0
            episode_loss = {"actor_loss": 0, "critic_loss": 0, "alpha_loss": 0, "entropy": 0}
            episode_steps = 0
            done = False
            
            # 에피소드 진행
            while not done and episode_steps < self.max_steps:
                # 행동 선택 (딕셔너리 형태의 상태를 그대로 전달)
                action = self.agent.select_action(state)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, info = self.env.step(action)
                
                # 경험 저장
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # 모델 업데이트
                if len(self.agent.replay_buffer) > self.batch_size:
                    loss = self.agent.update_parameters(self.batch_size)
                    
                    # 손실 누적
                    for k, v in loss.items():
                        if k in episode_loss:
                            episode_loss[k] += v
                        else:
                            episode_loss[k] = v  # 새로운 키가 있을 경우 추가
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            # 에피소드 통계 기록
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # 손실 평균 계산 및 기록
            if episode_steps > 0:
                for k in episode_loss:
                    episode_loss[k] /= episode_steps
            self.train_losses.append(episode_loss)
            
            # 진행 상황 로깅
            episode_time = time.time() - episode_start_time
            LOGGER.info(f"에피소드 {episode}/{self.num_episodes} - 보상: {episode_reward:.2f}, 스텝: {episode_steps}, 시간: {episode_time:.2f}초")
            
            # 주기적 평가
            if episode % self.evaluate_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                LOGGER.info(f"평가 결과 (에피소드 {episode}) - 보상: {eval_reward:.2f}")
            
            # 주기적 모델 저장
            if episode % self.save_interval == 0:
                model_path = self.agent.save_model(self.models_dir, f"episode_{episode}_")
                LOGGER.info(f"모델 저장 완료: {model_path}")
            
            # 학습 곡선 업데이트
            if episode % 10 == 0:
                self._plot_training_curves(timestamp)
        
        # 최종 모델 저장
        final_model_path = self.agent.save_model(self.models_dir, "final_")
        LOGGER.info(f"최종 모델 저장 완료: {final_model_path}")
        
        # 최종 학습 곡선 저장
        self._plot_training_curves(timestamp)
        
        # 학습 시간 계산
        total_time = time.time() - start_time
        LOGGER.info(f"학습 완료: 총 시간 {total_time:.2f}초 ({total_time/60:.2f}분)")

        # 학습 통계 반환
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "actor_losses": [loss["actor_loss"] for loss in self.train_losses],
            "critic_losses": [loss["critic_loss"] for loss in self.train_losses],
            "alpha_losses": [loss["alpha_loss"] for loss in self.train_losses],
            "entropy_values": [loss["entropy"] for loss in self.train_losses]
        }
    
    def evaluate(self, num_episodes: int = 1) -> float:
        """
        현재 정책 평가
        
        Args:
            num_episodes: 평가할 에피소드 수
            
        Returns:
            평균 에피소드 보상
        """
        total_reward = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 평가 모드에서 행동 선택
                action = self.agent.select_action(state, evaluate=True)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def _plot_training_curves(self, timestamp: str) -> None:
        """
        학습 곡선 시각화 및 저장
        
        Args:
            timestamp: 파일명에 사용할 타임스탬프
        """
        # 결과 저장 디렉토리
        result_dir = self.results_dir / f"training_{timestamp}"
        create_directory(result_dir)
        
        # 에피소드 보상 곡선
        plot_learning_curve(
            self.episode_rewards,
            ma_window=10,
            save_path=result_dir / "episode_rewards.png"
        )
        
        # 에피소드 길이 곡선
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_lengths)
        plt.title('에피소드 길이')
        plt.xlabel('에피소드')
        plt.ylabel('스텝 수')
        plt.grid(True, alpha=0.3)
        plt.savefig(result_dir / "episode_lengths.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 손실 곡선
        if self.train_losses:
            # Actor 손실
            plt.figure(figsize=(10, 6))
            plt.plot([loss["actor_loss"] for loss in self.train_losses])
            plt.title('Actor 손실')
            plt.xlabel('에피소드')
            plt.ylabel('손실')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "actor_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Critic 손실
            plt.figure(figsize=(10, 6))
            plt.plot([loss["critic_loss"] for loss in self.train_losses])
            plt.title('Critic 손실')
            plt.xlabel('에피소드')
            plt.ylabel('손실')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "critic_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Alpha 손실
            plt.figure(figsize=(10, 6))
            plt.plot([loss["alpha_loss"] for loss in self.train_losses])
            plt.title('Alpha 손실')
            plt.xlabel('에피소드')
            plt.ylabel('손실')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "alpha_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 엔트로피
            plt.figure(figsize=(10, 6))
            plt.plot([loss["entropy"] for loss in self.train_losses])
            plt.title('정책 엔트로피')
            plt.xlabel('에피소드')
            plt.ylabel('엔트로피')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "entropy.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 평가 보상 곡선
        if self.eval_rewards:
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(self.evaluate_interval, self.num_episodes + 1, self.evaluate_interval)), self.eval_rewards)
        plt.title('평가 보상')
        plt.xlabel('에피소드')
        plt.ylabel('평균 보상')
        plt.grid(True, alpha=0.3)
        plt.savefig(result_dir / "eval_rewards.png", dpi=300, bbox_inches='tight')
        plt.close()
