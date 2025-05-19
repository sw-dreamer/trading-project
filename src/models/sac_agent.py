"""
SAC (Soft Actor-Critic) 알고리즘 구현 모듈
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import deque
import random
import pickle
import time
from pathlib import Path
import sys
import os


from src.config.config import (
    DEVICE,
    HIDDEN_DIM,
    GAMMA,
    TAU,
    LEARNING_RATE_ACTOR,
    LEARNING_RATE_CRITIC,
    LEARNING_RATE_ALPHA,
    ALPHA_INIT,
    REPLAY_BUFFER_SIZE,
    TARGET_UPDATE_INTERVAL,
    BATCH_SIZE,
    MODELS_DIR,
    LOGGER
)
from src.models.networks import ActorNetwork, CriticNetwork, CNNActorNetwork, CNNCriticNetwork
from src.utils.utils import soft_update, create_directory


class ReplayBuffer:
    """
    경험 리플레이 버퍼
    RL 에이전트가 경험한 샘플을 저장하고 무작위로 샘플링
    """
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        """
        ReplayBuffer 클래스 초기화
        
        Args:
            capacity: 버퍼의 최대 용량
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        버퍼에 샘플 추가
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 종료 여부
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        버퍼에서 무작위로 샘플 추출
        
        Args:
            batch_size: 추출할 샘플 수
            
        Returns:
            (상태, 행동, 보상, 다음 상태, 종료 여부) 튜플
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        
        return state, action, reward, next_state, done
    
    def __len__(self) -> int:
        """
        버퍼의 현재 크기 반환
        
        Returns:
            버퍼의 현재 크기
        """
        return len(self.buffer)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        버퍼를 파일로 저장
        
        Args:
            path: 저장할 파일 경로
        """
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        파일에서 버퍼 로드
        
        Args:
            path: 로드할 파일 경로
        """
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
        self.position = len(self.buffer) % self.capacity


class SACAgent:
    """
    SAC 알고리즘 에이전트
    """
        
        # Fix in the __init__ method of SACAgent class
    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        actor_lr: float = LEARNING_RATE_ACTOR,
        critic_lr: float = LEARNING_RATE_CRITIC,
        alpha_lr: float = LEARNING_RATE_ALPHA,
        gamma: float = GAMMA,
        tau: float = TAU,
        alpha_init: float = ALPHA_INIT,
        target_update_interval: int = TARGET_UPDATE_INTERVAL,
        use_automatic_entropy_tuning: bool = True,
        device: torch.device = DEVICE,
        buffer_capacity: int = REPLAY_BUFFER_SIZE,
        input_shape: Tuple[int, int] = None,
        use_cnn: bool = False
    ):
        """
        SACAgent 클래스 초기화
        
        Args:
            state_dim: 상태 공간의 차원 (CNN 사용 시 None)
            action_dim: 행동 공간의 차원
            hidden_dim: 신경망 은닉층의 차원
            actor_lr: Actor 네트워크 학습률
            critic_lr: Critic 네트워크 학습률
            alpha_lr: 엔트로피 계수 학습률
            gamma: 할인 계수
            tau: 타겟 네트워크 소프트 업데이트 계수
            alpha_init: 초기 엔트로피 계수
            target_update_interval: 타겟 네트워크 업데이트 간격
            use_automatic_entropy_tuning: 자동 엔트로피 조정 사용 여부
            device: 학습에 사용할 장치
            buffer_capacity: 리플레이 버퍼 용량
            input_shape: 입력 데이터 형태 (CNN 사용 시 (window_size, feature_dim))
            use_cnn: CNN 모델 사용 여부
        """
        # Store all attributes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.tau = tau
        self.alpha_init = alpha_init
        self.target_update_interval = target_update_interval
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.device = device
        self.use_cnn = use_cnn
        self.input_shape = input_shape
        
        # 네트워크 초기화
        if use_cnn:
            if input_shape is None:
                raise ValueError("CNN 모델을 사용할 때는 input_shape가 필요합니다.")
            
            self.actor = CNNActorNetwork(
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device
            )
            
            self.critic = CNNCriticNetwork(
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device
            )
            
            self.critic_target = CNNCriticNetwork(
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device
            )
            
            # 현재 정책을 타겟 정책으로 복사
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                print(f'target_param : {target_param}')
                print(f'param : {param}')
                target_param.data.copy_(param.data)
                
        else:
            if state_dim is None:
                raise ValueError("일반 모델을 사용할 때는 state_dim이 필요합니다.")
            
            self.actor = ActorNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device
            )
            
            self.critic = CriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device
            )
            
            self.critic_target = CriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device
            )
            
            # 현재 정책을 타겟 정책으로 복사
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 자동 엔트로피 조정
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -action_dim  # 행동 차원에 따른 목표 엔트로피
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = torch.tensor(alpha_init, device=device)
        
        # 리플레이 버퍼
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # 훈련 단계 카운터
        self.train_step_counter = 0
        
        # 학습 통계
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.entropy_values = []
        
        LOGGER.info(f"SAC 에이전트 초기화 완료: 행동 차원 {action_dim}, {'CNN 사용' if use_cnn else '일반 모델 사용'}")
    
    def debug_state_format(state):
        """
        상태 객체의 형태를 자세히 출력하여 디버깅
        
        Args:
            state: 디버깅할 상태 객체
        """
        if isinstance(state, dict):
            print("State is a dictionary with keys:", list(state.keys()))
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key} shape: {value.shape}, dtype: {value.dtype}")
                    print(f"  {key} min: {value.min()}, max: {value.max()}, mean: {value.mean()}")
                    # 첫 몇 개 값 출력
                    print(f"  {key} first few values: {value.flatten()[:5]}")
                else:
                    print(f"  {key} type: {type(value)}")
        elif isinstance(state, np.ndarray):
            print(f"State is an ndarray with shape: {state.shape}, dtype: {state.dtype}")
            print(f"State min: {state.min()}, max: {state.max()}, mean: {state.mean()}")
            print(f"First few values: {state.flatten()[:5]}")
        else:
            print(f"State is of type: {type(state)}")
            
    def select_action(self, state: Any, evaluate: bool = False) -> np.ndarray:
        """
        상태에 따른 행동 선택
        
        Args:
            state: 현재 상태
            evaluate: 평가 모드 여부 (True일 경우 탐색 없이 평균 행동 선택)
            
        Returns:
            선택된 행동
        """
        # 상태를 텐서로 변환
        if self.use_cnn:
            # Dictionary 형태의 상태 처리
            state_tensor = {key: torch.FloatTensor(value).unsqueeze(0).to(self.device) 
                            if isinstance(value, np.ndarray) else value.unsqueeze(0).to(self.device) 
                            for key, value in state.items()}
        else:
            # 일반 배열 형태의 상태 처리
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                market_data = state['market_data']            # shape: (30, 40)
                portfolio_state = state['portfolio_state']    # shape: (2,)

                # Convert to tensors
                market_data_tensor = torch.tensor(market_data).float().to(self.device)     # (30, 40)
                portfolio_state_tensor = torch.tensor(portfolio_state).float().to(self.device)  # (2,)

                # Flatten market_data to (1200,) and concatenate with portfolio_state (2,) → (1202,)
                market_data_flat = market_data_tensor.reshape(-1)
                state_tensor = torch.cat([market_data_flat, portfolio_state_tensor], dim=0).unsqueeze(0)  # (1, 1202)


        # Evaluate mode: Choose average action (exploitative)
        if evaluate:
            _, _, action = self.actor.sample(state_tensor)
        else:
            # Training mode: Sample action from policy (exploratory)
            action, _, _ = self.actor.sample(state_tensor)
        
        # Return the action as a numpy array
        return action.detach().cpu().numpy()[0]

    
    def process_state_for_network(self, state: Any) -> Any:
        """
        네트워크 입력을 위한 상태 처리
        
        Args:
            state: 원본 상태
            
        Returns:
            처리된 상태
        """
        if self.use_cnn:
            if isinstance(state, dict):
                state_dict = {}
                LOGGER.debug(f"[CNN] Received state dict keys: {list(state.keys())}")
                for key, value in state.items():
                    if isinstance(value, np.ndarray):
                        state_dict[key] = torch.FloatTensor(value).to(self.device)
                    else:
                        state_dict[key] = value.to(self.device)
                return state_dict
            else:
                LOGGER.error("CNN 모델을 사용할 때는 상태가 딕셔너리 형태여야 합니다.")
                return None

        else:
            if isinstance(state, dict):
                LOGGER.debug(f"[MLP] Received state dict keys: {list(state.keys())}")
                
                # 'market_data'와 'portfolio_state'를 합침
                if 'market_data' in state and 'portfolio_state' in state:
                    market_data = state['market_data']  # shape: (30, N)
                    portfolio_state = state['portfolio_state']  # shape: (2,)
                    
                    # market_data 형태 확인 및 평탄화
                    if len(market_data.shape) == 2:
                        # market_data를 1차원으로 평탄화 (flatten)
                        flattened_market_data = market_data.flatten()
                        
                        # portfolio_state와 concatenate
                        combined = np.concatenate([flattened_market_data, portfolio_state])
                        return torch.FloatTensor(combined).unsqueeze(0).to(self.device)
                    else:
                        LOGGER.error(f"[MLP] market_data 형식이 예상과 다릅니다: {market_data.shape}")
                        return None
                else:
                    LOGGER.error(f"[MLP] dict 상태에 'market_data'와 'portfolio_state' 키가 없습니다: {list(state.keys())}")
                    return None

            elif isinstance(state, np.ndarray):
                return torch.FloatTensor(state).to(self.device)
            else:
                return state.to(self.device)



    
    def update_parameters(self, batch_size: int = BATCH_SIZE) -> Dict[str, float]:
        """
        네트워크 파라미터 업데이트 (학습)
        
        Args:
            batch_size: 미니배치 크기
            
        Returns:
            학습 통계 (actor_loss, critic_loss, alpha_loss, entropy)
        """
        # 버퍼에 충분한 샘플이 없으면 업데이트 건너뛰기
        if len(self.replay_buffer) < batch_size:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'alpha_loss': 0.0,
                'entropy': 0.0
            }
        
        # 리플레이 버퍼에서 미니배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 상태 및 행동 처리
        if self.use_cnn:
            # CNN 모델을 위한 배치 처리
            batched_states = {
                'market_data': torch.cat([torch.FloatTensor(s['market_data']) if isinstance(s['market_data'], np.ndarray) else s['market_data'] for s in states], dim=0).to(self.device),
                'portfolio_state': torch.cat([torch.FloatTensor(s['portfolio_state']) if isinstance(s['portfolio_state'], np.ndarray) else s['portfolio_state'] for s in states], dim=0).to(self.device)
            }
            
            batched_next_states = {
                'market_data': torch.cat([torch.FloatTensor(s['market_data']) if isinstance(s['market_data'], np.ndarray) else s['market_data'] for s in next_states], dim=0).to(self.device),
                'portfolio_state': torch.cat([torch.FloatTensor(s['portfolio_state']) if isinstance(s['portfolio_state'], np.ndarray) else s['portfolio_state'] for s in next_states], dim=0).to(self.device)
            }
        else:
            # 일반 모델을 위한 배치 처리
            batched_states = torch.cat([self.process_state_for_network(s) for s in states], dim=0)
            batched_next_states = torch.cat([self.process_state_for_network(s) for s in next_states], dim=0)
        
        batched_actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        batched_rewards = torch.FloatTensor(np.vstack(rewards)).to(self.device)
        batched_dones = torch.FloatTensor(np.vstack(dones)).to(self.device)
        
        # 현재 정책에서 다음 행동 샘플링
        next_actions, next_log_probs, _ = self.actor.sample(batched_next_states)
        
        # 타겟 Q 값 계산
        next_q1_target, next_q2_target = self.critic_target(batched_next_states, next_actions)
        next_q_target = torch.min(next_q1_target, next_q2_target)
        next_q_target = next_q_target - self.alpha * next_log_probs
        expected_q = batched_rewards + (1.0 - batched_dones) * self.gamma * next_q_target
        
        # Critic 업데이트
        current_q1, current_q2 = self.critic(batched_states, batched_actions)
        q1_loss = F.mse_loss(current_q1, expected_q.detach())
        q2_loss = F.mse_loss(current_q2, expected_q.detach())
        critic_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor 업데이트
        new_actions, log_probs, _ = self.actor.sample(batched_states)
        q1, q2 = self.critic(batched_states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 자동 엔트로피 조정
        alpha_loss = 0.0
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # 타겟 네트워크 소프트 업데이트
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        # 학습 통계 기록
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.alpha_losses.append(alpha_loss.item() if self.use_automatic_entropy_tuning else 0.0)
        self.entropy_values.append(-log_probs.mean().item())
        LOGGER.info(f"'actor_loss': {actor_loss.item()}")
        LOGGER.info(f"'critic_loss': {critic_loss.item()}")
        LOGGER.info(f"'alpha_loss': {alpha_loss.item() if self.use_automatic_entropy_tuning else 0.0}")
        LOGGER.info(f"'entropy': {log_probs.mean().item()}")
        LOGGER.info(f"'alpha': {actor_loss.item()}")
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item() if self.use_automatic_entropy_tuning else 0.0,
            'entropy': -log_probs.mean().item(),
            'alpha': self.alpha.item()
        }
    
    # Fix in the save_model method of SACAgent
    def save_model(self, save_dir: Union[str, Path] = None, prefix: str = '') -> None:
        """
        모델 저장
        
        Args:
            save_dir: 저장 디렉토리 (None인 경우 기본값 사용)
            prefix: 파일명 접두사
        """
        if save_dir is None:
            save_dir = MODELS_DIR
        
        create_directory(save_dir)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = Path(save_dir) / f"{prefix}sac_model_{timestamp}"
        create_directory(model_path)
        
        # 네트워크 가중치 저장
        torch.save(self.actor.state_dict(), model_path / "actor.pth")
        torch.save(self.critic.state_dict(), model_path / "critic.pth")
        torch.save(self.critic_target.state_dict(), model_path / "critic_target.pth")
        
        # 옵티마이저 상태 저장
        torch.save(self.actor_optimizer.state_dict(), model_path / "actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(), model_path / "critic_optimizer.pth")
        
        # Alpha 관련 상태 저장
        if self.use_automatic_entropy_tuning:
            torch.save(self.log_alpha, model_path / "log_alpha.pth")
            torch.save(self.alpha_optimizer.state_dict(), model_path / "alpha_optimizer.pth")
        
        # 학습 통계 저장
        training_stats = {
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'alpha_losses': self.alpha_losses,
            'entropy_values': self.entropy_values,
            'train_step_counter': self.train_step_counter
        }
        torch.save(training_stats, model_path / "training_stats.pth")
        
        # 설정 저장
        config = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha_init': self.alpha_init,
            'target_update_interval': self.target_update_interval,
            'use_automatic_entropy_tuning': self.use_automatic_entropy_tuning,
            'use_cnn': self.use_cnn,
            'input_shape': self.input_shape
        }
        torch.save(config, model_path / "config.pth")
        
        # Remove the incorrect save operations that reference critic1, critic2, etc.
        # These lines should be deleted:
        # torch.save(self.actor.state_dict(), model_path / "actor.pth")
        # torch.save(self.critic1.state_dict(), model_path / "critic1.pth")
        # torch.save(self.critic2.state_dict(), model_path / "critic2.pth")
        # torch.save(self.target_critic1.state_dict(), model_path / "target_critic1.pth")
        # torch.save(self.target_critic2.state_dict(), model_path / "target_critic2.pth")
            
        LOGGER.info(f"모델 저장 완료: {model_path}")
        
        return model_path
    
    def load_model(self, model_path: Union[str, Path]) -> None:
       
        """
        모델 로드
        
        Args:
            model_path: 모델 디렉토리 경로
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            LOGGER.error(f"모델 경로가 존재하지 않습니다: {model_path}")
            return
        
        try:
            # 먼저 원래 방식으로 로드 시도
            self.actor.load_state_dict(torch.load(model_path / "actor.pth", map_location=self.device))
            self.critic.load_state_dict(torch.load(model_path / "critic.pth", map_location=self.device))
            self.critic_target.load_state_dict(torch.load(model_path / "critic_target.pth", map_location=self.device))
            
            # 옵티마이저 상태 로드
            self.actor_optimizer.load_state_dict(torch.load(model_path / "actor_optimizer.pth", map_location=self.device))
            self.critic_optimizer.load_state_dict(torch.load(model_path / "critic_optimizer.pth", map_location=self.device))
            
            # Alpha 관련 로드
            if self.use_automatic_entropy_tuning:
                self.log_alpha = torch.load(model_path / "log_alpha.pth", map_location=self.device)
                self.log_alpha.requires_grad = True
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer.load_state_dict(torch.load(model_path / "alpha_optimizer.pth", map_location=self.device))
            else:
                self.alpha = torch.tensor(self.alpha_init, device=self.device)
            
            # 학습 통계 로드
            training_stats = torch.load(model_path / "training_stats.pth", map_location=self.device)
            self.actor_losses = training_stats.get('actor_losses', [])
            self.critic_losses = training_stats.get('critic_losses', [])
            self.alpha_losses = training_stats.get('alpha_losses', [])
            self.entropy_values = training_stats.get('entropy_values', [])
            self.train_step_counter = training_stats.get('train_step_counter', 0)
            
            LOGGER.info(f"모델 로드 완료: {model_path}")
        except RuntimeError as e:
            # 크기 불일치 오류 발생 시 크기 조정 로드 메서드 사용
            LOGGER.warning(f"기본 로드 실패, 크기 조정 방식으로 로드 시도: {e}")
            self.load_model_with_resize(model_path)
    
    
    def get_latest_model_path(self, save_dir: Union[str, Path] = None, prefix: str = '') -> Optional[Path]:
        """
        최신 모델 경로 반환
        
        Args:
            save_dir: 모델 디렉토리 (None인 경우 기본값 사용)
            prefix: 파일명 접두사
            
        Returns:
            최신 모델 경로 (없으면 None)
        """
        if save_dir is None:
            save_dir = MODELS_DIR
        
        save_dir = Path(save_dir)
        if not save_dir.exists():
            return None
        
        model_dirs = [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith(f"{prefix}sac_model_")]
        if not model_dirs:
            return None
        
        # 타임스탬프 기준으로 정렬
        model_dirs.sort(key=lambda d: d.name, reverse=True)
        return model_dirs[0]
    
    def load_model_with_resize(self, model_path):
        model_path = Path(model_path)
        
        # 저장된 상태 사전 로드
        saved_actor_dict = torch.load(model_path / "actor.pth", map_location=self.device)
        saved_critic_dict = torch.load(model_path / "critic.pth", map_location=self.device)
        
        # 현재 모델의 상태 사전
        current_actor_dict = self.actor.state_dict()
        current_critic_dict = self.critic.state_dict()
        
        # 크기가 일치하는 파라미터만 로드
        actor_dict = {k: v for k, v in saved_actor_dict.items() if k in current_actor_dict and v.shape == current_actor_dict[k].shape}
        critic_dict = {k: v for k, v in saved_critic_dict.items() if k in current_critic_dict and v.shape == current_critic_dict[k].shape}
        
        # 상태 사전 업데이트
        current_actor_dict.update(actor_dict)
        current_critic_dict.update(critic_dict)
        
        # 모델에 적용
        self.actor.load_state_dict(current_actor_dict)
        self.critic.load_state_dict(current_critic_dict)
        
        # critic_target 업데이트
        try:
            saved_critic_target_dict = torch.load(model_path / "critic_target.pth", map_location=self.device)
            current_critic_target_dict = self.critic_target.state_dict()
            critic_target_dict = {k: v for k, v in saved_critic_target_dict.items() if k in current_critic_target_dict and v.shape == current_critic_target_dict[k].shape}
            current_critic_target_dict.update(critic_target_dict)
            self.critic_target.load_state_dict(current_critic_target_dict)
        except:
            # critic_target 파일이 없으면 critic을 복사
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
        
        # 옵티마이저와 기타 상태 로드 (가능한 경우)
        try:
            self.actor_optimizer.load_state_dict(torch.load(model_path / "actor_optimizer.pth", map_location=self.device))
            self.critic_optimizer.load_state_dict(torch.load(model_path / "critic_optimizer.pth", map_location=self.device))
            
            if self.use_automatic_entropy_tuning:
                self.log_alpha = torch.load(model_path / "log_alpha.pth", map_location=self.device)
                self.log_alpha.requires_grad = True
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer.load_state_dict(torch.load(model_path / "alpha_optimizer.pth", map_location=self.device))
        except:
            LOGGER.warning(f"옵티마이저 또는 기타 상태 로드 실패. 기본값 유지.")
        
        # 학습 통계 로드 (가능한 경우)
        try:
            training_stats = torch.load(model_path / "training_stats.pth", map_location=self.device)
            self.actor_losses = training_stats.get('actor_losses', [])
            self.critic_losses = training_stats.get('critic_losses', [])
            self.alpha_losses = training_stats.get('alpha_losses', [])
            self.entropy_values = training_stats.get('entropy_values', [])
            self.train_step_counter = training_stats.get('train_step_counter', 0)
        except:
            LOGGER.warning(f"학습 통계 로드 실패. 기본값 유지.")
        
        LOGGER.info(f"모델 로드 완료(일부 파라미터 크기 불일치로 무시됨): {model_path}")

if __name__ == "__main__":
    # 모듈 테스트 코드
    # 일반 SAC 에이전트 테스트
    state_dim = 10
    action_dim = 1
    batch_size = 4
    
    # 에이전트 초기화
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    
    # 리플레이 버퍼에 샘플 추가
    for _ in range(batch_size * 2):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn(1)
        next_state = np.random.randn(state_dim)
        done = np.random.randint(0, 2, (1,))
        
        agent.replay_buffer.push(
            torch.FloatTensor(state).to(agent.device),
            torch.FloatTensor(action).to(agent.device),
            reward,
            torch.FloatTensor(next_state).to(agent.device),
            done
        )
    
    # 학습 테스트
    for _ in range(5):
        stats = agent.update_parameters()
        print(f"Learning stats: {stats}")
    
    # 모델 저장 및 로드 테스트
    model_path = agent.save_model()
    agent.load_model(model_path)
    
    # CNN SAC 에이전트 테스트
    window_size = 30
    feature_dim = 5
    
    # CNN 에이전트 초기화
    cnn_agent = SACAgent(
        action_dim=action_dim,
        input_shape=(window_size, feature_dim),
        use_cnn=True
    )
    
    # 리플레이 버퍼에 샘플 추가
    for _ in range(batch_size * 2):
        state = {
            'market_data': torch.FloatTensor(np.random.randn(window_size, feature_dim)).to(cnn_agent.device),
            'portfolio_state': torch.FloatTensor(np.random.randn(2)).to(cnn_agent.device)
        }
        action = np.random.randn(action_dim)
        reward = np.random.randn(1)
        next_state = {
            'market_data': torch.FloatTensor(np.random.randn(window_size, feature_dim)).to(cnn_agent.device),
            'portfolio_state': torch.FloatTensor(np.random.randn(2)).to(cnn_agent.device)
        }
        done = np.random.randint(0, 2, (1,))
        
        cnn_agent.replay_buffer.push(state, torch.FloatTensor(action).to(cnn_agent.device), reward, next_state, done)
    
    # 학습 테스트
    for _ in range(5):
        stats = cnn_agent.update_parameters()
        print(f"CNN Learning stats: {stats}")
    
    # 모델 저장 및 로드 테스트
    model_path = cnn_agent.save_model(prefix='cnn_')
    cnn_agent.load_model(model_path) 