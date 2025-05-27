"""
SAC (Soft Actor-Critic) 알고리즘 구현 모듈 (LSTM 지원 버전)
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

from src.models.networks import (
    ActorNetwork, 
    CriticNetwork, 
    CNNActorNetwork, 
    CNNCriticNetwork,
    LSTMActorNetwork,
    LSTMCriticNetwork
)

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
    SAC 알고리즘 에이전트 (MLP/CNN/LSTM 지원)
    """
        
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
        use_cnn: bool = False,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        lstm_dropout: float = 0.2
    ):
        """
        SACAgent 클래스 초기화
        
        Args:
            state_dim: 상태 공간의 차원 (CNN/LSTM 사용 시 None)
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
            input_shape: 입력 데이터 형태 (CNN/LSTM 사용 시 (window_size, feature_dim))
            use_cnn: CNN 모델 사용 여부
            use_lstm: LSTM 모델 사용 여부
            lstm_hidden_dim: LSTM 은닉층 차원
            num_lstm_layers: LSTM 레이어 수
            lstm_dropout: LSTM 드롭아웃 비율
        """
        # 속성 저장
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
        self.use_lstm = use_lstm
        self.input_shape = input_shape
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm_dropout = lstm_dropout
        
        # 모델 타입 검증 (상호 배타적)
        if use_cnn and use_lstm:
            raise ValueError("CNN과 LSTM 모델을 동시에 사용할 수 없습니다. 하나만 선택해주세요.")
        
        # 모델 타입 로깅
        if use_lstm:
            LOGGER.info("🧠 LSTM 기반 SAC 에이전트 초기화 중...")
            LOGGER.info("🔀 LSTM 모델 선택됨")
            LOGGER.info(f"   └─ LSTM 은닉 차원: {lstm_hidden_dim}")
            LOGGER.info(f"   └─ LSTM 레이어 수: {num_lstm_layers}")
            LOGGER.info(f"   └─ LSTM 드롭아웃: {lstm_dropout}")
            LOGGER.info("💡 시계열 패턴 학습에 최적화된 모델입니다")
        elif use_cnn:
            LOGGER.info("🖼️ CNN 기반 SAC 에이전트 초기화 중...")
            LOGGER.info("🔀 CNN 모델 선택됨")
            LOGGER.info("💡 공간적 특성 추출에 최적화된 모델입니다")
        else:
            LOGGER.info("🔢 MLP 기반 SAC 에이전트 초기화 중...")
            LOGGER.info("🔀 MLP 모델 선택됨 (기본값)")
            LOGGER.info("💡 일반적인 완전연결 신경망 모델입니다")
        
        # TradingEnvironment를 위한 상태 차원 자동 계산
        if not use_cnn and not use_lstm and state_dim is None:
            # TradingEnvironment의 기본 구조: market_data (30, 40) + portfolio_state (2,)
            if input_shape is None:
                raise ValueError("input_shape를 명시적으로 제공해야 합니다.")
            self.state_dim = input_shape[0] * input_shape[1] + 2  # 1200 + 2 = 1202
            state_dim = self.state_dim  # 네트워크 생성을 위해 local 변수도 업데이트
            LOGGER.info(f"📏 TradingEnvironment를 위한 상태 차원 자동 계산: {self.state_dim}")
        
        # 네트워크 초기화
        if use_lstm:
            if input_shape is None:
                raise ValueError("LSTM 모델을 사용할 때는 input_shape가 필요합니다.")
            
            LOGGER.info("🏗️ LSTM 네트워크 생성 중...")
            
            self.actor = LSTMActorNetwork(
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=lstm_dropout,
                device=device
            )
            
            self.critic = LSTMCriticNetwork(
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=lstm_dropout,
                device=device
            )
            
            self.critic_target = LSTMCriticNetwork(
                input_shape=input_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=lstm_dropout,
                device=device
            )
            
            LOGGER.info("✅ LSTM 네트워크 생성 완료")
            
        elif use_cnn:
            if input_shape is None:
                raise ValueError("CNN 모델을 사용할 때는 input_shape가 필요합니다.")
            
            LOGGER.info("🏗️ CNN 네트워크 생성 중...")
            
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
            
            LOGGER.info("✅ CNN 네트워크 생성 완료")
                
        else:
            # state_dim이 None인 경우는 이미 위에서 처리됨
            if state_dim is None:
                raise ValueError("일반 모델을 사용할 때는 state_dim이 필요합니다.")
            
            LOGGER.info("🏗️ MLP 네트워크 생성 중...")
            
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
            
            LOGGER.info("✅ MLP 네트워크 생성 완료")
            
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
        
        # 최종 초기화 로그
        model_type = "LSTM" if use_lstm else ("CNN" if use_cnn else "MLP")
        LOGGER.info(f"🎉 SAC 에이전트 초기화 완료!")
        LOGGER.info(f"   └─ 모델 타입: {model_type}")
        LOGGER.info(f"   └─ 행동 차원: {action_dim}")
        LOGGER.info(f"   └─ 상태 차원: {self.state_dim if not (use_cnn or use_lstm) else input_shape}")
        LOGGER.info(f"   └─ 장치: {device}")
    
    def select_action(self, state: Union[Dict[str, np.ndarray], Dict[str, torch.Tensor], np.ndarray], evaluate: bool = False) -> float:
        """
        상태에 따른 행동 선택 (실시간 트레이딩 호환)
        
        Args:
            state: 상태 (TradingEnvironment 형태 또는 기존 numpy 배열)
            evaluate: 평가 모드 여부 (True일 경우 탐색 없이 평균 행동 선택)
            
        Returns:
            선택된 행동 (단일 float 값)
        """
        try:
            # 상태를 네트워크 입력 형태로 변환
            if isinstance(state, dict):
                # TradingEnvironment 스타일 상태
                state_tensor = self._process_state_for_network(state)
            elif isinstance(state, np.ndarray):
                # 기존 스타일 상태 (numpy 배열)
                if self.use_cnn or self.use_lstm:
                    # CNN/LSTM 모델인데 플래튼된 배열을 받은 경우
                    # window_size와 feature_dim으로 reshape 시도
                    if self.input_shape and len(state) == (self.input_shape[0] * self.input_shape[1] + 2):
                        # MLP용으로 플래튼된 상태를 다시 분리
                        market_data_flat = state[:-2]  # 포트폴리오 상태 제외
                        portfolio_state = state[-2:]   # 마지막 2개 요소
                        
                        # market_data를 원래 형태로 reshape
                        market_data = market_data_flat.reshape(self.input_shape)
                        
                        # 딕셔너리 형태로 변환
                        state = {
                            'market_data': market_data,
                            'portfolio_state': portfolio_state
                        }
                        state_tensor = self._process_state_for_network(state)
                    else:
                        # 형태를 알 수 없는 경우 텐서로 변환
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:
                    # MLP 모델
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif isinstance(state, torch.Tensor):
                # 이미 텐서인 경우
                state_tensor = state.to(self.device)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
            else:
                raise ValueError(f"지원하지 않는 상태 타입: {type(state)}")
            
            # 행동 선택
            with torch.no_grad():
                if evaluate:
                    # 평가 모드: 평균 행동 선택 (탐색 없음)
                    _, _, action = self.actor.sample(state_tensor)
                else:
                    # 학습 모드: 탐색이 포함된 행동 선택
                    action, _, _ = self.actor.sample(state_tensor)
            
            # 단일 float 값으로 반환
            return float(action.detach().cpu().numpy().flatten()[0])
            
        except Exception as e:
            LOGGER.error(f"행동 선택 중 오류: {e}")
            LOGGER.error(f"상태 타입: {type(state)}")
            if isinstance(state, dict):
                LOGGER.error(f"상태 키: {state.keys()}")
                for k, v in state.items():
                    LOGGER.error(f"  {k}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
            elif hasattr(state, 'shape'):
                LOGGER.error(f"상태 형태: {state.shape}")
            
            # 오류 발생 시 기본 행동 반환
            return 0.0

    def _process_state_for_network(self, state: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        상태를 네트워크 입력 형태로 변환 (개선된 버전)
        
        Args:
            state: TradingEnvironment 상태 딕셔너리
            
        Returns:
            네트워크 입력용 텐서 또는 텐서 딕셔너리
        """
        try:
            if self.use_cnn or self.use_lstm:
                # CNN / LSTM: 상태를 딕셔너리 형태로 유지
                market_data = state["market_data"]
                portfolio_state = state["portfolio_state"]
                
                # numpy 배열을 텐서로 변환
                if isinstance(market_data, np.ndarray):
                    market_tensor = torch.FloatTensor(market_data).to(self.device)
                else:
                    market_tensor = market_data.to(self.device)
                    
                if isinstance(portfolio_state, np.ndarray):
                    portfolio_tensor = torch.FloatTensor(portfolio_state).to(self.device)
                else:
                    portfolio_tensor = portfolio_state.to(self.device)
                
                # 배치 차원 확인 및 추가
                if market_tensor.dim() == 2:  # (window_size, feature_dim)
                    market_tensor = market_tensor.unsqueeze(0)  # (1, window_size, feature_dim)
                if portfolio_tensor.dim() == 1:  # (2,)
                    portfolio_tensor = portfolio_tensor.unsqueeze(0)  # (1, 2)
                
                return {
                    "market_data": market_tensor,
                    "portfolio_state": portfolio_tensor
                }
            else:
                # MLP: 상태를 flatten
                market_data = state['market_data']
                portfolio_state = state['portfolio_state']
                
                # numpy 배열을 텐서로 변환
                if isinstance(market_data, np.ndarray):
                    market_flat = market_data.flatten()
                else:
                    market_flat = market_data.flatten().cpu().numpy()
                    
                if isinstance(portfolio_state, np.ndarray):
                    portfolio_flat = portfolio_state
                else:
                    portfolio_flat = portfolio_state.cpu().numpy()
                
                # 결합
                combined_state = np.concatenate([market_flat, portfolio_flat])
                return torch.FloatTensor(combined_state).unsqueeze(0).to(self.device)
            
        except Exception as e:
            LOGGER.error(f"상태 변환 중 오류: {e}")
            # 기본 상태 반환
            if self.use_cnn or self.use_lstm:
                return {
                    "market_data": torch.zeros((1, self.input_shape[0], self.input_shape[1]), device=self.device),
                    "portfolio_state": torch.zeros((1, 2), device=self.device)
                }
            else:
                return torch.zeros((1, self.state_dim), device=self.device)

    def _process_batch_states(self, states: List[Dict[str, np.ndarray]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        배치 상태들을 네트워크 입력으로 변환
        
        Args:
            states: TradingEnvironment 상태들의 리스트
        Returns:
            배치 텐서
        """
        
        if self.use_cnn or self.use_lstm:
            market_batch = []
            portfolio_batch = []
            for state in states:
                market_batch.append(state["market_data"])
                portfolio_batch.append(state["portfolio_state"])
            
            market_tensor = torch.FloatTensor(np.stack(market_batch)).to(self.device)  # (B, W, F)
            portfolio_tensor = torch.FloatTensor(np.stack(portfolio_batch)).to(self.device)  # (B, 2)

            return {
                "market_data": market_tensor,
                "portfolio_state": portfolio_tensor
            }
        else:
            batch_states = []
            for state in states:
                processed_state = self._process_state_for_network(state)
                batch_states.append(processed_state)
            return torch.cat(batch_states, dim=0)
        
    def add_experience(self, state: Dict[str, np.ndarray], action: float, reward: float, 
                      next_state: Dict[str, np.ndarray], done: bool) -> None:
        """
        경험을 리플레이 버퍼에 추가
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 종료 여부
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
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
                'entropy': 0.0,
                'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
            }
        
        # 리플레이 버퍼에서 미니배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 상태들을 배치 텐서로 변환
        batched_states = self._process_batch_states(states)
        batched_next_states = self._process_batch_states(next_states)
        
        # 행동, 보상, 완료 플래그를 텐서로 변환
        batched_actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)  # (batch_size, 1)
        batched_rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # (batch_size, 1)
        batched_dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)     # (batch_size, 1)
        
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
        stats = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item() if self.use_automatic_entropy_tuning else 0.0,
            'entropy': -log_probs.mean().item(),
            'alpha': self.alpha.item()
        }
        
        self.actor_losses.append(stats['actor_loss'])
        self.critic_losses.append(stats['critic_loss'])
        self.alpha_losses.append(stats['alpha_loss'])
        self.entropy_values.append(stats['entropy'])
        
        # 학습 통계 로깅 (DEBUG 레벨로 변경)
        LOGGER.debug(f"Actor Loss: {stats['actor_loss']:.6f}")
        LOGGER.debug(f"Critic Loss: {stats['critic_loss']:.6f}")
        LOGGER.debug(f"Alpha Loss: {stats['alpha_loss']:.6f}")
        LOGGER.debug(f"Entropy: {stats['entropy']:.6f}")
        LOGGER.debug(f"Alpha: {stats['alpha']:.6f}")
        
        return stats
    
    def save_model(self, save_dir: Union[str, Path] = None, prefix: str = "",
               model_type: str = None, symbol: str = None, symbols: List[str] = None) -> str:
        """
        모델 저장 (실시간 트레이딩 호환성 개선)

        Args:
            save_dir: 저장 디렉토리
            prefix: 파일명 접두사
            model_type: 모델 타입 ('mlp', 'cnn', 'lstm')
            symbol: 단일 심볼 (MLP용)
            symbols: 다중 심볼 리스트

        Returns:
            str: 저장된 모델의 경로
        """
        if save_dir is None:
            save_dir = MODELS_DIR

        create_directory(save_dir)

        # 타임스탬프 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 모델 타입 자동 감지 (지정되지 않은 경우)
        if model_type is None:
            if self.use_lstm:
                model_type = 'lstm'
            elif self.use_cnn:
                model_type = 'cnn'
            else:
                model_type = 'mlp'

        # 파일명 패턴 생성
        if model_type.lower() == 'mlp':
            # MLP: 심볼별로 구분
            if symbol:
                model_name = f"final_sac_model_{symbol}_{timestamp}"
            elif symbols and len(symbols) == 1:
                model_name = f"final_sac_model_{symbols[0]}_{timestamp}"
            else:
                # 다중 심볼이거나 심볼 정보 없음
                model_name = f"final_sac_model_multi_{timestamp}"
        else:
            # CNN, LSTM: 심볼 구분 없음
            model_name = f"final_{model_type.lower()}_sac_model_{timestamp}"

        # 접두사 적용
        if prefix:
            model_name = f"{prefix}{model_name}"

        # 저장 경로 생성
        model_path = Path(save_dir) / model_name
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

        # 설정 저장 (실시간 트레이딩 호환성 정보 포함)
        config = {
            # 기본 모델 정보
            'model_type': model_type.lower(),
            'symbol': symbol,
            'symbols': symbols,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            
            # 모델 아키텍처
            'use_cnn': self.use_cnn,
            'use_lstm': self.use_lstm,
            'input_shape': self.input_shape,
            
            # LSTM 전용 파라미터
            'lstm_hidden_dim': getattr(self, 'lstm_hidden_dim', None),
            'num_lstm_layers': getattr(self, 'num_lstm_layers', None),
            'lstm_dropout': getattr(self, 'lstm_dropout', None),
            
            # 실시간 트레이딩 호환성 정보
            'realtime_compatible': True,
            'supports_dict_state': self.use_cnn or self.use_lstm,
            'supports_flat_state': not (self.use_cnn or self.use_lstm),
            'state_format': 'dict' if (self.use_cnn or self.use_lstm) else 'flat',
            
            # 메타 정보
            'timestamp': timestamp,
            'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'device': str(self.device),
            'framework': 'pytorch',
            'sac_version': '1.0'
        }
        torch.save(config, model_path / "config.pth")

        model_type_display = model_type.upper()
        LOGGER.info(f"✅ {model_type_display} 모델 저장 완료: {model_path}")
        LOGGER.info(f"   └─ 모델 타입: {model_type_display}")
        LOGGER.info(f"   └─ 심볼: {symbol or symbols or 'Multi'}")
        LOGGER.info(f"   └─ 실시간 호환: {'YES' if config['realtime_compatible'] else 'NO'}")
        LOGGER.info(f"   └─ 상태 형식: {config['state_format'].upper()}")
        LOGGER.info(f"   └─ 타임스탬프: {timestamp}")

        return str(model_path)
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        모델 로드
        
        Args:
            model_path: 모델 디렉토리 경로
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            LOGGER.error(f"❌ 모델 경로가 존재하지 않습니다: {model_path}")
            return
        
        # 모델 타입 감지 및 로그
        model_name = model_path.name.lower()
        if "lstm" in model_name:
            LOGGER.info(f"🔄 LSTM SAC 모델 로드 중: {model_path}")
        elif "cnn" in model_name:
            LOGGER.info(f"🔄 CNN SAC 모델 로드 중: {model_path}")
        else:
            LOGGER.info(f"🔄 MLP SAC 모델 로드 중: {model_path}")
        
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
            
            # 성공 로그
            model_type = "LSTM" if self.use_lstm else ("CNN" if self.use_cnn else "MLP")
            LOGGER.info(f"✅ {model_type} 모델 로드 완료: {model_path}")
            LOGGER.info(f"   └─ 학습 스텝: {self.train_step_counter:,}")
            LOGGER.info(f"   └─ 버퍼 크기: {len(self.replay_buffer):,}")
            
        except RuntimeError as e:
            # 크기 불일치 오류 발생 시 크기 조정 로드 메서드 사용
            LOGGER.warning(f"⚠️ 표준 모델 로드 실패: {e}")
            LOGGER.info("🔄 크기 조정 방식으로 재시도 중...")
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
        
        # 모델 타입에 따른 패턴 확인
        patterns = [f"{prefix}lstm_sac_model_", f"{prefix}cnn_sac_model_", f"{prefix}mlp_sac_model_", f"{prefix}sac_model_"]
        
        model_dirs = []
        for pattern in patterns:
            model_dirs.extend([d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith(pattern)])
        
        if not model_dirs:
            return None
        
        # 타임스탬프 기준으로 정렬
        model_dirs.sort(key=lambda d: d.name, reverse=True)
        return model_dirs[0]
    
    def load_model_with_resize(self, model_path):
        """
        크기가 다른 모델을 부분적으로 로드
        
        Args:
            model_path: 모델 디렉토리 경로
        """
        model_path = Path(model_path)
        
        LOGGER.info("🔧 크기 조정 방식으로 모델 로드 시도 중...")
        
        # 저장된 상태 사전 로드
        saved_actor_dict = torch.load(model_path / "actor.pth", map_location=self.device)
        saved_critic_dict = torch.load(model_path / "critic.pth", map_location=self.device)
        
        # 현재 모델의 상태 사전
        current_actor_dict = self.actor.state_dict()
        current_critic_dict = self.critic.state_dict()
        
        # 크기가 일치하는 파라미터만 로드
        actor_dict = {k: v for k, v in saved_actor_dict.items() if k in current_actor_dict and v.shape == current_actor_dict[k].shape}
        critic_dict = {k: v for k, v in saved_critic_dict.items() if k in current_critic_dict and v.shape == current_critic_dict[k].shape}
        
        # 호환 통계 로그
        actor_loaded = len(actor_dict)
        actor_total = len(current_actor_dict)
        critic_loaded = len(critic_dict)
        critic_total = len(current_critic_dict)
        
        LOGGER.info(f"📊 모델 호환성 분석:")
        LOGGER.info(f"   └─ Actor: {actor_loaded}/{actor_total} 레이어 호환 ({actor_loaded/actor_total*100:.1f}%)")
        LOGGER.info(f"   └─ Critic: {critic_loaded}/{critic_total} 레이어 호환 ({critic_loaded/critic_total*100:.1f}%)")
        
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
            LOGGER.warning("⚠️ critic_target 파일이 없어 critic에서 복사합니다.")
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
            LOGGER.warning(f"⚠️ 옵티마이저 상태 로드 실패. 기본값 유지.")
        
        # 학습 통계 로드 (가능한 경우)
        try:
            training_stats = torch.load(model_path / "training_stats.pth", map_location=self.device)
            self.actor_losses = training_stats.get('actor_losses', [])
            self.critic_losses = training_stats.get('critic_losses', [])
            self.alpha_losses = training_stats.get('alpha_losses', [])
            self.entropy_values = training_stats.get('entropy_values', [])
            self.train_step_counter = training_stats.get('train_step_counter', 0)
        except:
            LOGGER.warning(f"⚠️ 학습 통계 로드 실패. 기본값 유지.")
        
        model_type = "LSTM" if self.use_lstm else ("CNN" if self.use_cnn else "MLP")
        LOGGER.info(f"✅ {model_type} 모델 크기 조정 로드 완료: {model_path}")
        LOGGER.info("💡 일부 파라미터는 새로 초기화됩니다.")


def train_sac_agent(env, agent, num_episodes: int = 1000, 
                   max_steps_per_episode: int = 1000, update_frequency: int = 1,
                   log_frequency: int = 100):
    """
    SAC 에이전트 학습 함수

    Args:
        env: TradingEnvironment 인스턴스
        agent: SAC 에이전트
        num_episodes: 학습할 에피소드 수
        max_steps_per_episode: 에피소드당 최대 스텝 수
        update_frequency: 업데이트 빈도
        log_frequency: 로그 출력 빈도
    
    Returns:
        학습된 SAC 에이전트
    """
    model_type = "LSTM" if agent.use_lstm else ("CNN" if agent.use_cnn else "MLP")
    LOGGER.info(f"🚀 {model_type} SAC 에이전트 학습 시작: {num_episodes} 에피소드")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # 행동 선택
            action = agent.select_action(state, evaluate=False)
            
            # 환경에서 스텝 실행
            next_state, reward, done, info = env.step(action)
            
            # 경험 저장
            agent.add_experience(state, action, reward, next_state, done)
            
            # 네트워크 업데이트
            if step % update_frequency == 0:
                stats = agent.update_parameters()
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 로그 출력
        if episode % log_frequency == 0:
            avg_reward = np.mean(episode_rewards[-log_frequency:])
            avg_length = np.mean(episode_lengths[-log_frequency:])
            
            LOGGER.info(f"Episode {episode}")
            LOGGER.info(f"  평균 보상: {avg_reward:.2f}")
            LOGGER.info(f"  평균 길이: {avg_length:.1f}")
            LOGGER.info(f"  포트폴리오 가치: ${info['portfolio_value']:.2f}")
            LOGGER.info(f"  총 수익률: {info['total_return'] * 100:.2f}%")
            
            if len(agent.actor_losses) > 0:
                LOGGER.info(f"  Actor Loss: {agent.actor_losses[-1]:.6f}")
                LOGGER.info(f"  Critic Loss: {agent.critic_losses[-1]:.6f}")
                LOGGER.info(f"  Alpha: {agent.alpha.item():.6f}")
    
    LOGGER.info(f"🎉 {model_type} SAC 에이전트 학습 완료!")
    return agent


if __name__ == "__main__":
    # 모듈 테스트 코드
    print("SAC 에이전트 테스트 시작...")
    
    # TradingEnvironment 스타일의 테스트 설정
    window_size = 30
    feature_dim = 40
    action_dim = 1
    batch_size = 4
    
    print("\n=== MLP SAC 에이전트 테스트 ===")
    
    # MLP 에이전트 초기화 (TradingEnvironment용)
    mlp_agent = SACAgent(
        state_dim=None,  # 자동 계산됨 (30*40 + 2 = 1202)
        action_dim=1,
        input_shape=(window_size, feature_dim),  # 자동 계산을 위한 힌트
        use_cnn=False,
        use_lstm=False
    )
    
    print(f"MLP 에이전트 초기화 완료 - 상태 차원: {mlp_agent.state_dim}")
    
    print("\n=== CNN SAC 에이전트 테스트 ===")
    
    # CNN 에이전트 초기화
    cnn_agent = SACAgent(
        state_dim=None,
        action_dim=1,
        input_shape=(window_size, feature_dim),
        use_cnn=True,
        use_lstm=False
    )
    
    print(f"CNN 에이전트 초기화 완료")
    
    print("\n=== LSTM SAC 에이전트 테스트 ===")
    
    # LSTM 에이전트 초기화
    lstm_agent = SACAgent(
        state_dim=None,
        action_dim=1,
        input_shape=(window_size, feature_dim),
        use_cnn=False,
        use_lstm=True,
        lstm_hidden_dim=64,
        num_lstm_layers=1,
        lstm_dropout=0.1
    )
    
    print(f"LSTM 에이전트 초기화 완료")
    
    # 모든 에이전트 테스트
    agents = [
        ("MLP", mlp_agent),
        ("CNN", cnn_agent), 
        ("LSTM", lstm_agent)
    ]
    
    for agent_name, agent in agents:
        print(f"\n--- {agent_name} 에이전트 테스트 ---")
        
        # TradingEnvironment 스타일의 상태로 테스트
        for i in range(batch_size):
            state = {
                'market_data': np.random.randn(window_size, feature_dim),
                'portfolio_state': np.random.randn(2)
            }
            next_state = {
                'market_data': np.random.randn(window_size, feature_dim),
                'portfolio_state': np.random.randn(2)
            }
            
            action = np.random.uniform(-1.0, 1.0)
            reward = np.random.randn()
            done = np.random.choice([True, False])
            
            # 경험 추가
            agent.add_experience(state, action, reward, next_state, done)
            
            # 행동 선택 테스트
            if i == 0:
                selected_action = agent.select_action(state, evaluate=False)
                print(f"  선택된 행동: {selected_action}")
        
        print(f"  리플레이 버퍼 크기: {len(agent.replay_buffer)}")
        
        # 학습 테스트
        print(f"  학습 테스트 중...")
        stats = agent.update_parameters(batch_size=batch_size)
        print(f"  학습 통계: Actor Loss {stats['actor_loss']:.6f}, Critic Loss {stats['critic_loss']:.6f}")
        
        # 모델 저장 테스트
        print(f"  모델 저장 테스트...")
        model_path = agent.save_model(prefix=f'{agent_name.lower()}_test_')
        print(f"  모델 저장 완료: {model_path}")
        
        # 모델 로드 테스트
        print(f"  모델 로드 테스트...")
        agent.load_model(model_path)
        print(f"  모델 로드 완료")
        
        # 로드 후 행동 선택 테스트
        test_state = {
            'market_data': np.random.randn(window_size, feature_dim),
            'portfolio_state': np.random.randn(2)
        }
        test_action = agent.select_action(test_state, evaluate=True)
        print(f"  로드 후 테스트 행동: {test_action}")
    
    print("\n🎉 모든 SAC 에이전트 테스트 완료!")