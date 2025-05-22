"""
SAC 알고리즘의 Actor와 Critic 네트워크 구현 모듈
"""
#  ActorNetwork, CriticNetwork, CNNActorNetwork, CNNCriticNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Union, Optional
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from src.config.config import (
    HIDDEN_DIM,
    DEVICE,
    LOGGER
)

class ActorNetwork(nn.Module):
    """
    SAC 알고리즘의 Actor 네트워크 (정책 네트워크)
    연속적인 행동 공간에 대한 확률적 정책을 모델링
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        device: torch.device = DEVICE
    ):
        """
        ActorNetwork 클래스 초기화
        
        Args:
            state_dim: 상태 공간의 차원
            action_dim: 행동 공간의 차원
            hidden_dim: 은닉층의 뉴런 수
            log_std_min: 로그 표준편차의 최소값
            log_std_max: 로그 표준편차의 최대값
            device: 모델이 실행될 장치
        """
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        
        # 공통 특성 추출 레이어
        print(f'state_dim : {state_dim}')
        print(f'hidden_dim : {hidden_dim}')
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 평균값 출력 레이어
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # 로그 표준편차 출력 레이어
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        LOGGER.info(f"Actor 네트워크 초기화 완료: 상태 차원 {state_dim}, 행동 차원 {action_dim}")
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 텐서
            
        Returns:
            평균과 로그 표준편차 텐서의 튜플
        """
        # 입력 차원 검사 및 조정
        if state.size(1) != self.state_dim:
            print(f"경고: 입력 상태 차원 ({state.size(1)})이 모델의 상태 차원 ({self.state_dim})과 일치하지 않습니다.")
            # 차원이 더 큰 경우 자르기
            if state.size(1) > self.state_dim:
                state = state[:, :self.state_dim]
                print(f"입력을 {self.state_dim} 차원으로 잘랐습니다.")
            # 차원이 더 작은 경우 패딩
            else:
                padding = torch.zeros(state.size(0), self.state_dim - state.size(1), device=state.device)
                state = torch.cat([state, padding], dim=1)
                print(f"입력을 {self.state_dim} 차원으로 패딩했습니다.")
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 평균값 계산
        mean = self.mean(x)
        
        # 로그 표준편차 계산 및 클리핑
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        상태에서 행동을 샘플링
        
        Args:
            state: 상태 텐서
            
        Returns:
            (행동, 로그 확률, 평균) 튜플
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 재매개변수화 트릭을 사용하여 정규 분포에서 샘플링
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 재매개변수화된 샘플
        
        # Tanh 변환을 통해 행동 범위 제한 (-1, 1)
        y_t = torch.tanh(x_t)
        
        # 정책의 로그 확률 계산
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean
    
    def to(self, device: torch.device) -> 'ActorNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(ActorNetwork, self).to(device)


class CriticNetwork(nn.Module):
    """
    SAC 알고리즘의 Critic 네트워크 (Q-함수)
    상태-행동 쌍의 가치를 평가
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        device: torch.device = DEVICE
    ):
        """
        CriticNetwork 클래스 초기화
        
        Args:
            state_dim: 상태 공간의 차원
            action_dim: 행동 공간의 차원
            hidden_dim: 은닉층의 뉴런 수
            device: 모델이 실행될 장치
        """
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Q1 네트워크
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 네트워크 (Double Q-learning)
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        LOGGER.info(f"Critic 네트워크 초기화 완료: 상태 차원 {state_dim}, 행동 차원 {action_dim}")
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 텐서
            action: 행동 텐서
            
        Returns:
            두 Q 값의 튜플
        """
        # 입력 차원 검사 및 조정
        if state.size(1) != self.state_dim:
            print(f"경고: 입력 상태 차원 ({state.size(1)})이 모델의 상태 차원 ({self.state_dim})과 일치하지 않습니다.")
            # 차원이 더 큰 경우 자르기
            if state.size(1) > self.state_dim:
                state = state[:, :self.state_dim]
                print(f"입력을 {self.state_dim} 차원으로 잘랐습니다.")
            # 차원이 더 작은 경우 패딩
            else:
                padding = torch.zeros(state.size(0), self.state_dim - state.size(1), device=state.device)
                state = torch.cat([state, padding], dim=1)
                print(f"입력을 {self.state_dim} 차원으로 패딩했습니다.")
        
        # 상태와 행동을 연결
        sa = torch.cat([state, action], 1)
        
        # Q1 값 계산
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.q1(q1)
        
        # Q2 값 계산
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = F.relu(self.fc6(q2))
        q2 = self.q2(q2)
        
        return q1, q2
    
    def to(self, device: torch.device) -> 'CriticNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(CriticNetwork, self).to(device)


class CNNActorNetwork(nn.Module):
    """
    CNN 기반 Actor 네트워크
    시계열 데이터의 특성을 추출하기 위한 1D 컨볼루션 레이어 사용
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        device: torch.device = DEVICE
    ):
        """
        CNNActorNetwork 클래스 초기화
        
        Args:
            input_shape: 입력 데이터의 형태 (window_size, feature_dim)
            action_dim: 행동 공간의 차원
            hidden_dim: 은닉층의 뉴런 수
            log_std_min: 로그 표준편차의 최소값
            log_std_max: 로그 표준편차의 최대값
            device: 모델이 실행될 장치
        """
        super(CNNActorNetwork, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        # print(f"hidden_dim: {hidden_dim}, type: {type(hidden_dim)}")

        # 1D 컨볼루션 레이어
        self.conv1 = nn.Conv1d(self.feature_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 풀링 레이어
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 컨볼루션 출력 크기 계산
        # 두 번의 풀링으로 크기가 1/4로 줄어듦
        self.conv_output_size = 128 * (self.window_size // 4)
        
        # 포트폴리오 상태 입력을 위한 레이어
        self.portfolio_fc = nn.Linear(2, hidden_dim // 4)
        
        # 컨볼루션 출력과 포트폴리오 상태를 결합하는 레이어
        self.fc1 = nn.Linear(self.conv_output_size + hidden_dim // 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 평균값 출력 레이어
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # 로그 표준편차 출력 레이어
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        LOGGER.info(f"CNN Actor 네트워크 초기화 완료: 입력 형태 {input_shape}, 행동 차원 {action_dim}")

    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 딕셔너리 {'market_data': 시장 데이터, 'portfolio_state': 포트폴리오 상태}
            
        Returns:
            평균과 로그 표준편차 텐서의 튜플
        """
        # 시장 데이터 처리 (B, W, F) -> (B, F, W) 형태로 변환
        market_data = state['market_data']
        
        # 차원 확인 및 처리
        if len(market_data.shape) == 3:  # (B, W, F) 형태
            market_data = market_data.permute(0, 2, 1)
        elif len(market_data.shape) == 2:  # (W, F) 형태 - 배치 차원 추가 필요
            market_data = market_data.unsqueeze(0).permute(0, 2, 1)
        
        # 포트폴리오 상태 처리
        portfolio_state = state['portfolio_state']
        
        # 포트폴리오 상태 차원 확인 및 처리
        if len(portfolio_state.shape) == 1:  # (2,) 형태 - 배치 차원 추가 필요
            portfolio_state = portfolio_state.unsqueeze(0)  # (1, 2) 형태로 변환
        
        # 컨볼루션 레이어 통과
        x = F.relu(self.conv1(market_data))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        
        # 컨볼루션 출력 평탄화
        x = x.view(x.size(0), -1)
        
        # 포트폴리오 상태 처리
        p = F.relu(self.portfolio_fc(portfolio_state))
        
        # 특성 결합
        combined = torch.cat([x, p], dim=1)
        
        # 완전 연결 레이어 통과
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # 평균값 계산
        mean = self.mean(x)
        
        # 로그 표준편차 계산 및 클리핑
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        상태에서 행동을 샘플링
        
        Args:
            state: 상태 딕셔너리
            
        Returns:
            (행동, 로그 확률, 평균) 튜플
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 재매개변수화 트릭을 사용하여 정규 분포에서 샘플링
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 재매개변수화된 샘플
        
        # Tanh 변환을 통해 행동 범위 제한 (-1, 1)
        y_t = torch.tanh(x_t)
        
        # 정책의 로그 확률 계산
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean
    
    def to(self, device: torch.device) -> 'CNNActorNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(CNNActorNetwork, self).to(device)


class CNNCriticNetwork(nn.Module):
    """
    CNN 기반 Critic 네트워크
    시계열 데이터의 특성을 추출하기 위한 1D 컨볼루션 레이어 사용
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        device: torch.device = DEVICE
    ):
        """
        CNNCriticNetwork 클래스 초기화
        
        Args:
            input_shape: 입력 데이터의 형태 (window_size, feature_dim)
            action_dim: 행동 공간의 차원
            hidden_dim: 은닉층의 뉴런 수
            device: 모델이 실행될 장치
        """
        super(CNNCriticNetwork, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 1D 컨볼루션 레이어 (Q1)
        self.q1_conv1 = nn.Conv1d(self.feature_dim, 32, kernel_size=3, stride=1, padding=1)
        self.q1_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.q1_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 1D 컨볼루션 레이어 (Q2)
        self.q2_conv1 = nn.Conv1d(self.feature_dim, 32, kernel_size=3, stride=1, padding=1)
        self.q2_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.q2_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 풀링 레이어
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 컨볼루션 출력 크기 계산
        # 수정: int 타입의 window_size를 사용하여 계산
        # 두 번의 풀링으로 크기가 1/4로 줄어듦
        self.conv_output_size = 128 * (self.window_size // 4)
        
        # 포트폴리오 상태 입력을 위한 레이어
        self.q1_portfolio_fc = nn.Linear(2, hidden_dim // 4)
        self.q2_portfolio_fc = nn.Linear(2, hidden_dim // 4)
        
        # 행동 입력을 위한 레이어
        self.q1_action_fc = nn.Linear(action_dim, hidden_dim // 4)
        self.q2_action_fc = nn.Linear(action_dim, hidden_dim // 4)
        
        # 컨볼루션 출력, 포트폴리오 상태, 행동을 결합하는 레이어
        self.q1_fc1 = nn.Linear(self.conv_output_size + hidden_dim // 2, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(self.conv_output_size + hidden_dim // 2, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        LOGGER.info(f"CNN Critic 네트워크 초기화 완료: 입력 형태 {input_shape}, 행동 차원 {action_dim}")
    
    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 딕셔너리 {'market_data': 시장 데이터, 'portfolio_state': 포트폴리오 상태}
            action: 행동 텐서
            
        Returns:
            두 Q 값의 튜플
        """
        # 시장 데이터 처리 (B, W, F) -> (B, F, W) 형태로 변환
        market_data = state['market_data']
        
        # 차원 확인 및 처리
        if len(market_data.shape) == 3:  # (B, W, F) 형태
            market_data = market_data.permute(0, 2, 1)
        elif len(market_data.shape) == 2:  # (W, F) 형태 - 배치 차원 추가 필요
            market_data = market_data.unsqueeze(0).permute(0, 2, 1)
        
        # 포트폴리오 상태 처리
        portfolio_state = state['portfolio_state']
        
        # 포트폴리오 상태 차원 확인 및 처리
        if len(portfolio_state.shape) == 1:  # (2,) 형태 - 배치 차원 추가 필요
            portfolio_state = portfolio_state.unsqueeze(0)  # (1, 2) 형태로 변환
        
        # 행동 차원 확인 및 처리
        if len(action.shape) == 1:  # (action_dim,) 형태 - 배치 차원 추가 필요
            action = action.unsqueeze(0)  # (1, action_dim) 형태로 변환
        
        # Q1 네트워크
        # 시장 데이터 처리
        q1_x = F.relu(self.q1_conv1(market_data))
        q1_x = self.pool(q1_x)
        q1_x = F.relu(self.q1_conv2(q1_x))
        q1_x = self.pool(q1_x)
        q1_x = F.relu(self.q1_conv3(q1_x))
        q1_x = q1_x.view(q1_x.size(0), -1)  # 평탄화
        
        # 포트폴리오 상태 처리
        q1_p = F.relu(self.q1_portfolio_fc(portfolio_state))
        
        # 행동 처리
        q1_a = F.relu(self.q1_action_fc(action))
        
        # 특성 결합
        q1_combined = torch.cat([q1_x, q1_p, q1_a], dim=1)
        
        # 완전 연결 레이어 통과
        q1_x = F.relu(self.q1_fc1(q1_combined))
        q1_x = F.relu(self.q1_fc2(q1_x))
        q1 = self.q1(q1_x)
        
        # Q2 네트워크
        # 시장 데이터 처리
        q2_x = F.relu(self.q2_conv1(market_data))
        q2_x = self.pool(q2_x)
        q2_x = F.relu(self.q2_conv2(q2_x))
        q2_x = self.pool(q2_x)
        q2_x = F.relu(self.q2_conv3(q2_x))
        q2_x = q2_x.view(q2_x.size(0), -1)  # 평탄화
        
        # 포트폴리오 상태 처리
        q2_p = F.relu(self.q2_portfolio_fc(portfolio_state))
        
        # 행동 처리
        q2_a = F.relu(self.q2_action_fc(action))
        
        # 특성 결합
        q2_combined = torch.cat([q2_x, q2_p, q2_a], dim=1)
        
        # 완전 연결 레이어 통과
        q2_x = F.relu(self.q2_fc1(q2_combined))
        q2_x = F.relu(self.q2_fc2(q2_x))
        q2 = self.q2(q2_x)
        
        return q1, q2
    
    def to(self, device: torch.device) -> 'CNNCriticNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(CNNCriticNetwork, self).to(device)


if __name__ == "__main__":
    # 모듈 테스트 코드
    # 일반 네트워크 테스트
    state_dim = 10
    action_dim = 1
    batch_size = 4
    
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim, action_dim)
    
    # 랜덤 데이터로 테스트
    state = torch.randn(batch_size, state_dim).to(DEVICE)
    action = torch.randn(batch_size, action_dim).to(DEVICE)
    
    # Actor 테스트
    mean, log_std = actor(state)
    action_sample, log_prob, _ = actor.sample(state)
    
    print(f"Actor 출력 - 평균: {mean.shape}, 로그 표준편차: {log_std.shape}")
    print(f"Actor 샘플 - 행동: {action_sample.shape}, 로그 확률: {log_prob.shape}")
    
    # Critic 테스트
    q1, q2 = critic(state, action)
    
    print(f"Critic 출력 - Q1: {q1.shape}, Q2: {q2.shape}")
    
    # CNN 네트워크 테스트
    window_size = 30
    feature_dim = 5
    
    cnn_actor = CNNActorNetwork((window_size, feature_dim), action_dim)
    cnn_critic = CNNCriticNetwork((window_size, feature_dim), action_dim)
    
    # 랜덤 데이터로 테스트
    market_data = torch.randn(batch_size, window_size, feature_dim).to(DEVICE)
    portfolio_state = torch.randn(batch_size, 2).to(DEVICE)
    state_dict = {
        'market_data': market_data,
        'portfolio_state': portfolio_state
    }
    print(f'state_dict : {state_dict}')
    
    # CNN Actor 테스트
    mean, log_std = cnn_actor(state_dict)
    action_sample, log_prob, _ = cnn_actor.sample(state_dict)
    
    print(f"CNN Actor 출력 - 평균: {mean.shape}, 로그 표준편차: {log_std.shape}")
    print(f"CNN Actor 샘플 - 행동: {action_sample.shape}, 로그 확률: {log_prob.shape}")
    
    # CNN Critic 테스트
    q1, q2 = cnn_critic(state_dict, action_sample)
    
    print(f"CNN Critic 출력 - Q1: {q1.shape}, Q2: {q2.shape}")