# SAC 트레이딩 시스템 문서

SAC(Soft Actor-Critic) 강화학습 알고리즘 기반 트레이딩 시스템의 설계, 구현 및 사용법에 관한 문서입니다.

## 시스템 아키텍처

### 클래스 다이어그램
![SAC 트레이딩 시스템 클래스 다이어그램](diagrams/class_diagram.png)

### 핵심 컴포넌트
- **SAC 모델**: 강화학습 기반 트레이딩 정책 모델
- **트레이딩 환경**: OpenAI Gym 인터페이스 기반 환경
- **대시보드**: 웹 기반 모니터링 및 관리 인터페이스
- **데이터 관리**: 파일 및 데이터베이스 기반 데이터 처리

## 프로세스 플로우차트

### 모델 학습 프로세스
![SAC 모델 학습 프로세스](flowcharts/training_flow.png)

[상세 설명](flowcharts/training_flow.md)

### 백테스트 프로세스
![백테스트 프로세스](flowcharts/backtest_flow.png)

[상세 설명](flowcharts/backtest_flow.md)

### 실시간 트레이딩 프로세스
![실시간 트레이딩 프로세스](flowcharts/live_trading_flow.png)

[상세 설명](flowcharts/live_trading_flow.md)

## 웹 대시보드

### 개요
SAC 트레이딩 시스템의 웹 대시보드는 트레이딩 성과 모니터링, 백테스트 결과 분석, 모델 관리 등의 기능을 제공합니다.

[웹 대시보드 개요](web/dashboard_overview.md)

### 주요 화면
- [대시보드 메인](web_screenshots/dashboard_main.png)
- [실시간 트레이딩](web_screenshots/live_trading.png)
- [백테스트 결과](web_screenshots/backtest_results.png)
- [모델 관리](web_screenshots/model_management.png)
- [설정 화면](web_screenshots/settings.png)

## 데이터 관리

### 파일 기반 데이터 관리
기본 모드로, JSON 및 CSV 파일을 사용하여 데이터를 저장합니다. 로컬 개발 및 테스트에 적합합니다.

### 데이터베이스 기반 데이터 관리
MySQL 데이터베이스를 사용하여 데이터를 저장하고 관리합니다. 실시간 운영 환경에 적합합니다.

[데이터베이스 통합 상세 설명](web/database_integration.md)

## 설치 및 실행 가이드

### 요구 사항
- Python 3.8 이상
- PyTorch 2.0 이상
- MySQL 5.7 이상 (데이터베이스 모드 사용 시)

### 설치
```bash
# 저장소 복제
git clone https://github.com/yourusername/sac-trading.git
cd sac-trading

# 의존성 설치
pip install -r requirements.txt
```

### 실행
```bash
# 파일 모드로 대시보드 실행
python src/dashboard/run_dashboard.py

# 데이터베이스 모드로 대시보드 실행
python src/dashboard/run_dashboard.py --db-mode
```

## API 문서

### 주요 API 엔드포인트
- 데이터 API
- 차트 API
- 모델 관리 API

[API 문서 상세 보기](api_docs.md)

## 고급 주제

### 모델 하이퍼파라미터 튜닝
- 학습률
- 보상 함수 설계
- 할인율(감마)
- 엔트로피 계수(알파)

### 커스텀 환경 개발
- 새로운 액션 공간 정의
- 관측 공간 확장
- 보상 함수 사용자 정의

### 성능 최적화
- 데이터베이스 인덱싱
- 연결 풀링
- 쿼리 최적화
- 캐싱 전략

## 기여 가이드
- 코드 스타일 가이드
- 이슈 보고 방법
- 풀 리퀘스트 프로세스

## 라이센스
이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요. 