# SAC 트레이딩 시스템 웹 대시보드

SAC 트레이딩 시스템의 웹 대시보드는 시스템의 모든 측면을 모니터링하고 관리할 수 있는 중앙 인터페이스를 제공합니다. 이 문서는 대시보드의 주요 기능과 구성 요소를 설명합니다.

## 주요 기능

### 1. 대시보드 메인 화면

![대시보드 메인 화면](../web_screenshots/dashboard_main.png)

- 포트폴리오 가치 변화 차트
- 현재 포지션 요약
- 최근 거래 내역
- 주요 성과 지표 (총 수익률, 일일 손익, 샤프 비율, 최대 낙폭)
- 시스템 상태 표시기

### 2. 실시간 트레이딩 모니터링

![실시간 트레이딩](../web_screenshots/live_trading.png)

- 실시간 시장 데이터 차트
- 매수/매도 신호 표시
- 현재 포지션 및 주문 상태
- 실시간 손익 계산
- 포트폴리오 가치 변화 추적

### 3. 백테스트 결과 비교

![백테스트 결과](../web_screenshots/backtest_results.png)

- 여러 모델의 백테스트 결과 비교
- 성능 지표 시각화 (총 수익률, 샤프 비율, 최대 낙폭, 승률 등)
- 레이더 차트를 통한 모델 성능 비교
- 거래 분포 분석
- 수익률 분포 및 통계

### 4. 모델 관리

![모델 관리](../web_screenshots/model_management.png)

- 학습된 모델 목록 및 정보
- 모델 성능 지표
- 모델 활성화/비활성화 설정
- 모델 비교 및 선택

### 5. 설정 및 구성

![설정 화면](../web_screenshots/settings.png)

- 트레이딩 파라미터 설정
- 데이터 소스 관리
- 알림 설정
- 시스템 로그 확인

## 기술 스택

### 프론트엔드
- HTML5/CSS3/JavaScript
- Charts.js 및 Plotly.js (데이터 시각화)
- Bootstrap (반응형 레이아웃)

### 백엔드
- Flask (Python 웹 프레임워크)
- SQLite/MySQL (데이터 저장)
- Pandas (데이터 처리)
- PyTorch (모델 처리)

## 데이터 관리

### 파일 기반 데이터 관리
- JSON 형식의 트레이딩 통계
- CSV 형식의 시장 데이터
- JSON 형식의 백테스트 결과
- PyTorch 모델 파일(.pt)

### 데이터베이스 기반 데이터 관리
- 관계형 데이터베이스(MySQL) 사용
- 실시간 데이터 업데이트
- 효율적인 쿼리 및 분석
- 데이터 무결성 보장

## API 엔드포인트

### 데이터 API
- `/api/trading-stats` - 트레이딩 통계 조회
- `/api/models` - 모델 정보 조회
- `/api/backtest-results` - 백테스트 결과 조회
- `/api/performance-metrics` - 성능 지표 조회
- `/api/market-data` - 시장 데이터 조회

### 차트 API
- `/api/charts/portfolio` - 포트폴리오 가치 차트
- `/api/charts/returns` - 수익률 차트
- `/api/charts/drawdown` - 낙폭 차트
- `/api/charts/trade-distribution` - 거래 분포 차트
- `/api/charts/price` - 가격 및 신호 차트
- `/api/charts/model-comparison` - 모델 비교 차트
- `/api/charts/radar` - 모델 레이더 차트

## 실행 방법

### 파일 모드 실행
```bash
python src/dashboard/run_dashboard.py --data-dir results
```

### 데이터베이스 모드 실행
```bash
python src/dashboard/run_dashboard.py --db-mode --db-user sac_user --db-password your_password
```

### 파일에서 데이터베이스로 동기화
```bash
python src/dashboard/run_dashboard.py --db-mode --sync-to-db
```

## 보안 및 접근 제어

- SSL/TLS 암호화 지원
- 사용자 인증 및 권한 설정 가능
- API 키 기반 접근 제어
- 로깅 및 활동 추적

## 오류 처리

- 데이터 불일치 감지 및 복구
- 연결 오류 자동 재시도
- 오류 로깅 및 알림
- 그레이스풀 디그레이드(Graceful degradation) 지원

## 확장성

- 새로운 시각화 및 차트 추가 가능
- 추가 성능 지표 통합 용이
- 다양한 데이터 소스 연결 가능
- 커스텀 알림 및 보고서 설정 