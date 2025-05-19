# SAC 트레이딩 시스템 - 데이터베이스 모드

이 문서는 SAC 트레이딩 시스템의 웹 대시보드를 MySQL 데이터베이스 모드로 실행하는 방법에 대해 설명합니다.

## 목차

1. [소개](#소개)
2. [요구사항](#요구사항)
3. [MySQL 설정](#mysql-설정)
4. [테이블 구조](#테이블-구조)
5. [실행 방법](#실행-방법)
6. [파일 동기화](#파일-동기화)
7. [문제 해결](#문제-해결)

## 소개

SAC 트레이딩 시스템은 두 가지 데이터 관리 모드를 지원합니다:

1. **파일 모드**: 기본 모드로, 모든 데이터를 JSON, CSV 파일로 저장합니다. 로컬 테스트 및 단위 테스트에 적합합니다.
2. **데이터베이스 모드**: MySQL 데이터베이스를 사용하여 모든 데이터를 관리합니다. 실시간 운영 환경에 적합합니다.

데이터베이스 모드를 사용하면 다음과 같은 장점이 있습니다:

- 실시간 데이터 업데이트 및 처리
- 동시 접근 관리
- 확장성 향상
- 데이터 무결성 및 일관성 유지

## 요구사항

데이터베이스 모드를 사용하기 위해서는 다음이 필요합니다:

- MySQL 서버 5.7 이상
- Python 3.8 이상
- `mysql-connector-python` 패키지 (requirements.txt에 포함)

패키지 설치:
```bash
pip install -r requirements.txt
```

## MySQL 설정

1. MySQL 서버 설치 및 실행

2. 데이터베이스 생성:
```sql
CREATE DATABASE sac_trading CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

3. 사용자 생성 및 권한 설정:
```sql
CREATE USER 'sac_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON sac_trading.* TO 'sac_user'@'localhost';
FLUSH PRIVILEGES;
```

## 테이블 구조

데이터베이스 모드를 실행하면 시스템이 자동으로 다음 테이블을 생성합니다:

- `trading_stats`: 트레이딩 통계 정보
- `trades`: 거래 내역
- `positions`: 포지션 정보
- `market_data`: 시장 데이터
- `models`: 모델 정보
- `backtest_results`: 백테스트 결과

테이블 구조는 `src/utils/database.py` 파일의 `create_tables()` 메서드에서 확인할 수 있습니다.

## 실행 방법

### 기본 실행 (데이터베이스 모드)

```bash
python src/dashboard/run_dashboard.py --db-mode --db-user sac_user --db-password your_password
```

### 모든 파라미터 옵션

```bash
python src/dashboard/run_dashboard.py --help
```

주요 파라미터:

- `--db-mode`: 데이터베이스 모드 활성화
- `--db-host`: 데이터베이스 호스트 (기본값: localhost)
- `--db-port`: 데이터베이스 포트 (기본값: 3306)
- `--db-user`: 데이터베이스 사용자 (기본값: root)
- `--db-password`: 데이터베이스 비밀번호
- `--db-name`: 데이터베이스 이름 (기본값: sac_trading)
- `--sync-to-db`: 파일 데이터를 DB로 동기화 후 실행

## 파일 동기화

기존 파일 기반 데이터를 데이터베이스로 동기화하려면 `--sync-to-db` 옵션을 사용합니다:

```bash
python src/dashboard/run_dashboard.py --db-mode --sync-to-db
```

이 명령은 다음 데이터를 동기화합니다:
- 트레이딩 통계 (JSON)
- 백테스트 결과 (JSON)
- 모델 파일 정보 (.pt)
- 시장 데이터 (CSV)

## 문제 해결

### 연결 오류

MySQL 연결 오류가 발생하는 경우:

1. MySQL 서버가 실행 중인지 확인
2. 사용자 및 비밀번호가 올바른지 확인
3. 데이터베이스가 존재하는지 확인
4. 네트워크 설정 확인 (방화벽, 접근 허용 등)

### 테이블 생성 오류

테이블 생성 오류가 발생하는 경우:

1. 데이터베이스 사용자에게 적절한 권한이 부여되었는지 확인
2. 데이터베이스 공간이 충분한지 확인
3. 로그 파일을 확인하여 구체적인 오류 파악 