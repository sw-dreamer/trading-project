#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델 호환성을 위한 빠른 설정 수정 스크립트
사용법: python quick_fix_config.py "모델경로" --update-config
"""
import os
import sys
import json
import torch
from pathlib import Path

def get_model_symbols(model_path):
    """모델에서 사용된 심볼 추출"""
    model_path = Path(model_path)
    
    print(f"🔍 모델 분석 중: {model_path}")
    
    # model_metadata.json에서 심볼 확인
    metadata_path = model_path / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            symbols = metadata.get('symbols', [])
            print(f"   └─ 메타데이터에서 심볼 발견: {symbols}")
            return symbols
        except Exception as e:
            print(f"   └─ 메타데이터 읽기 실패: {e}")
    
    # config.pth에서 심볼 정보 확인 시도
    config_path = model_path / "config.pth"
    if config_path.exists():
        try:
            config = torch.load(config_path, map_location='cpu')
            symbols = config.get('symbols', [])
            if symbols:
                print(f"   └─ 설정에서 심볼 발견: {symbols}")
                return symbols
        except Exception as e:
            print(f"   └─ 설정 읽기 실패: {e}")
    
    print("   └─ 심볼 정보를 찾을 수 없음. 기본 심볼 사용")
    # 일반적으로 학습에 사용되는 심볼들
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

def create_model_metadata(model_path, symbols):
    """모델 메타데이터 생성"""
    model_path = Path(model_path)
    metadata_path = model_path / "model_metadata.json"
    
    if metadata_path.exists():
        print(f"   └─ 메타데이터가 이미 존재합니다: {metadata_path}")
        return True
    
    metadata = {
        "training_date": "2025-05-23T12:06:24",
        "backtest_performance": {
            "total_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0
        },
        "symbols": symbols,
        "window_size": 30,
        "model_path": str(model_path)
    }
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   └─ 메타데이터 생성 완료: {metadata_path}")
        return True
    except Exception as e:
        print(f"   └─ 메타데이터 생성 실패: {e}")
        return False

def update_main_config(model_path):
    """메인 config.py 파일을 모델과 호환되도록 수정"""
    
    print(f"🔧 config.py 파일 수정 중...")
    
    # 모델에서 심볼 추출
    model_symbols = get_model_symbols(model_path)
    if not model_symbols:
        print("❌ 모델에서 심볼을 찾을 수 없습니다.")
        return False
    
    # 첫 번째 심볼을 테스트용으로 사용 (안전)
    test_symbol = model_symbols[0]
    print(f"   └─ 테스트용 심볼 선택: {test_symbol}")
    
    # 메타데이터 생성
    print(f"📝 모델 메타데이터 생성 중...")
    if not create_model_metadata(model_path, model_symbols):
        print("❌ 메타데이터 생성 실패")
        return False
    
    # config.py 경로
    config_path = Path("src/config/config.py")
    
    if not config_path.exists():
        print(f"❌ config.py 파일을 찾을 수 없습니다: {config_path}")
        return False
    
    try:
        # 백업 생성
        backup_path = config_path.with_suffix('.py.backup')
        if config_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)
            print(f"📦 기존 설정 백업: {backup_path}")
        
        # 새 설정 생성
        new_config_content = f'''"""
SAC 트레이딩 시스템 설정 파일 (모델 호환 버전)
자동 생성됨 - 모델: {model_path}
테스트용 심볼: {test_symbol}
"""
import os
import logging
import torch
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 데이터 관련 설정
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# 모델과 호환되는 심볼 설정 (테스트용으로 {test_symbol} 선택)
TARGET_SYMBOLS = ['{test_symbol}']
TRADING_SYMBOLS = ['{test_symbol}']

# 실시간 트레이딩 설정 (매우 안전한 테스트용)
ACTIVE_TRADING_SYMBOLS = {{
    '{test_symbol}': {{
        'model_path': r'{model_path}',
        'enabled': True,
        'max_position_size': 0.01,  # 1% (매우 안전)
        'trading_interval': 600,    # 10분
    }}
}}

# 현재 활성화된 트레이딩 종목들만 추출
trading_symbols = ['{test_symbol}']

# 매우 보수적인 리스크 관리 설정
GLOBAL_MAX_EXPOSURE = 0.05      # 전체 계좌의 5%까지만 투자
MAX_DRAWDOWN = 0.02             # 최대 2% 낙폭 허용
MAX_DAILY_LOSS = 0.01           # 일일 최대 손실 1%
EMERGENCY_STOP_LOSS = 0.03      # 응급 손절 3%

# 개별 종목 기본 설정 (매우 보수적)
DEFAULT_SYMBOL_CONFIG = {{
    'max_position_size': 0.01,      # 1%로 매우 낮게
    'trading_interval': 600,        # 10분 간격
    'min_trade_amount': 10.0,       # 최소 거래 금액 $10
    'max_trade_amount': 100.0,      # 최대 거래 금액 $100
}}

# TimescaleDB 데이터베이스 설정
DB_USER = "postgres"
DB_PASSWORD = "mysecretpassword"
DB_HOST = "192.168.40.193"
DB_PORT = 5432
DB_NAME = "mydb"
DB_SCHEMA = "public"
DB_TABLE_PREFIX = "ticker_"
DB_POOL_SIZE = 5
DB_TIMEOUT = 30
DB_MAX_OVERFLOW = 10
DB_RETRY_COUNT = 3
DB_RETRY_DELAY = 5

# 데이터 수집 설정
DATA_START_DATE = "2023-01-01"

# 데이터 전처리 설정
WINDOW_SIZE = 30
window_size = 30
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 트레이딩 환경 설정
INITIAL_BALANCE = 100000.0  # 페이퍼 트레이딩 초기 자본금
MAX_TRADING_UNITS = 10
TRANSACTION_FEE_PERCENT = 0.001

# SAC 모델 하이퍼파라미터
HIDDEN_DIM = 256
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4
LEARNING_RATE_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA_INIT = 0.2
TARGET_UPDATE_INTERVAL = 1
REPLAY_BUFFER_SIZE = 10000000

# 학습 설정
BATCH_SIZE = 256
NUM_EPISODES = 1000
EVALUATE_INTERVAL = 10
SAVE_MODEL_INTERVAL = 10
MAX_STEPS_PER_EPISODE = 1000

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alpaca API 설정 (페이퍼 트레이딩)
API_KEY = os.getenv("APCA_API_KEY_ID", "PK5SQ65O444Z6PRDV4UI")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "GIavNSovfJIwAnVZC14DI73pBhNBcAuY2CGhOKl5")
BASE_URL = "https://paper-api.alpaca.markets"  # 페이퍼 트레이딩 URL
DATA_FEED = 'iex'  # 무료 데이터 피드
DEBUG = True       # 디버그 모드 활성화
MAX_RETRIES = 5

# 백테스트 설정
BACKTEST_START_DATE = "2022-01-01"
BACKTEST_END_DATE = "2023-01-01"

# 로깅 설정
def setup_logger(name, log_file, level=logging.INFO):
    """로거 설정 함수"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

# 기본 로거 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"sac_trading_{{timestamp}}.log"
LOGGER = setup_logger("sac_trading", LOG_FILE)

# 데이터베이스 로거 설정
DB_LOG_FILE = LOGS_DIR / f"db_connection_{{timestamp}}.log"
DB_LOGGER = setup_logger("db_connection", DB_LOG_FILE)

# 헬퍼 함수들
def get_db_connection_string():
    """데이터베이스 연결 문자열 반환"""
    return f"postgresql+psycopg2://{{DB_USER}}:{{DB_PASSWORD}}@{{DB_HOST}}:{{DB_PORT}}/{{DB_NAME}}"

def get_active_symbol_config(symbol: str):
    """활성화된 심볼의 설정 반환"""
    return ACTIVE_TRADING_SYMBOLS.get(symbol, DEFAULT_SYMBOL_CONFIG)

def get_total_max_exposure():
    """전체 최대 노출도 반환"""
    return GLOBAL_MAX_EXPOSURE

# 데이터베이스 공통 쿼리 설정
DB_QUERIES = {{
    "get_table_list": """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = '{{schema}}' AND table_name LIKE '{{prefix}}%';
    """,
    "get_date_range": """
        SELECT 
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date,
            COUNT(*) as rows
        FROM {{table_name}};
    """,
    "get_daily_data": """
        SELECT *
        FROM {{table_name}}
        WHERE timestamp BETWEEN '{{start_date}}' AND '{{end_date}}'
        ORDER BY timestamp ASC;
    """
}}

class Config:
    # 디렉토리 설정
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    LOGS_DIR = LOGS_DIR
    MODELS_DIR = MODELS_DIR
    RESULTS_DIR = RESULTS_DIR
    
    # 심볼 설정
    TARGET_SYMBOLS = TARGET_SYMBOLS
    TRADING_SYMBOLS = TRADING_SYMBOLS
    trading_symbols = trading_symbols
    ACTIVE_TRADING_SYMBOLS = ACTIVE_TRADING_SYMBOLS
    DEFAULT_SYMBOL_CONFIG = DEFAULT_SYMBOL_CONFIG
    
    # 리스크 관리 설정
    GLOBAL_MAX_EXPOSURE = GLOBAL_MAX_EXPOSURE
    MAX_DRAWDOWN = MAX_DRAWDOWN
    MAX_DAILY_LOSS = MAX_DAILY_LOSS
    EMERGENCY_STOP_LOSS = EMERGENCY_STOP_LOSS
    
    # 개별 속성들 (트레이딩 스크립트에서 사용)
    max_position_size = DEFAULT_SYMBOL_CONFIG['max_position_size']
    max_trade_amount = DEFAULT_SYMBOL_CONFIG['max_trade_amount']
    trading_interval = DEFAULT_SYMBOL_CONFIG['trading_interval']
    
    # 데이터베이스 설정
    DB_USER = DB_USER
    DB_PASSWORD = DB_PASSWORD
    DB_HOST = DB_HOST
    DB_PORT = DB_PORT
    DB_NAME = DB_NAME
    DB_SCHEMA = DB_SCHEMA
    DB_TABLE_PREFIX = DB_TABLE_PREFIX
    DB_POOL_SIZE = DB_POOL_SIZE
    DB_TIMEOUT = DB_TIMEOUT
    DB_MAX_OVERFLOW = DB_MAX_OVERFLOW
    DB_RETRY_COUNT = DB_RETRY_COUNT
    DB_RETRY_DELAY = DB_RETRY_DELAY
    DB_QUERIES = DB_QUERIES
    
    # 데이터 설정
    DATA_START_DATE = DATA_START_DATE
    WINDOW_SIZE = WINDOW_SIZE
    window_size = WINDOW_SIZE
    TRAIN_RATIO = TRAIN_RATIO
    VALID_RATIO = VALID_RATIO
    TEST_RATIO = TEST_RATIO
    
    # 트레이딩 환경 설정
    INITIAL_BALANCE = INITIAL_BALANCE
    MAX_TRADING_UNITS = MAX_TRADING_UNITS
    TRANSACTION_FEE_PERCENT = TRANSACTION_FEE_PERCENT
    
    # 모델 설정
    HIDDEN_DIM = HIDDEN_DIM
    LEARNING_RATE_ACTOR = LEARNING_RATE_ACTOR
    LEARNING_RATE_CRITIC = LEARNING_RATE_CRITIC
    LEARNING_RATE_ALPHA = LEARNING_RATE_ALPHA
    GAMMA = GAMMA
    TAU = TAU
    ALPHA_INIT = ALPHA_INIT
    TARGET_UPDATE_INTERVAL = TARGET_UPDATE_INTERVAL
    REPLAY_BUFFER_SIZE = REPLAY_BUFFER_SIZE
    BATCH_SIZE = BATCH_SIZE
    NUM_EPISODES = NUM_EPISODES
    EVALUATE_INTERVAL = EVALUATE_INTERVAL
    SAVE_MODEL_INTERVAL = SAVE_MODEL_INTERVAL
    MAX_STEPS_PER_EPISODE = MAX_STEPS_PER_EPISODE
    DEVICE = DEVICE
    
    # 로거 설정
    LOGGER = LOGGER
    DB_LOGGER = DB_LOGGER
    
    # 백테스트 설정
    BACKTEST_START_DATE = BACKTEST_START_DATE
    BACKTEST_END_DATE = BACKTEST_END_DATE
    
    # 알파카 API 설정
    API_KEY = API_KEY
    API_SECRET = API_SECRET
    BASE_URL = BASE_URL
    DATA_FEED = DATA_FEED
    MAX_RETRIES = MAX_RETRIES
    DEBUG = DEBUG
    
    # 헬퍼 메서드들
    @staticmethod
    def get_db_connection_string():
        return get_db_connection_string()
    
    @staticmethod
    def get_active_symbol_config(symbol: str):
        return get_active_symbol_config(symbol)
    
    @staticmethod
    def get_total_max_exposure():
        return get_total_max_exposure()

    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO):
        return setup_logger(name, log_file, level)

# 글로벌 config 인스턴스
config = Config()

# 설정 확인 메시지
print("=" * 60)
print("🔧 모델 호환 설정 로드됨")
print("=" * 60)
print(f"📊 대상 심볼: {{TARGET_SYMBOLS}}")
print(f"🤖 모델 경로: {model_path}")
print(f"💰 최대 포지션: {{DEFAULT_SYMBOL_CONFIG['max_position_size']*100}}%")
print(f"💵 최대 거래금액: ${{DEFAULT_SYMBOL_CONFIG['max_trade_amount']}}")
print(f"⏰ 거래 간격: {{DEFAULT_SYMBOL_CONFIG['trading_interval']//60}}분")
print(f"🛡️ 일일 최대 손실: {{MAX_DAILY_LOSS*100}}%")
print("=" * 60)
'''
        
        # 새 설정 파일 작성
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_config_content)
        
        print(f"✅ config.py 업데이트 완료")
        print(f"📊 선택된 테스트 심볼: {test_symbol}")
        print(f"📊 모델의 모든 심볼: {model_symbols}")
        
        return True
        
    except Exception as e:
        print(f"❌ config.py 업데이트 실패: {e}")
        # 백업 복원 시도
        if backup_path.exists():
            import shutil
            shutil.copy2(backup_path, config_path)
            print(f"🔄 백업 파일 복원됨")
        return False

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='모델 호환성을 위한 빠른 설정 수정')
    parser.add_argument('model_path', help='모델 경로')
    parser.add_argument('--update-config', action='store_true', help='메인 config.py 파일 수정')
    
    args = parser.parse_args()
    
    print("🔧 모델 호환성 빠른 수정 도구")
    print("=" * 50)
    
    if args.update_config:
        print("📝 메인 config.py 파일을 모델과 호환되도록 수정합니다...")
        
        if update_main_config(args.model_path):
            print("\n" + "=" * 50)
            print("🎉 설정 수정 완료!")
            print("=" * 50)
            print("이제 실시간 트레이딩을 다시 시도해보세요:")
            print()
            print(f'python -m src.trading.run_live_trading --model_path="{args.model_path}" --dry_run')
            print()
            print("📊 적용된 안전 설정:")
            print("   └─ 최대 포지션 크기: 1%")
            print("   └─ 최대 거래 금액: $100")
            print("   └─ 거래 간격: 10분")
            print("   └─ 일일 최대 손실: 1%")
            print("   └─ 페이퍼 트레이딩만 사용")
        else:
            print("❌ 설정 수정 실패!")
    else:
        print("사용법:")
        print(f'python {sys.argv[0]} "{args.model_path}" --update-config')

if __name__ == "__main__":
    main()