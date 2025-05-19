"""
SAC 트레이딩 시스템 설정 파일
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

# 대상 주식 종목 (미국 빅테크 기업)
TARGET_SYMBOLS = [
    "AAPL",  # Apple Inc.
    "TSLA"
    ,'VALE', 'NEM', 'FCX', 'T', 'NFLX', 'WBD', 'AMZN', 'NKE', 'PEP', 'KVUE', 'PM', 'PBR', 'KMI', 'SHEL', 'ITUB', 'USB', 'MUFG', 'AZN', 'ABT', 'UNH', 'CSX', 'BA', 'RTX', 'NVDA', 'ERIC', 'YMM', 'NEE'
    ]

# TimescaleDB 데이터베이스 설정
DB_USER = "postgres"
DB_PASSWORD = "mysecretpassword"
DB_HOST = "192.168.40.193"
DB_PORT = 5432
DB_NAME = "mydb"
DB_SCHEMA = "public"
DB_TABLE_PREFIX = "ticker_"  # 테이블 이름 prefix (ticker_AAPL, ticker_MSFT 등)
DB_POOL_SIZE = 5
DB_TIMEOUT = 30
DB_MAX_OVERFLOW = 10
DB_RETRY_COUNT = 3
DB_RETRY_DELAY = 5  # 초 단위

# 데이터 수집 설정
DATA_START_DATE = "2023-01-01"  # 10년 데이터
#DATA_FREQUENCY = "daily"  # 일별 데이터

# 데이터 전처리 설정
WINDOW_SIZE = 30  # 관측 윈도우 크기
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 트레이딩 환경 설정
INITIAL_BALANCE = 100000.0  # 초기 자본금
MAX_TRADING_UNITS = 10  # 최대 거래 단위
TRANSACTION_FEE_PERCENT = 0.001  # 거래 수수료 (0.1%)

# SAC 모델 하이퍼파라미터
HIDDEN_DIM = 256
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4
LEARNING_RATE_ALPHA = 3e-4
GAMMA = 0.99  # 할인 계수
TAU = 0.005  # 타겟 네트워크 소프트 업데이트 계수
ALPHA_INIT = 0.2  # 초기 엔트로피 계수
TARGET_UPDATE_INTERVAL = 1
REPLAY_BUFFER_SIZE = 10000000

# 학습 설정
BATCH_SIZE = 256
NUM_EPISODES = 1000
EVALUATE_INTERVAL = 10
SAVE_MODEL_INTERVAL = 50
MAX_STEPS_PER_EPISODE = 1000

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
LOG_FILE = LOGS_DIR / f"sac_trading_{timestamp}.log"
LOGGER = setup_logger("sac_trading", LOG_FILE)

# 데이터베이스 로거 설정
DB_LOG_FILE = LOGS_DIR / f"db_connection_{timestamp}.log"
DB_LOGGER = setup_logger("db_connection", DB_LOG_FILE)

# 백테스트 설정
BACKTEST_START_DATE = "2022-01-01"
BACKTEST_END_DATE = "2023-01-01" 

# 데이터베이스 연결 재시도 설정
def get_db_connection_string():
    """데이터베이스 연결 문자열 반환"""
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 데이터베이스 공통 쿼리 설정
DB_QUERIES = {
    "get_table_list": """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = '{schema}' AND table_name LIKE '{prefix}%';
    """,
    "get_date_range": """
        SELECT 
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date,
            COUNT(*) as rows
        FROM {table_name};
    """,
    "get_daily_data": """
        SELECT *
        FROM {table_name}
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp ASC;
    """
}

class Config:
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    LOGS_DIR = LOGS_DIR
    MODELS_DIR = MODELS_DIR
    RESULTS_DIR = RESULTS_DIR
    TARGET_SYMBOLS = TARGET_SYMBOLS
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
    DATA_START_DATE = DATA_START_DATE
    WINDOW_SIZE = WINDOW_SIZE
    window_size = WINDOW_SIZE
    TRAIN_RATIO = TRAIN_RATIO
    VALID_RATIO = VALID_RATIO
    TEST_RATIO = TEST_RATIO
    INITIAL_BALANCE = INITIAL_BALANCE
    MAX_TRADING_UNITS = MAX_TRADING_UNITS
    TRANSACTION_FEE_PERCENT = TRANSACTION_FEE_PERCENT
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
    LOGGER = LOGGER
    DB_LOGGER = DB_LOGGER
    BACKTEST_START_DATE = BACKTEST_START_DATE
    BACKTEST_END_DATE = BACKTEST_END_DATE

    @staticmethod
    def get_db_connection_string():
        return get_db_connection_string()

    DB_QUERIES = DB_QUERIES

    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO):
        return setup_logger(name, log_file, level)

config = Config()
