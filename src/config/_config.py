"""
SAC 트레이딩 시스템 설정 파일 (모델 호환 버전)
테스트용 심볼: AAPL
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

TARGET_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
trading_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']

base_model_dir = r'models\{ticker}'
model_name_pattern = 'final_sac_model_{ticker}'

tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
ACTIVE_TRADING_SYMBOLS = {}

import glob
import re
from datetime import datetime
from pathlib import Path

# 모델 패턴 정의
MODEL_PATTERNS = {
    'mlp': r'final_sac_model_([A-Z]+)_(\d{8}_\d{6})',
    'cnn': r'final_cnn_sac_model_(\d{8}_\d{6})',
    'lstm': r'final_lstm_sac_model_(\d{8}_\d{6})',
    'transformer': r'final_transformer_sac_model_(\d{8}_\d{6})'
}


def find_latest_model_by_pattern(model_type: str, symbol: str = None, base_dir: str = None) -> str:
    """
    패턴별로 최신 모델 경로 찾기

    Args:
        model_type: 모델 타입 ('mlp', 'cnn', 'lstm', 'transformer')
        symbol: 주식 심볼 (MLP 모델의 경우 필수)
        base_dir: 모델 검색 기본 디렉토리 (기본값: models/)

    Returns:
        str: 최신 모델의 전체 경로, 없으면 None
    """
    if base_dir is None:
        base_dir = os.path.join(ROOT_DIR, "models")

    if not os.path.exists(base_dir):
        print(f"⚠️  모델 디렉토리가 존재하지 않습니다: {base_dir}")
        return None

    pattern = MODEL_PATTERNS.get(model_type.lower())
    if not pattern:
        print(f"⚠️  지원하지 않는 모델 타입: {model_type}")
        return None

    # 디렉토리 내 모든 폴더 스캔
    model_dirs = []

    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                match = re.match(pattern, item)
                if match:
                    # MLP 모델의 경우 심볼 매칭 확인
                    if model_type.lower() == 'mlp':
                        if symbol and match.group(1) == symbol:
                            timestamp = match.group(2)
                            model_dirs.append((timestamp, item_path))
                    else:
                        # CNN, LSTM 등은 심볼 무관
                        timestamp = match.group(1)
                        model_dirs.append((timestamp, item_path))

        if not model_dirs:
            if model_type.lower() == 'mlp' and symbol:
                print(f"⚠️  {symbol}에 대한 {model_type.upper()} 모델을 찾을 수 없습니다.")
            else:
                print(f"⚠️  {model_type.upper()} 모델을 찾을 수 없습니다.")
            return None

        # 타임스탬프 기준으로 정렬하여 최신 모델 선택
        model_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_model_path = model_dirs[0][1]

        # config.pth 파일 존재 여부 확인
        config_path = os.path.join(latest_model_path, 'config.pth')
        if os.path.exists(config_path):
            print(f"✅ 최신 {model_type.upper()} 모델 발견: {os.path.basename(latest_model_path)}")
            return latest_model_path
        else:
            print(f"⚠️  config.pth 파일이 없습니다: {latest_model_path}")
            return None

    except Exception as e:
        print(f"❌ 모델 검색 중 오류: {str(e)}")
        return None


# 모델 파일 존재 여부 확인 함수
def check_model_exists(ticker, custom_path=None):
    """
    모델 존재 여부 확인 (향상된 버전)

    Args:
        ticker: 주식 심볼
        custom_path: 사용자 지정 모델 경로 (선택사항)

    Returns:
        bool: 모델 존재 여부
    """
    if custom_path:
        # 사용자가 직접 경로를 지정한 경우
        config_path = os.path.join(custom_path, 'config.pth')
        return os.path.exists(config_path)
    else:
        # 기본 패턴으로 확인 (기존 방식)
        model_path = fr"{base_model_dir.format(ticker=ticker)}\{model_name_pattern.format(ticker=ticker)}"
        print(f'model_path : {model_path}')
        config_path = os.path.join(model_path, 'config.pth')
        print(f'config_path:{config_path}')
        if os.path.exists(config_path):
            return True

        # 기존 패턴에서 못 찾으면 최신 MLP 모델 검색
        latest_mlp = find_latest_model_by_pattern('mlp', ticker)
        return latest_mlp is not None


def get_model_path_for_ticker(ticker: str, model_type: str = 'mlp') -> str:
    """
    특정 ticker와 모델 타입에 대한 모델 경로 반환

    Args:
        ticker: 주식 심볼
        model_type: 모델 타입 ('mlp', 'cnn', 'lstm', 'transformer')

    Returns:
        str: 모델 경로, 없으면 None
    """
    # 1. 기본 패턴으로 먼저 확인 (기존 방식)
    if model_type.lower() == 'cnn':
        default_path = fr"{base_model_dir.format(ticker=ticker)}\{model_name_pattern.format(ticker=ticker)}"
        print(f'default_path:{default_path}')
        if os.path.exists(os.path.join(default_path, 'config.pth')):
            print(f'default_path:{default_path}')   
            return default_path

    # 2. 최신 모델 검색
    latest_model = find_latest_model_by_pattern(model_type, ticker)
    return latest_model


def setup_active_trading_symbols():
    """활성 트레이딩 심볼 설정 (향상된 버전)"""
    active_symbols = {}

    for ticker in tickers:
        # MLP 모델 우선 검색
        model_path = get_model_path_for_ticker(ticker, 'mlp')

        if model_path:
            active_symbols[ticker] = {
                'model_path': model_path,
                'model_type': 'mlp',
                'enabled': True,
                'max_position_size': 0.01,
                'trading_interval': 60,
            }
            print(f"✅ {ticker}: {os.path.basename(model_path)}")
        else:
            print(f"⚠️  {ticker} 모델이 없어 거래 대상에서 제외됩니다.")

    return active_symbols


# 수정된 버전 - 백테스트 시에는 모델 존재 여부 확인 건너뛰기
tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']


# 활성 심볼 설정 (백테스트 모드 고려)
def setup_active_trading_symbols_for_backtest():
    """백테스트용 활성 트레이딩 심볼 설정"""
    active_symbols = {}

    for ticker in tickers:
        # 라이브 트레이딩용: 실제 모델 검색
        model_path = get_model_path_for_ticker(ticker, 'mlp')

        if model_path:
            active_symbols[ticker] = {
                'model_path': model_path,
                'model_type': 'mlp',
                'enabled': True,
                'max_position_size': 0.01,
                'trading_interval': 60,
            }
            print(f"✅ {ticker}: {os.path.basename(model_path)}")
        else:
            # 백테스트용: 모델 경로는 None으로 설정 (동적으로 로드)
            active_symbols[ticker] = {
                'model_path': None,  # 백테스트 시 동적 설정
                'model_type': 'mlp',
                'enabled': True,
                'max_position_size': 0.01,
                'trading_interval': 60,
            }
            print(f"⚠️  {ticker}: 백테스트 모드에서 동적 로드 예정")

    return active_symbols


# 활성 심볼 설정 실행
ACTIVE_TRADING_SYMBOLS = setup_active_trading_symbols_for_backtest()


# 전역 모델 검색 함수 (백테스트에서 사용)
def find_model_for_backtest(symbol: str = None, model_type: str = None, model_path: str = None) -> dict:
    """
    백테스트용 모델 검색

    Args:
        symbol: 주식 심볼 (선택사항)
        model_type: 모델 타입 (선택사항: 'mlp', 'cnn', 'lstm')
        model_path: 직접 지정한 모델 경로 (선택사항)

    Returns:
        dict: {'path': str, 'type': str, 'symbol': str} 또는 None
    """
    # 1. 직접 경로 지정된 경우
    if model_path:
        if os.path.exists(os.path.join(model_path, 'config.pth')):
            # 경로에서 모델 타입 추론
            model_name = os.path.basename(model_path)
            detected_type = 'mlp'  # 기본값
            detected_symbol = None

            for mtype, pattern in MODEL_PATTERNS.items():
                if re.match(pattern, model_name):
                    detected_type = mtype
                    if mtype == 'mlp':
                        match = re.match(pattern, model_name)
                        if match:
                            detected_symbol = match.group(1)
                    break

            return {
                'path': model_path,
                'type': detected_type,
                'symbol': detected_symbol or symbol
            }

    # 2. 자동 검색
    if model_type and symbol:
        # 특정 타입과 심볼로 검색
        latest_model = find_latest_model_by_pattern(model_type, symbol)
        if latest_model:
            return {
                'path': latest_model,
                'type': model_type,
                'symbol': symbol
            }
    elif model_type:
        # 타입만 지정 (CNN, LSTM 등)
        latest_model = find_latest_model_by_pattern(model_type)
        if latest_model:
            return {
                'path': latest_model,
                'type': model_type,
                'symbol': symbol
            }
    elif symbol:
        # 심볼만 지정 (MLP 우선 검색)
        for mtype in ['mlp', 'cnn', 'lstm']:
            latest_model = find_latest_model_by_pattern(mtype, symbol if mtype == 'mlp' else None)
            if latest_model:
                return {
                    'path': latest_model,
                    'type': mtype,
                    'symbol': symbol
                }

    return None


# 사용 가능한 모델 목록 출력
def list_available_models():
    """사용 가능한 모든 모델 목록 출력"""
    print("\n" + "=" * 60)
    print("📂 사용 가능한 모델 목록")
    print("=" * 60)

    base_dir = os.path.join(ROOT_DIR, "models")
    if not os.path.exists(base_dir):
        print("❌ 모델 디렉토리가 존재하지 않습니다.")
        return

    models_found = []

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.pth')):
            # 모델 타입과 심볼 감지
            model_info = {'name': item, 'path': item_path, 'type': 'unknown', 'symbol': None}

            for mtype, pattern in MODEL_PATTERNS.items():
                match = re.match(pattern, item)
                if match:
                    model_info['type'] = mtype
                    if mtype == 'mlp' and len(match.groups()) >= 2:
                        model_info['symbol'] = match.group(1)
                    break

            models_found.append(model_info)

    # 타입별로 정렬
    models_found.sort(key=lambda x: (x['type'], x['symbol'] or '', x['name']))

    for model in models_found:
        symbol_info = f" ({model['symbol']})" if model['symbol'] else ""
        print(f"🤖 {model['type'].upper()}{symbol_info}: {model['name']}")

    print(f"\n📊 총 {len(models_found)}개 모델 발견")
    print("=" * 60 + "\n")


# 설정 완료 메시지
print("=" * 60)
print("🔧 향상된 모델 감지 시스템 로드됨")
print(f"📊 활성화된 심볼: {len(ACTIVE_TRADING_SYMBOLS)}개")
print("🎯 백테스트에서 자동 모델 감지 지원")
print("=" * 60)

# 매우 보수적인 리스크 관리 설정
GLOBAL_MAX_EXPOSURE = 0.05  # 전체 계좌의 5%까지만 투자
MAX_DRAWDOWN = 0.02  # 최대 2% 낙폭 허용
MAX_DAILY_LOSS = 0.01  # 일일 최대 손실 1%
EMERGENCY_STOP_LOSS = 0.03  # 응급 손절 3%

# 개별 종목 기본 설정 (매우 보수적)
DEFAULT_SYMBOL_CONFIG = {
    'max_position_size': 0.01,  # 1%로 매우 낮게
    'trading_interval': 60,  # 10분 간격
    'min_trade_amount': 10.0,  # 최소 거래 금액 $10
    'max_trade_amount': 100.0,  # 최대 거래 금액 $100
}

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
TRANSACTION_FEE_PERCENT = 0.0  # 수수료

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
DEBUG = True  # 디버그 모드 활성화
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
LOG_FILE = LOGS_DIR / f"sac_trading_{timestamp}.log"
LOGGER = setup_logger("sac_trading", LOG_FILE)

# 데이터베이스 로거 설정
DB_LOG_FILE = LOGS_DIR / f"db_connection_{timestamp}.log"
DB_LOGGER = setup_logger("db_connection", DB_LOG_FILE)


# 헬퍼 함수들
def get_db_connection_string():
    """데이터베이스 연결 문자열 반환"""
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_active_symbol_config(symbol: str):
    """활성화된 심볼의 설정 반환"""
    return ACTIVE_TRADING_SYMBOLS.get(symbol, DEFAULT_SYMBOL_CONFIG)


def get_total_max_exposure():
    """전체 최대 노출도 반환"""
    return GLOBAL_MAX_EXPOSURE


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
    # MySQL 데이터베이스 설정 (환경 변수 우선)
    MYSQL_HOST = os.getenv("MYSQL_HOST", "192.168.40.199")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "trading")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mysecretpassword")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))

    # 백테스트 DB 저장 설정
    SAVE_TO_DATABASE = True  # 기본적으로 DB 저장 활성화
    SKIP_DB_ON_ERROR = True  # DB 오류 시 백테스트는 계속 진행

    # TimescaleDB 설정 (환경 변수 우선)
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")
    DB_HOST = os.getenv("DB_HOST", "192.168.40.193")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "mydb")
    
    # 디렉토리 설정
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    LOGS_DIR = LOGS_DIR
    MODELS_DIR = MODELS_DIR
    RESULTS_DIR = RESULTS_DIR

    # 심볼 설정
    TARGET_SYMBOLS = TARGET_SYMBOLS
    TRADING_SYMBOLS = trading_symbols
    trading_symbols = trading_symbols
    ACTIVE_TRADING_SYMBOLS = ACTIVE_TRADING_SYMBOLS
    DEFAULT_SYMBOL_CONFIG = DEFAULT_SYMBOL_CONFIG

    # 리스크 관리 설정
    GLOBAL_MAX_EXPOSURE = GLOBAL_MAX_EXPOSURE
    MAX_DRAWDOWN = MAX_DRAWDOWN
    max_drawdown = MAX_DRAWDOWN
    MAX_DAILY_LOSS = MAX_DAILY_LOSS
    max_daily_loss = MAX_DAILY_LOSS
    EMERGENCY_STOP_LOSS = EMERGENCY_STOP_LOSS
    emergency_stop_loss = EMERGENCY_STOP_LOSS

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
    def get_mysql_connection_string():
        """MySQL 연결 문자열 반환"""
        return f"mysql+pymysql://{Config.MYSQL_USER}:{Config.MYSQL_PASSWORD}@{Config.MYSQL_HOST}:{Config.MYSQL_PORT}/{Config.MYSQL_DATABASE}"

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

# 백워드 호환성을 위한 전역 변수들 (backtester.py 등에서 직접 import 가능) - 수정됨
MYSQL_HOST = Config.MYSQL_HOST
MYSQL_DATABASE = Config.MYSQL_DATABASE  
MYSQL_USER = Config.MYSQL_USER
MYSQL_PASSWORD = Config.MYSQL_PASSWORD
MYSQL_PORT = Config.MYSQL_PORT
SAVE_TO_DATABASE = Config.SAVE_TO_DATABASE
SKIP_DB_ON_ERROR = Config.SKIP_DB_ON_ERROR

# 설정 확인 메시지
print("=" * 60)
print("🔧 모델 호환 설정 로드됨")
print("=" * 60)
print(f"📊 대상 심볼: {TARGET_SYMBOLS}")
print(f"💰 최대 포지션: {DEFAULT_SYMBOL_CONFIG['max_position_size'] * 100}%")
print(f"💵 최대 거래금액: ${DEFAULT_SYMBOL_CONFIG['max_trade_amount']}")
print(f"⏰ 거래 간격: {DEFAULT_SYMBOL_CONFIG['trading_interval'] // 60}분")
print(f"🛡️ 일일 최대 손실: {MAX_DAILY_LOSS * 100}%")
print(f"💾 MySQL 연결: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
print("=" * 60)