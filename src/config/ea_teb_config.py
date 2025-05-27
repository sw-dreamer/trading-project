"""
백테스팅/학습 전용 설정 파일
SAC 트레이딩 시스템 - 학습, 평가, 백테스트용
"""
import os
import logging
import torch
import glob
import re
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 디렉토리 설정
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# 심볼 설정
TARGET_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
trading_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']

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

# 데이터 전처리 설정
WINDOW_SIZE = 30
window_size = 30
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 트레이딩 환경 설정
INITIAL_BALANCE = 100000.0
MAX_TRADING_UNITS = 10
TRANSACTION_FEE_PERCENT = 0.0

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

# MySQL 설정 (백테스트 결과 저장용)
MYSQL_HOST = os.getenv("MYSQL_HOST", "192.168.40.199")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "trading")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mysecretpassword")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
SAVE_TO_DATABASE = True
SKIP_DB_ON_ERROR = True

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

# Config 클래스 (백워드 호환성)
class Config:
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

    # 데이터 설정
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

    # MySQL 설정
    MYSQL_HOST = MYSQL_HOST
    MYSQL_DATABASE = MYSQL_DATABASE
    MYSQL_USER = MYSQL_USER
    MYSQL_PASSWORD = MYSQL_PASSWORD
    MYSQL_PORT = MYSQL_PORT
    SAVE_TO_DATABASE = SAVE_TO_DATABASE
    SKIP_DB_ON_ERROR = SKIP_DB_ON_ERROR

    # 로거 설정
    LOGGER = LOGGER

    # 헬퍼 메서드들
    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO):
        return setup_logger(name, log_file, level)

# 글로벌 config 인스턴스
config = Config()

# 설정 확인 메시지
print("=" * 60)
print("🔧 백테스팅/학습 전용 설정 로드됨")
print("=" * 60)
print(f"📊 대상 심볼: {TARGET_SYMBOLS}")
print(f"🖥️  연산 장치: {DEVICE}")
print(f"📁 모델 디렉토리: {MODELS_DIR}")
print(f"💾 MySQL 연결: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
print(f"🔢 배치 크기: {BATCH_SIZE}")
print(f"🎯 학습 에피소드: {NUM_EPISODES}")
print("=" * 60)