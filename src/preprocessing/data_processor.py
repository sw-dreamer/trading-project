"""
데이터 전처리 및 특성 추출 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import RobustScaler, StandardScaler
import talib as ta
from datetime import datetime
from pathlib import Path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from src.config.config import (
    WINDOW_SIZE,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    DATA_DIR,
    LOGGER
)
from src.utils.utils import create_directory, save_to_csv, load_from_csv

class DataProcessor:
    """
    주식 데이터 전처리 및 특성 추출을 위한 클래스
    """
    
    def __init__(self, window_size: int = WINDOW_SIZE):
        """
        DataProcessor 클래스 초기화
        
        Args:
            window_size: 관측 윈도우 크기
        """
        self.window_size = window_size
        self.scalers = {}
        self.normalized_data_columns = {}  # 여기에 추가
        LOGGER.info(f"DataProcessor 초기화 완료: 윈도우 크기 {window_size}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        주식 데이터 전처리 (결측치 처리, 이상치 제거 등)
        
        Args:
            data: 원본 주식 데이터 데이터프레임
            
        Returns:
            전처리된 데이터프레임
        """
        if data.empty:
            LOGGER.warning("빈 데이터프레임이 입력되었습니다.")
            return data
        
        # 결측치 처리
        data = data.copy()
        data.fillna(method='ffill', inplace=True)  # 앞의 값으로 채우기
        data.fillna(method='bfill', inplace=True)  # 뒤의 값으로 채우기
        
        # 중복 인덱스 제거
        data = data[~data.index.duplicated(keep='first')]
        
        # 오름차순 정렬 (시간순)
        data = data.sort_index()
        
        # 필수 컬럼 존재 확인
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                LOGGER.warning(f"데이터에 필수 컬럼 '{col}'이 없습니다")
                # 가능한 경우 대문자 컬럼명으로 시도
                upper_col = col.upper()
                if upper_col in data.columns:
                    data[col] = data[upper_col]
                    data = data.drop(columns=[upper_col])
        
        # 데이터 유형 변환 - 모든 가격 데이터를 float64로 변환
        for col in required_cols:
            if col in data.columns:
                # 문자열이나 다른 유형의 데이터를 float로 변환 시도
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    LOGGER.info(f"{col} 컬럼을 float64 타입으로 변환했습니다")
                except Exception as e:
                    LOGGER.error(f"{col} 컬럼 변환 중 오류: {str(e)}")
        
        # 0 또는 음수 값 처리 (거래량은 0일 수 있음)
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                # 0 또는 음수 값을 가진 행 식별
                mask = data[col] <= 0
                if mask.any():
                    LOGGER.warning(f"{col} 열에서 {mask.sum()}개의 0 또는 음수 값이 발견되었습니다.")
                    # 해당 행을 제거하거나 대체할 수 있음
                    # 여기서는 이전 값으로 대체
                    data.loc[mask, col] = data[col].shift(1)[mask]
        
        # 결측치 다시 확인 및 처리
        if data.isna().any().any():
            LOGGER.warning(f"데이터에 결측치가 아직 있습니다: {data.isna().sum().sum()}개")
            # 추가 결측치 처리
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            # 그래도 남아있는 결측치는 컬럼 평균으로 대체
            for col in data.columns:
                if data[col].isna().any():
                    if data[col].dtype.kind in 'iufc':  # 숫자형 데이터만
                        data[col].fillna(data[col].mean(), inplace=True)
                    else:
                        data[col].fillna(0, inplace=True)
        
        LOGGER.info(f"데이터 전처리 완료: {len(data)} 행")
        return data
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표를 계산하여 특성 추출
        
        Args:
            data: 전처리된 주식 데이터 데이터프레임
            
        Returns:
            특성이 추가된 데이터프레임
        """
        if data.empty:
            LOGGER.warning("빈 데이터프레임이 입력되었습니다.")
            return data
        
        df = data.copy()
        
        # 기본 가격 데이터
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volumes = df['volume'].values
        
        try:
            # 이동평균선
            df['sma5'] = self._safe_ta(ta.SMA, close_prices, timeperiod=5)
            df['sma10'] = self._safe_ta(ta.SMA, close_prices, timeperiod=10)
            df['sma20'] = self._safe_ta(ta.SMA, close_prices, timeperiod=20)
            df['sma60'] = self._safe_ta(ta.SMA, close_prices, timeperiod=60)
            df['sma120'] = self._safe_ta(ta.SMA, close_prices, timeperiod=120)
            
            # EMA (지수이동평균)
            df['ema5'] = self._safe_ta(ta.EMA, close_prices, timeperiod=5)
            df['ema10'] = self._safe_ta(ta.EMA, close_prices, timeperiod=10)
            df['ema20'] = self._safe_ta(ta.EMA, close_prices, timeperiod=20)
            
            # MACD
            macd, macd_signal, macd_hist = self._safe_ta(ta.MACD, close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # RSI (상대강도지수)
            df['rsi14'] = self._safe_ta(ta.RSI, close_prices, timeperiod=14)
            
            # Bollinger Bands
            upper, middle, lower = self._safe_ta(ta.BBANDS, close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            # BB 너비 추가
            df['bb_width'] = (upper - lower) / middle
            
            # ATR (평균진실범위)
            df['atr14'] = self._safe_ta(ta.ATR, high_prices, low_prices, close_prices, timeperiod=14)
            
            # CCI (상품채널지수)
            df['cci14'] = self._safe_ta(ta.CCI, high_prices, low_prices, close_prices, timeperiod=14)
            
            # ROC (변화율)
            df['roc5'] = self._safe_ta(ta.ROC, close_prices, timeperiod=5)
            df['roc10'] = self._safe_ta(ta.ROC, close_prices, timeperiod=10)
            
            # OBV (거래량 가중치)
            df['obv'] = self._safe_ta(ta.OBV, close_prices, volumes)
            
            # Williams %R
            df['willr14'] = self._safe_ta(ta.WILLR, high_prices, low_prices, close_prices, timeperiod=14)
            
            # Stochastic
            slowk, slowd = self._safe_ta(ta.STOCH, high_prices, low_prices, close_prices, 
                                         fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # MFI (자금흐름지수)
            df['mfi14'] = self._safe_ta(ta.MFI, high_prices, low_prices, close_prices, volumes, timeperiod=14)
            
            # 일일 가격 변화율
            df['daily_return'] = df['close'].pct_change()
            
            # 거래량 변화율
            df['volume_change'] = df['volume'].pct_change()
            
            # 가격 변동성 (고가-저가)/종가
            df['volatility'] = (df['high'] - df['low']) / df['close']
            
            # 이동평균 교차 지표
            df['ma5_cross_ma20'] = (df['sma5'] > df['sma20']).astype(int)
            df['ma10_cross_ma60'] = (df['sma10'] > df['sma60']).astype(int)
            
            # 볼린저 밴드 관련 지표
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 이전 N일 종가 대비 현재 종가 비율
            for n in [1, 3, 5, 10]:
                df[f'close_ratio_{n}d'] = df['close'] / df['close'].shift(n)
            
        except Exception as e:
            LOGGER.error(f"특성 추출 중 오류 발생: {str(e)}")
            # 기본 특성만 사용
            df['daily_return'] = df['close'].pct_change()
        
        # 이상치 및 무한대 값 처리
        df = self._handle_outliers_and_infinities(df)
        
        # NaN 값 처리
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        LOGGER.info(f"특성 추출 완료: {len(df.columns)}개 특성")
        return df
    
    def _safe_ta(self, ta_func, *args, **kwargs):
        """
        TA-Lib 함수를 안전하게 호출하는 헬퍼 함수
        
        Args:
            ta_func: TA-Lib 함수
            *args: 위치 인자
            **kwargs: 키워드 인자
            
        Returns:
            TA-Lib 함수의 결과
        """
        try:
            # 입력 배열 타입 확인 및 강제 변환
            new_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    # 타입 체크 및 로깅
                    if arg.dtype != np.float64:
                        LOGGER.info(f"배열 타입 {arg.dtype}를 float64로 변환 시도")
                        arg = arg.astype(np.float64)
                    
                    # NaN 값 처리
                    if np.isnan(arg).any():
                        LOGGER.warning("입력 배열에 NaN 값이 있습니다. 처리 중...")
                        # NaN 값이 아닌 요소의 평균 계산
                        mean_val = np.nanmean(arg) if np.any(~np.isnan(arg)) else 0
                        # NaN 값을 평균으로 대체
                        arg = np.nan_to_num(arg, nan=mean_val)
                new_args.append(arg)
            
            # 변환된 인자로 함수 호출
            result = ta_func(*new_args, **kwargs)
            return result
        except Exception as e:
            LOGGER.warning(f"기술적 지표 계산 실패: {str(e)}")
            # 다중 출력 함수인지 확인
            if ta_func.__name__ in ['BBANDS', 'MACD', 'STOCH']:
                shapes = [len(args[0])] * 3  # 입력 크기와 동일한 3개의 배열
                return [np.full(shape, np.nan) for shape in shapes]
            else:
                return np.full(len(args[0]), np.nan)
    
    def _handle_outliers_and_infinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이상치 및 무한대 값 처리
        
        Args:
            df: 데이터프레임
            
        Returns:
            이상치가 처리된 데이터프레임
        """
        # 무한대 값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 극단치 처리: 각 컬럼별로 99.5%, 0.5% 경계를 넘어가는 값을 경계값으로 대체
        for col in df.columns:
            if df[col].dtype == np.float64 or df[col].dtype == np.int64:
                # 결측치가 너무 많은 컬럼은 건너뜀
                if df[col].isna().sum() > len(df) * 0.5:
                    continue
                    
                q_low = df[col].quantile(0.005)
                q_high = df[col].quantile(0.995)
                # 경계 값 적용
                df[col] = df[col].clip(lower=q_low, upper=q_high)
        
        # 남은 결측치 처리
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # NaN 체크 및 로그
        nan_counts = df.isna().sum().sum()
        if nan_counts > 0:
            LOGGER.warning(f"결측치 {nan_counts}개가 여전히 존재합니다")
        
        # 무한대 값 체크 및 로그
        inf_counts = np.isinf(df.values).sum()
        if inf_counts > 0:
            LOGGER.warning(f"무한대 값 {inf_counts}개가 여전히 존재합니다")
            # 무한대 값을 0으로 대체
            df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def normalize_features(self, data: pd.DataFrame, symbol: str, is_training: bool = True) -> pd.DataFrame:
        """
        특성 정규화
        
        Args:
            data: 특성이 추출된 데이터프레임
            symbol: 주식 심볼 (스케일러 구분용)
            is_training: 학습 데이터 여부
            
        Returns:
            정규화된 데이터프레임
        """
        if data.empty:
            return data
        
        df = data.copy()
        
        # 정규화에서 제외할 컬럼
        exclude_cols = ['date'] if 'date' in df.columns else []
        
        # 정규화할 컬럼
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
        
        if is_training:
            # 학습 데이터인 경우 새 스케일러 생성
            if symbol not in self.scalers:
                self.scalers[symbol] = {}
            
            scaler = RobustScaler()
            self.scalers[symbol]['scaler'] = scaler
            normalized_data = scaler.fit_transform(df[cols_to_normalize])
        else:
            # 테스트 데이터인 경우 기존 스케일러 사용
            if symbol not in self.scalers or 'scaler' not in self.scalers[symbol]:
                LOGGER.warning(f"{symbol}에 대한 스케일러가 없습니다. 새로 생성합니다.")
                scaler = RobustScaler()
                if symbol not in self.scalers:
                    self.scalers[symbol] = {}
                self.scalers[symbol]['scaler'] = scaler
                normalized_data = scaler.fit_transform(df[cols_to_normalize])
            else:
                scaler = self.scalers[symbol]['scaler']
                normalized_data = scaler.transform(df[cols_to_normalize])
        
        # 정규화된 데이터로 데이터프레임 생성
        normalized_df = pd.DataFrame(normalized_data, index=df.index, columns=cols_to_normalize)
        
        # 제외된 컬럼 다시 추가
        for col in exclude_cols:
            normalized_df[col] = df[col]
        
        LOGGER.info(f"{symbol} 데이터 정규화 완료")
        return normalized_df
    
    def create_window_samples(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 윈도우 샘플 생성
        
        Args:
            data: 정규화된 데이터프레임
            
        Returns:
            (X, y) 튜플: X는 윈도우 샘플, y는 다음 날 종가 변화율
        """
        if len(data) < self.window_size + 1:
            LOGGER.warning(f"데이터가 너무 적습니다. 최소 {self.window_size + 1}개 필요")
            return np.array([]), np.array([])
        
        # 다음 날 종가 변화율을 타겟으로 사용
        if 'daily_return' in data.columns:
            y = data['daily_return'].shift(-1).values[:-1]
        else:
            # daily_return이 없는 경우 계산
            y = (data['close'].shift(-1) / data['close'] - 1).values[:-1]
        
        # 윈도우 샘플 생성
        X = []
        for i in range(len(data) - self.window_size):
            X.append(data.iloc[i:i+self.window_size].values)
        
        X = np.array(X)
        y = y[-len(X):]  # X와 y의 길이 맞추기
        
        LOGGER.info(f"윈도우 샘플 생성 완료: X 형태 {X.shape}, y 형태 {y.shape}")
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터를 학습/검증/테스트 세트로 분할
        
        Args:
            X: 특성 데이터
            y: 타겟 데이터
            
        Returns:
            (X_train, X_valid, X_test, y_train, y_valid, y_test) 튜플
        """
        if len(X) == 0 or len(y) == 0:
            LOGGER.warning("빈 데이터가 입력되었습니다.")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # 데이터 분할 인덱스 계산
        train_idx = int(len(X) * TRAIN_RATIO)
        valid_idx = int(len(X) * (TRAIN_RATIO + VALID_RATIO))
        
        # 시간 순서대로 분할 (미래 데이터 누수 방지)
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
        X_test, y_test = X[valid_idx:], y[valid_idx:]
        
        LOGGER.info(f"데이터 분할 완료: 학습 {len(X_train)}개, 검증 {len(X_valid)}개, 테스트 {len(X_test)}개")
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def process_symbol_data(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        단일 심볼 데이터에 대한 전체 전처리 과정 수행
        
        Args:
            data: 원본 주식 데이터 데이터프레임
            symbol: 주식 심볼
            
        Returns:
            전처리된 데이터와 분할된 데이터셋을 포함하는 딕셔너리
        """
        LOGGER.info(f"{symbol} 데이터 전처리 시작")
        
        # 1. 데이터 전처리
        processed_data = self.preprocess_data(data)
        if processed_data.empty:
            LOGGER.error(f"{symbol} 데이터 전처리 실패")
            return {}
        
        # 2. 특성 추출
        featured_data = self.extract_features(processed_data)
        
        # 3. 특성 정규화
        normalized_data = self.normalize_features(featured_data, symbol, is_training=True)
        
        # 4. 윈도우 샘플 생성
        X, y = self.create_window_samples(normalized_data)
        if len(X) == 0:
            LOGGER.error(f"{symbol} 윈도우 샘플 생성 실패")
            return {}
        
        # 5. 데이터 분할
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.split_data(X, y)
        
        # 결과 반환
        result = {
            'processed_data': processed_data,
            'featured_data': featured_data,
            'normalized_data': normalized_data,
            'X_train': X_train,
            'X_valid': X_valid,
            'X_test': X_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test
        }
        
        LOGGER.info(f"{symbol} 데이터 전처리 완료")
        return result
    
    def process_all_symbols(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        모든 심볼 데이터에 대한 전처리 수행
        
        Args:
            data_dict: 심볼을 키로 하고 데이터프레임을 값으로 하는 딕셔너리
            
        Returns:
            심볼을 키로 하고 전처리 결과를 값으로 하는 딕셔너리
        """
        results = {}
        
        for symbol, data in data_dict.items():
            try:
                result = self.process_symbol_data(data, symbol)
                if result:
                    results[symbol] = result
            except Exception as e:
                LOGGER.error(f"{symbol} 처리 중 오류 발생: {str(e)}")
        
        LOGGER.info(f"총 {len(results)}/{len(data_dict)} 종목 데이터 전처리 완료")
        return results
    
    def save_processed_data(self, results: Dict[str, Dict[str, Any]], base_dir: Union[str, Path] = None) -> None:
        """
        전처리된 데이터를 파일로 저장
        
        Args:
            results: 전처리 결과 딕셔너리
            base_dir: 저장할 기본 디렉토리 (None인 경우 기본값 사용)
        """
        if base_dir is None:
            base_dir = DATA_DIR / "processed"
        
        create_directory(base_dir)
        
        for symbol, result in results.items():
            # 데이터프레임만 저장
            for key in ['processed_data', 'featured_data', 'normalized_data']:
                if key in result and isinstance(result[key], pd.DataFrame):
                    save_dir = base_dir / symbol
                    create_directory(save_dir)
                    file_path = save_dir / f"{key}.csv"
                    save_to_csv(result[key], file_path)
        
        LOGGER.info(f"전처리된 데이터 저장 완료: {base_dir}")


if __name__ == "__main__":
    # 테스트용 코드
    from src.data_collection.data_collector import DataCollector
    
    # 데이터 수집
    collector = DataCollector(symbols=["AAPL"])
    data = collector.load_all_data()
    
    # 데이터 전처리
    processor = DataProcessor(window_size=30)
    results = processor.process_all_symbols(data)
    processor.save_processed_data(results)
    
    # 첫 번째 종목의 결과 확인
    if results:
        symbol = list(results.keys())[0]
        print(f"\n{symbol} 전처리 결과:")
        print(f"원본 데이터 크기: {data[symbol].shape}")
        print(f"전처리 데이터 크기: {results[symbol]['processed_data'].shape}")
        print(f"특성 추출 데이터 크기: {results[symbol]['featured_data'].shape}")
        print(f"학습 데이터 크기: {results[symbol]['X_train'].shape}")
        print(f"검증 데이터 크기: {results[symbol]['X_valid'].shape}")
        print(f"테스트 데이터 크기: {results[symbol]['X_test'].shape}")