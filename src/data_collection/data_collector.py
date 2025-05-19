"""
TimescaleDB에서 주식 데이터를 수집하는 모듈
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from src.config.config import (
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    TARGET_SYMBOLS,
    DATA_DIR,
    LOGGER
)
from src.utils.utils import create_directory, save_to_csv, load_from_csv

class DataCollector:
    """
    TimescaleDB에서 주식 데이터를 수집하는 클래스
    """
    
    def __init__(
        self,
        user: str = DB_USER,
        password: str = DB_PASSWORD,
        host: str = DB_HOST,
        port: Union[str, int] = DB_PORT,
        db_name: str = DB_NAME,
        symbols: List[str] = None
    ):
        """
        DataCollector 클래스 초기화
        
        Args:
            user: 데이터베이스 사용자명
            password: 데이터베이스 비밀번호
            host: 데이터베이스 호스트
            port: 데이터베이스 포트
            db_name: 데이터베이스 이름
            symbols: 수집할 주식 심볼 리스트 (None인 경우 설정 파일의 기본값 사용)
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.symbols = symbols if symbols is not None else TARGET_SYMBOLS
        self.engine = None
        self.data_dir = DATA_DIR
        create_directory(self.data_dir)
        self._create_engine()
        LOGGER.info(f"DataCollector 초기화 완료: {len(self.symbols)}개 종목 대상")
    
    def _create_engine(self) -> None:
        """
        SQLAlchemy 엔진 생성
        """
        try:
            self.engine = create_engine(
                f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
            )
            LOGGER.info("DB 엔진 생성 성공")
        except SQLAlchemyError as e:
            LOGGER.error(f"DB 엔진 생성 실패: {str(e)}")
            raise
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        데이터베이스에서 특정 심볼의 주식 데이터 로드
        
        Args:
            symbol: 주식 심볼
            
        Returns:
            로드된 주식 데이터 데이터프레임
        """
        query = f"""
            SELECT *
            FROM ticker_{symbol}
            ORDER BY timestamp DESC;
        """
        try:
            LOGGER.info(f"{symbol} 데이터 로드 시작...")
            df = pd.read_sql(query, self.engine)
            
            if 'timestamp' not in df.columns:
                LOGGER.warning(f"{symbol} 테이블에 'timestamp' 컬럼이 없습니다.")
                return pd.DataFrame()
            
            # 데이터 전처리
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            LOGGER.info(f"{symbol} 데이터 로드 완료: {len(df)} 행")
            return df
            
        except SQLAlchemyError as e:
            LOGGER.error(f"{symbol} 데이터 로드 중 DB 오류 발생: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            LOGGER.error(f"{symbol} 데이터 로드 실패: {str(e)}")
            return pd.DataFrame()
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        모든 심볼에 대한 데이터 로드
        
        Returns:
            심볼을 키로 하고 데이터프레임을 값으로 하는 딕셔너리
        """
        all_data = {}
        
        for symbol in self.symbols:
            try:
                data = self.load_data(symbol)
                if not data.empty:
                    all_data[symbol] = data
            except Exception as e:
                LOGGER.error(f"{symbol} 처리 중 오류 발생: {str(e)}")
        
        LOGGER.info(f"총 {len(all_data)}/{len(self.symbols)} 종목 데이터 로드 완료")
        return all_data
    
    def save_data(self, data: Dict[str, pd.DataFrame], subdir: str = None) -> None:
        """
        수집된 데이터를 CSV 파일로 저장
        
        Args:
            data: 심볼을 키로 하고 데이터프레임을 값으로 하는 딕셔너리
            subdir: 저장할 하위 디렉토리 (None인 경우 날짜 기반으로 생성)
        """
        if subdir is None:
            subdir = datetime.now().strftime("%Y%m%d")
        
        save_dir = self.data_dir / subdir
        create_directory(save_dir)
        
        for symbol, df in data.items():
            file_path = save_dir / f"{symbol}.csv"
            save_to_csv(df, file_path)
        
        LOGGER.info(f"모든 데이터 저장 완료: {save_dir}")
    
    def check_tables(self) -> List[str]:
        """
        데이터베이스에 존재하는 테이블 확인
        
        Returns:
            테이블 이름 리스트
        """
        try:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name LIKE 'ticker_%';
            """
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                tables = [row[0] for row in result]
            
            LOGGER.info(f"총 {len(tables)}개의 테이블 확인됨")
            return tables
        
        except SQLAlchemyError as e:
            LOGGER.error(f"테이블 확인 중 오류 발생: {str(e)}")
            return []
    
    def get_data_range(self, symbol: str) -> Dict[str, datetime]:
        """
        특정 심볼의 데이터 기간 조회
        
        Args:
            symbol: 주식 심볼
            
        Returns:
            시작일과 종료일이 포함된 딕셔너리
        """
        query = f"""
            SELECT 
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM ticker_{symbol};
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
            if row and row[0] and row[1]:
                range_info = {
                    'start_date': row[0],
                    'end_date': row[1],
                    'days': (row[1] - row[0]).days
                }
                LOGGER.info(f"{symbol} 데이터 기간: {range_info['start_date']} ~ {range_info['end_date']} ({range_info['days']}일)")
                return range_info
            else:
                LOGGER.warning(f"{symbol} 테이블에 데이터가 없거나 날짜 정보가 없습니다.")
                return {'start_date': None, 'end_date': None, 'days': 0}
                
        except SQLAlchemyError as e:
            LOGGER.error(f"{symbol} 데이터 기간 조회 중 오류 발생: {str(e)}")
            return {'start_date': None, 'end_date': None, 'days': 0}
    
    def load_and_save(self) -> Dict[str, pd.DataFrame]:
        """
        데이터 로드 및 저장을 한 번에 수행
        
        Returns:
            로드된 데이터 딕셔너리
        """
        data = self.load_all_data()
        if data:
            self.save_data(data)
        return data


if __name__ == "__main__":
    # 모듈 테스트 코드
    collector = DataCollector(
        user='postgres',
        password='mysecretpassword',
        host='192.168.40.193',
        port=5432,
        db_name='mydb'
    )
    
    # 데이터베이스 테이블 확인
    tables = collector.check_tables()
    print(f"사용 가능한 테이블: {tables}")
    
    # 데이터 로드 및 저장
    data = collector.load_and_save()
    print(f"로드된 데이터 종목: {list(data.keys())}")
    
    # 첫 번째 종목의 데이터 샘플 출력
    if data:
        symbol = list(data.keys())[0]
        print(f"\n{symbol} 데이터 샘플:")
        print(data[symbol].head())