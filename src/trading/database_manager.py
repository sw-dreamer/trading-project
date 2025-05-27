#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime
from decimal import Decimal
import logging
from typing import Dict, List, Optional, Any
import time


class DatabaseManager:
    """MySQL 데이터베이스 연결 및 트레이딩 데이터 저장 관리 클래스"""
    
    def __init__(self, host='192.168.40.199', database='trading', 
                 user='root', password='mysecretpassword', logger=None):
        """
        데이터베이스 매니저 초기화
        
        Args:
            host: MySQL 서버 호스트
            database: 데이터베이스 명
            user: 사용자명
            password: 비밀번호
            logger: 로거 인스턴스
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.logger = logger or logging.getLogger(__name__)
        self.connection_config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'autocommit': True,
            'charset': 'utf8mb4',
            'connection_timeout': 30
        }
        
    def connect(self) -> bool:
        """데이터베이스 연결"""
        try:
            if self.connection and self.connection.is_connected():
                return True
                
            self.connection = mysql.connector.connect(**self.connection_config)
            
            if self.connection.is_connected():
                self.logger.info(f"✅ MySQL 데이터베이스 연결 성공: {self.host}/{self.database}")
                return True
            
        except Error as e:
            self.logger.error(f"❌ MySQL 연결 실패: {e}")
            return False
        
        return False
    
    def disconnect(self):
        """데이터베이스 연결 종료"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("🔌 MySQL 연결 종료")
    
    def is_connected(self) -> bool:
        """연결 상태 확인 및 재연결"""
        try:
            if self.connection and self.connection.is_connected():
                # 연결 상태 테스트
                try:
                    self.connection.ping(attempts=3, delay=1)
                    return True
                except:
                    # ping 실패 시 연결이 끊어진 것으로 간주
                    pass
        except:
            pass
        
        # 연결이 끊어진 경우 재연결 시도
        self.logger.warning("⚠️ MySQL 연결이 끊어졌습니다. 재연결을 시도합니다...")
        return self.connect()
    
    def execute_query(self, query: str, params: tuple = None, retry_count: int = 3) -> bool:
        """쿼리 실행 (INSERT, UPDATE, DELETE) - 재시도 로직 포함"""
        for attempt in range(retry_count):
            try:
                # 연결 상태 확인 및 재연결
                if not self.is_connected():
                    self.logger.error("❌ 데이터베이스 연결을 설정할 수 없습니다.")
                    if attempt < retry_count - 1:
                        time.sleep(1)
                        continue
                    return False
                
                cursor = self.connection.cursor()
                cursor.execute(query, params)
                cursor.close()
                return True
                
            except Error as e:
                self.logger.error(f"❌ 쿼리 실행 실패 (시도 {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    self.logger.info("🔄 1초 후 재시도...")
                    time.sleep(1)
                    # 연결 재설정
                    try:
                        if self.connection:
                            self.connection.close()
                    except:
                        pass
                    self.connection = None
                else:
                    self.logger.error(f"Query: {query}")
                    if params:
                        self.logger.error(f"Params: {params}")
                    
        return False
    
    def fetch_query(self, query: str, params: tuple = None, retry_count: int = 3) -> List[Dict]:
        """쿼리 실행 (SELECT) 및 결과 반환 - 재시도 로직 포함"""
        for attempt in range(retry_count):
            try:
                # 연결 상태 확인 및 재연결
                if not self.is_connected():
                    self.logger.error("❌ 데이터베이스 연결을 설정할 수 없습니다.")
                    if attempt < retry_count - 1:
                        time.sleep(1)
                        continue
                    return []
                
                cursor = self.connection.cursor(dictionary=True)
                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()
                return results
                
            except Error as e:
                self.logger.error(f"❌ 조회 쿼리 실행 실패 (시도 {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    self.logger.info("🔄 1초 후 재시도...")
                    time.sleep(1)
                    # 연결 재설정
                    try:
                        if self.connection:
                            self.connection.close()
                    except:
                        pass
                    self.connection = None
                else:
                    self.logger.error(f"Query: {query}")
                    if params:
                        self.logger.error(f"Params: {params}")
                    
        return []
    
    def save_trading_stats(self, portfolio_value: float, cash_balance: float, 
                          equity_value: float, daily_pnl: float = 0, 
                          total_pnl: float = 0) -> bool:
        """trading_stats 테이블에 통계 저장"""
        query = """
        INSERT INTO trading_stats 
        (timestamp, portfolio_value, cash_balance, equity_value, daily_pnl, total_pnl)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        params = (
            datetime.now(),
            Decimal(str(portfolio_value)),
            Decimal(str(cash_balance)),
            Decimal(str(equity_value)),
            Decimal(str(daily_pnl)),
            Decimal(str(total_pnl))
        )
        
        if self.execute_query(query, params):
            self.logger.debug("📊 trading_stats 저장 완료")
            return True
        return False
    
    def save_trade(self, symbol: str, side: str, quantity: float, 
                   price: float, fee: float = 0, pnl: Optional[float] = None,
                   model_id: Optional[str] = None) -> bool:
        """trades 테이블에 거래 저장"""
        query = """
        INSERT INTO trades 
        (timestamp, symbol, side, quantity, price, fee, pnl, model_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            datetime.now(),
            symbol,
            side,
            Decimal(str(quantity)),
            Decimal(str(price)),
            Decimal(str(fee)),
            Decimal(str(pnl)) if pnl is not None else None,
            model_id
        )
        
        if self.execute_query(query, params):
            self.logger.info(f"💰 거래 저장 완료: {symbol} {side} {quantity}@{price}")
            return True
        return False
    
    def save_position(self, symbol: str, quantity: float, avg_entry_price: float,
                     current_price: Optional[float] = None, 
                     unrealized_pnl: float = 0) -> bool:
        """positions 테이블에 포지션 저장/업데이트"""
        # UPSERT 쿼리 (존재하면 업데이트, 없으면 삽입)
        query = """
        INSERT INTO positions 
        (symbol, quantity, avg_entry_price, current_price, unrealized_pnl, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        quantity = VALUES(quantity),
        avg_entry_price = VALUES(avg_entry_price),
        current_price = VALUES(current_price),
        unrealized_pnl = VALUES(unrealized_pnl),
        timestamp = VALUES(timestamp)
        """
        
        params = (
            symbol,
            Decimal(str(quantity)),
            Decimal(str(avg_entry_price)),
            Decimal(str(current_price)) if current_price is not None else None,
            Decimal(str(unrealized_pnl)),
            datetime.now()
        )
        
        if self.execute_query(query, params):
            self.logger.debug(f"🏢 포지션 저장 완료: {symbol} {quantity}주")
            return True
        return False
    
    def save_market_data(self, symbol: str, timestamp: datetime, 
                        open_price: float, high: float, low: float, 
                        close: float, volume: float = 0) -> bool:
        """market_data 테이블에 시장 데이터 저장"""
        query = """
        INSERT INTO market_data 
        (symbol, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        open = VALUES(open),
        high = VALUES(high),
        low = VALUES(low),
        close = VALUES(close),
        volume = VALUES(volume)
        """
        
        params = (
            symbol,
            timestamp,
            Decimal(str(open_price)),
            Decimal(str(high)),
            Decimal(str(low)),
            Decimal(str(close)),
            Decimal(str(volume))
        )
        
        if self.execute_query(query, params):
            self.logger.debug(f"📈 시장 데이터 저장: {symbol} {timestamp}")
            return True
        return False
    
    def save_model_info(self, model_id: str, file_path: str, 
                       description: str = None, is_active: bool = False) -> bool:
        """models 테이블에 모델 정보 저장"""
        import os
        
        # 파일 정보 가져오기
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        created_time = datetime.fromtimestamp(os.path.getctime(file_path)) if os.path.exists(file_path) else datetime.now()
        modified_time = datetime.fromtimestamp(os.path.getmtime(file_path)) if os.path.exists(file_path) else datetime.now()
        
        query = """
        INSERT INTO models 
        (model_id, file_path, created_time, modified_time, file_size, description, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        file_path = VALUES(file_path),
        modified_time = VALUES(modified_time),
        file_size = VALUES(file_size),
        description = VALUES(description),
        is_active = VALUES(is_active)
        """
        
        params = (
            model_id,
            file_path,
            created_time,
            modified_time,
            file_size,
            description,
            is_active
        )
        
        if self.execute_query(query, params):
            self.logger.info(f"🤖 모델 정보 저장: {model_id}")
            return True
        return False
    
    def save_backtest_result(self, model_id: str, start_date: str, end_date: str,
                           initial_balance: float, final_balance: float,
                           total_return: float, annualized_return: Optional[float] = None,
                           sharpe_ratio: Optional[float] = None, 
                           max_drawdown: Optional[float] = None,
                           win_rate: Optional[float] = None,
                           profit_factor: Optional[float] = None,
                           total_trades: int = 0, parameters: Dict = None) -> bool:
        """backtest_results 테이블에 백테스트 결과 저장"""
        query = """
        INSERT INTO backtest_results 
        (model_id, backtest_date, start_date, end_date, initial_balance, final_balance,
         total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate,
         profit_factor, total_trades, parameters)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        backtest_date = VALUES(backtest_date),
        start_date = VALUES(start_date),
        end_date = VALUES(end_date),
        initial_balance = VALUES(initial_balance),
        final_balance = VALUES(final_balance),
        total_return = VALUES(total_return),
        annualized_return = VALUES(annualized_return),
        sharpe_ratio = VALUES(sharpe_ratio),
        max_drawdown = VALUES(max_drawdown),
        win_rate = VALUES(win_rate),
        profit_factor = VALUES(profit_factor),
        total_trades = VALUES(total_trades),
        parameters = VALUES(parameters)
        """
        
        params = (
            model_id,
            datetime.now(),
            start_date,
            end_date,
            Decimal(str(initial_balance)),
            Decimal(str(final_balance)),
            Decimal(str(total_return)),
            Decimal(str(annualized_return)) if annualized_return is not None else None,
            Decimal(str(sharpe_ratio)) if sharpe_ratio is not None else None,
            Decimal(str(max_drawdown)) if max_drawdown is not None else None,
            Decimal(str(win_rate)) if win_rate is not None else None,
            Decimal(str(profit_factor)) if profit_factor is not None else None,
            total_trades,
            json.dumps(parameters) if parameters else None
        )
        
        if self.execute_query(query, params):
            self.logger.info(f"📋 백테스트 결과 저장: {model_id}")
            return True
        return False
    
    def get_latest_trading_stats(self, limit: int = 10) -> List[Dict]:
        """최근 거래 통계 조회"""
        query = """
        SELECT * FROM trading_stats 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        return self.fetch_query(query, (limit,))
    
    def get_recent_trades(self, symbol: str = None, limit: int = 20) -> List[Dict]:
        """최근 거래 내역 조회"""
        if symbol:
            query = """
            SELECT * FROM trades 
            WHERE symbol = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            params = (symbol, limit)
        else:
            query = """
            SELECT * FROM trades 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            params = (limit,)
        
        return self.fetch_query(query, params)
    
    def get_current_positions(self) -> List[Dict]:
        """현재 포지션 조회 (수량이 0이 아닌 것만)"""
        query = """
        SELECT * FROM positions 
        WHERE ABS(quantity) > 0.000001
        ORDER BY timestamp DESC
        """
        return self.fetch_query(query)
    
    def get_active_models(self) -> List[Dict]:
        """활성화된 모델 목록 조회"""
        query = """
        SELECT * FROM models 
        WHERE is_active = TRUE 
        ORDER BY modified_time DESC
        """
        return self.fetch_query(query)
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """오래된 데이터 정리 (선택적)"""
        try:
            # market_data에서 30일 이전 데이터 삭제
            query = """
            DELETE FROM market_data 
            WHERE timestamp < DATE_SUB(NOW(), INTERVAL %s DAY)
            """
            self.execute_query(query, (days_to_keep,))
            
            # trading_stats에서 오래된 데이터 정리 (더 보수적으로)
            query = """
            DELETE FROM trading_stats 
            WHERE timestamp < DATE_SUB(NOW(), INTERVAL %s DAY)
            """
            self.execute_query(query, (days_to_keep * 3,))  # 90일 보관
            
            self.logger.info(f"🧹 {days_to_keep}일 이전 데이터 정리 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 정리 실패: {e}")
            return False