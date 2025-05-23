"""
    MYSQL 데이터베이스 기반 대시보드 데이터 관리 모듈
    database.py에서 제공하는 DatabaseManager 인스턴스를 사용. 
    이미 연결된 데이터베이스 연결을 이용하여 데이터 로직을 처리
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time

from src.utils.logger import Logger
from src.utils.database import DatabaseManager


class DBDataManager:
    """
    데이터베이스를 사용하는 대시보드 데이터 관리 클래스
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        data_dir: Optional[str] = None,
        logger: Optional[Logger] = None
    ):
        """
        DBDataManager 클래스 초기화
        
        Args:
            db_manager: 데이터베이스 관리자 객체
            data_dir: 데이터 디렉토리 경로 (필요시 사용, 옵션)
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        self.db_manager = db_manager
        self.data_dir = data_dir
        self.logger = logger
        
        if self.logger:
            self.logger.info("DBDataManager 초기화 완료")
    
    def get_trading_stats(self, refresh: bool = False) -> Dict[str, Any]:
        """
        트레이딩 통계 조회
        
        Args:
            refresh: 캐시 갱신 여부 (데이터베이스 방식에서는 무시됨)
            
        Returns:
            트레이딩 통계 정보
        """
        try:
            # 최근 트레이딩 통계 조회
            stats_query = """
            SELECT * FROM trading_stats 
            ORDER BY timestamp DESC 
            LIMIT 100
            """
            
            stats_rows = self.db_manager.execute_query(stats_query)
            
            if not stats_rows:
                if self.logger:
                    self.logger.warning("트레이딩 통계 데이터가 없습니다.")
                return {}
            
            # 최근 거래 내역 조회
            trades_query = """
            SELECT * FROM trades 
            ORDER BY timestamp DESC 
            LIMIT 20
            """
            
            trades = self.db_manager.execute_query(trades_query)
            
            # 현재 포지션 조회
            positions_query = """
            SELECT * FROM positions
            """
            
            positions = self.db_manager.execute_query(positions_query)
            
            # 통계 데이터 구성
            portfolio_values = [row['portfolio_value'] for row in reversed(stats_rows)]
            timestamps = [row['timestamp'].isoformat() for row in reversed(stats_rows)]
            
            # 일일 수익률 계산
            returns = []
            for i in range(1, len(portfolio_values)):
                daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(daily_return)
            returns.insert(0, 0)  # 첫 날의 수익률은 0
            
            # 낙폭 계산
            drawdowns = []
            peak = portfolio_values[0] if portfolio_values else 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                    drawdowns.append(0)
                else:
                    drawdown = (peak - value) / peak
                    drawdowns.append(drawdown)
            
            # 포지션 데이터 가공
            positions_dict = {}
            for position in positions:
                positions_dict[position['symbol']] = {
                    'quantity': float(position['quantity']),
                    'avg_entry_price': float(position['avg_entry_price']),
                    'current_price': float(position['current_price']) if position['current_price'] else 0,
                    'unrealized_pnl': float(position['unrealized_pnl']),
                    'timestamp': position['timestamp'].isoformat()
                }
            
            # 최신 트레이딩 통계 데이터
            latest_stats = stats_rows[0]
            
            # 결과 구성
            result = {
                'portfolio_values': portfolio_values,
                'timestamps': timestamps,
                'returns': returns,
                'drawdowns': drawdowns,
                'trades': [
                    {
                        'timestamp': trade['timestamp'].isoformat(),
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'quantity': float(trade['quantity']),
                        'price': float(trade['price']),
                        'fee': float(trade['fee']),
                        'pnl': float(trade['pnl']) if trade['pnl'] else 0
                    } for trade in trades
                ],
                'positions': positions_dict,
                'trading_stats': {
                    'portfolio_value': float(latest_stats['portfolio_value']),
                    'cash_balance': float(latest_stats['cash_balance']),
                    'equity_value': float(latest_stats['equity_value']),
                    'daily_pnl': float(latest_stats['daily_pnl']),
                    'total_pnl': float(latest_stats['total_pnl']),
                    'timestamp': latest_stats['timestamp'].isoformat()
                }
            }
            
            if self.logger:
                self.logger.info(f"트레이딩 통계 조회 완료: {len(stats_rows)}개 레코드")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"트레이딩 통계 조회 중 오류 발생: {e}")
            return {}
    
    def get_backtest_results(self, model_id: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
        """
        백테스트 결과 조회
        
        Args:
            model_id: 모델 ID (옵션, 특정 모델 결과만 조회)
            refresh: 캐시 갱신 여부 (데이터베이스 방식에서는 무시됨)
            
        Returns:
            백테스트 결과 정보
        """
        try:
            if model_id:
                # 특정 모델 결과만 조회
                query = """
                SELECT * FROM backtest_results
                WHERE model_id = %s
                ORDER BY backtest_date DESC
                """
                params = (model_id,)
            else:
                # 모든 모델 결과 조회
                query = """
                SELECT * FROM backtest_results
                ORDER BY model_id, backtest_date DESC
                """
                params = None
            
            results = self.db_manager.execute_query(query, params)
            
            if not results:
                if self.logger:
                    self.logger.warning(f"백테스트 결과가 없습니다: {model_id if model_id else '모든 모델'}")
                return {}
            
            # 모델별로 결과 정리
            backtest_results = {}
            
            for result in results:
                model_id_val = result['model_id']
                
                
                # 첫 번째 결과만 사용 (가장 최근 백테스트)
                if model_id_val not in backtest_results:
                    # JSON 파라미터 파싱
                    parameters = json.loads(result['parameters']) if result['parameters'] else {}
                    
                    backtest_results[model_id_val] = {
                        'model_id': model_id_val,
                        'backtest_date': result['backtest_date'].isoformat(),
                        'start_date': result['start_date'].isoformat(),
                        'end_date': result['end_date'].isoformat(),
                        'initial_balance': float(result['initial_balance']),
                        'final_balance': float(result['final_balance']),
                        'parameters': parameters,
                        'metrics': {
                            'total_return': float(result['total_return']),
                            'annualized_return': float(result['annualized_return']) if result['annualized_return'] else 0,
                            'sharpe_ratio': float(result['sharpe_ratio']) if result['sharpe_ratio'] else 0,
                            'max_drawdown': float(result['max_drawdown']) if result['max_drawdown'] else 0,
                            'win_rate': float(result['win_rate']) if result['win_rate'] else 0,
                            'profit_factor': float(result['profit_factor']) if result['profit_factor'] else 0,
                            'total_trades': result['total_trades'] or 0
                        }
                    }
            
            if self.logger:
                self.logger.info(f"백테스트 결과 조회 완료: {len(backtest_results)}개 모델")
            
            return backtest_results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"백테스트 결과 조회 중 오류 발생: {e}")
            return {}
    
    def get_model_info(self, refresh: bool = False) -> Dict[str, Any]:
        """
        학습된 모델 정보 조회
        
        Args:
            refresh: 캐시 갱신 여부 (데이터베이스 방식에서는 무시됨)
            
        Returns:
            모델 정보 목록
        """
        try:
            query = """
            SELECT * FROM models
            ORDER BY created_time DESC
            """
            
            models = self.db_manager.execute_query(query)
            
            if not models:
                if self.logger:
                    self.logger.warning("등록된 모델이 없습니다.")
                return {}
            
            # 모델 정보 정리
            model_info = {}
            
            for model in models:
                model_id = model['model_id']
                
                model_info[model_id] = {
                    'model_id': model_id,
                    'file_path': model['file_path'],
                    'created_time': model['created_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'modified_time': model['modified_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'file_size': model['file_size'],
                    'file_size_mb': round(model['file_size'] / (1024 * 1024), 2),
                    'description': model['description'],
                    'is_active': bool(model['is_active'])
                }
            
            if self.logger:
                self.logger.info(f"모델 정보 조회 완료: {len(model_info)}개")
            
            return model_info
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"모델 정보 조회 중 오류 발생: {e}")
            return {}
    
    def get_market_data(self, symbol: str, interval: str = '1d', limit: int = 100, refresh: bool = False) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            symbol: 심볼/티커
            interval: 시간 간격 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: 가져올 데이터 개수
            refresh: 캐시 갱신 여부 (데이터베이스 방식에서는 무시됨)
            
        Returns:
            시장 데이터 DataFrame
        """
        try:
            query = """
            SELECT * FROM market_data
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
            
            params = (symbol, limit)
            
            market_data = self.db_manager.execute_query(query, params)
            
            if not market_data:
                if self.logger:
                    self.logger.warning(f"{symbol} 시장 데이터가 없습니다.")
                return pd.DataFrame()
            
            # DataFrame으로 변환
            df = pd.DataFrame(market_data)
            
            # 날짜 열 이름 변경 (timestamp -> date)
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp']
            
            # 최신 순으로 정렬되어 있으므로 역순으로 변경
            df = df.iloc[::-1].reset_index(drop=True)
            
            if self.logger:
                self.logger.info(f"{symbol} 시장 데이터 조회 완료: {len(df)} 행")
            
            return df
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 시장 데이터 조회 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        모델 성능 지표 조회
        
        Args:
            model_id: 모델 ID (옵션, 특정 모델 결과만 조회)
            
        Returns:
            성능 지표 정보
        """
        # 백테스트 결과에서 성능 지표 추출
        backtest_results = self.get_backtest_results(model_id)
        
        metrics = {}
        for model_id, result in backtest_results.items():
            if 'metrics' in result:
                metrics[model_id] = result['metrics']
            else:
                # 기본 지표 구조 생성
                metrics[model_id] = {
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0),
                    'profit_factor': result.get('profit_factor', 0)
                }
        
        return metrics

    def sync_file_to_db(self, file_type: str, file_path: str) -> bool:
        """
        파일 데이터를 데이터베이스에 동기화
        
        Args:
            file_type: 파일 유형 (trading_stats, backtest_results, models, market_data)
            file_path: 파일 경로
            
        Returns:
            성공 여부
        """
        try:
            if not os.path.exists(file_path):
                if self.logger:
                    self.logger.warning(f"파일이 존재하지 않습니다: {file_path}")
                return False
            
            # 파일 로드
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                if self.logger:
                    self.logger.warning(f"지원하지 않는 파일 형식입니다: {file_path}")
                return False
            
            # 파일 유형에 따라 다른 처리
            if file_type == 'trading_stats':
                return self._sync_trading_stats(data, file_path)
            elif file_type == 'backtest_results':
                return self._sync_backtest_results(data, file_path)
            elif file_type == 'models':
                return self._sync_model_info(data, file_path)
            elif file_type == 'market_data':
                return self._sync_market_data(data, file_path)
            else:
                if self.logger:
                    self.logger.warning(f"지원하지 않는 파일 유형입니다: {file_type}")
                return False
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"파일 데이터베이스 동기화 중 오류 발생: {e}")
            return False
    
    def _sync_trading_stats(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        트레이딩 통계 동기화
        """
        try:
            # 파일명에서 타임스탬프 추출
            filename = os.path.basename(file_path)
            timestamp_str = filename.split('_')[-1].split('.')[0]
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            
            # 트레이딩 통계 삽입
            if 'portfolio_value' in data and 'cash_balance' in data:
                query = """
                INSERT INTO trading_stats 
                (timestamp, portfolio_value, cash_balance, equity_value, daily_pnl, total_pnl)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                portfolio_value = VALUES(portfolio_value),
                cash_balance = VALUES(cash_balance),
                equity_value = VALUES(equity_value),
                daily_pnl = VALUES(daily_pnl),
                total_pnl = VALUES(total_pnl)
                """
                
                params = (
                    timestamp,
                    data['portfolio_value'],
                    data['cash_balance'],
                    data.get('equity_value', data['portfolio_value'] - data['cash_balance']),
                    data.get('daily_pnl', 0),
                    data.get('total_pnl', 0)
                )
                
                self.db_manager.execute_query(query, params, fetch=False)
            
            # 포지션 정보 삽입
            if 'positions' in data:
                for symbol, position in data['positions'].items():
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
                        position['quantity'],
                        position['avg_entry_price'],
                        position.get('current_price', 0),
                        position.get('unrealized_pnl', 0),
                        timestamp
                    )
                    
                    self.db_manager.execute_query(query, params, fetch=False)
            
            # 거래 내역 삽입
            if 'trades' in data:
                for trade in data['trades']:
                    query = """
                    INSERT INTO trades
                    (timestamp, symbol, side, quantity, price, fee, pnl, model_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    trade_timestamp = datetime.fromisoformat(trade['timestamp']) if 'timestamp' in trade else timestamp
                    
                    params = (
                        trade_timestamp,
                        trade['symbol'],
                        trade['side'],
                        trade['quantity'],
                        trade['price'],
                        trade.get('fee', 0),
                        trade.get('pnl', None),
                        trade.get('model_id', None)
                    )
                    
                    self.db_manager.execute_query(query, params, fetch=False)
            
            if self.logger:
                self.logger.info(f"트레이딩 통계 동기화 완료: {file_path}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"트레이딩 통계 동기화 중 오류 발생: {e}")
            return False
    
    def _sync_backtest_results(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        백테스트 결과 동기화
        """
        try:
            # 파일명에서 모델 ID 추출
            filename = os.path.basename(file_path)
            model_id = filename.split('_')[1] if '_' in filename else filename.split('.')[0]
            
            # 백테스트 결과 삽입
            backtest_date = datetime.fromisoformat(data.get('backtest_date', datetime.now().isoformat()))
            start_date = datetime.fromisoformat(data.get('start_date', '')).date() if 'start_date' in data else None
            end_date = datetime.fromisoformat(data.get('end_date', '')).date() if 'end_date' in data else None
            
            # 지표 추출
            metrics = data.get('metrics', {})
            if not metrics and 'performance' in data:
                metrics = data['performance']
            
            query = """
            INSERT INTO backtest_results
            (model_id, backtest_date, start_date, end_date, initial_balance, final_balance,
             total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate,
             profit_factor, total_trades, parameters)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
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
                backtest_date,
                start_date,
                end_date,
                data.get('initial_balance', 0),
                data.get('final_balance', 0),
                metrics.get('total_return', 0),
                metrics.get('annualized_return', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('win_rate', 0),
                metrics.get('profit_factor', 0),
                metrics.get('total_trades', 0),
                json.dumps(data.get('parameters', {}))
            )
            
            self.db_manager.execute_query(query, params, fetch=False)
            
            if self.logger:
                self.logger.info(f"백테스트 결과 동기화 완료: {file_path}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"백테스트 결과 동기화 중 오류 발생: {e}")
            return False
    
    def _sync_model_info(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        모델 정보 동기화
        """
        try:
            model_id = data.get('model_id', os.path.basename(file_path).split('.')[0])
            
            # 파일 정보 추출
            file_stats = os.stat(file_path)
            created_time = datetime.fromtimestamp(file_stats.st_ctime)
            modified_time = datetime.fromtimestamp(file_stats.st_mtime)
            file_size = file_stats.st_size
            
            query = """
            INSERT INTO models
            (model_id, file_path, created_time, modified_time, file_size, description, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            file_path = VALUES(file_path),
            modified_time = VALUES(modified_time),
            file_size = VALUES(file_size),
            description = VALUES(description)
            """
            
            params = (
                model_id,
                file_path,
                created_time,
                modified_time,
                file_size,
                data.get('description', ''),
                data.get('is_active', False)
            )
            
            self.db_manager.execute_query(query, params, fetch=False)
            
            if self.logger:
                self.logger.info(f"모델 정보 동기화 완료: {file_path}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"모델 정보 동기화 중 오류 발생: {e}")
            return False
    
    def _sync_market_data(self, data: Union[Dict[str, Any], pd.DataFrame], file_path: str) -> bool:
        """
        시장 데이터 동기화
        """
        try:
            # 파일명에서 심볼 추출
            filename = os.path.basename(file_path)
            symbol = filename.split('_')[0]
            
            # DataFrame으로 변환
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # 데이터 삽입
            if len(df) > 0:
                # 일괄 삽입을 위한 파라미터 리스트 생성
                params_list = []
                
                for _, row in df.iterrows():
                    # 날짜 처리
                    if 'date' in row:
                        timestamp = pd.to_datetime(row['date'])
                    elif 'timestamp' in row:
                        timestamp = pd.to_datetime(row['timestamp'])
                    else:
                        continue
                    
                    params = (
                        symbol,
                        timestamp,
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row.get('volume', 0)
                    )
                    
                    params_list.append(params)
                
                if params_list:
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
                    
                    self.db_manager.execute_many(query, params_list)
            
            if self.logger:
                self.logger.info(f"시장 데이터 동기화 완료: {file_path}, {len(df)} 행")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"시장 데이터 동기화 중 오류 발생: {e}")
            return False 