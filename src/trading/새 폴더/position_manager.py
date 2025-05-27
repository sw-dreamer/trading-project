import pandas as pd
from typing import Dict, List, Any, Optional
import time

from src.utils.logger import Logger
from src.trading.api_connector import APIConnector


class PositionManager:
    """
    포지션 관리를 담당하는 매니저 클래스
    """
    
    def __init__(self, api_connector: APIConnector, logger: Optional[Logger] = None):
        """
        PositionManager 클래스 초기화
        
        Args:
            api_connector: API 커넥터 인스턴스
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        self.api = api_connector
        self.logger = logger
        self.positions = {}  # 심볼별 포지션 정보 캐시
        
        # 초기 포지션 정보 로드
        self.update_all_positions()
        
    def update_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 포지션 정보 업데이트
        
        Returns:
            업데이트된 포지션 정보 딕셔너리
        """
        try:
            # API에서 모든 포지션 정보 조회
            positions_list = self.api.get_all_positions()
            
            # 포지션 정보 캐시 업데이트
            self.positions = {}
            for position in positions_list:
                symbol = position.get("symbol", "")
                if symbol:
                    self.positions[symbol] = position
            
            if self.logger:
                self.logger.info(f"전체 포지션 정보 업데이트 완료: {len(self.positions)}개")
                
            return self.positions
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"전체 포지션 정보 업데이트 중 오류 발생: {e}")
            return self.positions
    
    def update_position(self, symbol: str) -> Dict[str, Any]:
        """
        특정 심볼의 포지션 정보 업데이트
        
        Args:
            symbol: 심볼/티커
            
        Returns:
            업데이트된 심볼의 포지션 정보
        """
        try:
            # API에서 심볼의 포지션 정보 조회
            position = self.api.get_position(symbol)
            
            # 포지션 정보 캐시 업데이트
            if position:
                self.positions[symbol] = position
            elif symbol in self.positions:
                # 포지션이 없는 경우 캐시에서 제거
                del self.positions[symbol]
            
            if self.logger:
                self.logger.info(f"{symbol} 포지션 정보 업데이트 완료")
                
            return position
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 포지션 정보 업데이트 중 오류 발생: {e}")
            
            # 캐시에서 조회
            return self.positions.get(symbol, {
                "symbol": symbol,
                "quantity": 0,
                "entry_price": 0,
                "unrealized_pnl": 0
            })
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        특정 심볼의 포지션 정보 조회
        
        Args:
            symbol: 심볼/티커
            
        Returns:
            심볼의 포지션 정보
        """
        # 캐시에서 조회
        if symbol in self.positions:
            return self.positions[symbol]
        
        # 없으면 API에서 조회 후 캐시 업데이트
        return self.update_position(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 포지션 정보 조회
        
        Returns:
            모든 포지션 정보 딕셔너리
        """
        return self.positions
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """
        포지션 정보 요약
        
        Returns:
            포지션 정보 요약 딕셔너리
        """
        total_positions = len(self.positions)
        total_long_positions = sum(1 for p in self.positions.values() if p.get("quantity", 0) > 0)
        total_short_positions = sum(1 for p in self.positions.values() if p.get("quantity", 0) < 0)
        total_value = sum(abs(p.get("market_value", 0)) for p in self.positions.values())
        total_unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in self.positions.values())
        
        return {
            "total_positions": total_positions,
            "long_positions": total_long_positions,
            "short_positions": total_short_positions,
            "total_value": total_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "positions": [{
                "symbol": symbol,
                "quantity": p.get("quantity", 0),
                "entry_price": p.get("entry_price", 0),
                "market_value": p.get("market_value", 0),
                "unrealized_pnl": p.get("unrealized_pnl", 0)
            } for symbol, p in self.positions.items()]
        }
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        특정 심볼의 포지션 청산
        
        Args:
            symbol: 심볼/티커
            
        Returns:
            청산 결과 정보
        """
        try:
            # 현재 포지션 확인
            position = self.get_position(symbol)
            quantity = position.get("quantity", 0)
            
            if quantity == 0:
                if self.logger:
                    self.logger.info(f"{symbol} 포지션이 없어 청산할 수 없습니다.")
                return {"success": False, "message": "포지션이 없습니다."}
            
            # 반대 방향 주문으로 청산
            side = "sell" if quantity > 0 else "buy"
            abs_quantity = abs(quantity)
            
            # 포지션 청산 주문 실행
            order_result = self.api.place_market_order(
                symbol=symbol,
                side=side,
                quantity=abs_quantity
            )
            
            if order_result.get("success", False):
                # 포지션 정보 업데이트
                time.sleep(1)  # 주문 처리 대기
                self.update_position(symbol)
                
                if self.logger:
                    self.logger.info(f"{symbol} 포지션 청산 성공")
                    
                return {
                    "success": True,
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs_quantity,
                    "order_id": order_result.get("order_id", "")
                }
            else:
                if self.logger:
                    self.logger.error(f"{symbol} 포지션 청산 실패: {order_result.get('error', 'Unknown error')}")
                    
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": order_result.get("error", "Unknown error")
                }
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 포지션 청산 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        모든 포지션 청산
        
        Returns:
            청산 결과 정보
        """
        results = {
            "success": True,
            "closed_positions": [],
            "failed_positions": []
        }
        
        # 모든 포지션 정보 업데이트
        self.update_all_positions()
        
        # 각 포지션 청산
        for symbol in list(self.positions.keys()):
            result = self.close_position(symbol)
            
            if result.get("success", False):
                results["closed_positions"].append({
                    "symbol": symbol,
                    "quantity": result.get("quantity", 0),
                    "side": result.get("side", ""),
                    "order_id": result.get("order_id", "")
                })
            else:
                results["failed_positions"].append({
                    "symbol": symbol,
                    "error": result.get("error", "Unknown error")
                })
                results["success"] = False
        
        if self.logger:
            self.logger.info(f"전체 포지션 청산 결과: 성공 {len(results['closed_positions'])}개, 실패 {len(results['failed_positions'])}개")
            
        return results
    
    def calculate_position_exposure(self) -> Dict[str, float]:
        """
        각 포지션의 익스포저(노출도) 계산
        
        Returns:
            심볼별 익스포저 딕셔너리 (계정 자본금 대비 %)
        """
        try:
            # 계정 정보 조회
            account_info = self.api.get_account_info()
            total_equity = float(account_info.get("equity", 1.0))
            
            if total_equity <= 0:
                if self.logger:
                    self.logger.error("계정 자본금이 0 이하입니다.")
                return {}
            
            # 각 포지션의 익스포저 계산
            exposures = {}
            for symbol, position in self.positions.items():
                market_value = abs(float(position.get("market_value", 0)))
                exposure_pct = (market_value / total_equity) * 100
                exposures[symbol] = exposure_pct
            
            return exposures
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"포지션 익스포저 계산 중 오류 발생: {e}")
            return {} 