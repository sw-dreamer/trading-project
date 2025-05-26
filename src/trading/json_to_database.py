#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime
from src.trading.database_manager import DatabaseManager



def parse_json_to_database(json_file_path: str):
    """JSON 파일을 읽어서 데이터베이스에 저장"""
    
    # 데이터베이스 연결
    db_manager = DatabaseManager()
    if not db_manager.connect():
        print("❌ 데이터베이스 연결 실패")
        return False
    
    
    try:
        # JSON 파일 읽기
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📁 JSON 파일 로드 완료: {json_file_path}")
        
        # 1. 거래 내역 저장 (trades 테이블)
        print("\n💰 거래 내역 저장 중...")
        trades_saved = 0
        for trade in data['trading_stats']['trades']:
            success = db_manager.save_trade(
                symbol=trade['symbol'],
                side=trade['side'],
                quantity=trade['quantity'],
                price=trade['price'],
                fee=0,  # JSON에 fee 정보가 없으므로 0으로 설정
                pnl=None,  # 개별 거래 손익은 계산 필요
                model_id="historical_data"  # JSON 파일에서 가져온 데이터임을 표시
            )
            if success:
                trades_saved += 1
        
        print(f"✅ 거래 내역 저장 완료: {trades_saved}개")
        
        # 2. 포지션 정보 저장 (positions 테이블)
        print("\n🏢 포지션 정보 저장 중...")
        positions_saved = 0
        for symbol, position in data['positions'].items():
            success = db_manager.save_position(
                symbol=symbol,
                quantity=float(position['qty']),
                avg_entry_price=float(position['avg_entry_price']),
                current_price=float(position['current_price']),
                unrealized_pnl=float(position['unrealized_pl'])
            )
            if success:
                positions_saved += 1
        
        print(f"✅ 포지션 정보 저장 완료: {positions_saved}개")
        
        # 3. 전체 통계 저장 (trading_stats 테이블)
        print("\n📊 전체 통계 저장 중...")
        account_info = data['account_info']
        trading_stats = data['trading_stats']
        
        # timestamp가 문자열이므로 datetime으로 변환
        timestamp_str = data['timestamp']
        
        success = db_manager.save_trading_stats(
            portfolio_value=float(account_info['portfolio_value']),
            cash_balance=float(account_info['cash']),
            equity_value=float(account_info['portfolio_value']) - float(account_info['cash']),
            daily_pnl=0,  # 일일 손익은 별도 계산 필요
            total_pnl=float(trading_stats['pnl'])
        )
        
        if success:
            print("✅ 전체 통계 저장 완료")
        
        # 4. 모델 정보 저장 (models 테이블)
        print("\n🤖 모델 정보 저장 중...")
        model_id = f"historical_json_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        success = db_manager.save_model_info(
            model_id=model_id,
            file_path=json_file_path,
            description=f"JSON 파일에서 가져온 히스토리컬 데이터 ({timestamp_str})",
            is_active=False  # 과거 데이터이므로 비활성
        )
        
        if success:
            print("✅ 모델 정보 저장 완료")
        
        # 5. 요약 정보 출력
        print("\n" + "="*60)
        print("📋 저장 완료 요약")
        print("="*60)
        print(f"📁 JSON 파일: {os.path.basename(json_file_path)}")
        print(f"⏰ 데이터 시점: {timestamp_str}")
        print(f"💰 거래 내역: {trades_saved}개")
        print(f"🏢 포지션: {positions_saved}개")
        print(f"📊 포트폴리오 가치: ${account_info['portfolio_value']:,.2f}")
        print(f"💵 현금 잔액: ${account_info['cash']:,.2f}")
        print(f"📈 총 손익: ${trading_stats['pnl']:+,.2f}")
        print(f"🎯 성공 거래: {trading_stats['successful_trades']}회")
        print(f"❌ 실패 거래: {trading_stats['failed_trades']}회")
        print("="*60)
        
        db_manager.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        db_manager.disconnect()
        return False

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JSON 파일을 데이터베이스로 변환')
    parser.add_argument('json_file', type=str, help='변환할 JSON 파일 경로')
    
    args = parser.parse_args()
    
    # JSON 파일 존재 확인
    if not os.path.exists(args.json_file):
        print(f"❌ 파일을 찾을 수 없습니다: {args.json_file}")
        return
    
    print("🔄 JSON 데이터를 데이터베이스로 변환 시작")
    print("=" * 60)
    
    # 변환 실행
    success = parse_json_to_database(args.json_file)
    
    if success:
        print("\n🎉 모든 데이터가 성공적으로 데이터베이스에 저장되었습니다!")
    else:
        print("\n❌ 데이터 저장 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()