# test_db.py 파일 생성
from database_manager import DatabaseManager


# DB 연결 테스트
db_manager = DatabaseManager()
if db_manager.connect():
    print("✅ DB 연결 성공!")
    
    # 테스트 데이터 저장
    db_manager.save_trading_stats(200000, 60000, 60000, 0, 0)
    print("✅ 테스트 데이터 저장 완료!")
    
    # 데이터 조회
    stats = db_manager.get_latest_trading_stats(limit=1)
    print(f"📊 저장된 데이터: {stats}")
    
    db_manager.disconnect()
else:
    print("❌ DB 연결 실패")