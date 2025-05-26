#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 실시간 트레이딩 테스트 스크립트
문제를 단계별로 해결하기 위한 최소 버전
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """모듈 import 테스트"""
    print("📦 필요한 모듈들 import 테스트...")
    
    try:
        from src.config.config import Config
        print("✅ Config 모듈 import 성공")
        
        config = Config()
        print("✅ Config 객체 생성 성공")
        
        # 기본 속성들 확인
        print(f"📊 설정 확인:")
        
        # 안전한 방식으로 속성 접근
        trading_symbols = getattr(config, 'TRADING_SYMBOLS', ['AAPL'])
        print(f"   활성화된 심볼: {trading_symbols}")
        
        api_key = getattr(config, 'API_KEY', 'None')
        base_url = getattr(config, 'BASE_URL', 'None')
        print(f"   API URL: {base_url}")
        print(f"   API 키 설정: {'예' if api_key != 'None' else '아니오'}")
        
        # DEFAULT_SYMBOL_CONFIG에서 값 가져오기
        default_config = getattr(config, 'DEFAULT_SYMBOL_CONFIG', {})
        max_position_size = default_config.get('max_position_size', 0.01)
        max_trade_amount = default_config.get('max_trade_amount', 100.0)
        trading_interval = default_config.get('trading_interval', 600)
        
        print(f"   최대 포지션 크기: {max_position_size*100}%")
        print(f"   최대 거래 금액: ${max_trade_amount}")
        print(f"   거래 간격: {trading_interval/60}분")
        
        return config
        
    except Exception as e:
        print(f"❌ Config 모듈 import 실패: {e}")
        return None

def test_api_connector(config):
    """API 커넥터 테스트"""
    print("\n🔌 API 커넥터 테스트...")
    
    try:
        from src.trading.api_connector import APIConnector
        print("✅ APIConnector 모듈 import 성공")
        
        # API 커넥터 생성
        client = APIConnector()
        print("✅ APIConnector 객체 생성 성공")
        
        # 연결 테스트
        print("🔗 알파카 API 연결 시도...")
        if client.connect():
            print("✅ API 연결 성공!")
            
            # 계정 정보 조회
            account = client.get_account_info()
            if account and 'cash' in account:
                print(f"💰 계정 현금: ${account.get('cash', 0):,.2f}")
                print(f"📊 포트폴리오: ${account.get('portfolio_value', 0):,.2f}")
                print(f"🏦 계정 상태: {account.get('status', 'Unknown')}")
                return client
            else:
                print(f"⚠️ 계정 정보 조회 실패: {account}")
                return None
        else:
            print("❌ API 연결 실패")
            print("💡 config.py의 API 키를 확인하세요")
            return None
            
    except Exception as e:
        print(f"❌ API 커넥터 테스트 실패: {e}")
        return None

def test_data_collection(client):
    """데이터 수집 테스트"""
    print("\n📈 데이터 수집 테스트...")
    
    try:
        print("📊 AAPL 데이터 수집 시도...")
        data = client.get_market_data("AAPL", limit=3)
        
        if not data.empty:
            print("✅ 데이터 수집 성공!")
            print(f"📊 데이터 형태: {data.shape}")
            print(f"💰 최신 AAPL 가격: ${data['close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ 데이터가 비어있습니다")
            return False
            
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        print("💡 pip install tvdatafeed 를 실행해보세요")
        return False

def test_model_directory(config):
    """모델 디렉토리 확인"""
    print("\n🤖 모델 디렉토리 확인...")
    
    try:
        models_dir = getattr(config, 'MODELS_DIR', './models')
        print(f"📁 모델 디렉토리: {models_dir}")
        
        if hasattr(models_dir, 'exists') and models_dir.exists():
            model_files = list(models_dir.iterdir())
            model_dirs = [f for f in model_files if f.is_dir()]
            
            if model_dirs:
                print(f"✅ 모델 발견: {len(model_dirs)}개")
                for model in model_dirs[:3]:
                    print(f"   - {model.name}")
                return True
            else:
                print("⚠️ 학습된 모델이 없습니다")
                print("💡 python run_training.py --symbols AAPL --num_episodes 10 으로 모델을 학습하세요")
                return False
        else:
            print("⚠️ models 디렉토리가 없습니다")
            return False
            
    except Exception as e:
        print(f"❌ 모델 디렉토리 확인 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 SAC 트레이딩 시스템 간단 테스트")
    print("=" * 50)
    
    # 테스트 결과 저장
    results = {}
    
    # 1. Config 테스트
    config = test_imports()
    results['config'] = config is not None
    
    if not config:
        print("\n❌ 기본 설정 로드 실패. 테스트 중단.")
        return
    
    # 2. API 커넥터 테스트
    client = test_api_connector(config)
    results['api'] = client is not None
    
    # 3. 데이터 수집 테스트 (API 연결 성공한 경우만)
    if client:
        results['data'] = test_data_collection(client)
        client.disconnect()
    else:
        results['data'] = False
    
    # 4. 모델 디렉토리 확인
    results['models'] = test_model_directory(config)
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"✅ 설정 로드: {'성공' if results['config'] else '실패'}")
    print(f"✅ API 연결: {'성공' if results['api'] else '실패'}")
    print(f"✅ 데이터 수집: {'성공' if results['data'] else '실패'}")
    print(f"✅ 모델 확인: {'성공' if results['models'] else '실패'}")
    
    print(f"\n📊 전체 점수: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed >= 3:
        print("\n🎉 기본 테스트 통과! 실시간 트레이딩 준비가 거의 완료되었습니다.")
        print("\n🚀 다음 단계:")
        if not results['models']:
            print("1. 모델 학습: python run_training.py --symbols AAPL --num_episodes 20")
        print("2. 백테스트: python run_backtest.py --model_path models/최신모델")
        print("3. 실시간 트레이딩 테스트: python run_live_trading.py --model_path models/최신모델 --dry_run")
    elif passed >= 2:
        print("\n👍 절반 이상 성공! 몇 가지만 해결하면 됩니다.")
        if not results['api']:
            print("💡 알파카 API 키 설정을 확인하세요")
        if not results['data']:
            print("💡 pip install tvdatafeed 를 실행하세요")
    else:
        print("\n⚠️ 기본 설정에 문제가 있습니다.")
        print("💡 프로젝트 구조와 config.py 파일을 확인하세요")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 테스트가 중단되었습니다")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        print("💡 프로젝트 루트에서 실행하고 있는지 확인하세요")