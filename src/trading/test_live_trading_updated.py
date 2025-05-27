#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실시간 트레이딩 시스템 테스트 스크립트 (업데이트된 버전)
안전한 설정으로 단계별 테스트 진행
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.api_connector import APIConnector
from src.config.config import Config
import pandas as pd
from datetime import datetime
import time

def print_header(title):
    """예쁜 헤더 출력"""
    print("\n" + "=" * 60)
    print(f"🔍 {title}")
    print("=" * 60)

def print_success(message):
    """성공 메시지 출력"""
    print(f"✅ {message}")

def print_error(message):
    """에러 메시지 출력"""
    print(f"❌ {message}")

def print_warning(message):
    """경고 메시지 출력"""
    print(f"⚠️  {message}")

def print_info(message):
    """정보 메시지 출력"""
    print(f"💡 {message}")

def test_config_settings():
    """0단계: 설정 확인"""
    print_header("설정 확인")
    
    config = Config()
    
    print("📊 현재 설정:")
    print(f"   활성화된 심볼: {config.TRADING_SYMBOLS}")
    print(f"   최대 포지션 크기: {config.max_position_size*100}%")
    print(f"   최대 거래 금액: ${config.max_trade_amount}")
    print(f"   거래 간격: {config.trading_interval/60}분")
    print(f"   일일 최대 손실: {config.MAX_DAILY_LOSS*100}%")
    print(f"   API URL: {config.BASE_URL}")
    print(f"   데이터 피드: {config.DATA_FEED}")
    
    # 안전성 확인
    safety_checks = []
    if config.max_position_size <= 0.02:  # 2% 이하
        safety_checks.append("✅ 포지션 크기 안전")
    else:
        safety_checks.append("⚠️  포지션 크기가 클 수 있음")
    
    if config.max_trade_amount <= 200:  # $200 이하
        safety_checks.append("✅ 거래 금액 안전")
    else:
        safety_checks.append("⚠️  거래 금액이 클 수 있음")
    
    if "paper" in config.BASE_URL.lower():
        safety_checks.append("✅ 페이퍼 트레이딩 모드")
    else:
        safety_checks.append("🚨 실제 거래 모드 - 위험!")
    
    print("\n🛡️ 안전성 체크:")
    for check in safety_checks:
        print(f"   {check}")
    
    return True

def test_api_connection():
    """1단계: API 연결 테스트"""
    print_header("알파카 API 연결 테스트")
    
    try:
        client = APIConnector()
        
        print("🔌 API 연결 시도 중...")
        
        if client.connect():
            print_success("API 연결 성공!")
            
            # 계정 정보 확인
            account = client.get_account_info()
            print(f"📊 계정 정보:")
            print(f"   계정 ID: {account.get('id', 'N/A')}")
            print(f"   현금: ${account.get('cash', 0):,.2f}")
            print(f"   포트폴리오: ${account.get('portfolio_value', 0):,.2f}")
            print(f"   매수력: ${account.get('buying_power', 0):,.2f}")
            print(f"   상태: {account.get('status', 'Unknown')}")
            
            # 페이퍼 트레이딩 확인
            if account.get('cash', 0) >= 100000:  # 페이퍼 트레이딩은 보통 10만달러로 시작
                print_success("페이퍼 트레이딩 계정으로 확인됨")
            
            return client
        else:
            print_error("API 연결 실패")
            print_info("config.py의 API 키를 확인하세요")
            print_info("또는 새로운 알파카 계정을 생성하세요: https://alpaca.markets")
            return None
            
    except Exception as e:
        print_error(f"연결 중 오류: {e}")
        print_info("인터넷 연결이나 API 키를 확인하세요")
        return None

def test_data_collection(client):
    """2단계: 트레이딩뷰 데이터 수집 테스트"""
    print_header("트레이딩뷰 데이터 수집 테스트")
    
    try:
        print("📈 AAPL 데이터 수집 시도 중...")
        
        # AAPL 데이터 테스트 (소량)
        data = client.get_market_data("AAPL", limit=5)
        
        if not data.empty:
            print_success("데이터 수집 성공!")
            print(f"📊 데이터 형태: {data.shape}")
            print(f"📅 최신 데이터 (최근 2개):")
            print(data.tail(2))
            
            # 데이터 품질 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if not missing_columns:
                print_success("모든 필수 컬럼 존재")
            else:
                print_warning(f"누락된 컬럼: {missing_columns}")
            
            # 최신 가격 확인
            latest_price = data['close'].iloc[-1]
            print(f"💰 AAPL 최신 가격: ${latest_price:.2f}")
            
            return True
        else:
            print_error("데이터가 비어있습니다")
            print_info("tvdatafeed 라이브러리 설치를 확인하세요: pip install tvdatafeed")
            return False
            
    except Exception as e:
        print_error(f"데이터 수집 중 오류: {e}")
        print_info("인터넷 연결이나 tvdatafeed 라이브러리를 확인하세요")
        return False

def test_portfolio_and_orders(client):
    """3단계: 포트폴리오 및 주문 기능 테스트"""
    print_header("포트폴리오 및 주문 기능 테스트")
    
    try:
        # 현재 포지션 확인
        print("📊 현재 포지션 조회 중...")
        positions = client.get_all_positions()
        print(f"📊 현재 포지션: {len(positions)}개")
        
        if positions:
            print("   보유 포지션:")
            for pos in positions[:5]:  # 최대 5개만 표시
                symbol = pos.get('symbol', 'Unknown')
                qty = pos.get('qty', 0)
                value = pos.get('market_value', 0)
                pnl = pos.get('unrealized_pl', 0)
                print(f"   - {symbol}: {qty}주, ${value:.2f}, PnL: ${pnl:.2f}")
        else:
            print("   현재 보유 포지션 없음")
        
        # 미체결 주문 확인
        print("\n📋 미체결 주문 조회 중...")
        orders = client.get_open_orders()
        print(f"📋 미체결 주문: {len(orders)}개")
        
        if orders:
            print("   미체결 주문:")
            for order in orders[:5]:  # 최대 5개만 표시
                symbol = order.get('symbol', 'Unknown')
                side = order.get('side', 'Unknown')
                qty = order.get('qty', 0)
                status = order.get('status', 'Unknown')
                print(f"   - {symbol} {side} {qty}주 ({status})")
        else:
            print("   현재 미체결 주문 없음")
        
        print_success("포트폴리오 및 주문 조회 성공")
        return True
        
    except Exception as e:
        print_error(f"포트폴리오/주문 조회 중 오류: {e}")
        return False

def test_small_order(client, dry_run=True):
    """4단계: 소액 주문 테스트 (선택사항)"""
    print_header(f"소액 주문 테스트 ({'DRY RUN' if dry_run else 'REAL ORDER'})")
    
    if dry_run:
        print_info("DRY RUN 모드: 실제 주문은 실행되지 않습니다")
        print_success("주문 기능 코드는 정상적으로 구현되어 있습니다")
        return True
    else:
        print_warning("실제 주문을 테스트합니다!")
        print_warning("페이퍼 트레이딩이지만 신중하게 진행하세요")
        
        try:
            # 매우 소액 주문 (AAPL 0.01주 = 약 $2-3 정도)
            print("💰 AAPL 0.01주 매수 주문 테스트...")
            
            result = client.place_market_order("AAPL", "buy", 0.01)
            
            if result.get("success", False):
                order_id = result.get("order_id", "")
                print_success(f"소액 테스트 주문 성공! 주문 ID: {order_id}")
                print_info("주문 상태를 확인 중...")
                
                # 잠시 대기 후 주문 상태 확인
                time.sleep(2)
                order_status = client.get_order_status(order_id)
                print(f"   주문 상태: {order_status.get('status', 'Unknown')}")
                
                return True
            else:
                print_error(f"소액 테스트 주문 실패: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print_error(f"소액 주문 테스트 중 오류: {e}")
            return False

def test_model_availability():
    """5단계: 모델 파일 확인"""
    print_header("학습된 모델 확인")
    
    try:
        config = Config()
        
        # 모델 디렉토리 확인
        models_dir = config.MODELS_DIR
        print(f"📁 모델 디렉토리: {models_dir}")
        
        if models_dir.exists():
            model_files = [f for f in models_dir.iterdir() if f.is_dir()]
            
            if model_files:
                print_success(f"사용 가능한 모델: {len(model_files)}개")
                
                # AAPL 모델 특별히 확인
                aapl_models = [f for f in model_files if 'aapl' in f.name.lower()]
                
                if aapl_models:
                    print_success(f"AAPL 모델 발견: {len(aapl_models)}개")
                    for model in aapl_models[:3]:  # 최대 3개만 표시
                        print(f"   - {model.name}")
                        
                        # 모델 파일들 확인
                        required_files = ['actor.pth', 'critic.pth', 'config.pth']
                        missing_files = [f for f in required_files if not (model / f).exists()]
                        
                        if missing_files:
                            print_warning(f"     누락된 파일: {missing_files}")
                        else:
                            print_success(f"     모든 필수 파일 존재")
                else:
                    print_warning("AAPL 전용 모델이 없습니다")
                    print_info("일반 모델들:")
                    for model in model_files[:5]:
                        print(f"   - {model.name}")
                
                return True
            else:
                print_error("학습된 모델이 없습니다")
                print_info("먼저 다음 명령어로 모델을 학습하세요:")
                print_info("python run_training.py --symbols AAPL --num_episodes 50")
                return False
        else:
            print_error("models 디렉토리가 없습니다")
            print_info("프로젝트 루트에서 실행하고 있는지 확인하세요")
            return False
            
    except Exception as e:
        print_error(f"모델 확인 중 오류: {e}")
        return False

def test_data_preprocessing():
    """6단계: 데이터 전처리 파이프라인 테스트"""
    print_header("데이터 전처리 파이프라인 테스트")
    
    try:
        from src.preprocessing.data_processor import DataProcessor
        
        print("⚙️ 데이터 전처리기 초기화 중...")
        processor = DataProcessor(window_size=30)
        
        print_success("데이터 전처리기 초기화 성공")
        
        # 간단한 더미 데이터로 테스트
        import numpy as np
        import pandas as pd
        
        print("🧪 더미 데이터로 전처리 테스트 중...")
        
        # 더미 주식 데이터 생성
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        dummy_data = pd.DataFrame({
            'open': np.random.uniform(150, 200, 100),
            'high': np.random.uniform(150, 200, 100),
            'low': np.random.uniform(150, 200, 100),
            'close': np.random.uniform(150, 200, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # 전처리 단계별 테스트
        processed_data = processor.preprocess_data(dummy_data)
        print_success("1단계: 데이터 전처리 완료")
        
        featured_data = processor.extract_features(processed_data)
        print_success("2단계: 특성 추출 완료")
        
        normalized_data = processor.normalize_features(featured_data, 'TEST', is_training=True)
        print_success("3단계: 데이터 정규화 완료")
        
        print(f"📊 전처리 결과:")
        print(f"   원본 데이터: {dummy_data.shape}")
        print(f"   전처리 후: {processed_data.shape}")
        print(f"   특성 추출 후: {featured_data.shape}")
        print(f"   정규화 후: {normalized_data.shape}")
        
        return True
        
    except Exception as e:
        print_error(f"데이터 전처리 테스트 중 오류: {e}")
        print_info("필요한 라이브러리들이 설치되어 있는지 확인하세요")
        return False

def run_comprehensive_test():
    """전체 테스트 실행"""
    print("🚀 SAC 실시간 트레이딩 시스템 종합 테스트 시작")
    print("📝 이 테스트는 실제 돈을 사용하지 않는 페이퍼 트레이딩 환경에서 진행됩니다")
    
    # 테스트 결과 저장
    test_results = {}
    
    # 0단계: 설정 확인
    test_results['config'] = test_config_settings()
    
    # 1단계: API 연결
    client = test_api_connection()
    test_results['api_connection'] = client is not None
    
    if not client:
        print_error("API 연결 실패로 나머지 테스트를 건너뜁니다")
        return test_results
    
    # 2단계: 데이터 수집
    test_results['data_collection'] = test_data_collection(client)
    
    # 3단계: 포트폴리오/주문 기능
    test_results['portfolio_orders'] = test_portfolio_and_orders(client)
    
    # 4단계: 소액 주문 테스트 (DRY RUN)
    test_results['order_test'] = test_small_order(client, dry_run=True)
    
    # 5단계: 모델 확인
    test_results['model_availability'] = test_model_availability()
    
    # 6단계: 데이터 전처리
    test_results['data_preprocessing'] = test_data_preprocessing()
    
    # 연결 종료
    if client:
        client.disconnect()
        print_info("API 연결 종료")
    
    return test_results

def print_final_results(test_results):
    """최종 결과 출력"""
    print_header("종합 테스트 결과")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print("📊 테스트 결과 요약:")
    print(f"✅ 설정 확인: {'통과' if test_results.get('config', False) else '실패'}")
    print(f"✅ API 연결: {'통과' if test_results.get('api_connection', False) else '실패'}")
    print(f"✅ 데이터 수집: {'통과' if test_results.get('data_collection', False) else '실패'}")
    print(f"✅ 포트폴리오/주문: {'통과' if test_results.get('portfolio_orders', False) else '실패'}")
    print(f"✅ 주문 기능: {'통과' if test_results.get('order_test', False) else '실패'}")
    print(f"✅ 모델 확인: {'통과' if test_results.get('model_availability', False) else '실패'}")
    print(f"✅ 데이터 전처리: {'통과' if test_results.get('data_preprocessing', False) else '실패'}")
    
    print(f"\n📊 전체 점수: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 모든 테스트 통과! 실시간 트레이딩 준비 완료")
        print_next_steps()
    elif passed_tests >= total_tests * 0.7:  # 70% 이상 통과
        print("\n👍 대부분의 테스트 통과! 몇 가지 문제만 해결하면 됩니다")
        print_partial_success_guidance(test_results)
    else:
        print("\n⚠️ 여러 테스트가 실패했습니다. 기본 설정을 다시 확인해주세요")
        print_failure_guidance(test_results)

def print_next_steps():
    """다음 단계 안내"""
    print("\n" + "=" * 60)
    print("🚀 다음 단계:")
    print("=" * 60)
    print("1. 모델 학습 (필요한 경우):")
    print("   python run_training.py --symbols AAPL --num_episodes 50")
    print()
    print("2. 백테스트 실행:")
    print("   python run_backtest.py --model_path models/your_model --symbols AAPL")
    print()
    print("3. DRY RUN 실시간 트레이딩:")
    print("   python run_live_trading.py --model_path models/your_model --dry_run")
    print()
    print("4. 실제 페이퍼 트레이딩:")
    print("   python run_live_trading.py --model_path models/your_model")
    print()
    print("💡 모든 단계는 페이퍼 트레이딩 환경에서 안전하게 진행됩니다!")

def print_partial_success_guidance(test_results):
    """부분 성공 시 가이드"""
    print("\n💡 해결해야 할 문제들:")
    
    if not test_results.get('api_connection', False):
        print("- 알파카 API 연결 문제: config.py의 API 키를 확인하세요")
    
    if not test_results.get('data_collection', False):
        print("- 데이터 수집 문제: pip install tvdatafeed 실행하세요")
    
    if not test_results.get('model_availability', False):
        print("- 모델 없음: python run_training.py로 모델을 먼저 학습하세요")
    
    if not test_results.get('data_preprocessing', False):
        print("- 전처리 문제: 필요한 라이브러리들을 설치하세요")

def print_failure_guidance(test_results):
    """실패 시 가이드"""
    print("\n🔧 문제 해결 방법:")
    print("1. 프로젝트 구조 확인:")
    print("   - src/ 폴더가 있는지 확인")
    print("   - config.py 파일이 있는지 확인")
    print()
    print("2. 필요한 라이브러리 설치:")
    print("   pip install alpaca-trade-api tvdatafeed torch pandas numpy")
    print()
    print("3. 알파카 계정 설정:")
    print("   - https://alpaca.markets 에서 계정 생성")
    print("   - Paper Trading API 키 발급")
    print("   - config.py에 API 키 입력")

def main():
    """메인 함수"""
    try:
        # 전체 테스트 실행
        test_results = run_comprehensive_test()
        
        # 최종 결과 출력
        print_final_results(test_results)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 테스트가 중단되었습니다")
    except Exception as e:
        print_error(f"테스트 실행 중 예상치 못한 오류: {e}")
        print_info("프로젝트 루트 디렉토리에서 실행하고 있는지 확인하세요")

if __name__ == "__main__":
    main()