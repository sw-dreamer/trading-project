# Alpaca Markets Python SDK References

## REST API 클래스 및 함수

### AlpacaRestClient

**설명**: Alpaca API에 접근하기 위한 REST 클라이언트

**주요 메서드**:
- `get_trading_account()` - 트레이딩 계정 정보 조회
- `get_account_configurations()` - 계정 구성 정보 조회
- `update_account_configurations()` - 계정 구성 업데이트
- `get_account_activities()` - 계정 활동 내역 조회
- `get_portfolio_history()` - 포트폴리오 이력 조회
- `list_orders()` - 주문 목록 조회
- `submit_order()` - 주문 제출
- `get_order()` - 특정 주문 정보 조회
- `replace_order()` - 주문 수정
- `cancel_order()` - 주문 취소
- `cancel_all_orders()` - 모든 주문 취소
- `list_positions()` - 포지션 목록 조회
- `get_position()` - 특정 포지션 정보 조회
- `close_position()` - 포지션 종료
- `close_all_positions()` - 모든 포지션 종료
- `get_assets()` - 자산 정보 조회
- `get_asset()` - 특정 자산 정보 조회
- `get_clock()` - 시장 시간 정보 조회
- `get_calendar()` - 시장 캘린더 조회
- `get_account_balance()` - 계정 잔고 조회
- `submit_journal()` - 저널 제출
- `get_journals()` - 저널 목록 조회
- `get_journal()` - 특정 저널 조회
- `get_watchlists()` - 관심 목록 조회
- `create_watchlist()` - 관심 목록 생성
- `get_watchlist()` - 특정 관심 목록 조회
- `update_watchlist()` - 관심 목록 업데이트
- `add_asset_to_watchlist()` - 관심 목록에 자산 추가
- `delete_watchlist()` - 관심 목록 삭제
- `remove_asset_from_watchlist()` - 관심 목록에서 자산 제거
- `get_corporate_announcements()` - 기업 공지 조회
- `get_fundamental_data()` - 기업 기본 데이터 조회
- `create_transfer()` - 자금 이체 생성
- `get_transfers()` - 자금 이체 내역 조회
- `get_transfer()` - 특정 자금 이체 정보 조회
- `cancel_transfer()` - 자금 이체 취소
- `create_funding_source()` - 자금 소스 생성
- `get_funding_sources()` - 자금 소스 목록 조회
- `delete_funding_source()` - 자금 소스 삭제

### MarketDataRestClient

**설명**: 시장 데이터 조회를 위한 REST 클라이언트

**주요 메서드**:
- `get_bars()` - 가격 바 데이터 조회
- `get_quotes()` - 호가 데이터 조회
- `get_trades()` - 거래 데이터 조회
- `get_latest_bar()` - 최신 바 데이터 조회
- `get_latest_bars()` - 여러 심볼의 최신 바 데이터 조회
- `get_latest_quote()` - 최신 호가 데이터 조회
- `get_latest_quotes()` - 여러 심볼의 최신 호가 데이터 조회
- `get_latest_trade()` - 최신 거래 데이터 조회
- `get_latest_trades()` - 여러 심볼의 최신 거래 데이터 조회
- `get_snapshot()` - 특정 자산의 스냅샷 조회
- `get_snapshots()` - 여러 자산의 스냅샷 조회
- `get_news()` - 뉴스 데이터 조회

## 스트리밍 API 클래스 및 함수

### AlpacaStreamClient

**설명**: Alpaca API의 실시간 데이터 스트리밍을 위한 클라이언트

**주요 메서드**:
- `subscribe_trades()` - 거래 데이터 구독
- `unsubscribe_trades()` - 거래 데이터 구독 취소
- `subscribe_quotes()` - 호가 데이터 구독
- `unsubscribe_quotes()` - 호가 데이터 구독 취소
- `subscribe_bars()` - 바 데이터 구독
- `unsubscribe_bars()` - 바 데이터 구독 취소
- `subscribe_updated_bars()` - 업데이트된 바 데이터 구독
- `unsubscribe_updated_bars()` - 업데이트된 바 데이터 구독 취소
- `subscribe_daily_bars()` - 일별 바 데이터 구독
- `unsubscribe_daily_bars()` - 일별 바 데이터 구독 취소
- `subscribe_statuses()` - 상태 데이터 구독
- `unsubscribe_statuses()` - 상태 데이터 구독 취소
- `subscribe_lulds()` - LULD 데이터 구독
- `unsubscribe_lulds()` - LULD 데이터 구독 취소
- `subscribe_news()` - 뉴스 구독
- `unsubscribe_news()` - 뉴스 구독 취소
- `subscribe_trade_updates()` - 거래 업데이트 구독
- `run()` - 스트리밍 클라이언트 실행

## 데이터 모델

### Account

**설명**: 트레이딩 계정 정보를 나타내는 모델

**주요 속성**:
- `account_id` - 계정 ID
- `status` - 계정 상태
- `currency` - 계정 통화
- `buying_power` - 매수 가능 금액
- `cash` - 현금 잔고
- `portfolio_value` - 포트폴리오 가치
- `pattern_day_trader` - 패턴 데이 트레이더 여부
- `trading_blocked` - 거래 차단 여부
- `account_blocked` - 계정 차단 여부
- `created_at` - 계정 생성 시간

### Order

**설명**: 주문 정보를 나타내는 모델

**주요 속성**:
- `order_id` - 주문 ID
- `client_order_id` - 클라이언트 주문 ID
- `symbol` - 주문 심볼
- `asset_class` - 자산 클래스
- `qty` - 수량
- `filled_qty` - 체결된 수량
- `type` - 주문 유형 (market, limit 등)
- `side` - 매수/매도 방향
- `time_in_force` - 주문 유효 시간
- `limit_price` - 지정가
- `stop_price` - 손절가
- `status` - 주문 상태
- `created_at` - 주문 생성 시간
- `updated_at` - 주문 업데이트 시간
- `submitted_at` - 주문 제출 시간
- `filled_at` - 주문 체결 시간
- `expired_at` - 주문 만료 시간
- `canceled_at` - 주문 취소 시간
- `failed_at` - 주문 실패 시간
- `replaced_at` - 주문 대체 시간

### Position

**설명**: 보유 포지션 정보를 나타내는 모델

**주요 속성**:
- `asset_id` - 자산 ID
- `symbol` - 심볼
- `exchange` - 거래소
- `asset_class` - 자산 클래스
- `avg_entry_price` - 평균 진입 가격
- `qty` - 수량
- `side` - 매수/매도 방향
- `market_value` - 시장 가치
- `cost_basis` - 원가 기준
- `unrealized_pl` - 미실현 손익
- `unrealized_plpc` - 미실현 손익 퍼센트
- `current_price` - 현재 가격
- `lastday_price` - 전일 가격
- `change_today` - 오늘의 변화

### Asset

**설명**: 거래 가능한 자산 정보를 나타내는 모델

**주요 속성**:
- `asset_id` - 자산 ID
- `symbol` - 심볼
- `name` - 이름
- `exchange` - 거래소
- `asset_class` - 자산 클래스
- `status` - 상태
- `tradable` - 거래 가능 여부
- `marginable` - 마진 거래 가능 여부
- `shortable` - 공매도 가능 여부
- `easy_to_borrow` - 쉽게 차입 가능 여부
- `fractionable` - 분할 거래 가능 여부

### Bar

**설명**: 가격 바 데이터를 나타내는 모델

**주요 속성**:
- `symbol` - 심볼
- `timestamp` - 타임스탬프
- `open` - 시가
- `high` - 고가
- `low` - 저가
- `close` - 종가
- `volume` - 거래량
- `trade_count` - 거래 횟수
- `vwap` - 거래량 가중 평균 가격

### Quote

**설명**: 호가 데이터를 나타내는 모델

**주요 속성**:
- `symbol` - 심볼
- `timestamp` - 타임스탬프
- `ask_price` - 매도호가
- `ask_size` - 매도호가 수량
- `bid_price` - 매수호가
- `bid_size` - 매수호가 수량

### Trade

**설명**: 거래 데이터를 나타내는 모델

**주요 속성**:
- `symbol` - 심볼
- `timestamp` - 타임스탬프
- `price` - 가격
- `size` - 수량
- `exchange` - 거래소
- `trade_id` - 거래 ID
- `tape` - 테이프

### Clock

**설명**: 시장 시간 정보를 나타내는 모델

**주요 속성**:
- `timestamp` - 현재 타임스탬프
- `is_open` - 시장 개장 여부
- `next_open` - 다음 개장 시간
- `next_close` - 다음 마감 시간

### Calendar

**설명**: 시장 캘린더 정보를 나타내는 모델

**주요 속성**:
- `date` - 날짜
- `open` - 개장 시간
- `close` - 마감 시간
- `session_open` - 세션 시작 시간
- `session_close` - 세션 종료 시간

### Watchlist

**설명**: 관심 종목 목록을 나타내는 모델

**주요 속성**:
- `id` - 관심 목록 ID
- `name` - 관심 목록 이름
- `account_id` - 계정 ID
- `created_at` - 생성 시간
- `updated_at` - 업데이트 시간
- `assets` - 자산 목록

## 주요 열거형 및 상수

### OrderSide

**설명**: 주문 방향

**값**:
- `BUY` - 매수
- `SELL` - 매도

### OrderType

**설명**: 주문 유형

**값**:
- `MARKET` - 시장가 주문
- `LIMIT` - 지정가 주문
- `STOP` - 스톱 주문
- `STOP_LIMIT` - 스톱 리밋 주문
- `TRAILING_STOP` - 추적 손절 주문

### OrderTimeInForce

**설명**: 주문 유효 시간

**값**:
- `DAY` - 당일 유효
- `GTC` - 취소될 때까지 유효
- `OPG` - 개장 시 유효
- `CLS` - 마감 시 유효
- `IOC` - 즉시 체결 아니면 취소
- `FOK` - 전량 체결 아니면 취소

### OrderStatus

**설명**: 주문 상태

**값**:
- `NEW` - 신규 주문
- `PARTIALLY_FILLED` - 부분 체결
- `FILLED` - 완전 체결
- `DONE_FOR_DAY` - 당일 완료
- `CANCELED` - 취소됨
- `EXPIRED` - 만료됨
- `REPLACED` - 대체됨
- `PENDING_CANCEL` - 취소 대기 중
- `PENDING_REPLACE` - 대체 대기 중
- `ACCEPTED` - 수락됨
- `PENDING_NEW` - 신규 대기 중
- `ACCEPTED_FOR_BIDDING` - 입찰 수락됨
- `STOPPED` - 중지됨
- `REJECTED` - 거절됨
- `SUSPENDED` - 일시 중지됨
- `CALCULATED` - 계산됨

### AssetStatus

**설명**: 자산 상태

**값**:
- `ACTIVE` - 활성
- `INACTIVE` - 비활성

### AssetExchange

**설명**: 자산 거래소

**값**:
- `AMEX` - 아메리칸 증권거래소
- `ARCA` - 아카 거래소
- `BATS` - BATS 거래소
- `NYSE` - 뉴욕 증권거래소
- `NASDAQ` - 나스닥
- `NYSEARCA` - NYSE 아카

### AssetClass

**설명**: 자산 클래스

**값**:
- `US_EQUITY` - 미국 주식
- `CRYPTO` - 암호화폐

### BarTimeframe

**설명**: 바 데이터 시간 프레임

**값**:
- `ONE_MIN` - 1분
- `FIVE_MIN` - 5분
- `FIFTEEN_MIN` - 15분
- `THIRTY_MIN` - 30분
- `ONE_HOUR` - 1시간
- `TWO_HOURS` - 2시간
- `FOUR_HOURS` - 4시간
- `ONE_DAY` - 1일
- `ONE_WEEK` - 1주
- `ONE_MONTH` - 1개월

### NewsType

**설명**: 뉴스 유형

**값**:
- `TOP_NEWS` - 주요 뉴스
- `MARKET_NEWS` - 시장 뉴스
- `CORPORATE_NEWS` - 기업 뉴스
- `BUSINESS_NEWS` - 비즈니스 뉴스
- `PERSONAL_FINANCE` - 개인 금융 뉴스
