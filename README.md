# trading-agent

스냅샷을 수집하고 Claude와 Codex의 의견을 합의 정책으로 판단한 뒤 Risk Guard를
거쳐 기록하는 paper trading 1단계입니다. **실계좌 주문은 전송하지 않습니다.**

## 설치와 실행 전제

Python 3.13 이상과 [uv](https://docs.astral.sh/uv/)를 설치한 뒤 프로젝트에서
의존성을 설치합니다.

```sh
uv sync
```

시장 데이터 수집을 위해 `trading-api`가 기본 주소 `http://localhost:9999`에서
실행 중이어야 합니다. 판단 CLI도 각각 설치하고 구독 계정에 로그인해야 합니다.

- Claude: `claude setup-token`으로 토큰을 만들고 `CLAUDE_CODE_OAUTH_TOKEN`에
  설정하거나 로그인 세션을 사용합니다.
- Codex: `codex login`으로 로그인합니다.

`ANTHROPIC_API_KEY` 또는 `OPENAI_API_KEY`가 설정돼 있어도 자식 CLI 환경에서
자동으로 제거합니다. 이 키를 사용하면 API 사용량 과금이 발생할 수 있기
때문입니다. 구독 로그인과 사용량 한도를 확인하고 실행하세요.

두 CLI는 사용자 전역 설정과 격리해 실행합니다. 매매 판단 역할의 system prompt를
따로 지정하고 도구는 쓰지 않습니다.

- Claude: `--safe-mode --strict-mcp-config --disable-slash-commands`.
  hooks, MCP, skills, plugins, CLAUDE.md를 로드하지 않습니다. `--bare`는 OAuth
  인증을 막으므로 쓰지 않습니다.
- Codex: `--json --ignore-user-config --ignore-rules`와 hooks·plugins·apps·
  multi_agent 등 `--disable`. 단, 전역 `~/.codex/AGENTS.md`는 끌 수 있는 설정이
  없어 계속 주입됩니다. system prompt에서 이 작업에는 적용되지 않는다고
  명시합니다.

## 설정

기본 설정 파일은 현재 작업 디렉터리의 `config.yaml`입니다. 파일이 없으면
내장 기본값과 환경변수를 사용하며 watchlist는 비어 있습니다. 계좌 보유 종목은
자동으로 대상에 포함됩니다. `--config`로 경로를 지정한 경우 파일이 없으면
실행 오류를 반환합니다.

```yaml
trading_api:
  base_url: http://localhost:9999
  timeout_seconds: 10
  market_timeouts:
    vkospi: 3
    vix: 5
watchlist:
  - "005930"
risk:
  max_position_weight: 0.2
  max_order_ratio: 0.1
  max_order_amount: 500000
  min_confidence: 0.6
  max_proposals_per_cycle: 5
  stop_loss_percent: 10
  take_profit_percent: 20
  take_profit_sell_fraction: 0.5
recommendation:
  universe: []        # 비우면 watchlist를 추천 후보로 사용
  max_results: 5      # recommend 출력에 표시할 BUY 수
  top_candidates: 5   # score 기준 상대강도 후보 수
  horizon_days: [10, 20]  # 판단 기간: 오름차순의 양의 거래일 목록
snapshot:
  news_lookback_trading_days: 3  # 오늘 이전 일봉 거래일 수 (1~10)
  news_limit: 5                  # 종목별 프롬프트 뉴스 수 (1~20)
consensus:
  policy: buy_one_side       # strict 또는 buy_one_side
  confidence_aggregation: mean  # min 또는 mean
price_guard:
  mode: adjust               # adjust, warn 또는 off
  min_stop_atr_multiple: 2.0
  min_reward_risk: 1.5
llm:
  claude:
    command: claude
    model: claude-opus-5-5
    effort: high
    timeout_seconds: 300
  codex:
    command: codex
    model: gpt-6.1-sol
    reasoning_effort: high
    timeout_seconds: 300
ledger:
  db_path: data/trading-agent.db
```

환경변수는 `TRADING_AGENT__RISK__MAX_ORDER_AMOUNT`처럼 중첩 키를
`__`로 구분합니다.

`risk.min_confidence`를 `config.yaml`에 직접 적으면 해당 값이 코드 기본값을
대신합니다. `TRADING_AGENT__CONSENSUS__POLICY`와
`TRADING_AGENT__CONSENSUS__CONFIDENCE_AGGREGATION` 환경변수로 합의 설정을
덮어쓸 수도 있습니다.

스냅샷은 KST 오늘 날짜와 그 날짜보다 앞선 최근 `snapshot.news_lookback_trading_days`
일봉 거래일 날짜로 종목 뉴스를 순차 조회합니다. 설정 범위는 1~10이며, 오늘 일봉이
있어도 오늘 이전 거래일 N개를 고르므로 종목당 요청은 최대 N+1회입니다. 일봉이 없으면
오늘 날짜만 조회합니다. 결과는 key 또는 제목이
같은 기사를 합친 뒤 최신순으로 정렬하고 `snapshot.news_limit`개까지 프롬프트에 넣습니다.
각 뉴스에는 제목, 출처, KST 게시 시각, 스냅샷 생성 시각부터의 경과 시간(시간 단위,
소수 첫째 자리), 15:30 KST 뒤 게시 여부가 포함됩니다. 장 마감 뒤 기사는 다음 거래일
판단에 반영될 수 있습니다. 환경변수 `TRADING_AGENT__SNAPSHOT__NEWS_LIMIT`과
`TRADING_AGENT__SNAPSHOT__NEWS_LOOKBACK_TRADING_DAYS`로 설정을 덮어쓸 수 있습니다.

프롬프트 스냅샷의 최상위 `data_basis`는 수정주가 여부와 코드에서 사용하는 RSI, MACD,
이동평균, ATR, 실현 변동성 기간 및 뉴스 조회 거래일 수를 기록합니다. trading-api 일봉
요청은 `adjusted_price: true`를 사용합니다.

### 추천 가격 가드 설정

| 설정 | 기본값 | 동작 |
| --- | --- | --- |
| `price_guard.mode` | `adjust` | `adjust`, `warn`, `off` 중 적용 모드 |
| `price_guard.min_stop_atr_multiple` | `2.0` | buy 손절·sell 철회 가격의 최소 ATR 배수 (`> 0`) |
| `price_guard.min_reward_risk` | `1.5` | 목표 폭 ÷ 손절 폭이 이 값보다 낮으면 경고 (`> 0`) |

`adjust`는 buy 손절가를 현재가에서 최소 `min_stop_atr_multiple × ATR` 아래로 낮추고
sell 철회 가격은 그만큼 높입니다. buy는 내림, sell은 올림해 원 단위로 보정합니다.
`warn`은 가격을 유지하면서 손절 폭이나 목표/손절 손익비 경고를 기록하고, `off`는
검사를 끕니다. 손익비가 `min_reward_risk`보다 낮으면 가격을 바꾸지 않습니다.
ATR이나 가격을 구할 수 없는 추천은 보정하지 않습니다. 합의 추천에는 보정 여부, 원래
손절가, 경고를 저장하고 모델별 원 의견은 그대로 보존합니다.

## 합의 정책

`consensus.policy`의 기본값은 `buy_one_side`입니다. `strict`는 두 모델의 유효
동작이 같을 때만 매수·매도를 채택하는 기존 규칙입니다. `buy_one_side`는 한 모델이
매수하고 다른 모델이 HOLD 또는 종목 응답 누락이면 매수를 채택합니다. 반대 모델이
매도라면 불일치로 남기고, 매도는 두 모델 모두 매도일 때만 채택합니다. 두 모델의
중복 응답은 어느 정책에서도 불일치입니다. 모델 하나가 실패하면 두 정책 모두
fail-closed로 제안을 승인하지 않습니다.

한쪽 매수 채택은 그 모델의 가격·수량·확신도를 그대로 사용합니다. 두 모델이
동의하면 매수 가격은 낮은 값, 매도 가격은 높은 값, 수량은 낮은 값을 사용합니다.
`consensus.confidence_aggregation` 기본값은 `mean`이며 두 모델이 동의한 항목의
확신도를 평균합니다. `min`은 기존처럼 낮은 확신도를 사용합니다. 한쪽 매수에는
매수 모델의 확신도를 사용합니다. 채택 항목의 `consensus_basis`는 `agreed`,
`claude_only`, `codex_only`로 기록됩니다.

## 판단 원칙과 투자 기간

추천과 매매 프롬프트는 `recommendation.horizon_days`를 투자 기간으로 사용합니다.
기본값 `[10, 20]`은 약 2~4주 스윙이며, 값은 양의 정수로 오름차순·중복 없이
설정해야 합니다. 값 하나를 지정하면 해당 거래일 수를 기간으로 표시합니다. 예를 들어
`[10]`은 10거래일입니다. 환경변수 `TRADING_AGENT__RECOMMENDATION__HORIZON_DAYS`로도
재정의할 수 있습니다.

과열(RSI 80 이상 또는 20·60일선과의 큰 이격)만으로 매도하지 않습니다. 매도는 MACD
하향 교차, 종가의 20일선 이탈, 거래량 증가를 동반한 하락 중 하나 이상으로 추세
훼손이 확인될 때만 고려합니다. 5일 수익률이 10% 이상 오른 직후에는 눌림 후 지지나
거래량 동반 돌파 같은 추가 근거 없이 매수하지 않습니다. 매매 근거가 약하거나
확신이 낮으면 hold를 유지합니다.

## 리스크 정책

주문 금액 한도는 계좌 총자산의 `max_order_ratio` 비율로 계산합니다(기본값
`0.1`). `max_order_amount`는 선택적인 절대 상한이며, 설정하면 비율 한도와 절대
상한 중 더 낮은 값을 적용합니다. 예시의 500,000원 상한은 총자산이 500만 원을
넘을 때 주문 크기를 추가로 제한합니다. 총자산을 계산할 수 없으면 제안을 거절합니다.

| 설정 | 기본값 | 동작 |
| --- | --- | --- |
| `max_position_weight` | `0.2` | 매수 후 한 종목이 총자산에서 차지할 수 있는 최대 비중 |
| `max_order_ratio` | `0.1` | 1회 주문 금액의 총자산 비율 상한 (`0 < 값 ≤ 1`) |
| `max_order_amount` | `null` | 선택적인 1회 주문 절대 금액 상한 |
| `min_confidence` | `0.6` | Risk Guard 최소 확신도. `config.yaml`에 직접 적은 값이 우선 |
| `stop_loss_percent` | `null` | 평균 매입가 대비 해당 하락률 이하이면 전량 매도 규칙을 평가 |
| `take_profit_percent` | `null` | 평균 매입가 대비 해당 상승률 이상이면 일부 매도 규칙을 평가 |
| `take_profit_sell_fraction` | `0.5` | 익절 시 매도할 보유 수량 비율 (`0 < 값 ≤ 1`, 내림 후 최소 1주) |

기존 설정의 `max_order_amount` 키는 계속 읽고 적용합니다. 업그레이드 후에도
기존 절대 상한을 유지하려면 해당 값을 남겨 두세요. LLM 합의 매도는 현재 비중이
한도를 넘은 종목에서 한도까지 줄이는 데 필요한 수량만 일반 주문 금액 한도에서
예외 처리하며, 추가 매도에는 일반 한도가 적용됩니다. 보유 수량을 넘는 매도는
계속 거절합니다.

보유 수량, 현재가, 총자산을 확인할 수 있으면 다음 규칙을 LLM 합의와 독립적으로
평가합니다. `stop_loss_percent`와 `take_profit_percent`가 `null`이면 각각의 규칙은
꺼집니다. 규칙 제안은 신뢰도 1.0으로 같은 Risk Guard를 통과해야 하고, 같은 종목의
합의 제안은 규칙 제안 뒤에 검토되어 중복으로 거절됩니다.

| 우선순위 | 규칙 | 제안 |
| --- | --- | --- |
| 1 | 손절 | 수익률이 기준 이하이면 보유 전량 매도 |
| 2 | 익절 | 수익률이 기준 이상이면 설정 비율을 매도 |
| 3 | 리밸런싱 | 종목 비중이 `max_position_weight`를 넘으면 한도까지 축소 매도 |

규칙에서 만든 손절·익절·리밸런싱 SELL은 주문 금액 한도를 넘더라도 보유 수량
이내이면 승인 검사를 계속합니다. LLM 합의 SELL에는 일반 주문 한도가 적용되고,
현재 비중이 한도를 넘은 종목은 필요한 축소 수량만 예외 처리합니다.

손절·익절 기본값은 비활성화되어 있습니다. 예시처럼 `10`과 `20`을 설정하면 평균
매입가 대비 -10% 이하에서 전량 손절, +20% 이상에서 기본 보유량의 절반을 익절합니다.
trading-api 계좌 응답의 평균 매입가 항목이 제공되지 않아, 계좌 손익률과 현재가에서
평균 매입가를 계산해 스냅샷에 채웁니다.

`trading_api.market_timeouts`에는 `kospi`, `kosdaq`, `sp500`, `vix`,
`vkospi`, `fear_and_greed`, `usd_krw`, `korea_3y_yield`, `us_10y_yield`별
timeout 초를 지정할 수 있습니다. 생략한 지표는 `trading_api.timeout_seconds`를
사용합니다. 이 설정은 `config.yaml`에 추가하지 않아도 기본값이 코드에 있습니다.

## 실행

```sh
uv run trading-agent --help
uv run trading-agent run --help
uv run trading-agent run
uv run trading-agent run --config ./config.yaml --db ./data/paper.sqlite
uv run trading-agent run --force
```

한 사이클은 계좌·시장 스냅샷 수집, Claude와 Codex의 독립 판단, 일치 제안의
합의, Risk Guard 검토, SQLite 기록 순서로 진행합니다. 두 모델은 스냅샷의 모든
종목에 대해 `buy`/`sell`/`hold` 중 하나를 정확히 한 번 냅니다. 응답에서 빠진
종목은 `hold`로 간주하므로 hold와 미응답은 같은 의견입니다. `buy_one_side`에서는
매수와 HOLD/미응답 조합을 채택하고 근거에 두 모델 의견을 남깁니다. 채택되지 않은
유효 동작 불일치는 불일치(`action_mismatch`) 이력에 남기며, 한 모델이라도 실패하면
그 사이클은 `model_failed`로 기록하고 제안을 승인하지 않습니다. 저장소는 기본적으로
`data/trading-agent.db`이며 필요한 부모 디렉터리를 만듭니다.

스냅샷에는 가장 최근 일봉 날짜인 `as_of_trading_date`와 KST 기준
`market_session`(open, closed, pre_open, holiday)이 포함됩니다. 시장 지표는 최근
14 calendar days, KOSPI·KOSDAQ 지수는 60일선을 계산할 수 있도록 100 calendar days,
일봉은 380 calendar days, 이동평균·RSI·MACD 지표는 기존 120 calendar days를
조회합니다. 일봉 조회 기간을 늘려도 지표 API 요청 기간은 늘지 않습니다. `run`은
주말에는 스냅샷 수집 없이, 평일 09:00 이후에는 실제로 받은
최신 일봉 날짜가 오늘보다 이전일 때 `skipped` 상태로 기록하고 LLM 판단 없이 종료
코드 0으로 끝납니다. 일봉을 받지 못했거나 조회가 실패해 날짜를 확인하지 못하면
휴장으로 단정하지 않고 누락 원인을 기록한 뒤 판단을 진행합니다. `--force`는 휴장
skip을 건너뜁니다. `recommend`는 휴장일에도 실행합니다.

스냅샷에는 다음 변동성·가격 범위 필드도 포함됩니다.

| 필드 | 정의 |
| --- | --- |
| `atr14_percent` | Wilder ATR(14) ÷ 현재가 × 100. 고가·저가·전일 종가 OHLC가 필요합니다. |
| `realized_volatility_20d_percent` | 최근 20개 단순 일간 수익률의 모집단 표준편차(%). 일간 값이며 연율화하지 않습니다. |
| `high_20d_distance_percent` | (현재가 − 최근 20거래일 최고 고가) ÷ 최고 고가 × 100. 최고가 아래면 음수입니다. |
| `low_20d_distance_percent` | (현재가 − 최근 20거래일 최저 저가) ÷ 최저 저가 × 100. 최저가 위면 양수입니다. |
| `high_52w_position_percent` | 최근 최대 252거래일 저가~고가 범위에서 현재가 위치. 저가가 0%, 고가가 100%이며 계산한 거래일 수는 `high_52w_period_trading_days`에 표시합니다. |

스냅샷의 시장 국면·상대강도 필드는 다음과 같습니다.

| 필드 | 정의 |
| --- | --- |
| `market_context.kospi_return_5d_percent`, `kospi_return_20d_percent` (KOSDAQ도 동일) | 지수 최신 값과 각각 5/20거래일 전 값의 수익률(%). |
| `market_context.kospi_ma20_deviation_percent`, `kospi_ma60_deviation_percent` (KOSDAQ도 동일) | 최신 지수 값과 최근 20/60거래일 단순 이동평균 괴리(%). 양수면 이동평균 위입니다. |
| `market_context.vkospi_change_5d_points` | 현재 VKOSPI에서 5거래일 전 값을 뺀 포인트 변화입니다. |
| `stocks[].industry` | 현재 종목 quote의 업종 메타데이터입니다. |
| `stocks[].relative_strength_5d_percent`, `relative_strength_20d_percent` | 종목 수익률에서 `market_name`으로 판별한 KOSPI 또는 KOSDAQ 수익률을 뺀 초과수익(%p). |
| `stocks[].universe_rank_percentile_20d` | 같은 스냅샷에서 20일 수익률이 유효한 종목의 백분위(0~100). 높을수록 강하고 동순위는 평균 순위입니다. 비교 종목이 하나뿐이면 null입니다. |
| `stocks[].industry_relative_strength_20d_percent` | 종목 20일 수익률에서 같은 업종의 다른 종목 평균을 뺀 초과수익(%p). 동종 비교 종목이 없으면 null입니다. |

`market_session = open`이면 최신 일봉에 당일 미완성 봉이 포함될 수 있습니다.

### 종목 추천

```sh
uv run trading-agent recommend
uv run trading-agent recommend --top 3 --db ./data/paper.sqlite
```

계좌 잔고·보유 여부와 무관하게 `recommendation.universe`(비어 있으면 watchlist)
종목마다 두 모델이 투자의견(`buy`/`hold`/`sell`), 확신도, 목표가,
`invalidation_price`를 냅니다. 응답에는 universe 안 상대 매력도 `score`(-2~+2),
universe 평균 대비 10거래일 기대 초과수익 `expected_excess_return_10d`(-50~+50%p),
최대 3개 `drivers`, 주요 위험 객체 `key_risk`도 포함됩니다. 점수와 기대수익은
유한한 숫자여야 하고 driver와 key risk 종류는 정해진 enum만 허용합니다. buy의
`invalidation_price`는 손절가이고, sell에서는 그 위로 오르면 매도 의견을 철회할
가격입니다. 기본 `buy_one_side` 정책에서는 한
모델의 buy와 다른 모델의 hold/미응답도 채택합니다. 동의한 의견의 확신도는 기본적으로
평균, 가격은 보수적인 쪽을 씁니다(buy: 낮은 목표가·높은 손절가). 한쪽 buy는 그
모델의 값과 확신도를 그대로 씁니다.
sell의 목표가는 예상 하락 가격입니다. 출력에는 rating과 무관하게 합의 score가 높은
`recommendation.top_candidates`(기본 5개) 상대강도 후보를 먼저 표시합니다. 이 목록은
두 모델이 의견을 달리한 종목의 모델별 점수도 보여줍니다. 그 뒤 BUY 확신도 순, SELL,
HOLD, 불일치 순으로 표시하고 `runs.kind = 'recommendation'`으로 기록합니다. 계좌를
조회하지 않으며 Risk Guard도 적용하지 않습니다. 현재가를 읽지 못한 종목은
`missing`에 적고 제외합니다.

상위 후보는 모델별 점수에서 다시 계산할 수 있어 별도 DB 행으로 저장하지 않습니다.
합의 추천의 `drivers_json`, `key_risk_type`, `key_risk_note`에는 claude와 codex의 근거를
각각 보존합니다.

buy 손절가는 현재가 − `2 × ATR` 또는 최근 20거래일 최저 저가 아래 중 더 낮은
가격을 기본으로 고려하고, sell 철회 가격은 현재가 + `2 × ATR` 또는 20거래일 최고
고가 위 중 더 높은 가격을 고려합니다. ATR 금액은 현재가 × `atr14_percent` ÷ 100으로
계산합니다. 목표 폭 ÷ 손절 폭 손익비는 1.5 이상을
요구합니다. `run` 프롬프트에는 ATR 등 스냅샷 필드 설명만 제공하며 매매 판단 응답에는
손절 가격을 추가하지 않습니다.

### 과거 시점 백테스트

`backtest run`은 과거 판단일의 입력 스냅샷을 재구성하고 기존 추천 유스케이스와
두 모델 합의를 실행합니다. 채점은 결과를 저장한 뒤 별도 분석에서 수행합니다.
먼저 모델을 호출하지 않는 `--snapshot-only`로 누수 차단과 누락 데이터를 확인하세요.

```sh
uv run trading-agent backtest run --snapshot-only \
  --date 20260915 --universe 005930 \
  --out data/backtest/pit-check
uv run trading-agent backtest run \
  --from 20260701 --dates 20 --seed 20261103 \
  --universe 005930,000660 --out data/backtest/sample
```

날짜를 지정하지 않으면 기본 시작일은 `20260701`이고, 종료일은 기본 horizons
`5,10,20` 중 가장 긴 horizon의 종가가 확보된 마지막 KOSPI 거래일입니다.
KST 15:30 전에는 당일 KOSPI 날짜를 달력에서 제외합니다. 계산된 `effective_to`는
manifest에 저장하고 재개 실행에서 다시 사용하므로 날짜 표본이 바뀌지 않습니다.
`--date YYYYMMDD`를 반복하거나 콤마로 묶어 날짜를 고를 수 있습니다. `--dates`는
기본 10이며 seed로 재현 가능한 무작위 표본을 뽑습니다. `--dates all`은 명시한 경우에만
구간의 모든 거래일을 실행합니다. 날짜는 기본 순차 실행이며 `--workers 2`까지
허용합니다. `completed`와 `snapshot_only` 결과는 재개 시 건너뛰고,
`model_failed`·`data_failed`·`error`는 다시 실행합니다.
기존 run을 재개할 때는 생성 당시 사용한 `--dates` 값을 그대로 넘기세요. 이전 기본값
`all`로 만든 run은 `--dates all`을 지정해야 설정 비교를 통과합니다.

판단 시각은 기본 KST 15:40이며 `--decision-time HH:MM`으로 조정할 수 있습니다.
과거 데이터는 다음 경계로 자릅니다.

| 데이터 | 백테스트에 허용하는 마지막 날짜 |
| --- | --- |
| 종목 일봉, 이동평균, RSI, RSI signal, MACD, MACD signal | 판단일 D 포함 |
| KOSPI, KOSDAQ, VKOSPI, 국고 3년 | 판단일 D 포함 |
| S&P 500, VIX, USD/KRW, 미 10년 | D 이전 |
| 뉴스 | 검색일이 D 이하이고 게시 시각이 판단 시각 이하 |
| 공포탐욕 지수 | 과거 조회 불가로 missing 처리 |
| 종목 업종 (`industry`) | 현재 quote 메타데이터를 사용합니다. 가격 시계열의 미래값이 아니므로 누수로 보지 않습니다. |

일봉에서 D 종가와 직전 거래일 종가로 현재가와 등락률을 재구성합니다. `--mask`는
모델 입력에서 날짜·종목 식별자·종목명·뉴스·업종을 감추고 종목별 가격 배율을 적용합니다.
결과에는 별도 역변환표와 복구한 추천을 함께 저장합니다.

기본 결과는 `data/backtest/<run-id>/` 아래 `manifest.json`, `dates/<D>.json`,
프롬프트 파일, `backtest.db`로 저장됩니다. `--out`과 `--db`로 각각 바꿀 수 있고,
운영 `ledger.db_path`와 같은 DB 경로는 거부합니다. 응답 캐시는
`data/backtest/cache/`를 사용하며 요청은 직렬화됩니다. `data/` 전체는 gitignore
대상입니다. `--snapshot-only`는 모델과 ledger를 실행하지 않습니다.

모델의 학습 기억은 제거할 수 없으므로 `20260701` 이전 판단에는 기억 누수 경고가
표시됩니다. 기본 분석 권장 구간도 이 날짜 이후입니다. trading-api의 수정주가 정책,
과거 trading-api 정정 데이터, 실제 운영 환경과 다른 15:40 판단 시각은 결과에 남는
한계입니다. 모델 실행은 날짜마다 Claude와 Codex를 한 번씩 호출하므로 구독 사용량을
확인하고, 먼저 `--snapshot-only`로 API 데이터와 결과 범위를 점검하세요.

저장한 결과는 `backtest score`로 실제 이후 일봉과 시장 지수를 조회해 별도 채점할 수
있습니다. 기본 horizon은 manifest의 `5,10,20`이며 결과 디렉터리에 `score.json`과
`score.md`를 씁니다. 점수 명령은 저장된 모델별 파싱 결과로 합의 정책을 다시 적용하므로
모델을 재호출하지 않습니다.

```sh
uv run trading-agent backtest score data/backtest/sample
uv run trading-agent backtest score data/backtest/sample \
  --policy buy_one_side --confidence-aggregation mean
uv run trading-agent backtest score data/backtest/sample --compare-policies
```

주 지표 `xs`는 같은 판단일 universe의 평균 수익률을 뺀 종목 상대수익(%p)입니다.
buy는 `xs > 0`, sell은 `xs < 0`, hold는 horizon별 기본 밴드 안(`5일 2.5`,
`10일 3.5`, `20일 5.0` %p)이면 적중으로 봅니다. 밴드는 `--hold-bands 5:2.5,10:3.5,20:5.0`
형식으로 바꿀 수 있습니다. IC는 의견 방향에 확신도를 곱한 점수와 xs 사이 날짜별
Spearman 상관의 평균이며, 날짜별 유효 종목이 3개 미만이면 IC에서 제외하고 합의
불일치는 0점으로 넣습니다. 신뢰구간은 날짜를 cluster로 재표집한 bootstrap 95% 구간이며,
정렬된 표본에서 `int(q × 반복 수)` 위치를 분위수로 씁니다. 확신도 보정은 buy/sell과
hold를 분리하고 `[lo, hi)` 구간(마지막 구간은 1.0 포함)을 사용합니다. 목표가와
무효화 가격은 이후 일봉 고가·저가에서 먼저 닿은 쪽을 출처·의견별로 셉니다. 가격 수준
하나라도 없거나 OHLC 경로가 완전하지 않으면 `no_price`로 기록합니다. 손절가·목표가의
중앙 변화율은 buy/sell별로 부호를 유지합니다. 종목 시장에 맞춰 KOSPI 또는 KOSDAQ
수익률을 차감한 `exc`와 절대 수익률 기준율도 함께 저장합니다.

점수 결과에는 모델별·합의 score의 날짜별 Spearman IC와 10거래일 기대 초과수익의
10일 xs IC를 추가합니다. 합의 score 상위 N 바스켓의 horizon별 평균 xs와 bootstrap
95% 구간, 출처·horizon·driver별 방향 적중률과 평균 xs, key risk 유형별 건수도
기록합니다. 이전 백테스트 결과처럼 신규 필드가 없으면 관련 표에 `데이터 없음`을
표시합니다.

상위 바스켓 크기는 `backtest score --top-n N`으로 바꿀 수 있습니다. 지정하지 않으면
run manifest의 `recommendation.top_candidates` 값을 사용하고, 해당 값이 없는 구형
manifest에서는 현재 설정을 사용합니다. 기본값은 5입니다.

가격·지수 조회 종료일은 계산한 horizon 여유 날짜와 KST 전날 중 이른 날로 제한합니다.
이를 통해 아직 끝나지 않은 오늘 응답을 영구 응답 캐시에 저장하지 않습니다.

2026-07 이전 판단일이 있으면 결과 상단에 모델 기억 누수 가능성을 알리고, 기간별
지표도 나눠 보여줍니다. 작은 표본이나 넓은 신뢰구간에서는 정책 간 순위를 단정하지
마세요. 결과 JSON의 누락 가격·벤치마크 건수와 source별 표본 수를 함께 확인하세요.

종료 코드는 cron 모니터링에 사용합니다. `run`과 `recommend`에 공통입니다.
SIGTERM은 `SystemExit(143)`으로 바뀝니다. 실행 중인 `run`·`recommend` 기록은
`error` 상태로 닫고, 이 실행기가 시작한 모델 CLI 프로세스 그룹에 SIGTERM을 보냅니다.
0.2초 안에 끝나지 않은 프로세스 그룹에는 SIGKILL을 보냅니다. 백테스트 중단 때는
완료되지 않은 날짜 결과가 없거나 `error`·`model_failed`로 남을 수 있으며, resume은
`completed`와 `snapshot_only`만 건너뛰므로 해당 날짜를 다시 실행합니다.

- `0`: `completed` (합의 제안이 0건이어도 정상 완료)
- `2`: `model_failed`
- `1`: `data_failed`, `error` 또는 실행 준비 실패

평일 장 시작 전과 장 마감 후 실행 예시입니다. 프로젝트 경로와 로그 경로를
환경에 맞게 바꾸세요.

```cron
CRON_TZ=Asia/Seoul
0 8 * * 1-5 cd /path/to/trading-agent && uv run trading-agent run >> /var/log/trading-agent.log 2>&1
0 16 * * 1-5 cd /path/to/trading-agent && uv run trading-agent run >> /var/log/trading-agent.log 2>&1
```

## 가상 계좌 모의투자

`paper` 명령은 실제 계좌 대신 SQLite에 저장한 가상 계좌(기본 1,000만원)로 `run`과
같은 판단 사이클을 실행하고, Risk Guard가 승인한 제안을 모의 체결해 수익률을
추적합니다. 시장 데이터는 trading-api의 실제 값을 쓰며 주문은 보내지 않습니다.

```sh
uv run trading-agent paper run      # 판단 + 승인 제안 모의 체결 + 평가 기록
uv run trading-agent paper mark     # 매매 없이 현재가로 평가 기록 (장 마감 후 권장)
uv run trading-agent paper report   # 누적 수익률, 벤치마크, 보유·주문 이력
```

```yaml
paper:
  initial_cash: 10000000
  commission_rate: 0.00015   # 매수·매도 수수료 (원 미만 버림)
  sell_tax_rate: 0.002       # 매도 거래세 (원 미만 버림)
  slippage_rate: 0.0         # 매수는 현재가×(1+값) 올림, 매도는 ×(1−값) 내림
  db_path: data/paper/portfolio.db
  ledger_db_path: data/paper/decisions.db  # 운영 ledger와 같은 경로는 거부
```

체결 규칙은 다음과 같습니다.

- 사이클 상태가 `completed` 또는 `model_failed`(규칙 제안만 승인)이고 스냅샷
  `market_session`이 `open`일 때만 체결합니다. 그 외에는 `미체결`로 기록합니다.
- 모델 판단이 끝난 뒤 종목 현재가를 다시 조회해 체결 기준가로 씁니다.
- 매수는 현재가가 지정가 이하일 때, 합의 매도는 현재가가 지정가 이상일 때
  현재가로 체결합니다. 손절·익절·리밸런싱 규칙 매도는 시장가로 봅니다.
  조건을 만족하지 않은 주문은 그 사이클에서 취소되며 다음 사이클로 넘기지 않습니다.
- 매수 수량은 수수료를 포함한 현금 한도로, 매도 수량은 보유 수량으로 줄입니다.
- 평균 매입가는 수수료를 제외한 체결가 가중평균이며, 실현손익은 매도 수수료와
  세금을 뺀 값입니다. 매수 수수료는 총평가(현금 감소)에 반영됩니다.

각 실행 후 보유·관심 종목 현재가와 KOSPI 최신 값으로 평가를 기록합니다. `report`는
마지막 평가 가격 기준으로 누적 수익률, 실현·평가 손익, 누적 수수료·세금, 초기
자금부터 계산한 최대 낙폭, 같은 기간 KOSPI 수익률, 첫 평가 시점에 관심 종목을
균등 매수해 보유했을 때의 수익률(비용 제외)과 초과수익을 표시합니다. 휴장 skip
사이클은 평가를 남기지 않습니다. 판단 상세는 `ledger_db_path`의 DB에 `run`과 같은
스키마로 저장됩니다.

평일 장중 매시 실행 예시는 다음과 같습니다. cron의 `PATH`에 `uv`, `claude`,
`codex`(node 포함)가 있어야 합니다.

```cron
CRON_TZ=Asia/Seoul
5 9-15 * * 1-5 cd /path/to/trading-agent && flock -n data/paper/.lock uv run trading-agent paper run >> data/paper/cron.log 2>&1
40 15 * * 1-5 cd /path/to/trading-agent && flock -w 900 data/paper/.lock uv run trading-agent paper mark && uv run trading-agent paper report >> data/paper/report.log 2>&1
```

## 기록 조회

최근 사이클의 리스크 승인 제안:

```sql
SELECT source, code, action, quantity, limit_price, reason
FROM risk_verdicts
WHERE approved = 1
  AND run_id = (
    SELECT id FROM runs WHERE kind = 'decision'
    ORDER BY started_at DESC, id DESC LIMIT 1
  );
```

최근 매매 합의의 정책, 집계 방식, 항목별 근거:

```sql
SELECT c.code, c.action, c.confidence, c.consensus_basis,
       r.policy, r.confidence_aggregation
FROM consensus_proposals AS c
JOIN consensus_results AS r ON r.run_id = c.run_id
WHERE c.run_id = (
  SELECT id FROM runs WHERE kind = 'decision'
  ORDER BY started_at DESC, id DESC LIMIT 1
)
ORDER BY c.proposal_index;
```

종목별 모델 불일치 이력:

```sql
SELECT r.started_at, r.kind, d.code, d.reason_code, c.policy,
       d.claude_actions_json, d.codex_actions_json
FROM consensus_disagreements AS d
JOIN runs AS r ON r.id = d.run_id
JOIN consensus_results AS c ON c.run_id = d.run_id
WHERE r.kind = 'decision'
ORDER BY r.started_at DESC, d.disagreement_index;
```

최근 추천 순위:

```sql
SELECT c.rank, c.code, c.rating, c.confidence, c.consensus_basis,
       c.target_price, c.invalidation_price, c.original_invalidation_price,
       c.price_adjusted, c.price_warnings_json, r.policy, r.confidence_aggregation
FROM consensus_recommendations AS c
JOIN consensus_results AS r ON r.run_id = c.run_id
WHERE c.run_id = (
  SELECT id FROM runs WHERE kind = 'recommendation'
  ORDER BY started_at DESC, id DESC LIMIT 1
)
ORDER BY c.rank;
```

기존 v1~v6 DB는 처음 열 때 v7으로 자동 이전됩니다. 기존 행의 새 점수와 근거 열은
`NULL`로 유지됩니다. v5에서 이전할 때 기존 합의
추천 행에는 보정 전 가격 `NULL`, 보정 여부 `0`, 경고 `[]`를 채웁니다. 기존 합의 결과에는 당시
동작인 `policy = 'strict'`, `confidence_aggregation = 'min'`, 항목 basis는
`agreed`가 채워집니다. 기존 리스크 판정에는
`source = 'consensus'`가 채워지고, 새 규칙 제안은 `source = 'rule'`과 규칙 종류를
함께 기록합니다. 오래된 `running` 행은 DB를 열 때 `stale_run_after_seconds`
(기본 3600초)보다 오래된 경우 `abandoned`로 정리합니다.

## 안전

현재 단계는 기록 전용이며 실계좌 주문을 보내지 않습니다. 다만 두 판단 CLI는
각 계정의 구독 사용량을 소비하므로 모델별 사용량 한도와 인증 상태를 확인하고
실행하세요.
