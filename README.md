# trading-agent

스냅샷을 수집하고 Claude와 Codex가 같은 판단에 합의한 경우에만 Risk Guard를
통과시켜 기록하는 paper trading 1단계입니다. **실계좌 주문은 전송하지 않습니다.**

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
watchlist:
  - "005930"
risk:
  max_position_weight: 0.2
  max_order_amount: 500000
  min_confidence: 0.7
  max_proposals_per_cycle: 5
recommendation:
  universe: []        # 비우면 watchlist를 추천 후보로 사용
  max_results: 5      # recommend 출력에 표시할 BUY 수
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

## 실행

```sh
uv run trading-agent --help
uv run trading-agent run --help
uv run trading-agent run
uv run trading-agent run --config ./config.yaml --db ./data/paper.sqlite
```

한 사이클은 계좌·시장 스냅샷 수집, Claude와 Codex의 독립 판단, 일치 제안의
합의, Risk Guard 검토, SQLite 기록 순서로 진행합니다. 두 모델은 스냅샷의 모든
종목에 대해 `buy`/`sell`/`hold` 중 하나를 정확히 한 번 냅니다. 응답에서 빠진
종목은 `hold`로 간주하므로 hold와 미응답은 같은 의견입니다. 유효 동작이 다른
종목은 불일치(`action_mismatch`) 이력에 남기며, 한 모델이라도 실패하면 그 사이클은
`model_failed`로 기록하고 제안을 승인하지 않습니다. 저장소는 기본적으로
`data/trading-agent.db`이며 필요한 부모 디렉터리를 만듭니다.

### 종목 추천

```sh
uv run trading-agent recommend
uv run trading-agent recommend --top 3 --db ./data/paper.sqlite
```

계좌 잔고·보유 여부와 무관하게 `recommendation.universe`(비어 있으면 watchlist)
종목마다 두 모델이 투자의견(`buy`/`hold`/`sell`), 확신도, 목표가, 손절가를
냅니다. 의견이 같은 종목만 채택하고, 확신도는 낮은 쪽, 목표가와 손절가는 보수적인
쪽을 씁니다(buy: 낮은 목표가·높은 손절가). sell 의견에서는 같은 두 값이 예상
하락 가격과 의견 철회 가격(이 위로 오르면 매도 판단이 틀린 것)을 뜻하며, 출력도
그렇게 표시합니다. 결과는 BUY 확신도 순, SELL, HOLD,
불일치 순서로 출력하고 `runs.kind = 'recommendation'`으로 기록합니다. 계좌를
조회하지 않으며 Risk Guard도 적용하지 않습니다. 현재가를 읽지 못한 종목은
`missing`에 적고 제외합니다.

종료 코드는 cron 모니터링에 사용합니다. `run`과 `recommend`에 공통입니다.

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

## 기록 조회

최근 사이클의 리스크 승인 제안:

```sql
SELECT code, action, quantity, limit_price, reason
FROM risk_verdicts
WHERE approved = 1
  AND run_id = (SELECT id FROM runs ORDER BY started_at DESC, id DESC LIMIT 1);
```

종목별 모델 불일치 이력:

```sql
SELECT r.started_at, d.code, d.reason_code,
       d.claude_actions_json, d.codex_actions_json
FROM consensus_disagreements AS d
JOIN runs AS r ON r.id = d.run_id
ORDER BY r.started_at DESC, d.disagreement_index;
```

최근 추천 순위:

```sql
SELECT c.rank, c.code, c.rating, c.confidence, c.target_price, c.stop_loss_price
FROM consensus_recommendations AS c
WHERE c.run_id = (
  SELECT id FROM runs WHERE kind = 'recommendation'
  ORDER BY started_at DESC, id DESC LIMIT 1
)
ORDER BY c.rank;
```

기존 v1 DB는 처음 열 때 v2로 자동 이전됩니다(`runs.kind` 열과 추천 테이블 추가).

## 안전

현재 단계는 기록 전용이며 실계좌 주문을 보내지 않습니다. 다만 두 판단 CLI는
각 계정의 구독 사용량을 소비하므로 모델별 사용량 한도와 인증 상태를 확인하고
실행하세요.
