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
합의, Risk Guard 검토, SQLite 기록 순서로 진행합니다. 두 모델의 동작이
일치하지 않는 제안은 불일치 이력에 남기며, 한 모델이라도 실패하면 그 사이클은
`model_failed`로 기록하고 제안을 승인하지 않습니다. 저장소는 기본적으로
`data/trading-agent.db`이며 필요한 부모 디렉터리를 만듭니다.

종료 코드는 cron 모니터링에 사용합니다.

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

## 안전

현재 단계는 기록 전용이며 실계좌 주문을 보내지 않습니다. 다만 두 판단 CLI는
각 계정의 구독 사용량을 소비하므로 모델별 사용량 한도와 인증 상태를 확인하고
실행하세요.
