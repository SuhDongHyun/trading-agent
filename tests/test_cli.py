import signal
from pathlib import Path

import pytest

from trading_agent.application.records import CycleStatus, DecisionCycleResult
from trading_agent.interface.cli import _handle_sigterm, main


class FakeRuntime[R]:
    def __init__(self, result: R) -> None:
        self.result = result
        self.closed = False

    def run(self) -> R:
        return self.result

    def close(self) -> None:
        self.closed = True


def test_sigterm_handler_raises_system_exit_143() -> None:
    with pytest.raises(SystemExit) as error:
        _handle_sigterm(signal.SIGTERM, None)

    assert error.value.code == 143


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        (CycleStatus.COMPLETED, 0),
        (CycleStatus.SKIPPED, 0),
        (CycleStatus.MODEL_FAILED, 2),
        (CycleStatus.DATA_FAILED, 1),
        (CycleStatus.ERROR, 1),
    ],
)
def test_run_command_parses_arguments_and_returns_status_exit_code(
    tmp_path: Path,
    monkeypatch,
    capsys,
    status: CycleStatus,
    expected_exit: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    runtime = FakeRuntime(
        DecisionCycleResult(
            run_id=1,
            status=status,
            error="sample failure" if status is not CycleStatus.COMPLETED else None,
        )
    )
    calls = []

    def runtime_factory(settings, db_path):
        calls.append((settings, db_path))
        return runtime

    exit_code = main(
        ["run", "--db", "records/override.sqlite"],
        runtime_factory=runtime_factory,
    )

    assert exit_code == expected_exit
    assert calls[0][1] == Path("records/override.sqlite")
    assert runtime.closed
    assert f"상태: {status.value}" in capsys.readouterr().out


def test_missing_explicit_config_returns_error_before_runtime_creation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    created = False

    def runtime_factory(settings, db_path):
        nonlocal created
        created = True
        raise AssertionError("runtime must not be created")

    exit_code = main(
        ["run", "--config", "missing.yaml"],
        runtime_factory=runtime_factory,
    )

    assert exit_code == 1
    assert not created
    assert "설정 파일을 찾을 수 없습니다" in capsys.readouterr().err


def test_run_force_flag_is_forwarded_to_runtime_factory(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runtime = FakeRuntime(DecisionCycleResult(run_id=1, status=CycleStatus.COMPLETED))
    calls = []

    def runtime_factory(settings, db_path, force):
        calls.append(force)
        return runtime

    assert main(["run", "--force"], runtime_factory=runtime_factory) == 0
    assert calls == [True]
    assert runtime.closed


@pytest.mark.parametrize("top", ["0", "-1"])
def test_recommend_top_must_be_positive(tmp_path: Path, monkeypatch, top: str) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as error:
        main(["recommend", "--top", top])

    assert error.value.code == 2


def test_run_prints_approved_and_rejected_sources(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from decimal import Decimal

    from trading_agent.domain.models import (
        Action,
        ConsensusBasis,
        ProposalSource,
        RuleType,
        TradeProposal,
    )
    from trading_agent.domain.risk import (
        RiskIssue,
        RiskReasonCode,
        RiskVerdict,
    )

    monkeypatch.chdir(tmp_path)
    rule = TradeProposal(
        "005930",
        Action.SELL,
        1,
        Decimal("100"),
        1.0,
        "손절",
        source=ProposalSource.RULE,
        rule_type=RuleType.STOP_LOSS,
    )
    consensus = TradeProposal(
        "000660",
        Action.BUY,
        1,
        Decimal("200"),
        0.9,
        "합의 매수",
        consensus_basis=ConsensusBasis.CLAUDE_ONLY,
    )
    result = DecisionCycleResult(
        run_id=1,
        status=CycleStatus.COMPLETED,
        verdicts=[
            RiskVerdict(rule, True, []),
            RiskVerdict(
                consensus,
                False,
                [
                    RiskIssue(
                        RiskReasonCode.DUPLICATE_CODE,
                        "중복 제안입니다.",
                    )
                ],
            ),
        ],
    )

    assert main(["run"], runtime_factory=lambda settings, db: FakeRuntime(result)) == 0

    output = capsys.readouterr().out
    assert "규칙(stop_loss) 005930 SELL" in output
    assert "합의(claude 단독 buy) 000660 BUY" in output
    assert "리스크 거절:" in output
    assert "중복 제안입니다." in output


def test_recommend_command_prints_ranked_buys_and_returns_exit_code(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from datetime import UTC, datetime
    from decimal import Decimal

    from trading_agent.application.consensus_service import (
        RecommendationConsensusOutcome,
    )
    from trading_agent.application.ports import ModelRun
    from trading_agent.application.records import RecommendationRunResult
    from trading_agent.domain.consensus import Disagreement, DisagreementReasonCode
    from trading_agent.domain.models import (
        Action,
        ConsensusBasis,
        MarketContext,
        ResearchSnapshot,
        StockSnapshot,
    )
    from trading_agent.domain.recommendation import (
        RecommendationCandidate,
        RecommendationConsensusResult,
        StockRecommendation,
    )

    monkeypatch.chdir(tmp_path)
    buys = [
        StockRecommendation(
            code, Action.BUY, conf, Decimal("1200"), Decimal("900"), "r"
        )
        for code, conf in (("000660", 0.9), ("005930", 0.8))
    ]
    buys[0] = StockRecommendation(
        "000660",
        Action.BUY,
        0.9,
        Decimal("1200"),
        Decimal("900"),
        "r",
        consensus_basis=ConsensusBasis.CLAUDE_ONLY,
        original_invalidation_price=Decimal("950"),
        price_adjusted=True,
        price_warnings=["손익비 경고"],
    )
    sell = StockRecommendation(
        "086520", Action.SELL, 0.5, Decimal("66000"), Decimal("79000"), "r"
    )
    consensus = RecommendationConsensusResult(
        recommendations=[*buys, sell],
        disagreements=[
            Disagreement(
                "086520", DisagreementReasonCode.ACTION_MISMATCH, (Action.BUY,), ()
            )
        ],
        missing_info=[],
        claude_summary="c",
        codex_summary="x",
    )
    run = ModelRun("claude", "", "", 0, 0.1)
    snapshot = ResearchSnapshot(
        created_at=datetime(2026, 11, 2, tzinfo=UTC),
        stocks=[
            StockSnapshot("000660", "SK하이닉스"),
            StockSnapshot("005930", "삼성전자"),
            StockSnapshot("086520", "에코프로"),
        ],
        market_context=MarketContext(),
        missing=[],
    )
    result = RecommendationRunResult(
        run_id=3,
        status=CycleStatus.COMPLETED,
        snapshot=snapshot,
        outcome=RecommendationConsensusOutcome(run, run, consensus),
        recommendations=[*buys, sell],
        top_candidates=[
            RecommendationCandidate(
                code="005930",
                score=Decimal("1.5"),
                expected_excess_return_10d=Decimal("3.2"),
                claude_score=Decimal("1"),
                codex_score=Decimal("2"),
            )
        ],
    )
    runtime = FakeRuntime(result)

    exit_code = main(
        ["recommend", "--top", "1"],
        recommendation_runtime_factory=lambda settings, db: runtime,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert runtime.closed
    assert "상대강도 상위 후보:" in out
    assert "1. 005930 삼성전자 score 1.5 / 10일 기대 초과수익 3.2%p" in out
    assert (
        "1. 000660 SK하이닉스 (claude 단독 buy) 확신도 0.90 목표가 1,200원 "
        "손절가 900원 (ATR 보정, 원래 950원)"
    ) in out
    assert "가격 경고: 손익비 경고" in out
    assert "005930 삼성전자 확신도" not in out
    assert "예상 하락가 66,000원, 79,000원 위로 오르면 매도 의견 철회" in out
    assert "불일치: 086520 에코프로 (claude buy / codex 미응답)" in out
