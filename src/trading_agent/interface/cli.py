"""명령행 진입점."""

import argparse
import signal
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from types import FrameType
from typing import Protocol

from trading_agent.application.records import (
    CycleStatus,
    DecisionCycleResult,
    RecommendationRunResult,
)
from trading_agent.domain.models import Action, TradeProposal
from trading_agent.domain.recommendation import StockRecommendation
from trading_agent.infrastructure.config import AgentSettings, load_settings
from trading_agent.interface.bootstrap import (
    create_recommendation_runtime,
    create_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    """CLI 도움말과 하위 명령 파서를 만든다."""

    parser = argparse.ArgumentParser(
        prog="trading-agent",
        description="스냅샷 기반 paper trading 에이전트",
    )
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run", help="결정 사이클을 한 번 실행")
    run_parser.add_argument("--config", type=Path, help="YAML 설정 파일 경로")
    run_parser.add_argument("--db", type=Path, help="SQLite 기록 파일 경로")
    run_parser.add_argument(
        "--force", action="store_true", help="휴장일에도 모델 판단을 강제로 실행"
    )
    recommend_parser = commands.add_parser(
        "recommend", help="계좌와 무관하게 후보 종목 투자의견을 한 번 생성"
    )
    recommend_parser.add_argument("--config", type=Path, help="YAML 설정 파일 경로")
    recommend_parser.add_argument("--db", type=Path, help="SQLite 기록 파일 경로")
    recommend_parser.add_argument(
        "--top",
        type=_positive_int,
        help="표시할 BUY 추천 수 (기본: recommendation.max_results)",
    )
    return parser


class _Runtime[R](Protocol):
    """CLI가 실행하고 정리할 런타임."""

    def run(self) -> R: ...

    def close(self) -> None: ...


def _handle_sigterm(_signum: int, _frame: FrameType | None) -> None:
    """SIGTERM을 종료 코드 143인 SystemExit로 바꿔 실행 정리를 허용한다."""

    raise SystemExit(143)


def main(
    argv: Sequence[str] | None = None,
    runtime_factory: Callable[..., _Runtime[DecisionCycleResult]] = create_runtime,
    recommendation_runtime_factory: Callable[
        ..., _Runtime[RecommendationRunResult]
    ] = create_recommendation_runtime,
) -> int:
    """CLI를 실행하고 cron에서 확인할 종료 코드를 반환한다."""

    signal.signal(signal.SIGTERM, _handle_sigterm)
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _execute(
            args, runtime_factory, lambda result, _settings: _print_result(result)
        )
    if args.command == "recommend":
        return _execute(
            args,
            recommendation_runtime_factory,
            lambda result, settings: _print_recommendations(
                result, args.top or settings.recommendation.max_results
            ),
        )
    return 0


def _execute[R: (DecisionCycleResult, RecommendationRunResult)](
    args: argparse.Namespace,
    factory: Callable[..., _Runtime[R]],
    print_result: Callable[[R, AgentSettings], None],
) -> int:
    """설정을 읽어 런타임을 실행하고 결과 요약과 종료 코드를 반환한다."""

    try:
        settings = load_settings(args.config)
        if getattr(args, "force", False):
            runtime = factory(settings, args.db, True)
        else:
            runtime = factory(settings, args.db)
    except Exception as error:
        print(f"실행 준비 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    try:
        result = runtime.run()
    except Exception as error:
        print(f"실행 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        runtime.close()

    print_result(result, settings)
    return _exit_code(result.status)


def _print_result(result: DecisionCycleResult) -> None:
    """사람이 확인할 사이클 요약을 stdout에 표시한다."""

    consensus = result.decision.consensus if result.decision is not None else None
    proposal_count = len(consensus.proposals) if consensus is not None else 0
    disagreement_count = len(consensus.disagreements) if consensus is not None else 0
    verdicts = result.verdicts or []
    approved = [verdict for verdict in verdicts if verdict.approved]
    rejected = [verdict for verdict in verdicts if not verdict.approved]
    missing_info = consensus.missing_info if consensus is not None else []
    print(f"상태: {result.status.value}")
    print(f"합의 채택: {proposal_count}건, 불일치: {disagreement_count}건")
    if approved:
        approved_summary = ", ".join(
            f"{_proposal_origin(item.proposal)} {item.proposal.code} "
            f"{item.proposal.action.value.upper()} "
            f"x{item.proposal.quantity} @ {item.proposal.limit_price}"
            for item in approved
        )
    else:
        approved_summary = "없음"
    print(f"리스크 승인: {approved_summary}")
    if rejected:
        rejected_summary = "; ".join(
            f"{_proposal_origin(item.proposal)} {item.proposal.code} "
            f"{item.proposal.action.value.upper()} x{item.proposal.quantity}: "
            f"{', '.join(reason.message for reason in item.reasons)}"
            for item in rejected
        )
        print(f"리스크 거절: {rejected_summary}")
    print(f"missing_info: {', '.join(missing_info) if missing_info else '없음'}")
    if result.error:
        label = "사유" if result.status is CycleStatus.SKIPPED else "오류"
        print(f"{label}: {result.error}")


def _proposal_origin(proposal: TradeProposal) -> str:
    """제안 출처와 규칙 종류를 CLI에 표시한다."""

    if proposal.rule_type is not None:
        return f"규칙({proposal.rule_type.value})"
    return "합의(consensus)"


def _print_recommendations(result: RecommendationRunResult, top: int) -> None:
    """합의 투자의견을 BUY 순위, SELL, HOLD, 불일치 순서로 표시한다."""

    print(f"상태: {result.status.value}")
    names = {
        stock.code: stock.name
        for stock in (result.snapshot.stocks if result.snapshot else [])
    }
    consensus = result.outcome.consensus if result.outcome is not None else None
    if consensus is not None:
        items = result.recommendations or []
        buys = [item for item in items if item.rating is Action.BUY]
        sells = [item for item in items if item.rating is Action.SELL]
        holds = [item for item in items if item.rating is Action.HOLD]
        print(
            f"합의 의견: {len(items)}종목, 불일치: {len(consensus.disagreements)}종목"
        )
        print("매수 추천 (확신도 순):" if buys else "매수 추천: 없음")
        for rank, item in enumerate(buys[:top], start=1):
            _print_recommendation(rank, item, names)
        if sells:
            print("매도·회피 의견:")
            for rank, item in enumerate(sells, start=1):
                _print_recommendation(rank, item, names)
        if holds:
            print(
                "관망(HOLD): " + ", ".join(_label(item.code, names) for item in holds)
            )
        for disagreement in consensus.disagreements:
            claude = "/".join(a.value for a in disagreement.claude_actions) or "미응답"
            codex = "/".join(a.value for a in disagreement.codex_actions) or "미응답"
            print(
                f"불일치: {_label(disagreement.code, names)} "
                f"(claude {claude} / codex {codex})"
            )
        missing = consensus.missing_info
        print(f"missing_info: {', '.join(missing) if missing else '없음'}")
    if result.snapshot is not None and result.snapshot.missing:
        print(f"수집 누락: {', '.join(result.snapshot.missing)}")
    if result.error:
        print(f"오류: {result.error}")


def _print_recommendation(
    rank: int, item: StockRecommendation, names: dict[str, str]
) -> None:
    """한 추천 항목과 가격 기준, 근거를 표시한다."""

    target = _price(item.target_price)
    invalidation = _price(item.invalidation_price)
    if item.rating is Action.SELL:
        levels = f"예상 하락가 {target}, {invalidation} 위로 오르면 매도 의견 철회"
    else:
        levels = f"목표가 {target} 손절가 {invalidation}"
    print(f"  {rank}. {_label(item.code, names)} 확신도 {item.confidence:.2f} {levels}")
    print(f"     근거: {item.reason}")


def _label(code: str, names: dict[str, str]) -> str:
    """종목 코드와 알려진 이름을 함께 표시한다."""

    name = names.get(code)
    return f"{code} {name}" if name else code


def _price(value: Decimal | None) -> str:
    """가격을 원 단위 천 단위 구분 문자열로 바꾼다."""

    return "-" if value is None else f"{value:,.0f}원"


def _positive_int(value: str) -> int:
    """CLI에서 1 이상의 정수만 받는다."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("양의 정수여야 합니다.") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("양의 정수여야 합니다.")
    return parsed


def _exit_code(status: CycleStatus) -> int:
    """요구된 성공·모델 실패·기타 실패 종료 코드를 반환한다."""

    if status in (CycleStatus.COMPLETED, CycleStatus.SKIPPED):
        return 0
    if status is CycleStatus.MODEL_FAILED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
