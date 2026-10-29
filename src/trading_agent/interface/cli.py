"""명령행 진입점."""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from trading_agent.application.records import CycleStatus, DecisionCycleResult
from trading_agent.infrastructure.config import AgentSettings, load_settings
from trading_agent.interface.bootstrap import TradingAgentRuntime, create_runtime


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
    return parser


def main(
    argv: Sequence[str] | None = None,
    runtime_factory: Callable[
        [AgentSettings, str | Path | None], TradingAgentRuntime
    ] = create_runtime,
) -> int:
    """CLI를 실행하고 cron에서 확인할 종료 코드를 반환한다."""

    args = build_parser().parse_args(argv)
    if args.command != "run":
        return 0

    try:
        settings = load_settings(args.config)
        runtime = runtime_factory(settings, args.db)
    except Exception as error:
        print(f"실행 준비 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    try:
        result = runtime.run()
    except Exception as error:
        print(f"사이클 실행 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        runtime.close()

    _print_result(result)
    return _exit_code(result.status)


def _print_result(result: DecisionCycleResult) -> None:
    """사람이 확인할 사이클 요약을 stdout에 표시한다."""

    consensus = result.decision.consensus if result.decision is not None else None
    proposal_count = len(consensus.proposals) if consensus is not None else 0
    disagreement_count = len(consensus.disagreements) if consensus is not None else 0
    approved = [verdict for verdict in result.verdicts or [] if verdict.approved]
    missing_info = consensus.missing_info if consensus is not None else []
    print(f"상태: {result.status.value}")
    print(f"합의 채택: {proposal_count}건, 불일치: {disagreement_count}건")
    if approved:
        approved_summary = ", ".join(
            f"{item.proposal.code} {item.proposal.action.value.upper()} "
            f"x{item.proposal.quantity} @ {item.proposal.limit_price}"
            for item in approved
        )
    else:
        approved_summary = "없음"
    print(f"리스크 승인: {approved_summary}")
    print(f"missing_info: {', '.join(missing_info) if missing_info else '없음'}")
    if result.error:
        print(f"오류: {result.error}")


def _exit_code(status: CycleStatus) -> int:
    """요구된 성공·모델 실패·기타 실패 종료 코드를 반환한다."""

    if status is CycleStatus.COMPLETED:
        return 0
    if status is CycleStatus.MODEL_FAILED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
