"""명령행 진입점."""

import argparse
import signal
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from types import FrameType
from typing import Protocol
from zoneinfo import ZoneInfo

from trading_agent.application.paper_trading import (
    PaperCycleOutcome,
    PaperReport,
    PaperValuation,
)
from trading_agent.application.records import (
    CycleStatus,
    DecisionCycleResult,
    RecommendationRunResult,
)
from trading_agent.domain.models import Action, ConsensusBasis, TradeProposal
from trading_agent.domain.paper import FillStatus
from trading_agent.domain.recommendation import StockRecommendation
from trading_agent.infrastructure.config import AgentSettings, load_settings
from trading_agent.interface.bootstrap import (
    create_backtest_runtime,
    create_backtest_score_runtime,
    create_paper_mark_runtime,
    create_paper_runtime,
    create_recommendation_runtime,
    create_runtime,
    load_paper_report,
)

KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


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
    backtest_parser = commands.add_parser("backtest", help="과거 시점 추천을 재현")
    backtest_commands = backtest_parser.add_subparsers(dest="backtest_command")
    backtest_parser.set_defaults(_print_backtest_help=backtest_parser.print_help)
    backtest_run = backtest_commands.add_parser(
        "run", help="과거 시점 스냅샷과 추천을 저장"
    )
    backtest_run.add_argument("--config", type=Path, help="YAML 설정 파일 경로")
    backtest_run.add_argument(
        "--from",
        dest="start_date",
        type=_date_arg,
        default="20260701",
        help="시작일 YYYYMMDD",
    )
    backtest_run.add_argument(
        "--to", dest="end_date", type=_date_arg, help="종료일 YYYYMMDD"
    )
    backtest_run.add_argument(
        "--dates",
        help="all 또는 무작위로 뽑을 거래일 수 (기본: 10)",
    )
    backtest_run.add_argument(
        "--date", action="append", help="명시 날짜 YYYYMMDD (반복·콤마 목록 가능)"
    )
    backtest_run.add_argument("--seed", type=int, default=20261103)
    backtest_run.add_argument(
        "--universe", type=_comma_values, help="추천 종목 코드 콤마 목록"
    )
    backtest_run.add_argument("--mask", action="store_true", help="모델 입력을 마스킹")
    backtest_run.add_argument(
        "--horizons", default="5,10,20", help="미래 거래일 horizon (기본: 5,10,20)"
    )
    backtest_run.add_argument("--db", type=Path, help="백테스트 전용 SQLite 경로")
    backtest_run.add_argument("--out", type=Path, help="결과 디렉터리")
    backtest_run.add_argument("--cache-dir", type=Path, help="응답 캐시 디렉터리")
    backtest_run.add_argument(
        "--snapshot-only", action="store_true", help="모델·ledger 없이 스냅샷만 저장"
    )
    backtest_run.add_argument("--workers", type=_worker_count, default=1)
    backtest_run.add_argument(
        "--decision-time",
        type=_decision_time,
        default=time(15, 40),
        help="판단 시각 KST HH:MM",
    )
    backtest_run.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="완료된 날짜 결과를 건너뜀 (기본: 사용)",
    )
    backtest_score = backtest_commands.add_parser(
        "score", help="저장된 백테스트를 이후 가격으로 채점·재채점"
    )
    backtest_score.add_argument(
        "result_dir", type=Path, help="backtest run 결과 디렉터리"
    )
    backtest_score.add_argument("--config", type=Path, help="YAML 설정 파일 경로")
    backtest_score.add_argument("--cache-dir", type=Path, help="응답 캐시 디렉터리")
    backtest_score.add_argument(
        "--horizons",
        type=_positive_list,
        help="채점 거래일 horizon 목록 (기본: manifest)",
    )
    backtest_score.add_argument(
        "--top-n",
        dest="top_candidates",
        type=_positive_int,
        help=(
            "합의 score 상위 바스켓 크기 "
            "(기본: manifest 또는 recommendation.top_candidates)"
        ),
    )
    backtest_score.add_argument(
        "--policy", choices=("strict", "buy_one_side"), help="재채점 합의 정책"
    )
    backtest_score.add_argument(
        "--confidence-aggregation", choices=("min", "mean"), help="합의 확신도 집계"
    )
    backtest_score.add_argument(
        "--compare-policies",
        action="store_true",
        help="전체 정책·확신도 조합 비교표 생성",
    )
    backtest_score.add_argument(
        "--bootstrap-iterations", type=_positive_int, default=4000
    )
    backtest_score.add_argument("--seed", type=int, default=20261103)
    backtest_score.add_argument(
        "--hold-bands",
        type=_hold_bands,
        help="hold 적중 밴드 예: 5:2.5,10:3.5,20:5.0 (xs percentage points)",
    )
    backtest_score.add_argument(
        "--output", type=Path, help="출력 JSON 파일 (기본: <out-dir>/score.json)"
    )
    paper_parser = commands.add_parser(
        "paper", help="가상 계좌 모의투자 (주문 전송 없음)"
    )
    paper_commands = paper_parser.add_subparsers(dest="paper_command")
    paper_parser.set_defaults(_print_paper_help=paper_parser.print_help)
    paper_run = paper_commands.add_parser(
        "run", help="가상 계좌로 판단 사이클을 실행하고 승인 제안을 모의 체결"
    )
    paper_run.add_argument(
        "--force", action="store_true", help="휴장일 skip 판정을 생략"
    )
    paper_run.add_argument(
        "--no-reuse",
        action="store_true",
        help="직전 판단을 재사용하지 않고 모델을 호출",
    )
    paper_run.add_argument("--ledger-db", type=Path, help="모의투자 판단 기록 DB")
    paper_mark = paper_commands.add_parser(
        "mark", help="매매 없이 현재가로 가상 계좌를 평가해 기록"
    )
    paper_report = paper_commands.add_parser("report", help="모의투자 수익률 요약")
    paper_report.add_argument(
        "--fills", type=_positive_int, default=10, help="표시할 최근 주문 수"
    )
    for sub in (paper_run, paper_mark, paper_report):
        sub.add_argument("--config", type=Path, help="YAML 설정 파일 경로")
        sub.add_argument("--paper-db", type=Path, help="가상 계좌 DB 경로")
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
    backtest_runtime_factory: Callable[..., _Runtime[int]] = create_backtest_runtime,
    backtest_score_runtime_factory: Callable[
        ..., _Runtime[None]
    ] = create_backtest_score_runtime,
    paper_runtime_factory: Callable[
        ..., _Runtime[PaperCycleOutcome]
    ] = create_paper_runtime,
    paper_mark_runtime_factory: Callable[
        ..., _Runtime[PaperValuation]
    ] = create_paper_mark_runtime,
    paper_report_loader: Callable[..., PaperReport | None] = load_paper_report,
) -> int:
    """CLI를 실행하고 cron에서 확인할 종료 코드를 반환한다."""

    signal.signal(signal.SIGTERM, _handle_sigterm)
    args = build_parser().parse_args(argv)
    if args.command == "backtest" and args.backtest_command is None:
        args._print_backtest_help()
        return 2
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
    if args.command == "backtest" and args.backtest_command == "run":
        return _execute_backtest(args, backtest_runtime_factory)
    if args.command == "backtest" and args.backtest_command == "score":
        return _execute_backtest_score(args, backtest_score_runtime_factory)
    if args.command == "paper":
        return _execute_paper(
            args,
            paper_runtime_factory,
            paper_mark_runtime_factory,
            paper_report_loader,
        )
    return 0


def _execute_paper(
    args: argparse.Namespace,
    run_factory: Callable[..., _Runtime[PaperCycleOutcome]],
    mark_factory: Callable[..., _Runtime[PaperValuation]],
    report_loader: Callable[..., PaperReport | None],
) -> int:
    """모의투자 하위 명령을 실행한다."""

    if args.paper_command is None:
        args._print_paper_help()
        return 2
    try:
        settings = load_settings(args.config)
    except Exception as error:
        print(f"실행 준비 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if args.paper_command == "report":
        try:
            report = report_loader(settings, args)
        except Exception as error:
            print(f"리포트 실패: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        if report is None:
            print(
                "가상 계좌가 아직 없습니다. paper run이나 paper mark를 먼저 실행하세요."
            )
            return 0
        _print_paper_report(report)
        return 0
    factory = run_factory if args.paper_command == "run" else mark_factory
    try:
        runtime = factory(settings, args)
    except Exception as error:
        print(f"실행 준비 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    try:
        outcome = runtime.run()
    except Exception as error:
        print(f"실행 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        runtime.close()
    if isinstance(outcome, PaperCycleOutcome):
        _print_result(outcome.result)
        _print_paper_cycle(outcome)
        return _exit_code(outcome.result.status)
    _print_valuation(outcome, settings.paper.initial_cash)
    return 0


def _won(value: Decimal | float | None) -> str:
    if value is None:
        return "-"
    return f"{Decimal(str(value)):,.0f}원"


def _pct(value: Decimal | None, points: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}{'%p' if points else '%'}"


def _print_paper_cycle(outcome: PaperCycleOutcome) -> None:
    """모의 체결과 사이클 직후 평가를 표시한다."""

    for fill in outcome.fills:
        proposal = fill.proposal
        head = (
            f"{_proposal_origin(proposal)} {proposal.code} {fill.name} "
            f"{proposal.action.value.upper()}"
        )
        if fill.status == FillStatus.FILLED:
            print(
                f"모의 체결: {head} {fill.quantity}주 @ {_won(fill.fill_price)} "
                f"(수수료 {_won(fill.commission)}, 세금 {_won(fill.tax)})"
                + (f" - {fill.note}" if fill.note else "")
            )
        else:
            print(
                f"미체결: {head} {proposal.quantity}주 지정가 "
                f"{_won(proposal.limit_price)}, 현재가 {_won(fill.quote_price)} - "
                f"{fill.note}"
            )
    if outcome.valuation is not None:
        _print_valuation(outcome.valuation, outcome.portfolio.initial_cash)


def _print_valuation(valuation: PaperValuation, initial_cash: Decimal | float) -> None:
    initial = Decimal(str(initial_cash))
    total = valuation.total_value
    print(
        f"가상 계좌: 총평가 {_won(total)} "
        f"(현금 {_won(valuation.cash)}, 주식 {_won(valuation.holdings_value)}), "
        f"누적 수익률 {_pct((total / initial - 1) * 100)}"
    )
    if valuation.stale_codes:
        print(f"현재가 조회 실패로 평균가 평가: {', '.join(valuation.stale_codes)}")


def _print_paper_report(report: PaperReport) -> None:
    """모의투자 성과 요약을 표시한다."""

    def fmt(moment: datetime | None) -> str:
        if moment is None:
            return "-"
        return moment.astimezone(KOREA_TIMEZONE).strftime("%Y-%m-%d %H:%M")

    print(f"모의투자 시작 {fmt(report.started_at)} / 기준 {fmt(report.as_of)} KST")
    print(
        f"초기 자금 {_won(report.initial_cash)} -> 총평가 {_won(report.total_value)} "
        f"(현금 {_won(report.cash)}, 주식 {_won(report.holdings_value)})"
    )
    print(
        f"누적 수익률 {_pct(report.total_return_percent)} | "
        f"손익 {_won(report.total_value - report.initial_cash)} "
        f"(실현 {_won(report.realized_pnl)}, 평가 {_won(report.unrealized_pnl)}) | "
        f"수수료·세금 {_won(report.fees_and_taxes)} | "
        f"최대 낙폭 {_pct(report.max_drawdown_percent)}"
    )
    benchmarks = []
    for label, value in (
        ("KOSPI", report.kospi_return_percent),
        ("관심종목 균등 보유", report.watchlist_return_percent),
    ):
        if value is not None:
            excess = _pct(report.total_return_percent - value, True)
            benchmarks.append(f"{label} {_pct(value)} (초과 {excess})")
    comparison = " | ".join(benchmarks) or "평가 기록 2회 이상부터 표시"
    print(f"비교: {comparison}")
    counts = ", ".join(f"{k} {v}" for k, v in sorted(report.status_counts.items()))
    print(
        f"사이클: {counts or '없음'} "
        f"(모델 호출 {report.model_call_count}회, 재사용 {report.reused_count}회) | "
        f"주문: 체결 {report.filled_count}건, "
        f"미체결 {report.unfilled_count}건"
    )
    if report.stale_codes:
        stale = ", ".join(report.stale_codes)
        print(f"마지막 평가에 현재가가 없어 평균가로 평가: {stale}")
    print("보유 종목:")
    if not report.holdings:
        print("  없음")
    for item in report.holdings:
        print(
            f"  {item.code} {item.name} {item.quantity}주 "
            f"평균 {_won(item.average_price)} 현재 {_won(item.price)} "
            f"평가 {_won(item.market_value)} "
            f"손익 {_won(item.unrealized_pnl)} ({_pct(item.return_percent)}) "
            f"비중 {item.weight_percent:.1f}%"
        )
    print("일별 마지막 평가:")
    if not report.daily:
        print("  없음")
    for day in report.daily:
        print(
            f"  {day.date} {_won(day.total_value)} "
            f"일간 {_pct(day.daily_return_percent)} "
            f"누적 {_pct(day.cumulative_return_percent)}"
        )
    print("최근 주문:")
    if not report.recent_fills:
        print("  없음")
    for row in report.recent_fills:
        if row.status == FillStatus.FILLED:
            detail = f"체결 {row.quantity}주 @ {_won(row.fill_price)}"
            if row.realized_pnl is not None:
                detail += f" 실현 {_won(row.realized_pnl)}"
        else:
            detail = f"미체결 {row.requested_quantity}주 지정가 {_won(row.limit_price)}"
        note = f" - {row.note}" if row.note else ""
        print(
            f"  {fmt(row.at)} {row.code} {row.name} {row.action.upper()} "
            f"[{row.source}] {detail}{note}"
        )


def _execute_backtest_score(
    args: argparse.Namespace, factory: Callable[..., _Runtime[None]]
) -> int:
    """채점 설정과 런타임을 만들고 저장된 결과를 평가한다."""

    runtime: _Runtime[None] | None = None
    try:
        settings = load_settings(args.config)
        runtime = factory(settings, args)
        runtime.run()
        return 0
    except Exception as error:
        print(f"백테스트 채점 실패: {type(error).__name__}", file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            runtime.close()


def _execute_backtest(
    args: argparse.Namespace, factory: Callable[..., _Runtime[int]]
) -> int:
    """설정과 런타임을 만들고 백테스트 종료 코드를 반환한다."""

    try:
        settings = load_settings(args.config)
    except Exception as error:
        print(f"백테스트 준비 실패: {type(error).__name__}", file=sys.stderr)
        return 1
    try:
        runtime = factory(settings, args)
    except Exception as error:
        print(f"백테스트 준비 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    try:
        return runtime.run()
    except Exception as error:
        print(f"백테스트 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        runtime.close()


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
    proposals = (
        consensus.proposals if consensus is not None else result.reused_proposals or []
    )
    proposal_count = len(proposals)
    disagreement_count = len(consensus.disagreements) if consensus is not None else 0
    verdicts = result.verdicts or []
    approved = [verdict for verdict in verdicts if verdict.approved]
    rejected = [verdict for verdict in verdicts if not verdict.approved]
    missing_info = consensus.missing_info if consensus is not None else []
    print(f"상태: {result.status.value}")
    print(f"합의 채택: {proposal_count}건, 불일치: {disagreement_count}건")
    if result.status is CycleStatus.REUSED:
        print(
            f"모델 판단 재사용: run_id {result.reuse_source_run_id} "
            f"({result.reuse_reason or '사유 없음'})"
        )
    elif result.reuse_reason is not None:
        call_label = "호출" if result.model_called else "미호출"
        print(f"모델 판단 {call_label}: {result.reuse_reason}")
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
    basis = _consensus_basis_label(proposal.consensus_basis)
    if basis:
        return f"합의({basis})"
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
        print("상대강도 상위 후보:")
        if result.top_candidates:
            for rank, candidate in enumerate(result.top_candidates, start=1):
                expected = (
                    "—"
                    if candidate.expected_excess_return_10d is None
                    else f"{candidate.expected_excess_return_10d}%p"
                )
                print(
                    f"{rank}. {_label(candidate.code, names)} "
                    f"score {candidate.score} / 10일 기대 초과수익 {expected} "
                    f"(claude {candidate.claude_score}, codex {candidate.codex_score})"
                )
        else:
            print("없음")
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
    if item.price_adjusted and item.original_invalidation_price is not None:
        invalidation += f" (ATR 보정, 원래 {_price(item.original_invalidation_price)})"
    if item.rating is Action.SELL:
        levels = f"예상 하락가 {target}, {invalidation} 위로 오르면 매도 의견 철회"
    else:
        levels = f"목표가 {target} 손절가 {invalidation}"
    basis = _consensus_basis_label(item.consensus_basis)
    basis_suffix = f" ({basis})" if basis else ""
    print(
        f"  {rank}. {_label(item.code, names)}{basis_suffix} "
        f"확신도 {item.confidence:.2f} {levels}"
    )
    print(f"     근거: {item.reason}")
    for warning in item.price_warnings or []:
        print(f"     가격 경고: {warning}")


def _consensus_basis_label(basis: ConsensusBasis | None) -> str | None:
    """한쪽 BUY 채택 사유를 CLI 표시용 문구로 만든다."""

    if basis is ConsensusBasis.CLAUDE_ONLY:
        return "claude 단독 buy"
    if basis is ConsensusBasis.CODEX_ONLY:
        return "codex 단독 buy"
    return None


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


def _positive_list(value: str) -> tuple[int, ...]:
    """콤마로 구분한 양수 horizon 목록을 읽는다."""

    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "horizons는 양의 정수 목록이어야 합니다."
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("horizons는 양의 정수 목록이어야 합니다.")
    return tuple(dict.fromkeys(values))


def _hold_bands(value: str) -> dict[int, float]:
    """horizon:percentage 쌍을 콤마로 구분해 읽는다."""

    try:
        result = {
            int(pair.split(":", 1)[0].strip()): float(pair.split(":", 1)[1].strip())
            for pair in value.split(",")
        }
    except (IndexError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "hold-bands 형식은 5:2.5,10:3.5 입니다."
        ) from error
    if not result or any(horizon <= 0 or band < 0 for horizon, band in result.items()):
        raise argparse.ArgumentTypeError("horizon과 hold 밴드는 음수가 될 수 없습니다.")
    return result


def _worker_count(value: str) -> int:
    """백테스트 동시 실행 수를 최대 2로 제한한다."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workers는 1 또는 2여야 합니다.") from error
    if parsed not in (1, 2):
        raise argparse.ArgumentTypeError("workers는 1 또는 2여야 합니다.")
    return parsed


def _date_arg(value: str) -> str:
    """YYYYMMDD 날짜 형식을 확인한다."""

    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError("날짜는 YYYYMMDD 형식이어야 합니다.")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise argparse.ArgumentTypeError("유효한 날짜가 아닙니다.") from error
    return value


def _comma_values(value: str) -> list[str]:
    """콤마로 구분한 6자리 종목 코드를 확인한다."""

    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(len(item) != 6 or not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError("universe는 6자리 코드 목록이어야 합니다.")
    return values


def _decision_time(value: str) -> time:
    """시각 인자를 HH:MM 형식으로 읽는다."""

    if len(value) != 5 or value[2] != ":":
        raise argparse.ArgumentTypeError("시각은 HH:MM 형식이어야 합니다.")
    try:
        return time.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("시각은 HH:MM 형식이어야 합니다.") from error


def _exit_code(status: CycleStatus) -> int:
    """요구된 성공·모델 실패·기타 실패 종료 코드를 반환한다."""

    if status in (CycleStatus.COMPLETED, CycleStatus.REUSED, CycleStatus.SKIPPED):
        return 0
    if status is CycleStatus.MODEL_FAILED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
