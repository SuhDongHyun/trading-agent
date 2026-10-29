"""Claude와 Codex CLI를 검증된 판단 모델 포트에 연결한다."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn

from trading_agent.application.decision_schema import (
    DECISION_RESPONSE_SCHEMA,
    parse_model_decision,
)
from trading_agent.application.errors import DecisionModelError
from trading_agent.application.ports import DecisionModel, ModelRun
from trading_agent.infrastructure.command_runner import (
    CommandExecutionError,
    CommandResult,
    CommandRunner,
    CommandTimeoutError,
)
from trading_agent.infrastructure.config import ClaudeSettings, CodexSettings


class ClaudeCliDecisionModel(DecisionModel):
    """구독 인증을 사용하는 헤드리스 Claude CLI 어댑터."""

    def __init__(self, settings: ClaudeSettings, runner: CommandRunner) -> None:
        self.settings = settings
        self.runner = runner

    def decide(self, prompt: str) -> ModelRun:
        """Claude structured output envelope를 도메인 판단으로 검증한다."""

        schema_text = json.dumps(DECISION_RESPONSE_SCHEMA, separators=(",", ":"))
        argv = [
            self.settings.command,
            "-p",
            "--model",
            self.settings.model,
            "--effort",
            self.settings.effort,
            "--output-format",
            "json",
            "--json-schema",
            schema_text,
            "--tools",
            "",
            "--no-session-persistence",
        ]
        with TemporaryDirectory(prefix="trading-agent-claude-") as working_directory:
            result = _run_command(
                self.runner,
                argv,
                prompt,
                working_directory,
                self.settings.timeout_seconds,
                "claude",
            )
        try:
            envelope = json.loads(result.stdout)
            if not isinstance(envelope, dict):
                raise ValueError("Claude 응답 envelope가 JSON object가 아닙니다.")
            if envelope.get("is_error") is True:
                raise ValueError("Claude CLI가 is_error=true를 반환했습니다.")
            if "structured_output" not in envelope:
                raise ValueError("Claude 응답에 structured_output이 없습니다.")
            structured_output = envelope["structured_output"]
            response_text = (
                structured_output
                if isinstance(structured_output, str)
                else json.dumps(structured_output, ensure_ascii=False)
            )
            decision = parse_model_decision(response_text, "claude")
        except (ValueError, TypeError) as error:
            _raise_invalid_response("claude", result, str(error), None)
        return ModelRun(
            model_id="claude",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            response_text=response_text,
            decision=decision,
        )


class CodexCliDecisionModel(DecisionModel):
    """일회성 read-only Codex CLI 실행 어댑터."""

    def __init__(self, settings: CodexSettings, runner: CommandRunner) -> None:
        self.settings = settings
        self.runner = runner

    def decide(self, prompt: str) -> ModelRun:
        """임시 schema와 마지막 응답 파일을 이용해 판단을 검증한다."""

        with TemporaryDirectory(prefix="trading-agent-codex-") as working_directory:
            schema_path = Path(working_directory) / "decision-schema.json"
            output_path = Path(working_directory) / "decision-output.json"
            schema_path.write_text(
                json.dumps(DECISION_RESPONSE_SCHEMA, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            argv = [
                self.settings.command,
                "exec",
                "-m",
                self.settings.model,
                "-c",
                f"model_reasoning_effort={self.settings.reasoning_effort}",
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "-",
            ]
            result = _run_command(
                self.runner,
                argv,
                prompt,
                working_directory,
                self.settings.timeout_seconds,
                "codex",
            )
            try:
                response_text = output_path.read_text(encoding="utf-8")
            except OSError as error:
                _raise_invalid_response(
                    "codex", result, f"Codex 응답 파일을 읽지 못했습니다: {error}", None
                )
            try:
                decision = parse_model_decision(response_text, "codex")
            except (ValueError, TypeError) as error:
                _raise_invalid_response("codex", result, str(error), response_text)

        return ModelRun(
            model_id="codex",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            response_text=response_text,
            decision=decision,
        )


def _run_command(
    runner: CommandRunner,
    argv: list[str],
    prompt: str,
    working_directory: str,
    timeout_seconds: float,
    model_id: str,
) -> CommandResult:
    """실행 오류와 비정상 종료를 원문이 포함된 포트 오류로 바꾼다."""

    try:
        result = runner.run(argv, prompt, working_directory, timeout_seconds)
    except CommandTimeoutError as error:
        run = ModelRun(
            model_id=model_id,
            stdout=error.stdout,
            stderr=error.stderr,
            exit_code=None,
            duration_seconds=error.duration_seconds,
            error="CLI timeout",
        )
        raise DecisionModelError("판단 CLI가 timeout 되었습니다.", run) from error
    except CommandExecutionError as error:
        run = ModelRun(
            model_id=model_id,
            stdout="",
            stderr=str(error),
            exit_code=None,
            duration_seconds=error.duration_seconds,
            error="CLI 실행 실패",
        )
        raise DecisionModelError("판단 CLI를 실행하지 못했습니다.", run) from error

    if result.exit_code != 0:
        run = ModelRun(
            model_id=model_id,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            error=f"CLI exit code {result.exit_code}",
        )
        raise DecisionModelError(
            f"판단 CLI가 정상 종료하지 않았습니다 (exit code {result.exit_code}).",
            run,
        )
    return result


def _raise_invalid_response(
    model_id: str,
    result: CommandResult,
    message: str,
    response_text: str | None,
) -> NoReturn:
    """응답 JSON과 도메인 검증 오류를 fail-closed 포트 오류로 만든다."""

    run = ModelRun(
        model_id=model_id,
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        duration_seconds=result.duration_seconds,
        response_text=response_text,
        error=message,
    )
    raise DecisionModelError(f"{model_id} 응답 검증 실패: {message}", run)
