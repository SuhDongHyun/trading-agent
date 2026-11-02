"""Claude와 Codex CLI를 검증된 판단 모델 포트에 연결한다."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn

from trading_agent.application.errors import DecisionModelError
from trading_agent.application.model_tasks import DECISION_TASK
from trading_agent.application.ports import DecisionModel, ModelRun, ModelTask
from trading_agent.infrastructure.command_runner import (
    CommandExecutionError,
    CommandResult,
    CommandRunner,
    CommandTimeoutError,
)
from trading_agent.infrastructure.config import ClaudeSettings, CodexSettings

# 사용자 전역 설정(hooks, MCP, skills, plugins, CLAUDE.md, slash command)을 끈다.
# --bare는 OAuth 구독 인증을 막으므로 쓰지 않는다.
CLAUDE_ISOLATION_ARGS = (
    "--safe-mode",
    "--strict-mcp-config",
    "--disable-slash-commands",
)

# 사용자 config.toml(MCP 서버, hook 신뢰 상태), execpolicy rules, 도구성 기능을 끈다.
# 인증은 계속 CODEX_HOME을 사용한다. 전역 AGENTS.md는 끌 수 있는 설정이 없어
# system prompt에서 적용 대상이 아니라고 명시한다.
CODEX_DISABLED_FEATURES = (
    "hooks",
    "plugins",
    "apps",
    "multi_agent",
    "goals",
    "image_generation",
    "browser_use",
    "computer_use",
    "in_app_browser",
)
CODEX_DISABLED_CONTEXT = (
    "project_doc_max_bytes=0",
    "include_environment_context=false",
    "include_permissions_instructions=false",
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
)


class ClaudeCliDecisionModel[T](DecisionModel[T]):
    """구독 인증을 사용하는 헤드리스 Claude CLI 어댑터."""

    def __init__(
        self,
        settings: ClaudeSettings,
        runner: CommandRunner,
        task: ModelTask[T] = DECISION_TASK,  # type: ignore[assignment]
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.task = task

    def decide(self, prompt: str) -> ModelRun[T]:
        """Claude structured output envelope를 도메인 판단으로 검증한다."""

        schema_text = json.dumps(self.task.response_schema, separators=(",", ":"))
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
            *CLAUDE_ISOLATION_ARGS,
            "--system-prompt",
            self.task.system_prompt,
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
            decision = self.task.parse(response_text, "claude")
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


class CodexCliDecisionModel[T](DecisionModel[T]):
    """일회성 read-only Codex CLI 실행 어댑터."""

    def __init__(
        self,
        settings: CodexSettings,
        runner: CommandRunner,
        task: ModelTask[T] = DECISION_TASK,  # type: ignore[assignment]
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.task = task

    def decide(self, prompt: str) -> ModelRun[T]:
        """임시 schema와 마지막 응답 파일을 이용해 판단을 검증한다."""

        with TemporaryDirectory(prefix="trading-agent-codex-") as working_directory:
            schema_path = Path(working_directory) / "response-schema.json"
            output_path = Path(working_directory) / "response-output.json"
            instructions_path = Path(working_directory) / "instructions.md"
            schema_path.write_text(
                json.dumps(self.task.response_schema, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            instructions_path.write_text(self.task.system_prompt, encoding="utf-8")
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
                "--json",
                "--ignore-user-config",
                "--ignore-rules",
                *_flag_pairs("--disable", CODEX_DISABLED_FEATURES),
                *_flag_pairs("-c", CODEX_DISABLED_CONTEXT),
                "-c",
                f"model_instructions_file={json.dumps(str(instructions_path))}",
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
                decision = self.task.parse(response_text, "codex")
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


def _flag_pairs(flag: str, values: tuple[str, ...]) -> list[str]:
    """같은 옵션을 값마다 반복한 인자 목록을 만든다."""

    return [item for value in values for item in (flag, value)]


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
