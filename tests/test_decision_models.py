import json
import os
import signal
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from time import sleep

import pytest

from trading_agent.application.decision_schema import DECISION_RESPONSE_SCHEMA
from trading_agent.application.errors import DecisionModelError
from trading_agent.infrastructure.command_runner import (
    CommandResult,
    CommandRunner,
    CommandTimeoutError,
    SubprocessCommandRunner,
)
from trading_agent.infrastructure.config import ClaudeSettings, CodexSettings
from trading_agent.infrastructure.decision_models import (
    ClaudeCliDecisionModel,
    CodexCliDecisionModel,
)

VALID_RESPONSE = {
    "summary": "uptrend",
    "proposals": [
        {
            "code": "005930",
            "action": "buy",
            "quantity": 2,
            "limit_price": 70000,
            "confidence": 0.85,
            "reason": "trend",
        }
    ],
    "missing_info": [],
}


class FakeCommandRunner(CommandRunner):
    def __init__(
        self,
        response: Callable[[Sequence[str], str, str], CommandResult],
    ) -> None:
        self.response = response
        self.calls: list[tuple[list[str], str, str, float]] = []

    def run(
        self,
        argv: Sequence[str],
        input_text: str,
        cwd: str | Path,
        timeout_seconds: float,
    ) -> CommandResult:
        directory = str(cwd)
        self.calls.append((list(argv), input_text, directory, timeout_seconds))
        return self.response(argv, input_text, directory)


def completed(
    stdout: str,
    *,
    stderr: str = "diagnostic",
    exit_code: int = 0,
) -> CommandResult:
    return CommandResult(stdout, stderr, exit_code, 1.25)


def claude_envelope(
    response: object = VALID_RESPONSE,
    *,
    is_error: bool = False,
    result_text: str | None = None,
) -> str:
    # 실제 CLI result envelope 형태이며 session과 비용 값은 테스트용 더미다.
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": is_error,
            "result": result_text if result_text is not None else json.dumps(response),
            "structured_output": response,
            "stop_reason": "tool_use",
            "num_turns": 2,
            "session_id": "fixture-session-id",
            "total_cost_usd": 0.0123,
        }
    )


def test_claude_adapter_uses_structured_output_and_expected_argv(
    tmp_path: Path,
) -> None:
    def claude_response(_argv: Sequence[str], _prompt: str, cwd: str) -> CommandResult:
        assert Path(cwd).is_dir()
        return completed(
            claude_envelope(
                result_text=json.dumps(
                    {
                        "summary": "result string must be ignored",
                        "proposals": [],
                        "missing_info": [],
                    }
                )
            )
        )

    runner = FakeCommandRunner(claude_response)
    model = ClaudeCliDecisionModel(
        ClaudeSettings(command="claude-test", timeout_seconds=45), runner
    )

    result = model.decide("prompt body")

    argv, prompt, cwd, timeout = runner.calls[0]
    assert argv[:8] == [
        "claude-test",
        "-p",
        "--model",
        "claude-opus-5-5",
        "--effort",
        "high",
        "--output-format",
        "json",
    ]
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema == DECISION_RESPONSE_SCHEMA
    assert argv[argv.index("--tools") + 1] == ""
    assert "--no-session-persistence" in argv
    assert prompt == "prompt body"
    assert Path(cwd) != tmp_path
    assert timeout == 45
    assert result.succeeded
    assert result.stdout.startswith('{"type": "result"')
    assert result.stderr == "diagnostic"
    assert result.response_text is not None
    assert result.decision is not None
    assert result.decision.summary == "uptrend"
    assert result.decision.proposals[0].quantity == 2


@pytest.mark.parametrize(
    ("stdout", "exit_code", "error_fragment"),
    [
        (claude_envelope(is_error=True), 0, "is_error"),
        ("not JSON", 0, "Expecting value"),
        (claude_envelope({"summary": "missing keys"}), 0, "누락 필드"),
        ("ignored", 17, "exit code 17"),
    ],
)
def test_claude_adapter_fails_closed_for_bad_responses(
    stdout: str, exit_code: int, error_fragment: str
) -> None:
    runner = FakeCommandRunner(
        lambda _argv, _prompt, _cwd: completed(stdout, exit_code=exit_code)
    )
    model = ClaudeCliDecisionModel(ClaudeSettings(), runner)

    with pytest.raises(DecisionModelError, match=error_fragment) as raised:
        model.decide("prompt")

    assert raised.value.run.stdout == stdout
    assert raised.value.run.stderr == "diagnostic"
    assert raised.value.run.exit_code == exit_code


def test_claude_adapter_preserves_timeout_output() -> None:
    def timeout(_argv: Sequence[str], _prompt: str, _cwd: str) -> CommandResult:
        raise CommandTimeoutError("partial stdout", "partial stderr", 45.0)

    model = ClaudeCliDecisionModel(ClaudeSettings(), FakeCommandRunner(timeout))

    with pytest.raises(DecisionModelError, match="timeout") as raised:
        model.decide("prompt")

    assert raised.value.run.stdout == "partial stdout"
    assert raised.value.run.stderr == "partial stderr"
    assert raised.value.run.exit_code is None
    assert raised.value.run.duration_seconds == 45.0


def test_codex_adapter_uses_schema_output_file_and_isolated_working_directory() -> None:
    def write_response(argv: Sequence[str], _prompt: str, cwd: str) -> CommandResult:
        schema_path = Path(argv[argv.index("--output-schema") + 1])
        output_path = Path(argv[argv.index("-o") + 1])
        assert schema_path.parent == Path(cwd)
        assert json.loads(schema_path.read_text(encoding="utf-8")) == (
            DECISION_RESPONSE_SCHEMA
        )
        # Codex without --json writes the schema-shaped response to -o and stdout.
        raw_response = json.dumps(VALID_RESPONSE)
        output_path.write_text(raw_response, encoding="utf-8")
        return completed(raw_response)

    runner = FakeCommandRunner(write_response)
    model = CodexCliDecisionModel(
        CodexSettings(command="codex-test", timeout_seconds=61), runner
    )

    result = model.decide("same prompt")

    argv, prompt, cwd, timeout = runner.calls[0]
    assert argv[:9] == [
        "codex-test",
        "exec",
        "-m",
        "gpt-6.1-sol",
        "-c",
        "model_reasoning_effort=high",
        "--output-schema",
        argv[7],
        "-o",
    ]
    assert argv[-3:] == ["--ephemeral", "--skip-git-repo-check", "-"]
    assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
    assert prompt == "same prompt"
    assert timeout == 61
    assert result.succeeded
    assert result.stdout == result.response_text
    assert result.response_text is not None
    assert result.decision is not None
    assert result.decision.proposals[0].code == "005930"


@pytest.mark.parametrize("response_text", ["not JSON", '{"summary":"bad"}'])
def test_codex_adapter_rejects_invalid_output_file(response_text: str) -> None:
    def write_response(argv: Sequence[str], _prompt: str, _cwd: str) -> CommandResult:
        Path(argv[argv.index("-o") + 1]).write_text(response_text, encoding="utf-8")
        return completed("event log")

    model = CodexCliDecisionModel(CodexSettings(), FakeCommandRunner(write_response))

    with pytest.raises(DecisionModelError):
        model.decide("prompt")


def test_subprocess_runner_removes_api_keys_and_keeps_oauth_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    removed = [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
    ]
    for variable in removed:
        monkeypatch.setenv(variable, "secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-token")

    class FakeProcess:
        returncode = 0

        def communicate(
            self, input: str | None, timeout: float | None
        ) -> tuple[str, str]:
            assert input == "stdin prompt"
            assert timeout == 10
            return "out", "err"

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert all(variable not in environment for variable in removed)
        assert environment["CLAUDE_CODE_OAUTH_TOKEN"] == "subscription-token"
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == tmp_path
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == subprocess.PIPE
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert argv == ["fake-cli", "--flag"]
        return FakeProcess()

    monkeypatch.setattr(
        "trading_agent.infrastructure.command_runner.subprocess.Popen",
        fake_popen,
    )

    result = SubprocessCommandRunner().run(
        ["fake-cli", "--flag"], "stdin prompt", tmp_path, 10
    )

    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.exit_code == 0


def test_timeout_terminates_child_process_group_and_preserves_output(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "background-sleep.pid"
    sleep_seconds = "37.413"
    script = (
        'printf "partial output\\n"; '
        'sleep "$1" & child_pid=$!; '
        'printf "%s\\n" "$child_pid" > "$2"; '
        "wait"
    )
    child_pid: int | None = None
    try:
        with pytest.raises(CommandTimeoutError) as raised:
            SubprocessCommandRunner().run(
                [
                    "bash",
                    "-c",
                    script,
                    "process-group-test",
                    sleep_seconds,
                    str(pid_file),
                ],
                "",
                tmp_path,
                0.35,
            )

        assert "partial output" in raised.value.stdout
        assert pid_file.is_file()
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        for _ in range(30):
            if not _is_process_running(child_pid):
                break
            sleep(0.02)
        assert not _is_process_running(child_pid)
    finally:
        if child_pid is None and pid_file.is_file():
            child_pid = int(pid_file.read_text(encoding="utf-8"))
        if child_pid is not None and _is_process_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def _is_process_running(process_id: int) -> bool:
    """Linux의 zombie는 실행 중인 프로세스로 세지 않고 PID 생존을 검사한다."""

    stat_path = Path(f"/proc/{process_id}/stat")
    try:
        stat = stat_path.read_text(encoding="utf-8")
    except OSError:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False
        return True
    state = stat.rsplit(")", maxsplit=1)[1].strip().split(maxsplit=1)[0]
    return state != "Z"
