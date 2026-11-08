import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep
from typing import cast

import pytest

from trading_agent.application.decision_schema import DECISION_RESPONSE_SCHEMA
from trading_agent.application.errors import DecisionModelError
from trading_agent.application.model_tasks import DECISION_TASK, RECOMMENDATION_TASK
from trading_agent.domain.recommendation import ModelRecommendation
from trading_agent.infrastructure.command_runner import (
    CommandExecutionError,
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


def claude_result_event(
    response: object = VALID_RESPONSE,
    *,
    is_error: bool = False,
    result_text: str | None = None,
    num_turns: int = 2,
) -> str:
    # 실제 CLI result 이벤트 형태이며 session과 비용 값은 테스트용 더미다.
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": is_error,
            "result": result_text if result_text is not None else json.dumps(response),
            "structured_output": response,
            "stop_reason": "tool_use",
            "num_turns": num_turns,
            "session_id": "fixture-session-id",
            "total_cost_usd": 0.0123,
        }
    )


def claude_stream(*events: dict[str, object]) -> str:
    return "\n".join(json.dumps(event) for event in events)


def assert_strict_object_schemas(schema: object) -> None:
    """Ensure every object schema passed to Codex requires all its properties."""

    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            assert set(schema.get("required", [])) == set(properties)
        for value in schema.values():
            assert_strict_object_schemas(value)
    elif isinstance(schema, list):
        for value in schema:
            assert_strict_object_schemas(value)


def test_claude_adapter_uses_structured_output_and_expected_argv(
    tmp_path: Path,
) -> None:
    stream = claude_stream(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "StructuredOutput",
                        "input": {"summary": "incomplete"},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "is_error": True,
                        "content": "Input failed schema validation",
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "retry"}]},
        },
        json.loads(
            claude_result_event(
                result_text=json.dumps(
                    {
                        "summary": "result string must be ignored",
                        "proposals": [],
                        "missing_info": [],
                    }
                ),
                num_turns=3,
            )
        ),
    )

    def claude_response(_argv: Sequence[str], _prompt: str, cwd: str) -> CommandResult:
        assert Path(cwd).is_dir()
        return completed(stream)

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
        "stream-json",
    ]
    assert "--verbose" in argv
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema == DECISION_RESPONSE_SCHEMA
    assert argv[argv.index("--tools") + 1] == ""
    assert "--no-session-persistence" in argv
    assert prompt == "prompt body"
    assert Path(cwd) != tmp_path
    assert timeout == 45
    assert result.succeeded
    stream_events = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.stdout == stream
    assert [event["type"] for event in stream_events] == [
        "assistant",
        "user",
        "assistant",
        "result",
    ]
    assert stream_events[-1]["num_turns"] == 3
    assert result.stderr == "diagnostic"
    assert result.response_text is not None
    assert result.decision is not None
    assert result.decision.summary == "uptrend"
    assert result.decision.proposals[0].quantity == 2


@pytest.mark.parametrize(
    ("stdout", "exit_code", "error_fragment"),
    [
        (claude_result_event(is_error=True), 0, "is_error"),
        ("not JSON", 0, "Expecting value"),
        (claude_result_event({"summary": "missing keys"}), 0, "누락 필드"),
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


def test_claude_adapter_requires_result_event() -> None:
    stdout = claude_stream({"type": "assistant", "message": {"content": []}})
    model = ClaudeCliDecisionModel(
        ClaudeSettings(), FakeCommandRunner(lambda _a, _p, _c: completed(stdout))
    )

    with pytest.raises(DecisionModelError, match="result 이벤트") as raised:
        model.decide("prompt")

    assert raised.value.run.stdout == stdout


def test_claude_adapter_requires_structured_output_in_result_event() -> None:
    stdout = claude_stream({"type": "result", "is_error": False})
    model = ClaudeCliDecisionModel(
        ClaudeSettings(), FakeCommandRunner(lambda _a, _p, _c: completed(stdout))
    )

    with pytest.raises(DecisionModelError, match="structured_output") as raised:
        model.decide("prompt")

    assert raised.value.run.stdout == stdout


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
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema == DECISION_RESPONSE_SCHEMA
        assert_strict_object_schemas(schema)
        # Codex --json writes JSONL events to stdout and the final answer to -o.
        raw_response = json.dumps(VALID_RESPONSE)
        output_path.write_text(raw_response, encoding="utf-8")
        return completed('{"type":"turn.completed"}')

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
    assert result.stdout == '{"type":"turn.completed"}'
    assert result.response_text == json.dumps(VALID_RESPONSE)
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
        pid = 123_456
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

    runner = SubprocessCommandRunner()
    result = runner.run(["fake-cli", "--flag"], "stdin prompt", tmp_path, 10)

    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.exit_code == 0
    assert runner._processes == {}


@pytest.mark.parametrize(
    ("ignore_sigterm", "expected_exit_code"),
    [
        (False, -signal.SIGTERM),
        (True, -signal.SIGKILL),
    ],
    ids=["term", "kill-after-grace"],
)
def test_terminate_all_stops_registered_process_groups(
    tmp_path: Path, ignore_sigterm: bool, expected_exit_code: int
) -> None:
    runner = SubprocessCommandRunner()
    ready_file = tmp_path / "ready"
    install_handler = (
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_sigterm else ""
    )
    child_code = (
        f"import signal, time; {install_handler}"
        f"open({str(ready_file)!r}, 'w').close(); time.sleep(60)"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            runner.run,
            [sys.executable, "-c", child_code],
            "",
            tmp_path,
            60,
        )
        try:
            deadline = monotonic() + 3
            while not ready_file.exists() and monotonic() < deadline:
                sleep(0.01)
            assert runner._processes
            assert ready_file.is_file()

            started = monotonic()
            runner.terminate_all()
            result = future.result(timeout=2)

            assert monotonic() - started < 1
            assert result.exit_code == expected_exit_code
            assert runner._processes == {}
        finally:
            runner.terminate_all()
            if not future.done():
                future.result(timeout=2)


def test_run_after_terminate_all_is_rejected_without_spawning(
    monkeypatch, tmp_path: Path
) -> None:
    runner = SubprocessCommandRunner()
    runner.terminate_all()
    popen_calls = []

    def unexpected_popen(*_args, **_kwargs):
        popen_calls.append(True)
        raise AssertionError("종료 중인 runner는 Popen을 호출하면 안 됩니다.")

    monkeypatch.setattr(
        "trading_agent.infrastructure.command_runner.subprocess.Popen",
        unexpected_popen,
    )

    with pytest.raises(CommandExecutionError, match="종료 중"):
        runner.run(["fake-cli"], "", tmp_path, 10)

    assert popen_calls == []


def test_terminate_all_ignores_stale_registry_groups_within_grace(
    monkeypatch,
) -> None:
    process_group_id = 123_456_789
    runner = SubprocessCommandRunner()
    runner._processes[process_group_id] = cast(subprocess.Popen[str], object())
    signals: list[tuple[int, int]] = []

    def missing_group(group_id: int, signal_number: int) -> None:
        signals.append((group_id, signal_number))
        raise ProcessLookupError

    monkeypatch.setattr(
        "trading_agent.infrastructure.command_runner.os.killpg", missing_group
    )

    started = monotonic()
    runner.terminate_all()

    assert monotonic() - started < 0.2
    assert signals == [
        (process_group_id, signal.SIGTERM),
        (process_group_id, 0),
    ]


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


def test_claude_adapter_isolates_user_configuration_and_sets_role() -> None:
    runner = FakeCommandRunner(lambda _a, _p, _c: completed(claude_result_event()))
    model = ClaudeCliDecisionModel(ClaudeSettings(), runner)

    model.decide("prompt")

    argv = runner.calls[0][0]
    for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in argv
    assert "--bare" not in argv
    assert argv[argv.index("--system-prompt") + 1] == DECISION_TASK.system_prompt


def test_codex_adapter_isolates_user_configuration_and_sets_role() -> None:
    seen: dict[str, str] = {}

    def write_response(argv: Sequence[str], _prompt: str, _cwd: str) -> CommandResult:
        setting = next(
            argv[index + 1]
            for index, item in enumerate(argv)
            if item == "-c" and argv[index + 1].startswith("model_instructions_file=")
        )
        instructions = Path(json.loads(setting.split("=", 1)[1]))
        seen["instructions"] = instructions.read_text(encoding="utf-8")
        Path(argv[argv.index("-o") + 1]).write_text(
            json.dumps(VALID_RESPONSE), encoding="utf-8"
        )
        return completed('{"type":"turn.completed"}')

    runner = FakeCommandRunner(write_response)
    CodexCliDecisionModel(CodexSettings(), runner).decide("prompt")

    argv = runner.calls[0][0]
    for flag in ("--json", "--ignore-user-config", "--ignore-rules"):
        assert flag in argv
    disabled = {argv[i + 1] for i, item in enumerate(argv) if item == "--disable"}
    assert {"hooks", "plugins", "apps", "multi_agent"} <= disabled
    assert "project_doc_max_bytes=0" in argv
    assert seen["instructions"] == DECISION_TASK.system_prompt


def test_adapters_use_the_given_task_schema_and_parser() -> None:
    response = {
        "summary": "s",
        "recommendations": [
            {
                "code": "005930",
                "rating": "hold",
                "confidence": 0.5,
                "target_price": None,
                "invalidation_price": None,
                "reason": "r",
                "score": 0.5,
                "expected_excess_return_10d": 1.25,
                "drivers": ["trend_up"],
                "key_risk": {"type": "event_risk", "note": "발표 일정"},
            }
        ],
        "missing_info": [],
    }
    claude_runner = FakeCommandRunner(
        lambda _a, _p, _c: completed(claude_result_event(response))
    )

    def write_response(argv: Sequence[str], _prompt: str, _cwd: str) -> CommandResult:
        schema = json.loads(
            Path(argv[argv.index("--output-schema") + 1]).read_text(encoding="utf-8")
        )
        assert schema == RECOMMENDATION_TASK.response_schema
        assert_strict_object_schemas(schema)
        Path(argv[argv.index("-o") + 1]).write_text(
            json.dumps(response), encoding="utf-8"
        )
        return completed("events")

    claude = ClaudeCliDecisionModel(
        ClaudeSettings(), claude_runner, RECOMMENDATION_TASK
    ).decide("p")
    codex = CodexCliDecisionModel(
        CodexSettings(), FakeCommandRunner(write_response), RECOMMENDATION_TASK
    ).decide("p")

    claude_schema = json.loads(
        claude_runner.calls[0][0][claude_runner.calls[0][0].index("--json-schema") + 1]
    )
    assert claude_schema == RECOMMENDATION_TASK.response_schema
    for run in (claude, codex):
        assert isinstance(run.decision, ModelRecommendation)
        assert run.decision.recommendations[0].target_price is None
