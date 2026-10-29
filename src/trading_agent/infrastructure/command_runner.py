"""환경과 시간 제한을 적용해 외부 CLI를 실행한다."""

import os
import signal
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter, sleep

_REMOVED_ENVIRONMENT_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
)
_PROCESS_TERMINATION_GRACE_SECONDS = 0.2
_PROCESS_GROUP_POLL_SECONDS = 0.01


@dataclass(frozen=True, slots=True)
class CommandResult:
    """완료된 프로세스의 원문 출력과 실행 시간을 담는다."""

    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float


class CommandTimeoutError(RuntimeError):
    """시간 제한으로 종료된 명령과 확보된 원문 출력."""

    def __init__(self, stdout: str, stderr: str, duration_seconds: float) -> None:
        super().__init__("명령 실행 시간이 제한을 초과했습니다.")
        self.stdout = stdout
        self.stderr = stderr
        self.duration_seconds = duration_seconds


class CommandExecutionError(RuntimeError):
    """운영체제가 외부 명령을 시작하지 못했다."""

    def __init__(self, message: str, duration_seconds: float) -> None:
        super().__init__(message)
        self.duration_seconds = duration_seconds


class CommandRunner(ABC):
    """CLI 호출을 격리된 실행기로 추상화한다."""

    @abstractmethod
    def run(
        self,
        argv: Sequence[str],
        input_text: str,
        cwd: str | Path,
        timeout_seconds: float,
    ) -> CommandResult:
        """shell 없이 stdin과 제한된 환경을 사용해 명령을 실행한다."""


class SubprocessCommandRunner(CommandRunner):
    """새 프로세스 세션에서 명령과 자식 프로세스를 함께 관리한다."""

    def run(
        self,
        argv: Sequence[str],
        input_text: str,
        cwd: str | Path,
        timeout_seconds: float,
    ) -> CommandResult:
        """자격 증명 키를 제거하고 timeout 시 프로세스 그룹을 종료한다."""

        environment = os.environ.copy()
        for variable in _REMOVED_ENVIRONMENT_VARIABLES:
            environment.pop(variable, None)
        started = perf_counter()
        try:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=environment,
                shell=False,
                start_new_session=True,
            )
        except OSError as error:
            raise CommandExecutionError(str(error), perf_counter() - started) from error

        try:
            stdout, stderr = process.communicate(
                input=input_text, timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired as error:
            stdout, stderr = _terminate_process_group(process, error)
            raise CommandTimeoutError(
                stdout,
                stderr,
                perf_counter() - started,
            ) from error

        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
            duration_seconds=perf_counter() - started,
        )


def _terminate_process_group(
    process: subprocess.Popen[str], timeout_error: subprocess.TimeoutExpired
) -> tuple[str, str]:
    """TERM을 보내고 유예 후 그룹 내 잔여 프로세스를 KILL한다."""

    process_group_id = process.pid
    _signal_process_group(process_group_id, signal.SIGTERM)
    deadline = monotonic() + _PROCESS_TERMINATION_GRACE_SECONDS
    try:
        stdout, stderr = process.communicate(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired as termination_timeout:
        timeout_error = termination_timeout
        _signal_process_group(process_group_id, signal.SIGKILL)
        stdout, stderr = process.communicate()
    else:
        _wait_for_process_group(process_group_id, deadline)
        if _process_group_exists(process_group_id):
            _signal_process_group(process_group_id, signal.SIGKILL)

    return (
        _combine_partial_output(stdout, timeout_error.stdout),
        _combine_partial_output(stderr, timeout_error.stderr),
    )


def _wait_for_process_group(process_group_id: int, deadline: float) -> None:
    """종료 유예 시간 동안 프로세스 그룹이 정리되는지 확인한다."""

    while _process_group_exists(process_group_id) and monotonic() < deadline:
        sleep(_PROCESS_GROUP_POLL_SECONDS)


def _process_group_exists(process_group_id: int) -> bool:
    """프로세스 그룹에 아직 종료되지 않은 항목이 있는지 확인한다."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    """이미 종료된 그룹도 허용하며 지정 신호를 전체 그룹에 보낸다."""

    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass


def _output_text(value: str | bytes | None) -> str:
    """timeout 예외에 포함된 text 또는 byte 출력 문자열을 복원한다."""

    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _combine_partial_output(
    final_output: str | bytes | None, partial_output: str | bytes | None
) -> str:
    """timeout 전후 communicate 출력에서 중복 없이 모든 원문을 보존한다."""

    final_text = _output_text(final_output)
    partial_text = _output_text(partial_output)
    if not partial_text or partial_text in final_text:
        return final_text
    overlap = min(len(partial_text), len(final_text))
    while overlap and partial_text[-overlap:] != final_text[:overlap]:
        overlap -= 1
    return partial_text + final_text[overlap:]
