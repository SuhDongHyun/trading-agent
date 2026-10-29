from pathlib import Path

import pytest

from trading_agent.application.records import CycleStatus, DecisionCycleResult
from trading_agent.interface.cli import main


class FakeRuntime:
    def __init__(self, result: DecisionCycleResult) -> None:
        self.result = result
        self.closed = False

    def run(self) -> DecisionCycleResult:
        return self.result

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        (CycleStatus.COMPLETED, 0),
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
