"""백테스트 결과 저장 포트의 파일 시스템 구현."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trading_agent.application.backtest import (
    BACKTEST_DATE_DIRECTORY,
    BACKTEST_DATE_RESULT_TEMPLATE,
    BACKTEST_MANIFEST_FILENAME,
    BACKTEST_PROMPT_TEMPLATE,
    BacktestResultStore,
    json_value,
)
from trading_agent.infrastructure.response_cache import atomic_write_text


class FileBacktestResultStore(BacktestResultStore):
    """manifest·일별 JSON·프롬프트를 임시 파일 교체 방식으로 저장한다."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def read_manifest(self) -> Mapping[str, Any] | None:
        """기존 manifest를 읽고 없으면 None을 반환한다."""

        path = self.output_dir / BACKTEST_MANIFEST_FILENAME
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("manifest.json은 object여야 합니다.")
        return value

    def read_date_result(self, date: str) -> Mapping[str, Any] | None:
        """기존 날짜 결과를 읽고 없으면 None을 반환한다."""

        path = self._date_path(date)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{date} 결과는 object여야 합니다.")
        return value

    def write_manifest(self, value: Mapping[str, Any]) -> None:
        """manifest를 JSON으로 원자적으로 교체한다."""

        self._write_json(self.output_dir / BACKTEST_MANIFEST_FILENAME, value)

    def write_date_result(self, date: str, value: Mapping[str, Any]) -> None:
        """날짜 결과 JSON을 원자적으로 교체한다."""

        self._write_json(self._date_path(date), value)

    def write_prompt(self, date: str, value: str) -> None:
        """해당 날짜의 정확한 모델 프롬프트를 원자적으로 저장한다."""

        path = (
            self.output_dir
            / BACKTEST_DATE_DIRECTORY
            / BACKTEST_PROMPT_TEMPLATE.format(date=date)
        )
        atomic_write_text(path, value)

    def _date_path(self, date: str) -> Path:
        return (
            self.output_dir
            / BACKTEST_DATE_DIRECTORY
            / BACKTEST_DATE_RESULT_TEMPLATE.format(date=date)
        )

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        encoded = json.dumps(
            json_value(value), ensure_ascii=False, indent=2, allow_nan=False
        )
        atomic_write_text(path, encoded + "\n")
