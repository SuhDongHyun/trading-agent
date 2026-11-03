"""trading-api JSON 응답의 디스크 캐시."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, Protocol


class JsonRequester(Protocol):
    """JSON 요청 클라이언트의 최소 인터페이스."""

    def request_json(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any: ...

    def close(self) -> None: ...


def atomic_write_text(path: str | Path, value: str) -> None:
    """임시 파일을 완성한 뒤 교체해 기존 파일을 보존한다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(value)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


class ResponseCache:
    """성공 응답만 원자적으로 저장하고 요청을 직렬화한다."""

    def __init__(self, inner: JsonRequester, cache_dir: str | Path) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        self._thread_counts: dict[int, list[int]] = {}

    def request_json(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """요청 내용을 해시한 파일에서 읽고, 없으면 응답을 저장한다."""

        payload = json.dumps(
            [method.upper(), path, json_body],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(payload).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        thread_id = get_ident()
        with self._lock:
            counts = self._thread_counts.setdefault(thread_id, [0, 0])
            if cache_file.is_file():
                try:
                    value = json.loads(cache_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    cache_file.unlink(missing_ok=True)
                else:
                    self.hits += 1
                    counts[0] += 1
                    return value
            self.misses += 1
            counts[1] += 1
            value = self.inner.request_json(method, path, json_body, timeout_seconds)
            encoded = json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            atomic_write_text(cache_file, encoded)
            return value

    def stats_for_current_thread(self) -> tuple[int, int]:
        """현재 백테스트 날짜 실행 스레드의 적중·미스 누계를 반환한다."""

        with self._lock:
            values = self._thread_counts.get(get_ident(), [0, 0])
            return values[0], values[1]

    def close(self) -> None:
        """내부 HTTP 클라이언트를 닫는다."""

        self.inner.close()
