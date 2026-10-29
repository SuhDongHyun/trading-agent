"""명령행 진입점."""

import argparse


def main() -> None:
    """현재 지원하는 명령행 옵션을 표시한다."""

    parser = argparse.ArgumentParser(
        prog="trading-agent",
        description="스냅샷 기반 paper trading 에이전트",
    )
    parser.parse_args()
