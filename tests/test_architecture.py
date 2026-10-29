import ast
import sys
from pathlib import Path

import pytest

_ALLOWED_TOP_LEVEL_MODULES = sys.stdlib_module_names | {"trading_agent"}


def _import_names(tree: ast.Module, package: str) -> list[str]:
    """AST에서 import된 모듈 이름을 절대 경로로 추출한다."""

    names: list[str] = []
    package_parts = package.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package_parts[: len(package_parts) - node.level + 1]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""

            if node.names:
                names.extend(
                    f"{base}.{alias.name}" if base else alias.name
                    for alias in node.names
                )
            elif base:
                names.append(base)
    return names


def _assert_domain_imports_allowed(tree: ast.Module) -> None:
    """도메인 import를 표준 라이브러리와 domain 내부로 제한한다."""

    for name in _import_names(tree, "trading_agent.domain"):
        top_level = name.split(".", maxsplit=1)[0]
        assert top_level in _ALLOWED_TOP_LEVEL_MODULES
        if top_level == "trading_agent":
            assert name == "trading_agent" or name.startswith("trading_agent.domain.")


def test_domain_imports_only_standard_library_or_domain_modules() -> None:
    domain_dir = Path(__file__).parents[1] / "src" / "trading_agent" / "domain"

    for source_file in domain_dir.glob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        _assert_domain_imports_allowed(tree)


def test_application_does_not_import_infrastructure() -> None:
    application_dir = (
        Path(__file__).parents[1] / "src" / "trading_agent" / "application"
    )

    for source_file in application_dir.glob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        imported_modules = _import_names(tree, "trading_agent.application")
        assert all(
            not module.startswith("trading_agent.infrastructure")
            for module in imported_modules
        )


@pytest.mark.parametrize(
    "source",
    [
        "from trading_agent import infrastructure",
        "import trading_agent.infrastructure",
        "from requests import get",
    ],
)
def test_domain_import_check_rejects_outer_and_third_party_modules(
    source: str,
) -> None:
    tree = ast.parse(source)

    with pytest.raises(AssertionError):
        _assert_domain_imports_allowed(tree)
