"""Public API surface is importable and stable."""

import langchain_engram


def test_exports() -> None:
    assert set(langchain_engram.__all__) == {
        "EngramMiddleware",
        "EngramStore",
        "__version__",
        "create_memory_tools",
    }
    for name in langchain_engram.__all__:
        assert hasattr(langchain_engram, name)
