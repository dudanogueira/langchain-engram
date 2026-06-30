"""Placeholder used by CI to verify the package compiles and installs."""

import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Used in compilation testing to prevent dependency installation."""
