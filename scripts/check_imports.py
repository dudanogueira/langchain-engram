"""Import every module in the package to catch import-time errors in CI."""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback

import langchain_engram

failures = 0
for module in pkgutil.walk_packages(langchain_engram.__path__, "langchain_engram."):
    try:
        importlib.import_module(module.name)
    except Exception:  # noqa: BLE001
        failures += 1
        print(f"Failed to import {module.name}:")
        traceback.print_exc()

sys.exit(1 if failures else 0)
