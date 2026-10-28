"""Inventory of retained sparse-computation families and available evidence."""

from __future__ import annotations

import json
from pathlib import Path


_CATALOG = Path(__file__).with_name("families.json")


def mechanism_catalog() -> dict:
    """Return a copy of the evidence-qualified mechanism catalog."""
    return json.loads(_CATALOG.read_text(encoding="utf-8"))
