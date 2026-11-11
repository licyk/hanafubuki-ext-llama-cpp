"""Puts the repository and the extension SDK on the import path (see ``support``)."""

from __future__ import annotations

import sys

from support import ROOT, SDK

for path in (ROOT, SDK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
