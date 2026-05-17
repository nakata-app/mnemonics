"""mnemonics — local-first AI memory with tier-aware decay."""
from mnemonics.store import Store

# Submodules `mnemonics.ingest` / `mnemonics.retrieve` are reached via the
# usual `from mnemonics.ingest import ingest` import path. We deliberately
# do NOT re-export those functions here: hoisting them into this namespace
# overwrites the submodule attribute on the package (`mnemonics.ingest`
# would resolve to a function instead of a module), and
# `mock.patch("mnemonics.ingest._get_encoder")` then fails to find the
# attribute on Python 3.10. The 3.12 build happened to mask this; CI on
# 3.10 caught it.

__all__ = ["Store"]
__version__ = "0.3.0"
