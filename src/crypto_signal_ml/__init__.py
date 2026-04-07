"""Top-level package for the crypto signal machine learning project.

Package layout:
- ``trading/``: signal, policy, portfolio, and trader-brain modules
- ``storage/``: database adapters and persistence helpers
- root modules such as ``app.py`` and ``engine_api.py``: orchestration and service entrypoints

Backward-compatible root-module shims are intentionally kept for now so older
imports continue to work during the cleanup.
"""

__version__ = "0.1.0"
