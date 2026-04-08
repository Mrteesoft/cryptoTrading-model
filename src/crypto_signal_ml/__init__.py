"""Top-level package for the crypto signal machine learning project.

Preferred package boundaries:
- ``service/``: HTTP APIs and monitor/runtime entrypoints
- ``application/``: orchestration workflows such as training and signal refresh
- ``ml/``: dataset, feature, labeling, and model facades
- ``tools/``: stable structured tool wrappers for LLM-callable integration
- ``trading/``: signal, policy, portfolio, and trader-brain domain modules
- ``storage/``: database adapters and persistence helpers
- ``chat/`` and ``retrieval/``: assistant session and RAG-related facades
- ``memory/``: future-facing session and long-term memory facades
- ``llm/``: provider-specific adapters that consume tools instead of business logic

Backward-compatible root-module shims are intentionally kept so older imports
continue to work while new code migrates toward the layered namespaces above.
"""

__version__ = "0.1.0"

LEGACY_SHIM_MODULES = (
    "database",
    "policy",
    "portfolio",
    "signal_store",
    "signals",
    "symbols",
    "trader_brain",
)

__all__ = ["LEGACY_SHIM_MODULES", "__version__"]
