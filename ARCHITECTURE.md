# Model-Service Architecture

This service is still a trading engine first. The refactor in this repository does not change that. It makes the existing layers easier to see and safer to extend.

## Current Layer Map

- `main.py`, `__main__.py`
  Canonical Python entrypoint for the unified monitor + engine API process.
- `src/crypto_signal_ml/service/`
  Facade package for API creation and runtime entrypoints.
- `src/crypto_signal_ml/application/`
  Facade package for orchestration workflows such as training, refresh, validation, backtesting, and signal publishing.
- `src/crypto_signal_ml/tools/`
  Stable structured tool wrappers that expose authoritative signal, trader, model, and retrieval operations for LLM callers.
- `src/crypto_signal_ml/app.py`
  Concrete application workflow implementations used by scripts and the monitor.
- `src/crypto_signal_ml/api.py`, `src/crypto_signal_ml/engine_api.py`, `src/crypto_signal_ml/monitor.py`
  HTTP/service layer and long-running runtime process.
- `src/crypto_signal_ml/data.py`, `src/crypto_signal_ml/features.py`, `src/crypto_signal_ml/labels.py`, `src/crypto_signal_ml/modeling.py`, `src/crypto_signal_ml/regime_modeling.py`, `src/crypto_signal_ml/pipeline.py`, `src/crypto_signal_ml/regimes.py`
  ML pipeline and market-data preparation modules.
- `src/crypto_signal_ml/portfolio_core/`
  Canonical portfolio-planning layer for trader-brain decisions, market stance, and action mapping.
- `src/crypto_signal_ml/trading/`
  Trading domain logic for policies, signal shaping, portfolio state, signal persistence, and compatibility planner surfaces.
- `src/crypto_signal_ml/storage/`
  Shared database adapter layer for SQLite/PostgreSQL-backed stores.
- `src/crypto_signal_ml/chat/`, `src/crypto_signal_ml/retrieval/`
  Facade packages around the current assistant and RAG modules.
- `src/crypto_signal_ml/memory/`
  Session-memory facade that can later grow beyond the current conversation store.
- `src/crypto_signal_ml/llm/`
  Provider-specific adapters that stay outside the trading domain.
- `src/crypto_signal_ml/assistant.py`, `src/crypto_signal_ml/rag.py`
  Current assistant/session orchestration and local retrieval store implementations.
- `scripts/`
  CLI entrypoints that call the application or service layer.

## What Is Structurally Good

- The trading domain already has a real package boundary in `src/crypto_signal_ml/trading/`.
- Persistence already has a real package boundary in `src/crypto_signal_ml/storage/`.
- The ML pipeline is already split by responsibility instead of being collapsed into one large file.
- Scripts are thin wrappers over reusable classes instead of embedding business logic.

## What Was Structurally Messy

- Root modules such as `portfolio.py` and `signals.py` coexisted with `trading/` implementations.
- Service and orchestration responsibilities lived in large root modules without a directory that made the layer obvious.
- Assistant and retrieval capabilities existed, but their placement looked experimental rather than intentional.

## Legacy Compatibility Shims

These root modules are compatibility bridges and should not grow new logic:

- `src/crypto_signal_ml/database.py`
- `src/crypto_signal_ml/policy.py`
- `src/crypto_signal_ml/portfolio.py`
- `src/crypto_signal_ml/signal_store.py`
- `src/crypto_signal_ml/signals.py`
- `src/crypto_signal_ml/symbols.py`
- `src/crypto_signal_ml/trader_brain.py`

## Refactor Decisions

- Keep runtime implementations in place to avoid a risky file move.
- Add explicit facade packages for `service`, `application`, `ml`, `chat`, and `retrieval`.
- Add `tools/` as the stable contract for LLM-callable access to the authoritative engine.
- Add `llm/` as an isolated adapter layer that can consume tool schemas later without owning business logic.
- Add `memory/` as a clearer home for future assistant/session memory evolution.
- Prefer those facades in new scripts, tests, and documentation.
- Keep root compatibility modules importable for external callers.

## Future LLM-Oriented Growth

If this service evolves into a more agentic system, the current layout leaves clean extension points:

- `chat/`
  Session APIs, conversation orchestration, response policies.
- `tools/`
  Structured signal, trader, model, and retrieval actions that both internal chat and external LLMs can call.
- `retrieval/`
  Knowledge indexing, search, and source management.
- `memory/`
  Future long-term user/session memory distinct from retrieval chunks.
- `llm/` or `agents/`
  Provider adapters, prompting, planner/executor flows, or agent policies.

The trading engine stays the source of truth for market logic. Any future LLM layer should sit on top of `application/`, `service/`, `trading/`, and `retrieval/`, not replace them.
