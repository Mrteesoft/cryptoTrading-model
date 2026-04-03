# Market Intelligence Roadmap

## Goal

Turn the current crypto signal service from a single short-horizon classifier into a layered market-intelligence system that can:

- understand market regime
- detect actionable setups
- explain the thesis behind a call
- apply risk and confidence gates before publishing an action

The target is not "one model that acts like a human."
The target is "a stack of specialized components that together behave more like a disciplined analyst."

## Current Baseline

The current system already has a solid foundation:

- Data loading and enrichment: `src/crypto_signal_ml/data.py`
- Feature engineering: `src/crypto_signal_ml/features.py`
- Labeling: `src/crypto_signal_ml/labels.py`
- Dataset assembly: `src/crypto_signal_ml/pipeline.py`
- Model training and persistence: `src/crypto_signal_ml/modeling.py`
- Training, validation, tuning, generation flows: `src/crypto_signal_ml/app.py`
- Live inference: `src/crypto_signal_ml/live.py`
- Signal formatting: `src/crypto_signal_ml/signals.py`
- Cached frontend snapshot API: `src/crypto_signal_ml/api.py`
- Full engine API and assistant hooks: `src/crypto_signal_ml/engine_api.py`, `src/crypto_signal_ml/assistant.py`, `src/crypto_signal_ml/rag.py`

The current bottleneck is architectural:

- one model is doing too much
- labels are still too close to "short-term class prediction"
- regime understanding is implicit instead of explicit
- context and risk are not first-class model layers yet

## Target Architecture

Build the system in three layers:

1. Signal Engine
2. Context Engine
3. Decision Engine

Each layer should have its own data contracts, tests, and metrics.

## Phase 1: Strong Signal Engine

### Objective

Make the model explicitly understand market state and trade quality before adding more external context.

### Existing Files To Extend

- `src/crypto_signal_ml/config.py`
- `src/crypto_signal_ml/data.py`
- `src/crypto_signal_ml/features.py`
- `src/crypto_signal_ml/labels.py`
- `src/crypto_signal_ml/pipeline.py`
- `src/crypto_signal_ml/modeling.py`
- `src/crypto_signal_ml/app.py`
- `src/crypto_signal_ml/signals.py`
- `src/crypto_signal_ml/live.py`
- `src/crypto_signal_ml/backtesting.py`

### New Files To Add

- `src/crypto_signal_ml/regimes.py`
- `src/crypto_signal_ml/targets.py`
- `src/crypto_signal_ml/ensemble.py`
- `tests/testRegimes.py`
- `tests/testTargets.py`
- `tests/testSignalEnsemble.py`

### New Classes To Add

In `regimes.py`:

- `MarketRegimeDetector`
- `VolatilityRegimeBuilder`
- `TrendRegimeBuilder`

In `targets.py`:

- `DirectionTargetBuilder`
- `ContinuationFailureTargetBuilder`
- `SetupQualityTargetBuilder`

In `modeling.py` or `ensemble.py`:

- `RegimeSignalModel`
- `DirectionSignalModel`
- `SetupQualitySignalModel`
- `SignalEnsembleModel`

### Concrete Backlog

#### 1. Multi-timeframe feature support

Edit:

- `src/crypto_signal_ml/config.py`
- `src/crypto_signal_ml/data.py`
- `src/crypto_signal_ml/features.py`
- `src/crypto_signal_ml/pipeline.py`

Add:

- configurable training timeframes such as `1h`, `4h`, `1d`
- aligned multi-timeframe inputs per asset
- feature groups for higher-timeframe trend, slope, and breakout context

Definition of done:

- dataset builder can produce aligned multi-timeframe rows
- no future leakage across timeframe joins
- tests cover alignment and missing-data handling

#### 2. Explicit regime labeling

Edit:

- `src/crypto_signal_ml/labels.py`
- `src/crypto_signal_ml/app.py`

Add:

- regime labels such as `trend`, `range`, `high_volatility`, `low_volatility`
- ability to train and validate a regime model separately

Definition of done:

- regime labels are exported into the prepared dataset
- walk-forward can score a regime model independently

#### 3. Separate target families

Edit:

- `src/crypto_signal_ml/labels.py`
- `src/crypto_signal_ml/pipeline.py`

Add:

- direction target
- continuation-vs-failure target
- setup-quality target

Definition of done:

- target generation is configurable
- target leakage tests exist
- each target has its own distribution report

#### 4. Split the single-model training flow

Edit:

- `src/crypto_signal_ml/modeling.py`
- `src/crypto_signal_ml/app.py`

Add:

- dedicated training path for regime model
- dedicated training path for direction model
- dedicated training path for setup-quality model
- ensemble combiner that merges those outputs into one final score

Definition of done:

- each model can be trained and saved separately
- ensemble predictions can be generated from saved artifacts

#### 5. Stronger output schema

Edit:

- `src/crypto_signal_ml/signals.py`
- `src/crypto_signal_ml/live.py`
- `src/crypto_signal_ml/frontend.py`
- `src/crypto_signal_ml/api.py`

Add to signal output:

- `marketState`
- `tradeBias`
- `setupQuality`
- `expectedHorizon`
- `invalidationReason`
- `minimumActionConfidence`

Definition of done:

- API output shows more than raw class labels
- frontend snapshot remains backward-compatible where possible

### Phase 1 Acceptance Criteria

- walk-forward balanced accuracy materially beats the current baseline
- actionable signals have better precision than the raw ungated model
- backtest after fees is at least near breakeven
- signal outputs contain regime plus setup-quality context

## Phase 2: Analyst-Style Context Engine

### Objective

Give the system the same situational awareness a human analyst uses before acting.

### Existing Files To Extend

- `src/crypto_signal_ml/data.py`
- `src/crypto_signal_ml/assistant.py`
- `src/crypto_signal_ml/rag.py`
- `src/crypto_signal_ml/live.py`
- `src/crypto_signal_ml/engine_api.py`

### New Files To Add

- `src/crypto_signal_ml/context_sources.py`
- `src/crypto_signal_ml/context_snapshot.py`
- `src/crypto_signal_ml/news_context.py`
- `src/crypto_signal_ml/market_breadth.py`
- `tests/testContextSnapshot.py`

### New Classes To Add

- `NewsContextLoader`
- `DerivativesContextLoader`
- `OnChainContextLoader`
- `MarketContextSnapshotBuilder`
- `ContextAwareSignalExplainer`

### Concrete Backlog

#### 1. Add derivatives and event context

Add loaders for:

- funding rate
- open interest
- liquidations
- listing and unlock events
- macro calendar or high-impact market events

Definition of done:

- context snapshots can be generated on schedule
- missing vendors do not break inference

#### 2. Add market narrative summaries

Use:

- `src/crypto_signal_ml/rag.py`
- `src/crypto_signal_ml/assistant.py`
- `src/crypto_signal_ml/engine_api.py`

Add:

- recent narrative summary
- active risk events
- sector rotation snapshot
- benchmark dominance summary

Definition of done:

- the assistant can explain a live signal using current context
- context is cached and queryable through the engine API

#### 3. Join structured ML output with retrieved context

Edit:

- `src/crypto_signal_ml/live.py`
- `src/crypto_signal_ml/signals.py`

Add:

- a merged live view where the numeric model edge is separate from contextual warnings
- explicit "no trade because context risk" behavior

### Phase 2 Acceptance Criteria

- live outputs reflect catalyst and derivatives context
- assistant explanations are grounded in both numeric signals and retrieved context
- event-heavy periods show fewer false-positive trades

## Phase 3: Decision And Risk Engine

### Objective

Move from "the model predicts a class" to "the system decides whether a trade is worth taking."

### Existing Files To Extend

- `src/crypto_signal_ml/backtesting.py`
- `src/crypto_signal_ml/live.py`
- `src/crypto_signal_ml/signals.py`
- `src/crypto_signal_ml/api.py`
- `src/crypto_signal_ml/engine_api.py`

### New Files To Add

- `src/crypto_signal_ml/decision.py`
- `src/crypto_signal_ml/calibration.py`
- `src/crypto_signal_ml/risk.py`
- `src/crypto_signal_ml/portfolio.py`
- `tests/testDecisionPolicy.py`
- `tests/testCalibration.py`
- `tests/testRiskEngine.py`

### New Classes To Add

- `ProbabilityCalibrator`
- `TradeDecisionPolicy`
- `SignalRiskEngine`
- `PortfolioExposurePolicy`

### Concrete Backlog

#### 1. Calibrate probabilities

Edit:

- `src/crypto_signal_ml/modeling.py`
- `src/crypto_signal_ml/decision.py`

Add:

- probability calibration for classifier outputs
- calibrated confidence used in all downstream gating

Definition of done:

- confidence means something numerically
- confidence thresholds are tuned on calibrated scores

#### 2. Add a real trade-decision layer

Edit:

- `src/crypto_signal_ml/live.py`
- `src/crypto_signal_ml/signals.py`

Add:

- trade/no-trade decision
- expected risk-reward
- invalidation level or thesis failure condition
- setup grade

Definition of done:

- live API returns a decision object, not just a class label

#### 3. Portfolio-aware backtesting

Edit:

- `src/crypto_signal_ml/backtesting.py`

Add:

- exposure caps
- correlation-aware position limits
- regime-based sizing
- drawdown limits

Definition of done:

- walk-forward backtest evaluates portfolio behavior, not just isolated entries

### Phase 3 Acceptance Criteria

- positive expectancy after fees in walk-forward tests
- controlled drawdown
- stable confidence calibration
- decision quality is consistent across multiple market regimes

## Immediate Prioritized Backlog

Build in this order:

1. Multi-timeframe dataset support
2. Regime labels and regime model
3. Separate direction and setup-quality targets
4. Ensemble model layer
5. Expanded output schema
6. Context ingestion
7. Context-aware explanations
8. Probability calibration
9. Decision policy
10. Portfolio-aware backtesting

## First 2-Week Sprint

### Sprint Goal

Deliver the first real version of Phase 1:

- multi-timeframe features
- regime labeling
- separate regime model training path
- updated outputs that expose market state

### Sprint Scope

#### Week 1

Day 1:

- add new config fields in `src/crypto_signal_ml/config.py`
- define timeframe settings and regime-related settings

Day 2:

- extend `src/crypto_signal_ml/data.py` to load and align higher-timeframe data
- add tests for timestamp alignment

Day 3:

- extend `src/crypto_signal_ml/features.py` with higher-timeframe trend and volatility features
- update `src/crypto_signal_ml/pipeline.py`

Day 4:

- add `src/crypto_signal_ml/regimes.py`
- implement regime builders and regime label generation

Day 5:

- add `tests/testRegimes.py`
- add smoke tests for multi-timeframe feature generation

#### Week 2

Day 6:

- extend `src/crypto_signal_ml/modeling.py` with regime model support
- add save/load support for regime artifacts

Day 7:

- extend `src/crypto_signal_ml/app.py` with regime training and validation flow
- add or update scripts if needed

Day 8:

- extend `src/crypto_signal_ml/signals.py` to include `marketState`
- extend `src/crypto_signal_ml/live.py` and `src/crypto_signal_ml/api.py`

Day 9:

- run walk-forward validation
- compare baseline vs regime-aware version

Day 10:

- clean up outputs
- document metrics
- cut scope for Sprint 2 based on measured results

### Sprint Deliverables

- regime-aware prepared dataset
- saved regime model artifact
- updated live/API output with market state
- walk-forward comparison report

### Sprint Success Metrics

- dataset builds without leakage
- live output remains stable
- walk-forward balanced accuracy improves versus current baseline
- signal explanations contain market-state context

## Suggested Script Additions

After Sprint 1, add these runnable entrypoints:

- `scripts/trainRegimeModel.py`
- `scripts/runRegimeValidation.py`
- `scripts/trainSignalEnsemble.py`
- `scripts/buildContextSnapshot.py`

## Non-Negotiable Engineering Rules

- every new target must have a leakage test
- every live feature must be reproducible from information available at that timestamp
- every confidence threshold must be validated out of sample
- every model output used for trading must be auditable from saved artifacts

## Recommended Next Action

Do not start Phase 2 yet.

Start Sprint 1 with:

1. `config.py` multi-timeframe and regime settings
2. `data.py` timeframe alignment
3. `features.py` higher-timeframe features
4. `regimes.py` initial regime builder

That is the shortest path from the current repo to analyst-style market understanding.
