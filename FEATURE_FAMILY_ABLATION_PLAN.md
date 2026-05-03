# Feature Family Ablation Plan

## Goal

Measure whether the current signal model is helped by each feature family, or whether
some families mainly duplicate evidence that is already represented elsewhere in the
model and post-model pipeline.

This repo already organizes the feature space by family in:

- `src/crypto_signal_ml/features.py`
- `src/crypto_signal_ml/features_core/feature_registry.py`

The current full feature set is 112 columns:

- 55 local asset / structure features
- 57 market / context features

That makes the current model a near 50/50 local-vs-context hybrid. The core audit
question is whether that balance helps alpha or suppresses alt-specific setups.

## Current Families

The active families and counts are:

| Family | Count | Role |
| --- | ---: | --- |
| `returns` | 7 | Short-horizon directional change |
| `momentum` | 3 | Continuation / exhaustion |
| `volume` | 6 | Participation / confirmation |
| `volatility` | 4 | Range expansion / risk |
| `trend` | 9 | MA-distance trend state |
| `market_context` | 22 | Cross-asset and BTC-relative context |
| `market_structure` | 5 | Breakout / range structure |
| `time_context` | 4 | Calendar / session effects |
| `higher_timeframe` | 12 | 4h / 1d aligned context |
| `regime` | 5 | Derived market phase summary |
| `fundamentals_context` | 17 | CoinMarketCap asset context |
| `market_intelligence` | 18 | CoinMarketCap macro market context |

## Structural Risks To Test

The main overlap zones are:

1. Directional overlap

- `returns`
- `momentum`
- `trend`

2. Market-phase overlap

- `trend`
- `higher_timeframe`
- `regime`

3. Macro overlap

- `market_context`
- `market_intelligence`
- post-model market stance and late fusion

4. Risk overlap

- in-model `volatility`
- post-model volatility-aware policy gating

## Experiment Matrix

Use the current walk-forward validation flow as the base harness:

- `python scripts/runWalkForwardValidation.py`

Run one baseline and then the ablations below. The desired output is not just the best
classifier metric; it is the best signal system behavior after model scoring and
decision logic.

### Baseline

| Run | Pack / Change | Question |
| --- | --- | --- |
| `baseline_all` | All 112 features | What is the current reference behavior? |

### Single-Family Ablations

| Run | Change | Question |
| --- | --- | --- |
| `ablation_regime` | Remove `regime` | Does explicit regime add value beyond `trend` + `higher_timeframe`? |
| `ablation_higher_timeframe` | Remove `higher_timeframe` | Does HTF context add genuine signal, or mostly repeat trend state? |
| `ablation_market_intelligence` | Remove `market_intelligence` | Does macro CMC context improve alpha, or mainly make the system defensive? |
| `ablation_fundamentals_context` | Remove `fundamentals_context` | Does asset metadata help predictive quality, or mainly bias toward large / liquid names? |
| `ablation_time_context` | Remove `time_context` | Do time/session features contribute anything material? |
| `ablation_market_context` | Remove `market_context` | How dependent is the model on BTC-relative and cross-asset conditioning? |
| `ablation_trend` | Remove `trend`, keep `market_structure` | Can structure carry setups without overlapping MA-derived trend features? |

### Local-vs-Context Shape Tests

| Run | Change | Question |
| --- | --- | --- |
| `local_core_only` | Keep only `returns`, `volume`, `volatility`, `market_structure` | How strong is the pure alt-local signal without broad context? |
| `local_plus_context` | Local core + `market_context` + `market_intelligence` | What does broad context add on top of local edge? |
| `core_pack` | Existing `core` pack | Is the repo's default local-heavy pack already good enough? |
| `core_plus_market_pack` | Existing `core_plus_market` pack | How much is gained from market-relative context alone? |
| `core_plus_context_pack` | Existing `core_plus_context` pack | Does HTF + time context help after market context is present? |
| `core_plus_fundamentals_pack` | Existing `core_plus_fundamentals` pack | What do CMC asset + macro context add on top of the broader technical pack? |

## Metrics To Record

Do not judge families only by raw classification quality.

For every run, record:

1. Model metrics

- accuracy
- balanced accuracy
- average fold balanced accuracy
- fold balanced accuracy variance
- calibration summary and calibrated Brier score when available

2. Trading / signal metrics

- walk-forward trade count
- strategy total return
- benchmark total return
- max drawdown
- win rate
- average trade return

3. Publication-quality metrics

- policy-filtered trade count
- published-signal count per cycle
- actionable-signal count per cycle
- average confidence of published signals
- average decision score of published signals

4. Alt-sensitivity metrics

- breakout capture rate for `market_structure`-led BUY signals
- share of published signals coming from non-BTC / non-ETH alts
- acceptance rate by market regime
- rejection rate caused by policy gating
- false suppression rate for high-confidence HOLD-downgraded rows

## Decision Criteria

Use these rules after each ablation:

### Keep

Keep a family when it:

- improves average fold balanced accuracy or calibration quality
- improves strategy return without sharply increasing drawdown
- improves publication quality without suppressing too many actionable alts

### Demote

Demote a family when it:

- adds little predictive lift
- mainly suppresses signal flow
- makes the model noticeably more macro-dependent or trend-late

These families may still be useful as light-ranking inputs or confidence modifiers.

### Move To Post-Model Logic

Move a family out of the model and into gating / ranking when it:

- behaves more like a tradability filter than predictive alpha
- is useful for vetoing or soft-conditioning decisions
- duplicates evidence already encoded in model inputs

Typical candidates:

- parts of `fundamentals_context`
- parts of `market_intelligence`
- parts of `time_context`

### Drop

Drop a family when it:

- adds no meaningful predictive lift
- hurts calibration or trade quality
- increases overlap and complexity without improving the published signal set

## Recommended Priority Order

Run the ablations in this order:

1. `baseline_all`
2. `ablation_regime`
3. `ablation_higher_timeframe`
4. `ablation_market_intelligence`
5. `ablation_market_context`
6. `ablation_trend`
7. `local_core_only`
8. `local_plus_context`
9. `ablation_fundamentals_context`
10. `ablation_time_context`

This order tests the highest-overlap families first.

## Working Policy For Altcoin Focus

For an altcoin-focused engine, use this rule:

- local structure should lead
- context should condition

That means the hardest families to remove should be:

- `returns`
- `volume`
- `volatility`
- `market_structure`

These are closest to direct altcoin behavior.

The most overlap-prone families are:

- `trend`
- `higher_timeframe`
- `regime`
- `market_context`
- `market_intelligence`

These must justify themselves with walk-forward results, not intuition.

## Architecture Rule

Adopt this as a hard rule for future revisions:

> If a thesis is already heavily represented in model inputs, post-model fusion should
> confirm it lightly, veto it only when risk is strong, and avoid heavily re-scoring
> the same thesis again.

This is especially important for:

- trend
- volatility
- macro state

## Expected Outcome Shape

If the current model is too macro-heavy, the likely symptoms are:

- fewer early alt breakouts captured
- more confirmation lag
- better defense in weak tape
- lower sensitivity to single-name rotation

If the right ablation succeeds, you should see:

- similar or better calibration
- similar or better walk-forward return
- less false suppression
- more responsive alt-specific BUY publication

## Notes On Execution

The repo already supports named feature packs through `FEATURE_PACK` and the registry in:

- `src/crypto_signal_ml/features.py`
- `src/crypto_signal_ml/features_core/feature_registry.py`

For single-family ablations that are not already represented by existing packs, create
temporary experiment variants by excluding the target family from the active feature
list before training and walk-forward validation. Keep the rest of the pipeline
unchanged so the comparison remains valid.

The output of each run should be logged into one experiment table with:

- run name
- included families
- excluded families
- fold metrics
- backtest summary
- publication-quality notes
- keep / demote / move-to-post-model / drop recommendation

## Repo Support For This Plan

The repo now supports family-level selection through config and environment variables:

- `FEATURE_PACK`
- `FEATURE_INCLUDE_GROUPS`
- `FEATURE_EXCLUDE_GROUPS`

Selection is applied as:

`feature_pack + include_groups - exclude_groups`

Walk-forward runs are now written to per-run folders so ablations do not overwrite
each other:

- `outputs/walkForwardRuns/<timestamp>__<runLabel>/`
- `outputs/regimeWalkForwardRuns/<timestamp>__<runLabel>/`

The `runLabel` is derived from the active feature-family selection, for example:

- `pack-all`
- `pack-all__exc-regime`
- `pack-core__inc-market-context__exc-regime`

Examples:

```env
# Baseline
FEATURE_PACK=all
FEATURE_INCLUDE_GROUPS=
FEATURE_EXCLUDE_GROUPS=
```

```env
# Remove regime from the full pack
FEATURE_PACK=all
FEATURE_EXCLUDE_GROUPS=regime
```

```env
# Remove higher timeframe from the full pack
FEATURE_PACK=all
FEATURE_EXCLUDE_GROUPS=higher_timeframe
```

```env
# Local-core-only experiment
FEATURE_PACK=all
FEATURE_INCLUDE_GROUPS=returns,volume,volatility,market_structure
FEATURE_EXCLUDE_GROUPS=momentum,trend,market_context,time_context,higher_timeframe,regime,fundamentals_context,market_intelligence,news
```

```env
# Core pack plus market context, without regime
FEATURE_PACK=core
FEATURE_INCLUDE_GROUPS=market_context
FEATURE_EXCLUDE_GROUPS=regime
```

If a configuration resolves to zero features, the run now fails loudly instead of
silently falling back to the full feature set.
