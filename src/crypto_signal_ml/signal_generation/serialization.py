"""Adapters between legacy summary dictionaries and typed stage contracts."""

from __future__ import annotations

from typing import Any, Mapping

from .audit import build_signal_contribution_ledger
from .contracts import (
    ActionableSignal,
    ChartConfirmedCandidate,
    ContextEnrichedCandidate,
    GatedSignalCandidate,
    PublishedSignal,
    ScoredSignalCandidate,
    SignalContributionLedger,
)


_UPPER_TO_LOWER_PROBABILITY_KEYS = {
    "BUY": "buy",
    "HOLD": "hold",
    "TAKE_PROFIT": "take_profit",
}
_LOWER_TO_UPPER_PROBABILITY_KEYS = {
    value: key
    for key, value in _UPPER_TO_LOWER_PROBABILITY_KEYS.items()
}


def _safe_float(value: Any, default_value: float = 0.0) -> float:
    """Convert one optional numeric value without raising."""

    if value is None:
        return default_value

    try:
        return float(value)
    except (TypeError, ValueError):
        return default_value


def _normalize_probability_map(raw_probabilities: Mapping[str, Any] | None) -> dict[str, float]:
    """Normalize probability keys to the internal uppercase class names."""

    if not isinstance(raw_probabilities, Mapping):
        return {"BUY": 0.0, "HOLD": 0.0, "TAKE_PROFIT": 0.0}

    normalized_probabilities = {"BUY": 0.0, "HOLD": 0.0, "TAKE_PROFIT": 0.0}
    for raw_key, raw_value in raw_probabilities.items():
        normalized_key = str(raw_key).strip().upper()
        normalized_key = _LOWER_TO_UPPER_PROBABILITY_KEYS.get(normalized_key.lower(), normalized_key)
        if normalized_key in normalized_probabilities:
            normalized_probabilities[normalized_key] = _safe_float(raw_value)

    return normalized_probabilities


def _denormalize_probability_map(probabilities: Mapping[str, Any]) -> dict[str, float]:
    """Convert internal uppercase probability keys to the stable summary format."""

    normalized_probabilities = _normalize_probability_map(probabilities)
    return {
        _UPPER_TO_LOWER_PROBABILITY_KEYS[class_name]: float(probability)
        for class_name, probability in normalized_probabilities.items()
    }


def _build_ledger_from_summary(signal_summary: Mapping[str, Any]) -> SignalContributionLedger:
    """Rebuild one ledger object from a summary row or fall back to a minimal ledger."""

    raw_ledger = signal_summary.get("contributionLedger")
    if isinstance(raw_ledger, Mapping):
        return SignalContributionLedger(
            rawProbabilities=_normalize_probability_map(raw_ledger.get("rawProbabilities")),
            calibratedProbabilities=_normalize_probability_map(raw_ledger.get("calibratedProbabilities")),
            rawConfidence=_safe_float(
                raw_ledger.get("rawConfidence"),
                _safe_float(signal_summary.get("rawConfidence"), _safe_float(signal_summary.get("confidence"))),
            ),
            calibratedConfidence=_safe_float(raw_ledger.get("calibratedConfidence"), _safe_float(signal_summary.get("confidence"))),
            probabilityMargin=_safe_float(raw_ledger.get("probabilityMargin"), _safe_float(signal_summary.get("probabilityMargin"))),
            policyStatus=str(raw_ledger.get("policyStatus", "unknown") or "unknown"),
            rejectionReasons=tuple(
                str(reason).strip()
                for reason in list(raw_ledger.get("rejectionReasons") or [])
                if str(reason).strip()
            ),
            confidenceAdjustments={
                str(key): _safe_float(value)
                for key, value in dict(raw_ledger.get("confidenceAdjustments") or {}).items()
            },
            finalDecisionScore=_safe_float(raw_ledger.get("finalDecisionScore"), _safe_float(signal_summary.get("policyScore"))),
            publicationReason=str(
                raw_ledger.get("publicationReason", signal_summary.get("publicationReason", "candidate_only"))
                or "candidate_only"
            ),
        )

    return build_signal_contribution_ledger(
        raw_probabilities=_normalize_probability_map(signal_summary.get("rawProbabilities")),
        calibrated_probabilities=_normalize_probability_map(signal_summary.get("probabilities")),
        raw_confidence=_safe_float(signal_summary.get("rawConfidence"), _safe_float(signal_summary.get("confidence"))),
        calibrated_confidence=_safe_float(signal_summary.get("confidence")),
        probability_margin=_safe_float(signal_summary.get("probabilityMargin")),
        policy_status="passed" if bool(signal_summary.get("actionable", False)) else "standby",
        rejection_reasons=list(signal_summary.get("gateReasons") or []),
        required_action_confidence=_safe_float(signal_summary.get("requiredActionConfidence")),
        final_decision_score=_safe_float(signal_summary.get("policyScore")),
        publication_reason=str(signal_summary.get("publicationReason", "candidate_only") or "candidate_only"),
    )


def _collect_metadata(signal_summary: Mapping[str, Any], known_keys: set[str]) -> dict[str, Any]:
    """Preserve non-contract summary fields while migrating through typed stages."""

    return {
        str(key): value
        for key, value in signal_summary.items()
        if str(key) not in known_keys
    }


def gated_candidate_from_summary(signal_summary: Mapping[str, Any]) -> GatedSignalCandidate:
    """Convert one legacy signal summary into the typed gated-candidate contract."""

    known_keys = {
        "time_step",
        "timestamp",
        "close",
        "signal_name",
        "predicted_signal",
        "spotAction",
        "actionable",
        "symbol",
        "coinSymbol",
        "pairSymbol",
        "productId",
        "baseCurrency",
        "quoteCurrency",
        "coinName",
        "coinCategory",
        "confidence",
        "rawConfidence",
        "minimumActionConfidence",
        "requiredActionConfidence",
        "confidenceGateApplied",
        "riskGateApplied",
        "modelSignalName",
        "modelPredictedSignal",
        "modelSpotAction",
        "primaryProbability",
        "probabilityMargin",
        "tradeReadiness",
        "policyScore",
        "policyNotes",
        "gateReasons",
        "setupScore",
        "probabilities",
        "rawProbabilities",
        "reasonItems",
        "reasonSummary",
        "signalChat",
        "chartContext",
        "chartConfirmationScore",
        "chartConfirmationStatus",
        "chartSetupType",
        "chartConfirmationNotes",
        "chartPatternLabel",
        "chartPatternReasons",
        "chartDecision",
        "executionContext",
        "marketContext",
        "marketState",
        "eventContext",
        "contributionLedger",
        "publicationReason",
    }
    ledger = _build_ledger_from_summary(signal_summary)

    return GatedSignalCandidate(
        productId=str(signal_summary.get("productId", "") or ""),
        timestamp=(str(signal_summary.get("timestamp")).strip() or None)
        if signal_summary.get("timestamp") is not None
        else None,
        close=_safe_float(signal_summary.get("close")),
        symbol=str(signal_summary.get("symbol") or signal_summary.get("coinSymbol") or "").strip().upper(),
        pairSymbol=str(signal_summary.get("pairSymbol") or signal_summary.get("productId") or "").strip().upper(),
        baseCurrency=(str(signal_summary.get("baseCurrency", "")).strip().upper() or None),
        quoteCurrency=(str(signal_summary.get("quoteCurrency", "")).strip().upper() or None),
        coinName=(str(signal_summary.get("coinName", "")).strip() or None),
        coinCategory=(str(signal_summary.get("coinCategory", "")).strip() or None),
        timeStep=int(round(_safe_float(signal_summary.get("time_step")))),
        rawSignalName=str(signal_summary.get("modelSignalName") or signal_summary.get("signal_name") or "HOLD").strip().upper(),
        rawPredictedSignal=int(
            round(
                _safe_float(
                    signal_summary.get("modelPredictedSignal"),
                    _safe_float(signal_summary.get("predicted_signal")),
                )
            )
        ),
        rawSpotAction=str(signal_summary.get("modelSpotAction") or signal_summary.get("spotAction") or "wait").strip().lower(),
        rawProbabilities=_normalize_probability_map(signal_summary.get("rawProbabilities")),
        calibratedProbabilities=_normalize_probability_map(signal_summary.get("probabilities")),
        rawConfidence=_safe_float(signal_summary.get("rawConfidence"), _safe_float(signal_summary.get("confidence"))),
        calibratedConfidence=_safe_float(signal_summary.get("confidence")),
        primaryProbability=_safe_float(signal_summary.get("primaryProbability")),
        probabilityMargin=_safe_float(signal_summary.get("probabilityMargin")),
        hasProbabilityColumns=bool(signal_summary.get("probabilities")),
        minimumActionConfidence=_safe_float(signal_summary.get("minimumActionConfidence")),
        setupScore=_safe_float(signal_summary.get("setupScore")),
        chartContext=dict(signal_summary.get("chartContext") or {}),
        chartConfirmationScore=_safe_float(signal_summary.get("chartConfirmationScore")),
        chartSetupType=str(
            signal_summary.get("chartSetupType", signal_summary.get("chartPatternLabel", "no_clean_setup"))
            or "no_clean_setup"
        ),
        chartConfirmationStatus=str(
            signal_summary.get("chartConfirmationStatus", signal_summary.get("chartDecision", "early")) or "early"
        ),
        chartPatternLabel=str(
            signal_summary.get("chartPatternLabel", signal_summary.get("chartSetupType", "no_clean_setup"))
            or "no_clean_setup"
        ),
        chartDecision=str(
            signal_summary.get("chartDecision", signal_summary.get("chartConfirmationStatus", "early")) or "early"
        ),
        chartConfirmationNotes=tuple(
            str(note).strip()
            for note in list(signal_summary.get("chartConfirmationNotes") or [])
            if str(note).strip()
        ),
        chartPatternReasons=tuple(
            str(note).strip()
            for note in list(signal_summary.get("chartPatternReasons") or signal_summary.get("chartConfirmationNotes") or [])
            if str(note).strip()
        ),
        executionContext=dict(signal_summary.get("executionContext") or {}),
        marketContext=dict(signal_summary.get("marketContext") or {}),
        marketState=dict(signal_summary.get("marketState") or {}),
        eventContext=dict(signal_summary.get("eventContext") or {}),
        metadata=_collect_metadata(signal_summary, known_keys),
        signalName=str(signal_summary.get("signal_name") or "HOLD").strip().upper(),
        predictedSignal=int(round(_safe_float(signal_summary.get("predicted_signal")))),
        spotAction=str(signal_summary.get("spotAction") or "wait").strip().lower(),
        actionable=bool(signal_summary.get("actionable", False)),
        requiredActionConfidence=_safe_float(signal_summary.get("requiredActionConfidence")),
        confidenceGateApplied=bool(signal_summary.get("confidenceGateApplied", False)),
        riskGateApplied=bool(signal_summary.get("riskGateApplied", False)),
        tradeReadiness=str(signal_summary.get("tradeReadiness", "standby") or "standby"),
        policyScore=_safe_float(signal_summary.get("policyScore")),
        policyNotes=tuple(
            str(note).strip()
            for note in list(signal_summary.get("policyNotes") or [])
            if str(note).strip()
        ),
        gateReasons=tuple(
            str(reason).strip()
            for reason in list(signal_summary.get("gateReasons") or [])
            if str(reason).strip()
        ),
        reasonItems=tuple(
            str(reason).strip()
            for reason in list(signal_summary.get("reasonItems") or [])
            if str(reason).strip()
        ),
        reasonSummary=str(signal_summary.get("reasonSummary", "") or ""),
        signalChat=str(signal_summary.get("signalChat", "") or ""),
        ledger=ledger,
    )


def context_enriched_candidate_from_summary(signal_summary: Mapping[str, Any]) -> ContextEnrichedCandidate:
    """Convert one context-enriched summary into the typed context contract."""

    gated_candidate = gated_candidate_from_summary(signal_summary)
    return ContextEnrichedCandidate(
        **gated_candidate.__dict__,
        newsContext=dict(signal_summary.get("newsContext") or {}),
        trendContext=dict(signal_summary.get("trendContext") or {}),
        contextEvidence=dict(signal_summary.get("contextEvidence") or {}),
    )


def actionable_signal_from_summary(signal_summary: Mapping[str, Any]) -> ActionableSignal:
    """Convert one trader-brain-enriched summary into the actionable-signal contract."""

    context_candidate = context_enriched_candidate_from_summary(signal_summary)
    final_decision_score = _safe_float(
        (signal_summary.get("brain") or {}).get("decisionScore"),
        _safe_float(signal_summary.get("finalDecisionScore"), _safe_float(signal_summary.get("policyScore"))),
    )

    candidate_payload = dict(context_candidate.__dict__)
    candidate_payload.update(
        {
            "executionContext": dict(signal_summary.get("executionContext") or context_candidate.executionContext),
            "confidenceCalibration": dict(signal_summary.get("confidenceCalibration") or {}),
            "adaptiveContext": dict(signal_summary.get("adaptiveContext") or {}),
            "finalDecisionScore": final_decision_score,
            "portfolioDecision": str(
                (signal_summary.get("brain") or {}).get("decision", signal_summary.get("portfolioDecision", "watchlist"))
                or "watchlist"
            ),
            "tradeContext": dict(signal_summary.get("tradeContext") or {}),
            "brain": dict(signal_summary.get("brain") or {}),
            "watchlistState": dict(signal_summary.get("watchlistState") or {}),
            "watchlistPromotion": dict(signal_summary.get("watchlistPromotion") or {}),
        }
    )
    return ActionableSignal(**candidate_payload)


def published_signal_from_summary(signal_summary: Mapping[str, Any]) -> PublishedSignal:
    """Convert one published signal summary into the publication contract."""

    actionable_signal = actionable_signal_from_summary(signal_summary)
    return PublishedSignal(
        **actionable_signal.__dict__,
        publicationReason=str(
            signal_summary.get("publicationReason", actionable_signal.ledger.publicationReason)
            or actionable_signal.ledger.publicationReason
        ),
        publicSignalType=str(signal_summary.get("publicSignalType", "candidate") or "candidate"),
        watchlistFallback=bool(signal_summary.get("watchlistFallback", False)),
        published=bool(signal_summary.get("published", True)),
    )


def gated_candidate_to_summary(candidate: GatedSignalCandidate) -> dict[str, Any]:
    """Serialize one gated candidate into the stable summary shape."""

    summary = dict(candidate.metadata)
    summary.update(
        {
            "time_step": int(candidate.timeStep),
            "timestamp": candidate.timestamp,
            "close": float(candidate.close),
            "predicted_signal": int(candidate.predictedSignal),
            "signal_name": str(candidate.signalName),
            "spotAction": str(candidate.spotAction),
            "actionable": bool(candidate.actionable),
            "symbol": str(candidate.symbol),
            "coinSymbol": str(candidate.symbol),
            "pairSymbol": str(candidate.pairSymbol),
            "productId": str(candidate.productId),
            "baseCurrency": candidate.baseCurrency,
            "quoteCurrency": candidate.quoteCurrency,
            "coinName": candidate.coinName,
            "coinCategory": candidate.coinCategory,
            "confidence": float(candidate.calibratedConfidence),
            "rawConfidence": float(candidate.rawConfidence),
            "minimumActionConfidence": float(candidate.minimumActionConfidence),
            "requiredActionConfidence": float(candidate.requiredActionConfidence),
            "confidenceGateApplied": bool(candidate.confidenceGateApplied),
            "riskGateApplied": bool(candidate.riskGateApplied),
            "modelPredictedSignal": int(candidate.rawPredictedSignal),
            "modelSignalName": str(candidate.rawSignalName),
            "modelSpotAction": str(candidate.rawSpotAction),
            "primaryProbability": float(candidate.primaryProbability),
            "probabilityMargin": float(candidate.probabilityMargin),
            "tradeReadiness": str(candidate.tradeReadiness),
            "policyScore": float(candidate.policyScore),
            "policyNotes": list(candidate.policyNotes),
            "gateReasons": list(candidate.gateReasons),
            "setupScore": float(candidate.setupScore),
            "probabilities": _denormalize_probability_map(candidate.calibratedProbabilities),
            "rawProbabilities": _denormalize_probability_map(candidate.rawProbabilities),
            "reasonItems": list(candidate.reasonItems),
            "reasonSummary": str(candidate.reasonSummary),
            "signalChat": str(candidate.signalChat),
            "chartContext": dict(candidate.chartContext),
            "chartConfirmationScore": float(candidate.chartConfirmationScore),
            "chartSetupType": str(candidate.chartSetupType),
            "chartConfirmationStatus": str(candidate.chartConfirmationStatus),
            "chartConfirmationNotes": list(candidate.chartConfirmationNotes),
            "chartPatternLabel": str(candidate.chartPatternLabel),
            "chartPatternReasons": list(candidate.chartPatternReasons),
            "chartDecision": str(candidate.chartDecision),
            "executionContext": dict(candidate.executionContext),
            "marketContext": dict(candidate.marketContext),
            "marketState": dict(candidate.marketState),
            "eventContext": dict(candidate.eventContext),
            "contributionLedger": candidate.ledger.to_dict(),
            "publicationReason": str(candidate.ledger.publicationReason),
        }
    )
    return summary


def context_enriched_candidate_to_summary(candidate: ContextEnrichedCandidate) -> dict[str, Any]:
    """Serialize one context-enriched candidate into the stable summary shape."""

    summary = gated_candidate_to_summary(candidate)
    summary.update(
        {
            "newsContext": dict(candidate.newsContext),
            "trendContext": dict(candidate.trendContext),
            "contextEvidence": dict(candidate.contextEvidence),
        }
    )
    return summary


def scored_candidate_to_summary(candidate: ScoredSignalCandidate) -> dict[str, Any]:
    """Serialize one scored candidate into the stable summary shape."""

    summary = context_enriched_candidate_to_summary(candidate)
    summary.update(
        {
            "executionContext": dict(candidate.executionContext),
            "confidenceCalibration": dict(candidate.confidenceCalibration),
            "adaptiveContext": dict(candidate.adaptiveContext),
            "finalDecisionScore": float(candidate.finalDecisionScore),
            "contributionLedger": candidate.ledger.to_dict(),
            "publicationReason": str(candidate.ledger.publicationReason),
        }
    )
    return summary


def actionable_signal_to_summary(signal: ActionableSignal) -> dict[str, Any]:
    """Serialize one actionable signal into the stable summary shape."""

    summary = scored_candidate_to_summary(signal)
    summary.update(
        {
            "portfolioDecision": str(signal.portfolioDecision),
            "tradeContext": dict(signal.tradeContext),
            "brain": dict(signal.brain),
            "watchlistState": dict(signal.watchlistState),
            "watchlistPromotion": dict(signal.watchlistPromotion),
        }
    )
    return summary


def published_signal_to_summary(signal: PublishedSignal) -> dict[str, Any]:
    """Serialize one published signal into the stable summary shape."""

    summary = actionable_signal_to_summary(signal)
    summary.update(
        {
            "publicationReason": str(signal.publicationReason),
            "publicSignalType": str(signal.publicSignalType),
            "watchlistFallback": bool(signal.watchlistFallback),
            "published": bool(signal.published),
        }
    )
    return summary
