"""Fundamental and market-intelligence feature families."""

from __future__ import annotations


FUNDAMENTALS_CONTEXT_FEATURES = (
    "cmc_context_available",
    "cmc_rank_score",
    "cmc_market_cap_log",
    "cmc_volume_24h_log",
    "cmc_percent_change_24h",
    "cmc_percent_change_7d",
    "cmc_percent_change_30d",
    "cmc_circulating_supply_ratio",
    "cmc_num_market_pairs_log",
    "cmc_tags_count",
    "cmc_platform_present",
    "cmc_is_mineable",
    "cmc_has_defi_tag",
    "cmc_has_ai_tag",
    "cmc_has_layer1_tag",
    "cmc_has_gaming_tag",
    "cmc_has_meme_tag",
)

MARKET_INTELLIGENCE_FEATURES = (
    "cmc_market_intelligence_available",
    "cmc_market_total_market_cap_log",
    "cmc_market_total_volume_24h_log",
    "cmc_market_total_market_cap_change_24h",
    "cmc_market_total_volume_change_24h",
    "cmc_market_altcoin_share",
    "cmc_market_btc_dominance",
    "cmc_market_btc_dominance_change_24h",
    "cmc_market_eth_dominance",
    "cmc_market_stablecoin_share",
    "cmc_market_defi_market_cap_log",
    "cmc_market_defi_volume_24h_log",
    "cmc_market_derivatives_volume_24h_log",
    "cmc_market_fear_greed_score",
    "cmc_market_is_fear",
    "cmc_market_is_greed",
    "cmc_market_is_extreme_fear",
    "cmc_market_is_extreme_greed",
)
