"""ML-layer facade for dataset, feature, label, and model components.

The underlying implementations remain in the root package modules so this
package can stay small and non-invasive while offering a clearer namespace for
new imports.
"""

from ..data import (
    BaseApiPriceDataLoader,
    BasePriceDataEnricher,
    BasePriceDataLoader,
    BinancePublicDataPriceDataLoader,
    CoinbaseExchangePriceDataLoader,
    CoinMarketCalEventEnricher,
    CoinMarketCapContextEnricher,
    CoinMarketCapLatestQuotesPriceDataLoader,
    CoinMarketCapMarketIntelligenceEnricher,
    CoinMarketCapOhlcvPriceDataLoader,
    CoinMarketCapRateLimitError,
    CsvPriceDataLoader,
    EnrichedCsvPriceDataLoader,
    EnrichedPriceDataLoader,
    KrakenOhlcPriceDataLoader,
    create_market_data_loader,
)
from ..features import TechnicalFeatureEngineer, build_features, calculate_rsi
from ..labels_core import (
    BaseSignalLabeler,
    FutureReturnSignalLabeler,
    MarketRegimeLabeler,
    TripleBarrierSignalLabeler,
    add_regime_labels,
    add_signal_labels,
    create_labeler_from_config,
    create_regime_labeler_from_config,
    signal_to_text,
)
from ..modeling import (
    BaseSignalModel,
    HistGradientBoostingSignalModel,
    LogisticRegressionSignalModel,
    RandomForestSignalModel,
    create_model_from_config,
    get_model_class,
    register_model,
)
from ..pipeline import CryptoDatasetBuilder, build_feature_table, build_labeled_dataset
from ..regime_modeling import MarketRegimeModel
from ..regimes import MarketRegimeDetector, TrendRegimeBuilder, VolatilityRegimeBuilder
from .models import (
    LightGBMClassifierSignalModel,
    LightGBMRankerSignalModel,
    RiverOnlineSignalModel,
    TFTSequenceSignalModel,
    XGBoostClassifierSignalModel,
    ensure_registry_loaded,
    resolve_model_type,
)

__all__ = [
    "BaseApiPriceDataLoader",
    "BasePriceDataEnricher",
    "BasePriceDataLoader",
    "BaseSignalLabeler",
    "BaseSignalModel",
    "BinancePublicDataPriceDataLoader",
    "CoinbaseExchangePriceDataLoader",
    "CoinMarketCalEventEnricher",
    "CoinMarketCapContextEnricher",
    "CoinMarketCapLatestQuotesPriceDataLoader",
    "CoinMarketCapMarketIntelligenceEnricher",
    "CoinMarketCapOhlcvPriceDataLoader",
    "CoinMarketCapRateLimitError",
    "CryptoDatasetBuilder",
    "CsvPriceDataLoader",
    "EnrichedCsvPriceDataLoader",
    "EnrichedPriceDataLoader",
    "FutureReturnSignalLabeler",
    "HistGradientBoostingSignalModel",
    "KrakenOhlcPriceDataLoader",
    "LogisticRegressionSignalModel",
    "MarketRegimeDetector",
    "MarketRegimeLabeler",
    "MarketRegimeModel",
    "RandomForestSignalModel",
    "TechnicalFeatureEngineer",
    "TrendRegimeBuilder",
    "TripleBarrierSignalLabeler",
    "VolatilityRegimeBuilder",
    "add_regime_labels",
    "add_signal_labels",
    "build_feature_table",
    "build_features",
    "build_labeled_dataset",
    "calculate_rsi",
    "create_labeler_from_config",
    "create_market_data_loader",
    "create_model_from_config",
    "create_regime_labeler_from_config",
    "ensure_registry_loaded",
    "get_model_class",
    "LightGBMClassifierSignalModel",
    "LightGBMRankerSignalModel",
    "register_model",
    "resolve_model_type",
    "RiverOnlineSignalModel",
    "signal_to_text",
    "TFTSequenceSignalModel",
    "XGBoostClassifierSignalModel",
]
