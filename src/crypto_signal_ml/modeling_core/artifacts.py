"""Artifact bundle and manifest persistence helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
from typing import Any

import pandas as pd

from ..config import TrainingConfig, config_to_dict
from ..features_core.diagnostics import build_feature_registry_snapshot
from ..labels_core.contracts import LabelRecipe


@dataclass(frozen=True)
class ModelArtifactManifest:
    """Versioned manifest for one persisted trained model bundle."""

    modelVersion: str
    featureVersion: str
    labelVersion: str
    calibrationVersion: str
    trainingConfigHash: str
    gitCommitHash: str | None
    trainedAt: str
    modelType: str
    featurePack: str
    classOrder: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest into a JSON-friendly dictionary."""

        return asdict(self)


def _hash_training_config(config: TrainingConfig) -> str:
    """Create a stable content hash for the active training configuration."""

    payload = json.dumps(config_to_dict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_git_commit_hash(project_root: Path) -> str | None:
    """Resolve the current git commit hash when available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    commit_hash = completed.stdout.strip()
    return commit_hash or None


def save_model_artifact_bundle(
    *,
    artifact_dir: Path,
    model,
    metrics: dict[str, Any],
    prediction_df: pd.DataFrame,
    label_recipe: LabelRecipe,
    dataset_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> ModelArtifactManifest:
    """Persist the production-grade artifact directory for one trained model."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[3]
    trained_at = datetime.now(timezone.utc).isoformat()
    calibration_summary = dict(getattr(model, "calibration_summary", {}) or {})
    calibration_version = "sigmoid-v1" if calibration_summary.get("enabled") else "uncalibrated-v1"

    manifest = ModelArtifactManifest(
        modelVersion=f"{model.model_type}-{trained_at}",
        featureVersion="feature-registry-v1",
        labelVersion=str(label_recipe.version),
        calibrationVersion=calibration_version,
        trainingConfigHash=_hash_training_config(model.config),
        gitCommitHash=_resolve_git_commit_hash(project_root),
        trainedAt=trained_at,
        modelType=str(model.model_type),
        featurePack=str(model.config.feature_pack),
        classOrder=[-1, 0, 1],
    )

    with (artifact_dir / "estimator.pkl").open("wb") as output_file:
        pickle.dump(model.estimator, output_file)
    with (artifact_dir / "calibrator.pkl").open("wb") as output_file:
        pickle.dump(getattr(model, "calibrator", None), output_file)
    with (artifact_dir / "manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest.to_dict(), output_file, indent=2)
    with (artifact_dir / "label_recipe.json").open("w", encoding="utf-8") as output_file:
        json.dump(label_recipe.to_dict(), output_file, indent=2)
    with (artifact_dir / "feature_registry_snapshot.json").open("w", encoding="utf-8") as output_file:
        json.dump(build_feature_registry_snapshot(model.feature_columns), output_file, indent=2)
    with (artifact_dir / "evaluation_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "datasetPath": str(dataset_path),
                "trainRows": int(len(train_df)),
                "testRows": int(len(test_df)),
                "metrics": metrics,
                "predictionRowCount": int(len(prediction_df)),
                "calibration": calibration_summary,
                "config": config_to_dict(model.config),
            },
            output_file,
            indent=2,
        )

    prediction_df.to_csv(artifact_dir / "predictions.csv", index=False)
    return manifest
