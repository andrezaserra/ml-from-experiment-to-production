import json
import os
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

MODEL_NAME = "satellite-anomaly-classifier"
MODEL_ALIAS = "champion"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = PROJECT_ROOT / "release"
TARGET_MODEL_DIR = RELEASE_DIR / "model"
MANIFEST_PATH = RELEASE_DIR / "model-manifest.json"


def export_release() -> tuple[Path, Path]:
    """Resolve the champion alias and export an immutable model version."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    version = str(model_version.version)

    # Important: after resolving the alias, use the immutable version URI.
    model_uri = f"models:/{MODEL_NAME}/{version}"

    downloaded_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
        )
    )

    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)

    RELEASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        downloaded_path,
        TARGET_MODEL_DIR,
    )

    manifest = {
        "model_name": MODEL_NAME,
        "source_alias": MODEL_ALIAS,
        "model_version": version,
        "source_run_id": model_version.run_id,
        "model_uri": model_uri,
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TARGET_MODEL_DIR, MANIFEST_PATH


def main():
    model_dir, manifest_path = export_release()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )

    print("Release created.")
    print(f"Model:   {manifest['model_name']}")
    print(f"Alias:   {manifest['source_alias']}")
    print(f"Version: {manifest['model_version']}")
    print(f"Run ID:  {manifest['source_run_id']}")
    print(f"URI:     {manifest['model_uri']}")
    print(f"Target:  {model_dir}")


if __name__ == "__main__":
    main()
