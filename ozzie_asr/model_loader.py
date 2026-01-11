"""Model loading utilities for NVIDIA NeMo ASR models.

Supports:
1. Loading local .nemo checkpoint files via restore_from()
2. Downloading models from NGC via ngc CLI
3. Listing available Arabic ASR models from NeMo

Target model: speechtotext_ar_ar_conformer (trainable_v3.0)
File: Conformer-CTC-L_spe128_ar-AR_3.0.nemo
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# NGC model identifiers
NGC_MODEL_ID = "speechtotext_ar_ar_conformer"
NGC_MODEL_VERSION = "trainable_v3.0"
NGC_MODEL_ORG = "nvidia/riva"
NGC_NEMO_FILENAME = "Conformer-CTC-L_spe128_ar-AR_3.0.nemo"


def check_ngc_cli() -> bool:
    """Check if NGC CLI is installed and configured.

    Returns:
        True if NGC CLI is available and configured.
    """
    try:
        result = subprocess.run(
            ["ngc", "config", "current"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def download_model_ngc(
    model_id: str = NGC_MODEL_ID,
    version: str = NGC_MODEL_VERSION,
    output_dir: Path = Path("./models"),
) -> Path:
    """Download a model from NGC using ngc CLI.

    Args:
        model_id: NGC model identifier (e.g., 'speechtotext_ar_ar_conformer').
        version: Model version (e.g., 'trainable_v3.0').
        output_dir: Directory to save the downloaded model.

    Returns:
        Path to the downloaded .nemo file.

    Raises:
        RuntimeError: If NGC CLI is not available or download fails.
    """
    if not check_ngc_cli():
        raise RuntimeError(
            "NGC CLI is not installed or configured.\n"
            "Install: https://org.ngc.nvidia.com/setup/installers/cli\n"
            "Configure: ngc config set"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full NGC model path
    ngc_model_path = f"{NGC_MODEL_ORG}/{model_id}:{version}"

    logger.info(f"Downloading model from NGC: {ngc_model_path}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Download using NGC CLI
        result = subprocess.run(
            [
                "ngc",
                "registry",
                "model",
                "download-version",
                ngc_model_path,
                "--dest",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for large models
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"NGC download failed:\n{result.stderr}\n{result.stdout}"
            )

        logger.info("Download completed successfully")

    except subprocess.TimeoutExpired:
        raise RuntimeError("NGC download timed out (30 minutes)")

    # Find the downloaded .nemo file
    # NGC downloads to: output_dir/model_id_version/filename
    download_subdir = output_dir / f"{model_id}_v{version.replace('_', '')}"

    # Try multiple possible paths
    possible_paths = [
        download_subdir / NGC_NEMO_FILENAME,
        output_dir / f"{model_id}_{version}" / NGC_NEMO_FILENAME,
        output_dir / model_id / version / NGC_NEMO_FILENAME,
    ]

    # Also search recursively for any .nemo file
    nemo_files = list(output_dir.rglob("*.nemo"))

    for path in possible_paths:
        if path.exists():
            logger.info(f"Found model at: {path}")
            return path

    if nemo_files:
        logger.info(f"Found model at: {nemo_files[0]}")
        return nemo_files[0]

    raise RuntimeError(
        f"Download succeeded but .nemo file not found in {output_dir}\n"
        f"Searched: {possible_paths}\n"
        f"Found .nemo files: {nemo_files}"
    )


def load_nemo_model(model_path: Path, device: str = "cuda"):
    """Load a NeMo ASR model from a local .nemo checkpoint.

    Args:
        model_path: Path to the .nemo checkpoint file.
        device: Device to load model on ('cuda' or 'cpu').

    Returns:
        Loaded NeMo EncDecCTCModel.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ImportError: If NeMo is not installed.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not model_path.suffix == ".nemo":
        raise ValueError(f"Expected .nemo file, got: {model_path.suffix}")

    try:
        from nemo.collections.asr.models import EncDecCTCModel
    except ImportError as e:
        raise ImportError(
            "NeMo ASR toolkit not installed. Install with:\n"
            "pip install nemo_toolkit[asr]"
        ) from e

    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Target device: {device}")

    # Load the model
    model = EncDecCTCModel.restore_from(
        restore_path=str(model_path),
        map_location=device,
    )

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully: {type(model).__name__}")

    return model


def list_available_models(filter_arabic: bool = True) -> List[dict]:
    """List available NeMo ASR models.

    Args:
        filter_arabic: If True, only return Arabic models.

    Returns:
        List of model info dicts with 'name' and 'description' keys.
    """
    try:
        from nemo.collections.asr.models import EncDecCTCModel
    except ImportError:
        logger.warning("NeMo not installed, cannot list models")
        return []

    models = []
    try:
        available = EncDecCTCModel.list_available_models()
        for model_info in available:
            name = model_info.pretrained_model_name
            if filter_arabic and "ar" not in name.lower():
                continue
            models.append({
                "name": name,
                "description": getattr(model_info, "description", ""),
            })
    except Exception as e:
        logger.warning(f"Could not list models: {e}")

    return models


def get_model_info(model_path: Optional[Path] = None) -> dict:
    """Get information about the model for logging/reproducibility.

    Args:
        model_path: Path to the model file (optional).

    Returns:
        Dict with model metadata.
    """
    info = {
        "ngc_model_id": NGC_MODEL_ID,
        "ngc_model_version": NGC_MODEL_VERSION,
        "ngc_org": NGC_MODEL_ORG,
        "expected_filename": NGC_NEMO_FILENAME,
    }

    if model_path:
        model_path = Path(model_path)
        info["local_path"] = str(model_path)
        info["file_exists"] = model_path.exists()
        if model_path.exists():
            info["file_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)

    return info


class ModelLoader:
    """High-level model loader with caching and configuration.

    Handles both NGC download and local .nemo loading with a unified interface.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_id: str = NGC_MODEL_ID,
        model_version: str = NGC_MODEL_VERSION,
        cache_dir: Path = Path("./models"),
        device: str = "cuda",
    ):
        """Initialize model loader.

        Args:
            model_path: Path to local .nemo file (takes precedence if provided).
            model_id: NGC model ID for download.
            model_version: NGC model version for download.
            cache_dir: Directory to cache downloaded models.
            device: Device for model loading ('cuda' or 'cpu').
        """
        self.model_path = Path(model_path) if model_path else None
        self.model_id = model_id
        self.model_version = model_version
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._model = None

    def load(self, force_download: bool = False):
        """Load the model, downloading if necessary.

        Args:
            force_download: If True, re-download even if cached.

        Returns:
            Loaded NeMo model.
        """
        # If local path provided, use it directly
        if self.model_path and self.model_path.exists() and not force_download:
            logger.info(f"Using local model: {self.model_path}")
            self._model = load_nemo_model(self.model_path, self.device)
            return self._model

        # Check cache
        cached_path = self._find_cached_model()
        if cached_path and not force_download:
            logger.info(f"Using cached model: {cached_path}")
            self._model = load_nemo_model(cached_path, self.device)
            return self._model

        # Download from NGC
        logger.info("Model not found locally, downloading from NGC...")
        downloaded_path = download_model_ngc(
            model_id=self.model_id,
            version=self.model_version,
            output_dir=self.cache_dir,
        )

        self._model = load_nemo_model(downloaded_path, self.device)
        return self._model

    def _find_cached_model(self) -> Optional[Path]:
        """Find a cached model in the cache directory.

        Returns:
            Path to cached .nemo file, or None if not found.
        """
        if not self.cache_dir.exists():
            return None

        # Look for the expected filename
        for nemo_file in self.cache_dir.rglob("*.nemo"):
            if NGC_NEMO_FILENAME in nemo_file.name:
                return nemo_file

        return None

    @property
    def model(self):
        """Get the loaded model (loads if not already loaded)."""
        if self._model is None:
            self.load()
        return self._model

    def get_info(self) -> dict:
        """Get model information for logging."""
        return get_model_info(self.model_path or self._find_cached_model())

