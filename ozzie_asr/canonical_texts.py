"""Canonical Quran text loader for Al-Fatiha and Al-Ikhlas.

Loads pre-normalized canonical texts from configs/canonical_texts.json.
All texts are stored WITHOUT diacritics for evaluation purposes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# Default path to canonical texts config
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "canonical_texts.json"


class CanonicalTexts:
    """Loader and accessor for canonical Quran surah texts."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional custom config path.

        Args:
            config_path: Path to canonical_texts.json. Defaults to configs/canonical_texts.json.
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._data: Optional[Dict] = None

    def _load(self) -> Dict:
        """Load and cache the config data."""
        if self._data is None:
            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Canonical texts config not found: {self.config_path}\n"
                    "Please ensure configs/canonical_texts.json exists."
                )
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    @property
    def surahs(self) -> Dict:
        """Get all surah data."""
        return self._load()["surahs"]

    def get_surah_ids(self) -> List[str]:
        """Get list of available surah IDs (e.g., ['fatiha', 'ikhlas'])."""
        return list(self.surahs.keys())

    def get_text(self, surah_id: str, with_diacritics: bool = False) -> str:
        """Get the full text for a surah.

        Args:
            surah_id: Surah identifier ('fatiha' or 'ikhlas')
            with_diacritics: If True, return text_with_diacritics if available,
                           otherwise fall back to text_normalized.

        Returns:
            Full text as a single string.

        Raises:
            KeyError: If surah_id is not found.
        """
        if surah_id not in self.surahs:
            raise KeyError(
                f"Unknown surah: '{surah_id}'. Available: {self.get_surah_ids()}"
            )
        
        surah_data = self.surahs[surah_id]
        
        if with_diacritics and "text_with_diacritics" in surah_data:
            return surah_data["text_with_diacritics"]
        
        return surah_data["text_normalized"]

    def get_ayat(self, surah_id: str) -> List[str]:
        """Get individual ayat (verses) for a surah.

        Args:
            surah_id: Surah identifier ('fatiha' or 'ikhlas')

        Returns:
            List of normalized ayat strings.
        """
        if surah_id not in self.surahs:
            raise KeyError(
                f"Unknown surah: '{surah_id}'. Available: {self.get_surah_ids()}"
            )
        return self.surahs[surah_id]["ayat"]

    def get_surah_name(self, surah_id: str) -> str:
        """Get the English name of a surah."""
        return self.surahs[surah_id]["name"]

    def get_all_texts(self, with_diacritics: bool = False) -> Dict[str, str]:
        """Get a dict mapping surah_id -> text for all surahs.
        
        Args:
            with_diacritics: If True, return texts with diacritics if available.
        
        Returns:
            Dict of surah_id -> text.
        """
        return {sid: self.get_text(sid, with_diacritics=with_diacritics) for sid in self.get_surah_ids()}


# Module-level singleton for convenience
_default_instance: Optional[CanonicalTexts] = None


def get_canonical_texts(config_path: Optional[Path] = None) -> CanonicalTexts:
    """Get canonical texts instance (singleton for default path).

    Args:
        config_path: Optional custom config path. If provided, creates new instance.

    Returns:
        CanonicalTexts instance.
    """
    global _default_instance
    if config_path is not None:
        return CanonicalTexts(config_path)
    if _default_instance is None:
        _default_instance = CanonicalTexts()
    return _default_instance


# Convenience constants for direct import
SURAH_FATIHA = "fatiha"
SURAH_IKHLAS = "ikhlas"
SURAH_UNKNOWN = "unknown"

