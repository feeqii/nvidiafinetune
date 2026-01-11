"""Arabic text normalization for ASR evaluation.

Implements a deterministic normalization pipeline for Arabic text:
1. Remove diacritics (harakat, shadda, sukun, etc.)
2. Remove tatweel (kashida)
3. Normalize alef forms (إأآا → ا)
4. Normalize yaa/alef maqsurah (ى → ي)
5. Optionally normalize taa marbuta (ة → ه) - OFF by default
6. Remove punctuation
7. Collapse whitespace

All operations are deterministic and reversible where noted.
"""

import re
import unicodedata
from typing import Optional


# Arabic diacritics (harakat) Unicode ranges
# Fatha, Damma, Kasra, Fathatan, Dammatan, Kasratan, Shadda, Sukun
ARABIC_DIACRITICS = (
    "\u064B"  # FATHATAN
    "\u064C"  # DAMMATAN
    "\u064D"  # KASRATAN
    "\u064E"  # FATHA
    "\u064F"  # DAMMA
    "\u0650"  # KASRA
    "\u0651"  # SHADDA
    "\u0652"  # SUKUN
    "\u0670"  # SUPERSCRIPT ALEF (dagger alef)
    "\u0657"  # INVERTED DAMMA
    "\u0658"  # MARK NOON GHUNNA
    "\u065C"  # VOWEL SIGN DOT BELOW
    "\u065D"  # REVERSED DAMMA
    "\u065E"  # FATHA WITH TWO DOTS
    "\u0656"  # SUBSCRIPT ALEF
)

# Tatweel (kashida) - Arabic elongation character
TATWEEL = "\u0640"

# Alef forms to normalize
ALEF_FORMS = {
    "\u0622": "\u0627",  # ALEF WITH MADDA ABOVE → ALEF
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE → ALEF
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW → ALEF
    "\u0671": "\u0627",  # ALEF WASLA → ALEF
    "\u0672": "\u0627",  # ALEF WITH WAVY HAMZA ABOVE → ALEF
    "\u0673": "\u0627",  # ALEF WITH WAVY HAMZA BELOW → ALEF
}

# Hamza forms (standalone hamza normalization)
HAMZA_FORMS = {
    "\u0624": "\u0648",  # WAW WITH HAMZA ABOVE → WAW (optional, not applied by default)
    "\u0626": "\u064A",  # YEH WITH HAMZA ABOVE → YEH (optional, not applied by default)
}

# Yaa / Alef Maqsurah normalization
ALEF_MAQSURAH = "\u0649"  # ى
YAA = "\u064A"  # ي

# Taa Marbuta
TAA_MARBUTA = "\u0629"  # ة
HAA = "\u0647"  # ه

# Arabic punctuation to remove
ARABIC_PUNCTUATION = (
    "\u060C"  # Arabic comma
    "\u061B"  # Arabic semicolon
    "\u061F"  # Arabic question mark
    "\u06D4"  # Arabic full stop
    "\u066A"  # Arabic percent sign
    "\u066B"  # Arabic decimal separator
    "\u066C"  # Arabic thousands separator
)

# General punctuation pattern
PUNCTUATION_PATTERN = re.compile(r'[^\w\s\u0600-\u06FF]', re.UNICODE)


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (harakat) from text.

    Args:
        text: Arabic text potentially containing diacritics.

    Returns:
        Text with all diacritics removed.
    """
    for char in ARABIC_DIACRITICS:
        text = text.replace(char, "")
    return text


def remove_tatweel(text: str) -> str:
    """Remove tatweel (kashida) elongation character.

    Args:
        text: Arabic text potentially containing tatweel.

    Returns:
        Text with tatweel removed.
    """
    return text.replace(TATWEEL, "")


def normalize_alef(text: str) -> str:
    """Normalize all alef forms to plain alef (ا).

    Converts: آ أ إ ٱ → ا

    Args:
        text: Arabic text with various alef forms.

    Returns:
        Text with unified alef form.
    """
    for old, new in ALEF_FORMS.items():
        text = text.replace(old, new)
    return text


def normalize_alef_maqsurah(text: str) -> str:
    """Normalize alef maqsurah (ى) to yaa (ي).

    Args:
        text: Arabic text potentially containing alef maqsurah.

    Returns:
        Text with alef maqsurah converted to yaa.
    """
    return text.replace(ALEF_MAQSURAH, YAA)


def normalize_taa_marbuta(text: str) -> str:
    """Normalize taa marbuta (ة) to haa (ه).

    WARNING: This can distort matching in some cases.
    Use with caution - disabled by default.

    Args:
        text: Arabic text potentially containing taa marbuta.

    Returns:
        Text with taa marbuta converted to haa.
    """
    return text.replace(TAA_MARBUTA, HAA)


def remove_punctuation(text: str) -> str:
    """Remove punctuation marks (Arabic and general).

    Args:
        text: Text potentially containing punctuation.

    Returns:
        Text with punctuation removed.
    """
    # Remove Arabic-specific punctuation
    for char in ARABIC_PUNCTUATION:
        text = text.replace(char, " ")

    # Remove general punctuation but keep Arabic letters and spaces
    text = PUNCTUATION_PATTERN.sub(" ", text)

    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to single space and strip.

    Args:
        text: Text potentially containing multiple spaces.

    Returns:
        Text with normalized whitespace.
    """
    return " ".join(text.split())


def normalize_arabic(
    text: str,
    remove_diacritics_flag: bool = True,
    remove_tatweel_flag: bool = True,
    normalize_alef_flag: bool = True,
    normalize_alef_maqsurah_flag: bool = True,
    normalize_taa_marbuta_flag: bool = False,  # OFF by default per requirements
    remove_punctuation_flag: bool = True,
    collapse_whitespace_flag: bool = True,
) -> str:
    """Apply full Arabic text normalization pipeline.

    Default settings remove diacritics and normalize character forms
    for ASR evaluation, but preserve taa marbuta distinction.

    Args:
        text: Arabic text to normalize.
        remove_diacritics_flag: Remove harakat (default: True).
        remove_tatweel_flag: Remove tatweel/kashida (default: True).
        normalize_alef_flag: Unify alef forms (default: True).
        normalize_alef_maqsurah_flag: Convert ى to ي (default: True).
        normalize_taa_marbuta_flag: Convert ة to ه (default: False).
        remove_punctuation_flag: Remove punctuation (default: True).
        collapse_whitespace_flag: Normalize whitespace (default: True).

    Returns:
        Normalized Arabic text.
    """
    if not text:
        return ""

    # Apply normalization steps in order
    if remove_diacritics_flag:
        text = remove_diacritics(text)

    if remove_tatweel_flag:
        text = remove_tatweel(text)

    if normalize_alef_flag:
        text = normalize_alef(text)

    if normalize_alef_maqsurah_flag:
        text = normalize_alef_maqsurah(text)

    if normalize_taa_marbuta_flag:
        text = normalize_taa_marbuta(text)

    if remove_punctuation_flag:
        text = remove_punctuation(text)

    if collapse_whitespace_flag:
        text = collapse_whitespace(text)

    return text


class ArabicNormalizer:
    """Configurable Arabic text normalizer.

    Provides a reusable normalizer instance with consistent settings.
    """

    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        normalize_alef: bool = True,
        normalize_alef_maqsurah: bool = True,
        normalize_taa_marbuta: bool = False,
        remove_punctuation: bool = True,
        collapse_whitespace: bool = True,
    ):
        """Initialize normalizer with configuration.

        Args:
            remove_diacritics: Remove harakat (default: True).
            remove_tatweel: Remove tatweel/kashida (default: True).
            normalize_alef: Unify alef forms (default: True).
            normalize_alef_maqsurah: Convert ى to ي (default: True).
            normalize_taa_marbuta: Convert ة to ه (default: False).
            remove_punctuation: Remove punctuation (default: True).
            collapse_whitespace: Normalize whitespace (default: True).
        """
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.normalize_alef = normalize_alef
        self.normalize_alef_maqsurah = normalize_alef_maqsurah
        self.normalize_taa_marbuta = normalize_taa_marbuta
        self.remove_punctuation = remove_punctuation
        self.collapse_whitespace = collapse_whitespace

    def __call__(self, text: str) -> str:
        """Normalize text using configured settings.

        Args:
            text: Arabic text to normalize.

        Returns:
            Normalized text.
        """
        return normalize_arabic(
            text,
            remove_diacritics_flag=self.remove_diacritics,
            remove_tatweel_flag=self.remove_tatweel,
            normalize_alef_flag=self.normalize_alef,
            normalize_alef_maqsurah_flag=self.normalize_alef_maqsurah,
            normalize_taa_marbuta_flag=self.normalize_taa_marbuta,
            remove_punctuation_flag=self.remove_punctuation,
            collapse_whitespace_flag=self.collapse_whitespace,
        )

    def get_config(self) -> dict:
        """Return current configuration as a dict."""
        return {
            "remove_diacritics": self.remove_diacritics,
            "remove_tatweel": self.remove_tatweel,
            "normalize_alef": self.normalize_alef,
            "normalize_alef_maqsurah": self.normalize_alef_maqsurah,
            "normalize_taa_marbuta": self.normalize_taa_marbuta,
            "remove_punctuation": self.remove_punctuation,
            "collapse_whitespace": self.collapse_whitespace,
        }


# Default normalizer instance (taa marbuta OFF)
default_normalizer = ArabicNormalizer()

