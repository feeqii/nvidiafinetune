"""Unit tests for Arabic text normalization.

Run with: python -m pytest scripts/test_normalize.py -v
Or: make test
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ozzie_asr.text_normalize import (
    ArabicNormalizer,
    collapse_whitespace,
    normalize_alef,
    normalize_alef_maqsurah,
    normalize_arabic,
    normalize_taa_marbuta,
    remove_diacritics,
    remove_punctuation,
    remove_tatweel,
)


class TestRemoveDiacritics:
    """Test diacritics removal."""

    def test_remove_fatha(self):
        """Test removal of fatha."""
        assert remove_diacritics("بَ") == "ب"

    def test_remove_damma(self):
        """Test removal of damma."""
        assert remove_diacritics("بُ") == "ب"

    def test_remove_kasra(self):
        """Test removal of kasra."""
        assert remove_diacritics("بِ") == "ب"

    def test_remove_shadda(self):
        """Test removal of shadda."""
        assert remove_diacritics("بّ") == "ب"

    def test_remove_sukun(self):
        """Test removal of sukun."""
        assert remove_diacritics("بْ") == "ب"

    def test_remove_tanween(self):
        """Test removal of tanween (fathatan, dammatan, kasratan)."""
        assert remove_diacritics("بً") == "ب"
        assert remove_diacritics("بٌ") == "ب"
        assert remove_diacritics("بٍ") == "ب"

    def test_full_word_with_diacritics(self):
        """Test removal from a fully vocalized word."""
        # بِسْمِ (bismi) with diacritics
        assert remove_diacritics("بِسْمِ") == "بسم"

    def test_bismillah_with_diacritics(self):
        """Test removal from bismillah with full tashkeel."""
        # بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
        input_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        expected = "بسم الله الرحمن الرحيم"
        assert remove_diacritics(input_text) == expected


class TestRemoveTatweel:
    """Test tatweel (kashida) removal."""

    def test_remove_single_tatweel(self):
        """Test removal of single tatweel."""
        assert remove_tatweel("بـسم") == "بسم"

    def test_remove_multiple_tatweel(self):
        """Test removal of multiple tatweels."""
        assert remove_tatweel("بـــسم") == "بسم"

    def test_no_tatweel(self):
        """Test text without tatweel remains unchanged."""
        assert remove_tatweel("بسم") == "بسم"


class TestNormalizeAlef:
    """Test alef form normalization."""

    def test_alef_with_madda(self):
        """Test آ → ا."""
        assert normalize_alef("آ") == "ا"
        assert normalize_alef("القرآن") == "القران"

    def test_alef_with_hamza_above(self):
        """Test أ → ا."""
        assert normalize_alef("أ") == "ا"
        assert normalize_alef("أحمد") == "احمد"

    def test_alef_with_hamza_below(self):
        """Test إ → ا."""
        assert normalize_alef("إ") == "ا"
        assert normalize_alef("إسلام") == "اسلام"

    def test_alef_wasla(self):
        """Test ٱ → ا."""
        assert normalize_alef("ٱ") == "ا"

    def test_mixed_alef_forms(self):
        """Test text with multiple alef forms."""
        input_text = "آية إلى أحد"
        expected = "اية الى احد"
        assert normalize_alef(input_text) == expected


class TestNormalizeAlefMaqsurah:
    """Test alef maqsurah to yaa normalization."""

    def test_alef_maqsurah_to_yaa(self):
        """Test ى → ي."""
        assert normalize_alef_maqsurah("ى") == "ي"

    def test_word_ending_with_alef_maqsurah(self):
        """Test word ending with alef maqsurah."""
        assert normalize_alef_maqsurah("على") == "علي"
        assert normalize_alef_maqsurah("إلى") == "إلي"  # Note: alef not normalized here

    def test_multiple_alef_maqsurah(self):
        """Test multiple occurrences."""
        assert normalize_alef_maqsurah("على وإلى") == "علي وإلي"


class TestNormalizeTaaMarbuta:
    """Test taa marbuta to haa normalization."""

    def test_taa_marbuta_to_haa(self):
        """Test ة → ه."""
        assert normalize_taa_marbuta("ة") == "ه"

    def test_word_ending_with_taa_marbuta(self):
        """Test word ending with taa marbuta."""
        assert normalize_taa_marbuta("الفاتحة") == "الفاتحه"
        assert normalize_taa_marbuta("سورة") == "سوره"

    def test_multiple_taa_marbuta(self):
        """Test multiple occurrences."""
        assert normalize_taa_marbuta("سورة الفاتحة") == "سوره الفاتحه"


class TestRemovePunctuation:
    """Test punctuation removal."""

    def test_arabic_comma(self):
        """Test removal of Arabic comma."""
        assert remove_punctuation("بسم، الله").strip() == "بسم  الله"

    def test_arabic_question_mark(self):
        """Test removal of Arabic question mark."""
        result = remove_punctuation("ما هذا؟")
        assert "؟" not in result

    def test_general_punctuation(self):
        """Test removal of general punctuation."""
        result = remove_punctuation("بسم. الله!")
        assert "." not in result
        assert "!" not in result


class TestCollapseWhitespace:
    """Test whitespace normalization."""

    def test_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        assert collapse_whitespace("بسم   الله") == "بسم الله"

    def test_leading_trailing_spaces(self):
        """Test stripping leading/trailing spaces."""
        assert collapse_whitespace("  بسم الله  ") == "بسم الله"

    def test_tabs_and_newlines(self):
        """Test handling tabs and newlines."""
        assert collapse_whitespace("بسم\n\tالله") == "بسم الله"


class TestNormalizeArabic:
    """Test full normalization pipeline."""

    def test_full_normalization_default(self):
        """Test full pipeline with default settings (taa marbuta OFF)."""
        # Input with diacritics, various alef forms, alef maqsurah
        input_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        expected = "بسم الله الرحمن الرحيم"
        assert normalize_arabic(input_text) == expected

    def test_full_normalization_preserves_taa_marbuta(self):
        """Test that taa marbuta is preserved by default."""
        input_text = "سُورَةُ الفَاتِحَة"
        result = normalize_arabic(input_text)
        assert "ة" in result  # Taa marbuta should be preserved

    def test_full_normalization_with_taa_marbuta(self):
        """Test full pipeline with taa marbuta normalization ON."""
        input_text = "سُورَةُ الفَاتِحَة"
        result = normalize_arabic(input_text, normalize_taa_marbuta_flag=True)
        assert "ة" not in result
        assert "ه" in result

    def test_empty_string(self):
        """Test handling of empty string."""
        assert normalize_arabic("") == ""

    def test_already_normalized(self):
        """Test text that's already normalized."""
        input_text = "بسم الله الرحمن الرحيم"
        assert normalize_arabic(input_text) == input_text

    def test_fatiha_first_ayah(self):
        """Test normalization of Al-Fatiha first ayah with full tashkeel."""
        input_text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"
        expected = "بسم الله الرحمن الرحيم"
        assert normalize_arabic(input_text) == expected

    def test_ikhlas_full_with_tashkeel(self):
        """Test normalization of Al-Ikhlas with tashkeel."""
        input_text = "قُلْ هُوَ ٱللَّهُ أَحَدٌ"
        expected = "قل هو الله احد"
        assert normalize_arabic(input_text) == expected


class TestArabicNormalizer:
    """Test ArabicNormalizer class."""

    def test_default_normalizer(self):
        """Test default normalizer settings."""
        normalizer = ArabicNormalizer()
        input_text = "بِسْمِ اللَّهِ"
        expected = "بسم الله"
        assert normalizer(input_text) == expected

    def test_normalizer_with_taa_marbuta(self):
        """Test normalizer with taa marbuta enabled."""
        normalizer = ArabicNormalizer(normalize_taa_marbuta=True)
        input_text = "الفاتحة"
        result = normalizer(input_text)
        assert "ه" in result
        assert "ة" not in result

    def test_normalizer_config(self):
        """Test get_config returns correct settings."""
        normalizer = ArabicNormalizer(normalize_taa_marbuta=True)
        config = normalizer.get_config()
        assert config["normalize_taa_marbuta"] is True
        assert config["remove_diacritics"] is True

    def test_normalizer_preserves_taa_marbuta_by_default(self):
        """Test that default normalizer preserves taa marbuta."""
        normalizer = ArabicNormalizer()
        config = normalizer.get_config()
        assert config["normalize_taa_marbuta"] is False


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_mixed_arabic_english(self):
        """Test text with mixed Arabic and English."""
        input_text = "سورة Al-Fatiha الفاتحة"
        result = normalize_arabic(input_text)
        assert "Al" in result or "al" in result.lower()

    def test_numbers(self):
        """Test text with numbers."""
        input_text = "آية 1"
        result = normalize_arabic(input_text)
        assert "1" in result

    def test_only_diacritics(self):
        """Test string of only diacritics."""
        input_text = "ًٌٍَُِّْ"
        result = normalize_arabic(input_text)
        assert result == ""

    def test_unicode_normalization_consistency(self):
        """Test that output is consistent regardless of Unicode normalization form."""
        # Same text in different Unicode normalization forms should produce same output
        import unicodedata

        input_nfc = unicodedata.normalize("NFC", "بِسْمِ")
        input_nfd = unicodedata.normalize("NFD", "بِسْمِ")

        result_nfc = normalize_arabic(input_nfc)
        result_nfd = normalize_arabic(input_nfd)

        assert result_nfc == result_nfd


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

