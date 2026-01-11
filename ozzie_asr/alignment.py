"""Word-level alignment for snap-to-canonical evaluation.

Implements edit-distance based alignment to find the best matching
substring in a canonical text for partial clip evaluation.
"""

import difflib
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class AlignmentResult:
    """Result of aligning a hypothesis to a reference.

    Attributes:
        hypothesis: Original hypothesis text.
        reference: Full reference text.
        snapped_text: Best matching substring from reference.
        start_word_idx: Start index (word-level) in reference.
        end_word_idx: End index (word-level) in reference.
        similarity_ratio: Similarity score (0-1) between hypothesis and snapped text.
        alignment_ops: List of alignment operations.
    """

    hypothesis: str
    reference: str
    snapped_text: str
    start_word_idx: int
    end_word_idx: int
    similarity_ratio: float
    alignment_ops: Optional[List[Tuple[str, str, str]]] = None


def word_tokenize(text: str) -> List[str]:
    """Simple word tokenization by splitting on whitespace.

    Args:
        text: Text to tokenize.

    Returns:
        List of word tokens.
    """
    return text.split()


def compute_similarity(seq1: List[str], seq2: List[str]) -> float:
    """Compute similarity ratio between two word sequences.

    Uses difflib.SequenceMatcher for robust similarity calculation.

    Args:
        seq1: First word sequence.
        seq2: Second word sequence.

    Returns:
        Similarity ratio between 0 and 1.
    """
    if not seq1 or not seq2:
        return 0.0

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    return matcher.ratio()


def find_best_alignment(
    hypothesis: str,
    reference: str,
    min_overlap: float = 0.3,
) -> AlignmentResult:
    """Find the best matching substring in reference for the hypothesis.

    Uses a sliding window approach with edit-distance scoring to find
    the portion of the reference that best matches the hypothesis.

    Args:
        hypothesis: Predicted/transcribed text.
        reference: Full canonical reference text.
        min_overlap: Minimum overlap ratio to consider valid (default: 0.3).

    Returns:
        AlignmentResult with best matching substring and indices.
    """
    hyp_words = word_tokenize(hypothesis)
    ref_words = word_tokenize(reference)

    if not hyp_words:
        return AlignmentResult(
            hypothesis=hypothesis,
            reference=reference,
            snapped_text="",
            start_word_idx=0,
            end_word_idx=0,
            similarity_ratio=0.0,
        )

    if not ref_words:
        return AlignmentResult(
            hypothesis=hypothesis,
            reference=reference,
            snapped_text="",
            start_word_idx=0,
            end_word_idx=0,
            similarity_ratio=0.0,
        )

    hyp_len = len(hyp_words)
    ref_len = len(ref_words)

    best_score = 0.0
    best_start = 0
    best_end = ref_len

    # Try different window sizes around hypothesis length
    # Allow windows from 50% to 150% of hypothesis length
    min_window = max(1, int(hyp_len * 0.5))
    max_window = min(ref_len, int(hyp_len * 1.5) + 1)

    for window_size in range(min_window, max_window + 1):
        for start in range(ref_len - window_size + 1):
            end = start + window_size
            window_words = ref_words[start:end]

            score = compute_similarity(hyp_words, window_words)

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    # Also try the full reference
    full_score = compute_similarity(hyp_words, ref_words)
    if full_score > best_score:
        best_score = full_score
        best_start = 0
        best_end = ref_len

    snapped_words = ref_words[best_start:best_end]
    snapped_text = " ".join(snapped_words)

    return AlignmentResult(
        hypothesis=hypothesis,
        reference=reference,
        snapped_text=snapped_text,
        start_word_idx=best_start,
        end_word_idx=best_end,
        similarity_ratio=best_score,
    )


def get_alignment_operations(
    hypothesis: str,
    reference: str,
) -> List[Tuple[str, str, str]]:
    """Get detailed alignment operations between hypothesis and reference.

    Args:
        hypothesis: Predicted text.
        reference: Reference text.

    Returns:
        List of (operation, hyp_word, ref_word) tuples.
        Operations: 'equal', 'replace', 'insert', 'delete'
    """
    hyp_words = word_tokenize(hypothesis)
    ref_words = word_tokenize(reference)

    matcher = difflib.SequenceMatcher(None, hyp_words, ref_words)
    operations = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                operations.append(("equal", hyp_words[i1 + k], ref_words[j1 + k]))
        elif op == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                hyp_word = hyp_words[i1 + k] if i1 + k < i2 else ""
                ref_word = ref_words[j1 + k] if j1 + k < j2 else ""
                operations.append(("replace", hyp_word, ref_word))
        elif op == "insert":
            for k in range(j2 - j1):
                operations.append(("insert", "", ref_words[j1 + k]))
        elif op == "delete":
            for k in range(i2 - i1):
                operations.append(("delete", hyp_words[i1 + k], ""))

    return operations


def align_to_canonical(
    hypothesis: str,
    canonical_texts: dict,
    normalize_fn=None,
) -> Tuple[str, AlignmentResult]:
    """Align hypothesis to the best matching canonical text.

    First classifies which surah the hypothesis belongs to,
    then aligns to that surah's canonical text.

    Args:
        hypothesis: Predicted/transcribed text.
        canonical_texts: Dict mapping surah_id -> canonical text.
        normalize_fn: Optional normalization function to apply.

    Returns:
        Tuple of (surah_id, AlignmentResult).
    """
    if normalize_fn:
        hypothesis = normalize_fn(hypothesis)
        canonical_texts = {
            k: normalize_fn(v) for k, v in canonical_texts.items()
        }

    # Find best matching surah
    best_surah = None
    best_alignment = None
    best_score = -1

    for surah_id, canonical_text in canonical_texts.items():
        alignment = find_best_alignment(hypothesis, canonical_text)

        if alignment.similarity_ratio > best_score:
            best_score = alignment.similarity_ratio
            best_surah = surah_id
            best_alignment = alignment

    if best_surah is None:
        # Default to first surah if no match
        first_surah = list(canonical_texts.keys())[0]
        best_alignment = find_best_alignment(
            hypothesis, canonical_texts[first_surah]
        )
        best_surah = first_surah

    return best_surah, best_alignment

