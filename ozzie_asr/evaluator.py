"""ASR evaluation with FULL_SURAH and SNAP_TO_CANONICAL modes.

Implements:
1. Surah classification based on similarity to canonical texts
2. FULL_SURAH mode: Compare to full canonical text
3. SNAP_TO_CANONICAL mode: Align to best matching substring (for partial clips)
4. CER/WER calculation using jiwer
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer
import pandas as pd

from .alignment import AlignmentResult, find_best_alignment
from .canonical_texts import (
    SURAH_FATIHA,
    SURAH_IKHLAS,
    SURAH_UNKNOWN,
    get_canonical_texts,
)
from .text_normalize import ArabicNormalizer, normalize_arabic

logger = logging.getLogger(__name__)


class EvalMode(Enum):
    """Evaluation mode."""

    FULL_SURAH = "full"
    SNAP_TO_CANONICAL = "snap"


@dataclass
class ClassificationResult:
    """Result of surah classification.

    Attributes:
        predicted_surah: Predicted surah ID ('fatiha', 'ikhlas', 'unknown').
        confidence: Confidence score (similarity ratio).
        scores: Dict of surah_id -> similarity score.
    """

    predicted_surah: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single transcription.

    Attributes:
        file: Original filename.
        duration_sec: Audio duration in seconds.
        surah_pred: Predicted surah ID.
        surah_confidence: Classification confidence.
        predicted_text_raw: Raw transcription from model.
        predicted_text_normalized: Normalized transcription.
        reference_text: Reference text used for evaluation.
        mode: Evaluation mode used.
        cer: Character Error Rate.
        wer: Word Error Rate.
        snapped_text: Snapped reference text (SNAP mode only).
        alignment_start: Start word index in canonical (SNAP mode only).
        alignment_end: End word index in canonical (SNAP mode only).
        notes: Additional notes or warnings.
    """

    file: str
    duration_sec: float
    surah_pred: str
    surah_confidence: float
    predicted_text_raw: str
    predicted_text_normalized: str
    reference_text: str
    mode: str
    cer: float
    wer: float
    snapped_text: Optional[str] = None
    alignment_start: Optional[int] = None
    alignment_end: Optional[int] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame/CSV."""
        return {
            "file": self.file,
            "duration_sec": self.duration_sec,
            "surah_pred": self.surah_pred,
            "surah_confidence": round(self.surah_confidence, 4),
            "predicted_text_raw": self.predicted_text_raw,
            "predicted_text_normalized": self.predicted_text_normalized,
            "reference_text": self.reference_text,
            "mode": self.mode,
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "snapped_text": self.snapped_text or "",
            "alignment_start": self.alignment_start,
            "alignment_end": self.alignment_end,
            "notes": self.notes,
        }


class Evaluator:
    """ASR evaluation with surah classification and multiple modes.

    Supports:
    - FULL_SURAH: Compare transcription to full canonical text
    - SNAP_TO_CANONICAL: Align to best matching substring (for partial clips)
    """

    def __init__(
        self,
        mode: EvalMode = EvalMode.SNAP_TO_CANONICAL,
        normalize_taa_marbuta: bool = False,
        canonical_config_path: Optional[Path] = None,
        unknown_threshold: float = 0.2,
        remove_diacritics: bool = True,
    ):
        """Initialize evaluator.

        Args:
            mode: Evaluation mode (default: SNAP_TO_CANONICAL).
            normalize_taa_marbuta: Whether to normalize ة to ه (default: False).
            canonical_config_path: Path to canonical_texts.json.
            unknown_threshold: Minimum similarity to classify as known surah.
            remove_diacritics: Whether to remove diacritics (default: True).
        """
        self.mode = mode
        self.unknown_threshold = unknown_threshold
        self.remove_diacritics = remove_diacritics

        # Initialize normalizer
        self.normalizer = ArabicNormalizer(
            normalize_taa_marbuta=normalize_taa_marbuta,
            remove_diacritics=remove_diacritics,
        )

        # Load canonical texts
        self.canonical = get_canonical_texts(canonical_config_path)
        self._canonical_normalized = {
            surah_id: self.normalizer(text)
            for surah_id, text in self.canonical.get_all_texts(
                with_diacritics=not remove_diacritics
            ).items()
        }

    def classify_surah(self, text: str) -> ClassificationResult:
        """Classify which surah a transcription belongs to.

        Uses character-level similarity to canonical texts.

        Args:
            text: Normalized transcription text.

        Returns:
            ClassificationResult with predicted surah and confidence.
        """
        if not text.strip():
            return ClassificationResult(
                predicted_surah=SURAH_UNKNOWN,
                confidence=0.0,
                scores={},
            )

        # Normalize input
        text_norm = self.normalizer(text)

        # Compare to each canonical text
        scores = {}
        for surah_id, canonical_text in self._canonical_normalized.items():
            # Use alignment to get best similarity
            alignment = find_best_alignment(text_norm, canonical_text)
            scores[surah_id] = alignment.similarity_ratio

        # Find best match
        best_surah = max(scores, key=scores.get)
        best_score = scores[best_surah]

        # Check threshold
        if best_score < self.unknown_threshold:
            return ClassificationResult(
                predicted_surah=SURAH_UNKNOWN,
                confidence=best_score,
                scores=scores,
            )

        return ClassificationResult(
            predicted_surah=best_surah,
            confidence=best_score,
            scores=scores,
        )

    def compute_metrics(
        self,
        hypothesis: str,
        reference: str,
    ) -> Tuple[float, float]:
        """Compute CER and WER between hypothesis and reference.

        Args:
            hypothesis: Predicted text.
            reference: Reference text.

        Returns:
            Tuple of (CER, WER).
        """
        if not reference:
            return 1.0, 1.0

        if not hypothesis:
            return 1.0, 1.0

        # Use jiwer for metrics
        try:
            # Character Error Rate
            cer = jiwer.cer(reference, hypothesis)

            # Word Error Rate
            wer = jiwer.wer(reference, hypothesis)

            return cer, wer

        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
            return 1.0, 1.0

    def evaluate_single(
        self,
        transcription: str,
        filename: str,
        duration_sec: float = 0.0,
        force_surah: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a single transcription.

        Args:
            transcription: Raw transcription from model.
            filename: Original audio filename.
            duration_sec: Audio duration in seconds.
            force_surah: Force a specific surah (skip classification).

        Returns:
            EvaluationResult with all metrics and metadata.
        """
        # Normalize transcription
        text_normalized = self.normalizer(transcription)

        # Classify surah
        if force_surah:
            classification = ClassificationResult(
                predicted_surah=force_surah,
                confidence=1.0,
                scores={force_surah: 1.0},
            )
        else:
            classification = self.classify_surah(transcription)

        surah_pred = classification.predicted_surah
        notes = []

        # Handle unknown classification
        if surah_pred == SURAH_UNKNOWN:
            # Default to fatiha for evaluation
            surah_pred = SURAH_FATIHA
            notes.append(f"Low confidence ({classification.confidence:.2f}), defaulted to fatiha")

        # Get canonical text
        canonical_text = self._canonical_normalized.get(surah_pred, "")

        # Evaluate based on mode
        if self.mode == EvalMode.FULL_SURAH:
            reference_text = canonical_text
            snapped_text = None
            alignment_start = None
            alignment_end = None

        else:  # SNAP_TO_CANONICAL
            alignment = find_best_alignment(text_normalized, canonical_text)
            reference_text = alignment.snapped_text
            snapped_text = alignment.snapped_text
            alignment_start = alignment.start_word_idx
            alignment_end = alignment.end_word_idx

            if alignment.similarity_ratio < 0.3:
                notes.append(f"Low alignment score ({alignment.similarity_ratio:.2f})")

        # Compute metrics
        cer, wer = self.compute_metrics(text_normalized, reference_text)

        return EvaluationResult(
            file=filename,
            duration_sec=duration_sec,
            surah_pred=classification.predicted_surah,
            surah_confidence=classification.confidence,
            predicted_text_raw=transcription,
            predicted_text_normalized=text_normalized,
            reference_text=reference_text,
            mode=self.mode.value,
            cer=cer,
            wer=wer,
            snapped_text=snapped_text,
            alignment_start=alignment_start,
            alignment_end=alignment_end,
            notes="; ".join(notes),
        )

    def evaluate_batch(
        self,
        transcriptions: List[Dict],
    ) -> List[EvaluationResult]:
        """Evaluate a batch of transcriptions.

        Args:
            transcriptions: List of dicts with 'transcription', 'filename',
                          and optionally 'duration_sec'.

        Returns:
            List of EvaluationResult objects.
        """
        results = []

        for item in transcriptions:
            result = self.evaluate_single(
                transcription=item.get("transcription", ""),
                filename=item.get("filename", "unknown"),
                duration_sec=item.get("duration_sec", 0.0),
                force_surah=item.get("force_surah"),
            )
            results.append(result)

        return results

    def get_config(self) -> Dict:
        """Get evaluator configuration."""
        return {
            "mode": self.mode.value,
            "normalize_taa_marbuta": self.normalizer.normalize_taa_marbuta,
            "remove_diacritics": self.remove_diacritics,
            "unknown_threshold": self.unknown_threshold,
            "normalizer_config": self.normalizer.get_config(),
        }


def compute_summary_metrics(results: List[EvaluationResult]) -> Dict:
    """Compute summary statistics from evaluation results.

    Args:
        results: List of EvaluationResult objects.

    Returns:
        Dict with summary statistics.
    """
    if not results:
        return {"error": "No results to summarize"}

    cers = [r.cer for r in results]
    wers = [r.wer for r in results]

    import statistics

    summary = {
        "total_files": len(results),
        "mean_cer": statistics.mean(cers),
        "median_cer": statistics.median(cers),
        "std_cer": statistics.stdev(cers) if len(cers) > 1 else 0,
        "min_cer": min(cers),
        "max_cer": max(cers),
        "mean_wer": statistics.mean(wers),
        "median_wer": statistics.median(wers),
        "std_wer": statistics.stdev(wers) if len(wers) > 1 else 0,
        "min_wer": min(wers),
        "max_wer": max(wers),
    }

    # Count by surah
    surah_counts = {}
    for r in results:
        surah_counts[r.surah_pred] = surah_counts.get(r.surah_pred, 0) + 1
    summary["surah_distribution"] = surah_counts

    # Total duration
    total_duration = sum(r.duration_sec for r in results)
    summary["total_duration_sec"] = total_duration
    summary["total_duration_min"] = total_duration / 60

    return summary


def get_worst_files(
    results: List[EvaluationResult],
    n: int = 5,
    metric: str = "cer",
) -> List[EvaluationResult]:
    """Get the worst performing files by a metric.

    Args:
        results: List of EvaluationResult objects.
        n: Number of worst files to return.
        metric: Metric to sort by ('cer' or 'wer').

    Returns:
        List of worst EvaluationResult objects.
    """
    key_fn = lambda r: r.cer if metric == "cer" else r.wer
    sorted_results = sorted(results, key=key_fn, reverse=True)
    return sorted_results[:n]


def results_to_dataframe(results: List[EvaluationResult]) -> pd.DataFrame:
    """Convert evaluation results to a pandas DataFrame.

    Args:
        results: List of EvaluationResult objects.

    Returns:
        DataFrame with all result fields.
    """
    return pd.DataFrame([r.to_dict() for r in results])


def generate_summary_markdown(
    results: List[EvaluationResult],
    config: Dict,
) -> str:
    """Generate a markdown summary report.

    Args:
        results: List of EvaluationResult objects.
        config: Evaluator configuration dict.

    Returns:
        Markdown formatted summary string.
    """
    summary = compute_summary_metrics(results)
    worst = get_worst_files(results, n=5, metric="cer")

    lines = [
        "# ASR Baseline Evaluation Summary",
        "",
        "## Configuration",
        f"- Mode: {config.get('mode', 'unknown')}",
        f"- Normalize Taa Marbuta: {config.get('normalize_taa_marbuta', False)}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Files | {summary['total_files']} |",
        f"| Total Duration | {summary['total_duration_min']:.1f} min |",
        f"| Mean CER | {summary['mean_cer']:.4f} ({summary['mean_cer']*100:.2f}%) |",
        f"| Median CER | {summary['median_cer']:.4f} ({summary['median_cer']*100:.2f}%) |",
        f"| Std CER | {summary['std_cer']:.4f} |",
        f"| Mean WER | {summary['mean_wer']:.4f} ({summary['mean_wer']*100:.2f}%) |",
        f"| Median WER | {summary['median_wer']:.4f} ({summary['median_wer']*100:.2f}%) |",
        f"| Std WER | {summary['std_wer']:.4f} |",
        "",
        "## Surah Distribution",
        "",
    ]

    for surah, count in summary.get("surah_distribution", {}).items():
        pct = count / summary["total_files"] * 100
        lines.append(f"- {surah}: {count} ({pct:.1f}%)")

    lines.extend([
        "",
        "## Worst 5 Files (by CER)",
        "",
        "| File | CER | WER | Surah | Notes |",
        "|------|-----|-----|-------|-------|",
    ])

    for r in worst:
        lines.append(
            f"| {r.file} | {r.cer:.4f} | {r.wer:.4f} | {r.surah_pred} | {r.notes} |"
        )

    lines.extend([
        "",
        "## Go/No-Go Assessment",
        "",
    ])

    # Simple assessment based on CER thresholds
    mean_cer = summary["mean_cer"]
    if mean_cer < 0.15:
        assessment = "**GO** - Model performs well (CER < 15%)"
    elif mean_cer < 0.30:
        assessment = "**CONDITIONAL GO** - Model performs adequately (CER 15-30%), fine-tuning recommended"
    else:
        assessment = "**NO-GO** - Model needs significant improvement (CER > 30%), fine-tuning required"

    lines.append(assessment)
    lines.append("")
    lines.append("### Interpretation Notes")
    lines.append("- CER < 10%: Excellent recognition")
    lines.append("- CER 10-20%: Good, minor errors")
    lines.append("- CER 20-30%: Acceptable, noticeable errors")
    lines.append("- CER > 30%: Poor, significant fine-tuning needed")
    lines.append("")
    lines.append("### Common Error Patterns")
    lines.append("(Review worst files above for specific patterns)")
    lines.append("")

    return "\n".join(lines)

