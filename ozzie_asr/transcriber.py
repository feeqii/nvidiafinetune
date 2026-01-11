"""Batch transcription wrapper for NeMo ASR models.

Provides a clean interface for transcribing audio files using
NVIDIA NeMo Conformer-CTC models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


class Transcriber:
    """Batch transcription wrapper for NeMo ASR models.

    Handles:
    - Single file and batch transcription
    - Progress tracking
    - Error handling and logging
    """

    def __init__(self, model, batch_size: int = 8):
        """Initialize transcriber with a loaded NeMo model.

        Args:
            model: Loaded NeMo EncDecCTCModel.
            batch_size: Batch size for transcription (default: 8).
        """
        self.model = model
        self.batch_size = batch_size

    def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to audio file (WAV, 16kHz mono preferred).

        Returns:
            Transcribed text.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # NeMo transcribe expects a list
        results = self.model.transcribe([str(audio_path)])

        # Handle different return formats
        if isinstance(results, list):
            return results[0] if results else ""
        return str(results)

    def transcribe_batch(
        self,
        audio_paths: List[Path],
        show_progress: bool = True,
    ) -> List[Tuple[Path, str, Optional[str]]]:
        """Transcribe a batch of audio files.

        Args:
            audio_paths: List of paths to audio files.
            show_progress: Show progress bar (default: True).

        Returns:
            List of (path, transcription, error) tuples.
            Error is None if transcription succeeded.
        """
        results = []

        # Process in batches
        total = len(audio_paths)
        iterator = range(0, total, self.batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Transcribing", unit="batch")

        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, total)
            batch_paths = audio_paths[batch_start:batch_end]

            # Filter to existing files
            valid_paths = []
            for path in batch_paths:
                path = Path(path)
                if path.exists():
                    valid_paths.append(path)
                else:
                    results.append((path, "", f"File not found: {path}"))

            if not valid_paths:
                continue

            try:
                # Transcribe batch
                transcriptions = self.model.transcribe(
                    [str(p) for p in valid_paths]
                )

                # Handle return format
                if not isinstance(transcriptions, list):
                    transcriptions = [transcriptions]

                # Pair results with paths
                for path, text in zip(valid_paths, transcriptions):
                    results.append((path, str(text), None))

            except Exception as e:
                logger.error(f"Batch transcription failed: {e}")
                # Mark all files in batch as failed
                for path in valid_paths:
                    results.append((path, "", str(e)))

        return results

    def transcribe_directory(
        self,
        audio_dir: Path,
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Tuple[Path, str, Optional[str]]]:
        """Transcribe all audio files in a directory.

        Args:
            audio_dir: Directory containing audio files.
            extensions: File extensions to include (default: ['.wav']).
            max_files: Maximum files to transcribe (for testing).
            show_progress: Show progress bar.

        Returns:
            List of (path, transcription, error) tuples.
        """
        audio_dir = Path(audio_dir)

        if not audio_dir.exists():
            raise FileNotFoundError(f"Directory not found: {audio_dir}")

        if extensions is None:
            extensions = [".wav"]

        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
            audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))

        # Remove duplicates and sort
        audio_files = sorted(set(audio_files))

        if max_files:
            audio_files = audio_files[:max_files]

        logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")

        return self.transcribe_batch(audio_files, show_progress=show_progress)


def transcribe_with_model(
    model,
    audio_paths: Union[Path, List[Path]],
    batch_size: int = 8,
    show_progress: bool = True,
) -> Union[str, List[Tuple[Path, str, Optional[str]]]]:
    """Convenience function for transcription.

    Args:
        model: Loaded NeMo model.
        audio_paths: Single path or list of paths.
        batch_size: Batch size for transcription.
        show_progress: Show progress bar for batch transcription.

    Returns:
        Single transcription string or list of (path, text, error) tuples.
    """
    transcriber = Transcriber(model, batch_size=batch_size)

    if isinstance(audio_paths, (str, Path)):
        return transcriber.transcribe_file(Path(audio_paths))

    return transcriber.transcribe_batch(
        [Path(p) for p in audio_paths],
        show_progress=show_progress,
    )


class TranscriptionResult:
    """Container for transcription results with metadata.

    Provides structured access to transcription outputs.
    """

    def __init__(
        self,
        file_path: Path,
        transcription: str,
        duration_sec: float = 0.0,
        error: Optional[str] = None,
    ):
        """Initialize transcription result.

        Args:
            file_path: Path to the audio file.
            transcription: Transcribed text (empty if error).
            duration_sec: Audio duration in seconds.
            error: Error message if transcription failed.
        """
        self.file_path = Path(file_path)
        self.transcription = transcription
        self.duration_sec = duration_sec
        self.error = error

    @property
    def success(self) -> bool:
        """Check if transcription succeeded."""
        return self.error is None

    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return self.file_path.name

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "file": str(self.file_path),
            "filename": self.filename,
            "transcription": self.transcription,
            "duration_sec": self.duration_sec,
            "error": self.error,
            "success": self.success,
        }

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error}"
        return f"TranscriptionResult({self.filename}, {status})"


def create_transcription_results(
    transcriptions: List[Tuple[Path, str, Optional[str]]],
    durations: Optional[Dict[str, float]] = None,
) -> List[TranscriptionResult]:
    """Create TranscriptionResult objects from raw transcription output.

    Args:
        transcriptions: List of (path, text, error) tuples from transcriber.
        durations: Optional dict mapping filename to duration in seconds.

    Returns:
        List of TranscriptionResult objects.
    """
    durations = durations or {}

    results = []
    for path, text, error in transcriptions:
        path = Path(path)
        duration = durations.get(path.name, durations.get(str(path), 0.0))

        results.append(TranscriptionResult(
            file_path=path,
            transcription=text,
            duration_sec=duration,
            error=error,
        ))

    return results

