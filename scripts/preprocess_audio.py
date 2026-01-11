"""Audio preprocessing utilities for ASR evaluation.

Converts various audio formats (mp3, m4a, wav) to NeMo-compatible format:
- WAV format
- 16kHz sample rate
- Mono channel

Uses ffmpeg via subprocess for reliable conversion.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported input formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

# Target format for NeMo
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1  # Mono


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed and available.

    Returns:
        True if ffmpeg is available.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_audio_info(audio_path: Path) -> Dict:
    """Get audio file information using ffprobe.

    Args:
        audio_path: Path to audio file.

    Returns:
        Dict with duration, sample_rate, channels, format.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        # Use ffprobe to get audio info
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"ffprobe failed for {audio_path}: {result.stderr}")
            return {"error": result.stderr}

        import json
        data = json.loads(result.stdout)

        # Extract relevant info
        info = {
            "path": str(audio_path),
            "format": audio_path.suffix.lower(),
        }

        # Get duration from format
        if "format" in data:
            info["duration_sec"] = float(data["format"].get("duration", 0))

        # Get audio stream info
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                info["sample_rate"] = int(stream.get("sample_rate", 0))
                info["channels"] = int(stream.get("channels", 0))
                info["codec"] = stream.get("codec_name", "unknown")
                break

        return info

    except subprocess.TimeoutExpired:
        return {"error": "ffprobe timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse ffprobe output: {e}"}
    except Exception as e:
        return {"error": str(e)}


def convert_audio(
    input_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> Tuple[Path, Dict]:
    """Convert audio file to NeMo-compatible format.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output WAV file. If None, creates temp file.
        sample_rate: Target sample rate (default: 16000).
        channels: Target number of channels (default: 1 for mono).

    Returns:
        Tuple of (output_path, info_dict).

    Raises:
        FileNotFoundError: If input file doesn't exist.
        RuntimeError: If ffmpeg is not available or conversion fails.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg is not installed. Install with:\n"
            "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  RunPod: apt-get update && apt-get install -y ffmpeg"
        )

    # Get input info
    input_info = get_audio_info(input_path)

    # Determine output path
    if output_path is None:
        output_path = input_path.with_suffix(".converted.wav")
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if conversion is needed
    needs_conversion = True
    if input_path.suffix.lower() == ".wav":
        if (input_info.get("sample_rate") == sample_rate and
            input_info.get("channels") == channels):
            needs_conversion = False
            logger.debug(f"File already in target format: {input_path}")

    if not needs_conversion and input_path != output_path:
        # Just copy the file
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path, input_info

    # Run ffmpeg conversion
    logger.debug(f"Converting {input_path} -> {output_path}")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i", str(input_path),
                "-ar", str(sample_rate),
                "-ac", str(channels),
                "-acodec", "pcm_s16le",  # 16-bit PCM
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg conversion timed out for {input_path}")

    # Get output info
    output_info = get_audio_info(output_path)
    output_info["original_path"] = str(input_path)
    output_info["original_format"] = input_path.suffix.lower()
    output_info["original_duration_sec"] = input_info.get("duration_sec", 0)

    return output_path, output_info


def preprocess_audio_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    max_files: Optional[int] = None,
) -> Tuple[List[Path], List[Dict], List[Dict]]:
    """Preprocess all audio files in a directory.

    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory for converted files. If None, converts in-place.
        sample_rate: Target sample rate.
        channels: Target number of channels.
        max_files: Maximum number of files to process (for testing).

    Returns:
        Tuple of (processed_paths, file_infos, errors).
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files (recursive search)
    audio_files = []
    for ext in SUPPORTED_FORMATS:
        audio_files.extend(input_dir.rglob(f"*{ext}"))
        audio_files.extend(input_dir.rglob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    audio_files = sorted(set(audio_files))

    if max_files:
        audio_files = audio_files[:max_files]

    logger.info(f"Found {len(audio_files)} audio files in {input_dir}")

    processed_paths = []
    file_infos = []
    errors = []

    for audio_path in audio_files:
        try:
            # Determine output path
            if output_dir:
                out_path = output_dir / f"{audio_path.stem}.wav"
            else:
                out_path = audio_path.with_suffix(".wav")

            # Convert
            converted_path, info = convert_audio(
                audio_path,
                out_path,
                sample_rate=sample_rate,
                channels=channels,
            )

            processed_paths.append(converted_path)
            file_infos.append(info)

            logger.debug(f"Processed: {audio_path.name} -> {converted_path.name}")

        except Exception as e:
            error_info = {
                "file": str(audio_path),
                "error": str(e),
            }
            errors.append(error_info)
            logger.error(f"Failed to process {audio_path}: {e}")

    logger.info(
        f"Preprocessing complete: {len(processed_paths)} succeeded, "
        f"{len(errors)} failed"
    )

    return processed_paths, file_infos, errors


class AudioPreprocessor:
    """Audio preprocessor with configuration and caching.

    Provides a consistent interface for preprocessing audio files
    for NeMo ASR inference.
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        channels: int = TARGET_CHANNELS,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize preprocessor.

        Args:
            sample_rate: Target sample rate (default: 16000).
            channels: Target channels (default: 1).
            cache_dir: Directory to cache converted files.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, audio_path: Path) -> Tuple[Path, Dict]:
        """Process a single audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (processed_path, info_dict).
        """
        audio_path = Path(audio_path)

        if self.cache_dir:
            output_path = self.cache_dir / f"{audio_path.stem}.wav"
        else:
            output_path = None

        return convert_audio(
            audio_path,
            output_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def process_directory(
        self,
        input_dir: Path,
        max_files: Optional[int] = None,
    ) -> Tuple[List[Path], List[Dict], List[Dict]]:
        """Process all audio files in a directory.

        Args:
            input_dir: Directory containing audio files.
            max_files: Maximum files to process.

        Returns:
            Tuple of (processed_paths, file_infos, errors).
        """
        return preprocess_audio_directory(
            input_dir,
            output_dir=self.cache_dir,
            sample_rate=self.sample_rate,
            channels=self.channels,
            max_files=max_files,
        )

    def get_config(self) -> Dict:
        """Get preprocessor configuration."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio files for ASR")
    parser.add_argument("input", type=Path, help="Input file or directory")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--max-files", type=int, help="Max files to process")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.input.is_file():
        path, info = convert_audio(args.input, args.output)
        print(f"Converted: {path}")
        print(f"Duration: {info.get('duration_sec', 0):.2f}s")
    else:
        paths, infos, errors = preprocess_audio_directory(
            args.input,
            args.output,
            max_files=args.max_files,
        )
        print(f"Processed {len(paths)} files")
        if errors:
            print(f"Errors: {len(errors)}")
            for err in errors:
                print(f"  - {err['file']}: {err['error']}")

