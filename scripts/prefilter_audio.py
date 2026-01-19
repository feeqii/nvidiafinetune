#!/usr/bin/env python3
"""Pre-filter audio files before uploading to RunPod.

Filters out:
1. Files too short (< MIN_DURATION_SEC)
2. Files that are mostly silence
3. Corrupted/unreadable files

Outputs:
- List of files that pass filtering (for upload)
- Report of filtered-out files with reasons
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Thresholds
MIN_DURATION_SEC = 1.0  # Minimum duration to keep
MAX_DURATION_SEC = 120.0  # Maximum reasonable duration
SILENCE_THRESHOLD_DB = -40  # dB threshold for silence detection
MIN_NON_SILENCE_RATIO = 0.3  # At least 30% of audio should be non-silent


def get_audio_info(audio_path: Path) -> Dict:
    """Get audio file information using ffprobe."""
    try:
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
            return {"error": f"ffprobe failed: {result.stderr}"}
        
        data = json.loads(result.stdout)
        
        info = {
            "path": str(audio_path),
            "filename": audio_path.name,
            "surah": audio_path.parent.name,
        }
        
        if "format" in data:
            info["duration_sec"] = float(data["format"].get("duration", 0))
            info["size_bytes"] = int(data["format"].get("size", 0))
        
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                info["sample_rate"] = int(stream.get("sample_rate", 0))
                info["channels"] = int(stream.get("channels", 0))
                info["codec"] = stream.get("codec_name", "unknown")
                break
        
        return info
        
    except subprocess.TimeoutExpired:
        return {"error": "ffprobe timed out", "path": str(audio_path)}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "path": str(audio_path)}
    except Exception as e:
        return {"error": str(e), "path": str(audio_path)}


def detect_silence_ratio(audio_path: Path, threshold_db: float = -40) -> float:
    """Detect ratio of non-silent audio using ffmpeg silencedetect.
    
    Returns ratio of non-silent duration to total duration (0-1).
    """
    try:
        # Get total duration first
        info = get_audio_info(audio_path)
        if "error" in info or "duration_sec" not in info:
            return 0.0
        
        total_duration = info["duration_sec"]
        if total_duration <= 0:
            return 0.0
        
        # Use ffmpeg silencedetect filter
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(audio_path),
                "-af", f"silencedetect=noise={threshold_db}dB:d=0.1",
                "-f", "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        # Parse silence periods from stderr
        stderr = result.stderr
        silence_duration = 0.0
        
        # Look for "silence_duration: X.XXX" patterns
        import re
        for match in re.finditer(r"silence_duration:\s*([\d.]+)", stderr):
            silence_duration += float(match.group(1))
        
        non_silence_ratio = max(0, (total_duration - silence_duration) / total_duration)
        return non_silence_ratio
        
    except Exception as e:
        print(f"  Warning: silence detection failed for {audio_path.name}: {e}")
        return 1.0  # Assume it's fine if detection fails


def prefilter_files(
    input_dir: Path,
    min_duration: float = MIN_DURATION_SEC,
    max_duration: float = MAX_DURATION_SEC,
    check_silence: bool = True,
    min_non_silence: float = MIN_NON_SILENCE_RATIO,
) -> Tuple[List[Dict], List[Dict]]:
    """Pre-filter audio files.
    
    Returns:
        Tuple of (passed_files, filtered_files)
    """
    input_dir = Path(input_dir)
    
    # Find all WAV files
    wav_files = list(input_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")
    
    passed = []
    filtered = []
    
    for i, audio_path in enumerate(sorted(wav_files)):
        print(f"[{i+1}/{len(wav_files)}] Checking {audio_path.name}...", end=" ")
        
        info = get_audio_info(audio_path)
        
        # Check for errors
        if "error" in info:
            info["filter_reason"] = f"Error: {info['error']}"
            filtered.append(info)
            print(f"FILTERED (error)")
            continue
        
        # Check duration
        duration = info.get("duration_sec", 0)
        
        if duration < min_duration:
            info["filter_reason"] = f"Too short: {duration:.2f}s < {min_duration}s"
            filtered.append(info)
            print(f"FILTERED (too short: {duration:.2f}s)")
            continue
        
        if duration > max_duration:
            info["filter_reason"] = f"Too long: {duration:.2f}s > {max_duration}s"
            filtered.append(info)
            print(f"FILTERED (too long: {duration:.2f}s)")
            continue
        
        # Check silence (optional, slower)
        if check_silence:
            non_silence = detect_silence_ratio(audio_path)
            info["non_silence_ratio"] = non_silence
            
            if non_silence < min_non_silence:
                info["filter_reason"] = f"Mostly silence: {non_silence:.1%} non-silent < {min_non_silence:.0%}"
                filtered.append(info)
                print(f"FILTERED (mostly silence: {non_silence:.1%})")
                continue
        
        # Passed all checks
        passed.append(info)
        print(f"OK ({duration:.2f}s)")
    
    return passed, filtered


def main():
    parser = argparse.ArgumentParser(description="Pre-filter audio files before upload")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_DURATION_SEC,
        help=f"Minimum duration in seconds (default: {MIN_DURATION_SEC})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION_SEC,
        help=f"Maximum duration in seconds (default: {MAX_DURATION_SEC})",
    )
    parser.add_argument(
        "--skip-silence-check",
        action="store_true",
        help="Skip silence detection (faster)",
    )
    parser.add_argument(
        "--min-non-silence",
        type=float,
        default=MIN_NON_SILENCE_RATIO,
        help=f"Minimum non-silence ratio (default: {MIN_NON_SILENCE_RATIO})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--output-list",
        type=Path,
        default=None,
        help="Output text file with list of passed files (one per line)",
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("AUDIO PRE-FILTER")
    print(f"{'='*60}")
    print(f"Input directory: {args.input_dir}")
    print(f"Min duration: {args.min_duration}s")
    print(f"Max duration: {args.max_duration}s")
    print(f"Silence check: {'disabled' if args.skip_silence_check else 'enabled'}")
    print(f"{'='*60}\n")
    
    passed, filtered = prefilter_files(
        args.input_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        check_silence=not args.skip_silence_check,
        min_non_silence=args.min_non_silence,
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:    {len(passed) + len(filtered)}")
    print(f"Passed:         {len(passed)}")
    print(f"Filtered out:   {len(filtered)}")
    
    if passed:
        durations = [f["duration_sec"] for f in passed]
        total_dur = sum(durations)
        print(f"\nPassed files duration:")
        print(f"  Total:   {total_dur:.1f}s ({total_dur/60:.1f} min)")
        print(f"  Average: {total_dur/len(passed):.1f}s")
        print(f"  Min:     {min(durations):.1f}s")
        print(f"  Max:     {max(durations):.1f}s")
        
        # By surah
        by_surah = {}
        for f in passed:
            surah = f.get("surah", "unknown")
            by_surah[surah] = by_surah.get(surah, 0) + 1
        print(f"\nBy surah:")
        for surah, count in sorted(by_surah.items()):
            print(f"  {surah}: {count}")
    
    if filtered:
        print(f"\nFiltered files:")
        for f in filtered:
            print(f"  - {f.get('filename', f.get('path', 'unknown'))}: {f.get('filter_reason', 'unknown')}")
    
    # Save results
    if args.output:
        results = {
            "passed": passed,
            "filtered": filtered,
            "summary": {
                "total": len(passed) + len(filtered),
                "passed_count": len(passed),
                "filtered_count": len(filtered),
            }
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    if args.output_list:
        with open(args.output_list, "w") as f:
            for item in passed:
                f.write(item["path"] + "\n")
        print(f"File list saved to: {args.output_list}")
    
    print(f"\n{'='*60}")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())



