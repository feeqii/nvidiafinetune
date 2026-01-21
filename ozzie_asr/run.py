"""Main CLI entrypoint for Ozzie Arabic ASR Baseline Evaluation.

Usage:
    # Download model from NGC
    python -m ozzie_asr.run download --output_dir ./models

    # Run evaluation with local .nemo model
    python -m ozzie_asr.run eval --audio_dir ./data/audio --nemo_model_path ./models/model.nemo

    # Run evaluation with FULL_SURAH mode
    python -m ozzie_asr.run eval --audio_dir ./data/audio --nemo_model_path ./models/model.nemo --assume_full_surah

    # List available models
    python -m ozzie_asr.run list-models
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ozzie_asr.canonical_texts import get_canonical_texts
from ozzie_asr.evaluator import (
    EvalMode,
    Evaluator,
    generate_summary_markdown,
    results_to_dataframe,
)
from ozzie_asr.model_loader import (
    NGC_MODEL_ID,
    NGC_MODEL_VERSION,
    ModelLoader,
    download_model_ngc,
    get_model_info,
    list_available_models,
    load_nemo_model,
)
from ozzie_asr.transcriber import Transcriber, create_transcription_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_run_info(
    model_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    args: Optional[argparse.Namespace] = None,
) -> dict:
    """Generate run information for reproducibility.

    Args:
        model_path: Path to model file.
        audio_dir: Path to audio directory.
        args: Parsed command line arguments.

    Returns:
        Dict with run metadata.
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Get git commit if available
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    # Get GPU info if available
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    # Model info
    if model_path:
        info["model_path"] = str(model_path)
        info["model_info"] = get_model_info(model_path)

    # Audio info
    if audio_dir:
        info["audio_dir"] = str(audio_dir)

    # Command line args
    if args:
        info["args"] = vars(args)

    return info


def cmd_download(args: argparse.Namespace) -> int:
    """Handle 'download' command - download model from NGC.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code (0 for success).
    """
    logger.info("Downloading model from NGC...")
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Version: {args.model_version}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        model_path = download_model_ngc(
            model_id=args.model_id,
            version=args.model_version,
            output_dir=Path(args.output_dir),
        )
        logger.info(f"Model downloaded successfully: {model_path}")
        return 0

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


def cmd_list_models(args: argparse.Namespace) -> int:
    """Handle 'list-models' command - list available NeMo models.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    logger.info("Listing available Arabic ASR models...")

    models = list_available_models(filter_arabic=not args.all)

    if not models:
        logger.warning("No models found. NeMo may not be installed or no Arabic models available.")
        print("\nTarget model for this project:")
        print(f"  NGC Model ID: {NGC_MODEL_ID}")
        print(f"  Version: {NGC_MODEL_VERSION}")
        print("\nDownload with:")
        print(f"  python -m ozzie_asr.run download --output_dir ./models")
        return 0

    print("\nAvailable models:")
    for model in models:
        print(f"  - {model['name']}")
        if model.get("description"):
            print(f"    {model['description']}")

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Handle 'eval' command - run transcription and evaluation.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)

    # Validate inputs
    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        return 1

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {run_dir}")

    # Initialize error log
    error_log_path = run_dir / "errors.log"
    error_handler = logging.FileHandler(error_log_path)
    error_handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(error_handler)

    errors = []

    try:
        # Step 1: Preprocess audio
        logger.info("Step 1: Preprocessing audio files...")

        from scripts.preprocess_audio import AudioPreprocessor

        preprocessor = AudioPreprocessor(
            cache_dir=run_dir / "preprocessed",
        )

        processed_paths, file_infos, preprocess_errors = preprocessor.process_directory(
            audio_dir,
            max_files=args.max_files,
        )

        errors.extend(preprocess_errors)

        if not processed_paths:
            logger.error("No audio files processed successfully")
            return 1

        # Build duration map
        duration_map = {}
        for info in file_infos:
            if "original_path" in info:
                orig_name = Path(info["original_path"]).name
                duration_map[orig_name] = info.get("duration_sec", 0)
            if "path" in info:
                duration_map[Path(info["path"]).name] = info.get("duration_sec", 0)

        logger.info(f"Preprocessed {len(processed_paths)} files")

        # Step 2: Load model
        logger.info("Step 2: Loading ASR model...")

        model_path = None
        if args.nemo_model_path:
            model_path = Path(args.nemo_model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return 1

        device = "cuda" if not args.cpu else "cpu"

        loader = ModelLoader(
            model_path=model_path,
            model_id=args.model_id,
            model_version=args.model_version,
            cache_dir=Path(args.model_cache_dir),
            device=device,
        )

        model = loader.load()
        logger.info("Model loaded successfully")

        # Step 3: Transcribe
        logger.info("Step 3: Transcribing audio files...")

        transcriber = Transcriber(model, batch_size=args.batch_size)

        transcriptions = transcriber.transcribe_batch(
            processed_paths,
            show_progress=True,
        )

        # Create transcription results with durations
        transcription_results = create_transcription_results(
            transcriptions,
            durations=duration_map,
        )

        logger.info(f"Transcribed {len(transcription_results)} files")

        # Log transcription errors
        for tr in transcription_results:
            if not tr.success:
                errors.append({"file": tr.filename, "error": tr.error})

        # Step 4: Evaluate
        logger.info("Step 4: Evaluating transcriptions...")

        mode = EvalMode.FULL_SURAH if args.assume_full_surah else EvalMode.SNAP_TO_CANONICAL

        evaluator = Evaluator(
            mode=mode,
            normalize_taa_marbuta=args.normalize_taa_marbuta,
            remove_diacritics=not args.preserve_diacritics,
        )

        # Prepare evaluation input
        eval_input = []
        for tr in transcription_results:
            if tr.success:
                # Map preprocessed filename back to original
                orig_name = tr.filename
                for info in file_infos:
                    if info.get("path", "").endswith(tr.filename):
                        orig_name = Path(info.get("original_path", tr.filename)).name
                        break

                eval_input.append({
                    "transcription": tr.transcription,
                    "filename": orig_name,
                    "duration_sec": tr.duration_sec,
                })

        eval_results = evaluator.evaluate_batch(eval_input)

        logger.info(f"Evaluated {len(eval_results)} files")

        # Step 5: Generate outputs
        logger.info("Step 5: Generating output files...")

        # predictions.csv
        predictions_data = []
        for tr in transcription_results:
            if tr.success:
                # Find corresponding eval result
                eval_result = None
                for er in eval_results:
                    if er.predicted_text_raw == tr.transcription:
                        eval_result = er
                        break

                predictions_data.append({
                    "file": tr.filename,
                    "duration_sec": tr.duration_sec,
                    "surah_pred": eval_result.surah_pred if eval_result else "unknown",
                    "predicted_text_raw": tr.transcription,
                    "predicted_text_normalized": eval_result.predicted_text_normalized if eval_result else "",
                })

        predictions_df = results_to_dataframe(eval_results)[
            ["file", "duration_sec", "surah_pred", "predicted_text_raw", "predicted_text_normalized"]
        ] if eval_results else None

        if predictions_df is not None:
            predictions_path = run_dir / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Saved: {predictions_path}")

        # metrics.csv
        metrics_df = results_to_dataframe(eval_results)
        if not metrics_df.empty:
            metrics_path = run_dir / "metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved: {metrics_path}")

        # summary.md
        summary_md = generate_summary_markdown(eval_results, evaluator.get_config())
        summary_path = run_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        logger.info(f"Saved: {summary_path}")

        # run_info.json
        run_info = get_run_info(
            model_path=model_path or loader._find_cached_model(),
            audio_dir=audio_dir,
            args=args,
        )
        run_info["preprocessing"] = preprocessor.get_config()
        run_info["evaluation"] = evaluator.get_config()
        run_info["num_files_processed"] = len(processed_paths)
        run_info["num_files_evaluated"] = len(eval_results)
        run_info["num_errors"] = len(errors)

        run_info_path = run_dir / "run_info.json"
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, default=str)
        logger.info(f"Saved: {run_info_path}")

        # errors.log (append any remaining errors)
        if errors:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write("\n--- Processing Errors ---\n")
                for err in errors:
                    f.write(f"{err.get('file', 'unknown')}: {err.get('error', 'unknown error')}\n")

        # Print summary to console
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {run_dir}")
        print(f"Files processed: {len(processed_paths)}")
        print(f"Files evaluated: {len(eval_results)}")
        print(f"Errors: {len(errors)}")

        if eval_results:
            from ozzie_asr.evaluator import compute_summary_metrics
            summary = compute_summary_metrics(eval_results)
            print(f"\nMean CER: {summary['mean_cer']:.4f} ({summary['mean_cer']*100:.2f}%)")
            print(f"Mean WER: {summary['mean_wer']:.4f} ({summary['mean_wer']*100:.2f}%)")

        print("\nGenerated files:")
        print(f"  - {predictions_path.name}")
        print(f"  - {metrics_path.name}")
        print(f"  - {summary_path.name}")
        print(f"  - {run_info_path.name}")
        if errors:
            print(f"  - {error_log_path.name}")

        return 0

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ozzie Arabic ASR Baseline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download model from NGC
  python -m ozzie_asr.run download --output_dir ./models

  # Run evaluation
  python -m ozzie_asr.run eval --audio_dir ./data/audio --nemo_model_path ./models/model.nemo

  # Run with FULL_SURAH mode
  python -m ozzie_asr.run eval --audio_dir ./data/audio --nemo_model_path ./models/model.nemo --assume_full_surah

  # List available models
  python -m ozzie_asr.run list-models
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download model from NGC",
    )
    download_parser.add_argument(
        "--model_id",
        type=str,
        default=NGC_MODEL_ID,
        help=f"NGC model ID (default: {NGC_MODEL_ID})",
    )
    download_parser.add_argument(
        "--model_version",
        type=str,
        default=NGC_MODEL_VERSION,
        help=f"NGC model version (default: {NGC_MODEL_VERSION})",
    )
    download_parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Output directory for downloaded model (default: ./models)",
    )

    # List models command
    list_parser = subparsers.add_parser(
        "list-models",
        help="List available NeMo ASR models",
    )
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="List all models, not just Arabic",
    )

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run transcription and evaluation",
    )
    eval_parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files",
    )
    eval_parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)",
    )
    eval_parser.add_argument(
        "--nemo_model_path",
        type=str,
        help="Path to local .nemo model file",
    )
    eval_parser.add_argument(
        "--model_id",
        type=str,
        default=NGC_MODEL_ID,
        help=f"NGC model ID for download (default: {NGC_MODEL_ID})",
    )
    eval_parser.add_argument(
        "--model_version",
        type=str,
        default=NGC_MODEL_VERSION,
        help=f"NGC model version (default: {NGC_MODEL_VERSION})",
    )
    eval_parser.add_argument(
        "--model_cache_dir",
        type=str,
        default="./models",
        help="Directory to cache downloaded models (default: ./models)",
    )
    eval_parser.add_argument(
        "--assume_full_surah",
        action="store_true",
        help="Use FULL_SURAH mode (default: SNAP_TO_CANONICAL)",
    )
    eval_parser.add_argument(
        "--normalize_taa_marbuta",
        action="store_true",
        help="Normalize taa marbuta (ة) to haa (ه) - OFF by default",
    )
    eval_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for transcription (default: 8)",
    )
    eval_parser.add_argument(
        "--max_files",
        type=int,
        help="Maximum number of files to process (for testing)",
    )
    eval_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )
    eval_parser.add_argument(
        "--preserve-diacritics",
        action="store_true",
        help="Preserve diacritics in evaluation (default: remove diacritics)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "download":
        return cmd_download(args)
    elif args.command == "list-models":
        return cmd_list_models(args)
    elif args.command == "eval":
        return cmd_eval(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

