"""VisionOrbit — SpaceNet 7 Urban Change Detection Command Line Interface.

Unified entry point for inference, evaluation, testing, and pipeline tasks.
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    banner = r"""
======================================================================
     _    _ _     _             ____       _     _ _   
    | |  | (_)   (_)           / __ \     | |   (_) |  
    | |  | |_ ___ _  ___  _ __| |  | |_ __| |__  _| |_ 
    | |/\| | / __| |/ _ \| '_ \ |  | | '__| '_ \| | __|
    \  /\  / \__ \ | (_) | | | | |__| | |  | |_) | | |_ 
     \/  \/|_|___/_|\___/|_| |_|\____/|_|  |_.__/|_|\__|
                                                       
   SpaceNet 7 Multi-Temporal Satellite Urban Change Detection
======================================================================
"""
    print(banner)


def cmd_test(args):
    """Run end-to-end test detector pipeline."""
    print("Running end-to-end change detector test...")
    import subprocess
    cmd = [sys.executable, str(PROJECT_ROOT / "tests" / "test_detector.py")]
    subprocess.run(cmd, check=True)


def cmd_detect(args):
    """Run change detection between two GeoTIFF images."""
    from inference.change_detector import detect_change
    if not args.before or not args.after:
        print("Error: Both --before and --after GeoTIFF paths are required for detection.")
        print("Example: python main.py detect --before path/to/before.tif --after path/to/after.tif")
        sys.exit(1)

    result = detect_change(args.before, args.after, threshold=args.threshold)
    print("\nSummary:")
    for k, v in result.get("summary", {}).items():
        print(f"  {k}: {v}")

    print(f"\nExtracted {len(result.get('events', []))} change events.")

    if args.output:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to: {args.output}")


def cmd_evaluate(args):
    """Run evaluation sweep on best-interval validation dataset."""
    import subprocess
    script = PROJECT_ROOT / "evaluation" / "evaluate_best_interval.py"
    print(f"Executing evaluation: {script}")
    subprocess.run([sys.executable, str(script)], check=True)


def cmd_rag(args):
    """Generate filtered and merged RAG input events."""
    import subprocess
    script = PROJECT_ROOT / "rag" / "make_final_rag_input.py"
    print(f"Generating RAG events: {script}")
    subprocess.run([sys.executable, str(script)], check=True)


def cmd_info(args):
    """Print project details and directory structure overview."""
    print_banner()
    print("Project Configuration & Status:")
    print(f"  Project Root:     {PROJECT_ROOT}")
    print(f"  Champion Weights: {PROJECT_ROOT / 'checkpoints' / 'best_model_best_interval.pth'}")
    print(f"  Model Exists:     {(PROJECT_ROOT / 'checkpoints' / 'best_model_best_interval.pth').exists()}")
    print(f"  Data Directory:   {PROJECT_ROOT / 'data'}")
    print(f"  Outputs:          {PROJECT_ROOT / 'outputs'}")
    print("\nKey Modules:")
    print("  - models/         PyTorch U-Net change detection architecture")
    print("  - datasets/       SpaceNet 7 multi-temporal dataset loaders & samplers")
    print("  - preprocessing/  Patch extraction & building mask rasterization")
    print("  - training/       Model training scripts (baseline, temporal, best interval)")
    print("  - evaluation/     IoU / Dice sweeps and difference baselines")
    print("  - inference/      End-to-end sliding window detector and predictor")
    print("  - rag/            Spatial clustering, confidence filtering & RAG input generator")
    print("  - visualization/  Satellite overlay and change heatmap visualizers")
    print("  - tests/          Automated end-to-end verification tests")


def main():
    parser = argparse.ArgumentParser(
        description="VisionOrbit — SpaceNet 7 Urban Change Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    p_info = subparsers.add_parser("info", help="Display project configuration and module overview")
    p_info.set_defaults(func=cmd_info)

    # test
    p_test = subparsers.add_parser("test", help="Run end-to-end test on sample satellite imagery")
    p_test.set_defaults(func=cmd_test)

    # detect
    p_detect = subparsers.add_parser("detect", help="Run change detection on two satellite images")
    p_detect.add_argument("--before", type=str, required=True, help="Path to earlier 4-band GeoTIFF")
    p_detect.add_argument("--after", type=str, required=True, help="Path to later 4-band GeoTIFF")
    p_detect.add_argument("--threshold", type=float, default=0.3, help="Detection probability threshold (default: 0.3)")
    p_detect.add_argument("--output", type=str, default=None, help="Path to save result JSON")
    p_detect.set_defaults(func=cmd_detect)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run threshold evaluation sweep on validation set")
    p_eval.set_defaults(func=cmd_evaluate)

    # rag
    p_rag = subparsers.add_parser("rag", help="Generate filtered and merged RAG input events")
    p_rag.set_defaults(func=cmd_rag)

    args = parser.parse_args()

    if args.command is None:
        cmd_info(args)
        print("\nRun 'python main.py --help' to see available subcommands.")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
