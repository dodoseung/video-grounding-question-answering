#!/usr/bin/env python3
"""Command-line tool for spatio-temporal video grounding inference."""

import sys
from pathlib import Path
import argparse
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vgqa.inference import grounding


def main() -> None:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Run spatio-temporal video grounding inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--query", required=True, help="Text query for grounding")
    parser.add_argument("--cfg", default="configs/grounding_vidstg.yaml", help="Path to config YAML")
    parser.add_argument("--ckpt", default="checkpoints/grounding/tastvg_vidstg.pth", help="Path to checkpoint file")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (unused, kept for compatibility)")

    args = parser.parse_args()

    try:
        result = grounding.predict(
            video_path=args.video,
            query=args.query,
            cfg_path=args.cfg,
            ckpt_path=args.ckpt,
            device_str=args.device,
            batch_size=args.batch_size
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(json.dumps({"error": str(e), "traceback": error_details}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
