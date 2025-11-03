#!/usr/bin/env python3
"""Command-line tool for video question answering inference."""

import sys
from pathlib import Path
import argparse
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vgqa.inference import qa


def main() -> None:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Run video question answering inference with InternVideo2.5-Chat-8B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--question", required=True, help="Question to ask about the video")
    parser.add_argument("--bound-start", type=float, default=None, help="Temporal start in seconds (optional)")
    parser.add_argument("--bound-end", type=float, default=None, help="Temporal end in seconds (optional)")
    parser.add_argument("--model-dir", default="checkpoints/qa/InternVideo2_5_Chat_8B", help="Path to model directory")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames to sample")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--input-size", type=int, default=448, help="Input image size")
    parser.add_argument("--max-num", type=int, default=1, help="Maximum number of image patches per frame")

    args = parser.parse_args()

    # Parse temporal bounds
    bound = None
    if args.bound_start is not None and args.bound_end is not None:
        bound = (float(args.bound_start), float(args.bound_end))

    try:
        result = qa.predict(
            video_path=args.video,
            question=args.question,
            bound=bound,
            model_dir=args.model_dir,
            num_frames=args.num_frames,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            input_size=args.input_size,
            max_num=args.max_num,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(json.dumps({"error": str(e), "traceback": error_details}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
