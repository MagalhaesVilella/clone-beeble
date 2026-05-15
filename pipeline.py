import argparse
import json
import sys
from pathlib import Path

from mask_pipeline import run_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--points", default="")
    parser.add_argument("--user-mask", default="")
    parser.add_argument("--preview", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    pts = args.points or None
    if pts:
        print(f"[switchx:mask] pipeline: --points recebido ({len(pts)} chars)", file=sys.stderr, flush=True)
    info = run_batch(
        input_dir=args.frames_dir,
        output_dir=args.masks_dir,
        mode=args.mode,
        points_json=pts,
        user_mask_path=args.user_mask or None,
        preview_path=args.preview or None,
    )
    report = {
        "ok": True,
        "mode": args.mode,
        **info,
    }
    Path(args.report).write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
