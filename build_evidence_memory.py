#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mars.evidence_memory import build_memory, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MARS evidence memory for CASTLE.")
    parser.add_argument("--castle-root", type=Path, default=Path("."))
    parser.add_argument("--video-summary-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("code/outputs/evidence_memory.jsonl"))
    parser.add_argument("--max-transcript-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory = build_memory(
        castle_root=args.castle_root,
        video_summary_dir=args.video_summary_dir,
        max_transcript_chars=args.max_transcript_chars,
    )
    write_jsonl(args.output, memory)
    print(f"Wrote {len(memory)} evidence entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

