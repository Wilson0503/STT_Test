from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Canary meeting manifest (.jsonl) from audio directory.")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory containing meeting audio files.")
    parser.add_argument("--output-manifest", required=True, type=Path, help="Output manifest path (.jsonl).")
    parser.add_argument("--audio-ext", default="wav", help="Audio extension to scan (default: wav).")
    parser.add_argument("--source-lang", default="en", help="Canary source language code.")
    parser.add_argument("--target-lang", default="en", help="Canary target language code.")
    parser.add_argument(
        "--task",
        default="asr",
        choices=["asr", "s2t_translation"],
        help="Canary task name (asr or s2t_translation).",
    )
    parser.add_argument("--pnc", default="yes", choices=["yes", "no"], help="Enable punctuation/capitalization prompt.")
    parser.add_argument(
        "--duration",
        type=float,
        default=10000.0,
        help="Fallback duration value for manifest records if real duration is not precomputed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_dir = args.audio_dir.resolve()
    output_manifest = args.output_manifest.resolve()

    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    files = sorted(audio_dir.rglob(f"*.{args.audio_ext.lower()}"))
    if not files:
        raise RuntimeError(f"No '*.{args.audio_ext}' files found under: {audio_dir}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with output_manifest.open("w", encoding="utf-8") as fout:
        for path in files:
            row = {
                "audio_filepath": str(path.resolve()),
                "duration": args.duration,
                "taskname": args.task,
                "source_lang": args.source_lang,
                "target_lang": args.target_lang,
                "pnc": args.pnc,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(files)} items -> {output_manifest}")


if __name__ == "__main__":
    main()
