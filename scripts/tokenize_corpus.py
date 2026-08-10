"""Build the frozen FineWeb-Edu Pilot tokenized corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline import (  # noqa: E402
    TokenizedDataError,
    build_tokenized_corpus,
    preflight_summary,
    prepare_build_context,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode the frozen FineWeb-Edu Pilot corpus into resumable "
            "little-endian uint16 token and document-index shards."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/tokenized_data.yaml",
        help="Tokenized-data YAML configuration relative to the project root.",
    )
    parser.add_argument(
        "--profile",
        choices=("pilot",),
        default="pilot",
        help="Executable data profile. Day 5 intentionally exposes Pilot only.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output override under data/tokenized/. Its sibling "
            ".<name>.inprogress directory is used for staging."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an identity-matching staging directory at shard boundaries.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-shard progress messages.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    context = prepare_build_context(
        args.config,
        project_root=PROJECT_ROOT,
        profile=args.profile,
        output_dir_override=args.output_dir,
    )
    for line in preflight_summary(context, resume=args.resume):
        print(line)

    result = build_tokenized_corpus(
        context,
        resume=args.resume,
        progress=None if args.no_progress else print,
    )
    totals = result.manifest["totals"]
    status = "already complete" if result.already_complete else "complete"
    print(f"Tokenized corpus {status}")
    print(f"  records: {totals['records']}")
    print(f"  model tokens: {totals['model_tokens']}")
    print(f"  payload bytes: {totals['token_payload_bytes']}")
    print(f"  storage shards: {totals['storage_shards']}")
    print(f"  output: {result.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print(
            "Tokenization interrupted; the formal output was not published. "
            "A verified staging checkpoint may be resumed with --resume.",
            file=sys.stderr,
        )
        return 130
    except (TokenizedDataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

