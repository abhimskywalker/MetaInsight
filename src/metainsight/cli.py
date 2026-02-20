"""CLI for the Python implementation scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .core import load_data, setup_configure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MetaInsight CLI (Python scaffold)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )

    parser.add_argument(
        "data_file",
        nargs="?",
        help="Path to a study-level input CSV/TSV/Excel/Parquet file.",
    )
    parser.add_argument("--outcome", default="Continuous")
    parser.add_argument("--outcome-measure", default="MD", choices=["MD", "SMD", "OR", "RR", "HR"])
    parser.add_argument("--reference", default="Placebo")
    parser.add_argument("--model", default="random", choices=["random", "fixed"])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if not args.data_file:
        parser.error("data_file is required unless --version is set")

    try:
        data = load_data(Path(args.data_file))
        bundle = setup_configure(
            data=data,
            outcome=args.outcome,
            outcome_measure=args.outcome_measure,
            reference_treatment=args.reference,
            model_type=args.model,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("Loaded studies:", bundle.study_count)
    print("Treatments:", bundle.treatment_count)
    print("Reference:", bundle.config.reference_treatment)
    print("Model:", bundle.config.model_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
