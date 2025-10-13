"""Command line interface for controlling overwrite behavior."""

import argparse


def parse_args(description: str) -> argparse.Namespace:
    """Parse command line arguments for to control overwriting existing file behavior.

    Allows the selection of selection of overwriting behavior for output files.
    The script supports two flags:
        --force (-f): Force overwrite of existing output files without prompting.
        --prompt (-p): Prompt before overwriting existing output files. This overrides the --force flag.

    Returns:
        namespace (argparse.Namespace): Parsed command line arguments.

    """
    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwrite of existing output files.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        action="store_true",
        help="Prompt before overwriting existing output files. Overrides --force.",
    )
    return parser.parse_args()
