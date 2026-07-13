"""Command line interface for trackedit."""
# Usage: pixi run python cli.py convert tiff-to-zarr path/in -o path/out

from pathlib import Path

import click

from trackedit.utils.geff import convert_geff_to_db
from trackedit.utils.zarr_converter import convert_tiff_to_zarr


@click.group()
@click.version_option()
def cli():
    """TrackeEdit: A tool for editing and converting tracking data."""


@cli.group()
def convert():
    """Convert between different tracking data formats."""


@convert.command("geff-to-db")
@click.argument("geff_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output database path (default: <input_stem>_from_geff.db)",
)
def geff_to_db(geff_path: Path, output: Path = None):
    """Convert GEFF file to ULTrack SQLite database.

    Args:
        geff_path: Path to the input GEFF file
        output: Optional output database path
    """
    convert_geff_to_db(geff_path, output)


@convert.command("tiff-to-zarr")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output .zarr path (default: <input_stem>.zarr next to input)",
)
@click.option(
    "--group", "-g",
    default="data",
    show_default=True,
    help="Group name inside the zarr store",
)
def tiff_to_zarr(input_path: Path, output: Path, group: str):
    """Convert a tiff file or folder of tiffs to a zarr store.

    INPUT_PATH can be:
      - a single 4D (or 3D/2D) .tif/.tiff stack, or
      - a folder containing per-timepoint .tif/.tiff files (sorted order).

    The resulting zarr array has shape (T, [Z,] Y, X) and is written to --output.
    """
    convert_tiff_to_zarr(input_path, output, group)


if __name__ == "__main__":
    cli()
