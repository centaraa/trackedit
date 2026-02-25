"""Utility to convert tiff image stacks to zarr format."""

from pathlib import Path

import numpy as np
import tifffile
import zarr
from tqdm import tqdm


def convert_tiff_to_zarr(
    input_path: Path,
    output_path: Path = None,
    group: str = "data",
) -> Path:
    """Convert a tiff file or folder of per-timepoint tiffs to a zarr store.

    Supports:
    - A single multi-dimensional tiff (2D/3D/4D), treated as (T, [Z,] Y, X).
    - A folder of tiff files, one per timepoint, sorted alphabetically.
      Each file can be 2D (Y, X) or 3D (Z, Y, X).

    Args:
        input_path: Path to a .tif/.tiff file, or a directory of tiff files.
        output_path: Output .zarr path. Defaults to <input_stem>.zarr next to input.
        group: Group name inside the zarr store. Defaults to "data".

    Returns:
        Path to the created zarr store.
    """
    input_path = Path(input_path)

    if output_path is None:
        stem = input_path.stem if input_path.is_file() else input_path.name
        output_path = input_path.parent / f"{stem}.zarr"
    output_path = Path(output_path)

    if input_path.is_file():
        _convert_single_tiff(input_path, output_path, group)
    elif input_path.is_dir():
        _convert_tiff_folder(input_path, output_path, group)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    return output_path


def _convert_single_tiff(tiff_path: Path, zarr_path: Path, group: str) -> None:
    """Convert a single (possibly multi-dimensional) tiff to zarr."""
    print(f"Reading {tiff_path} ...")
    data = tifffile.imread(str(tiff_path))
    print(f"  Shape: {data.shape}  dtype: {data.dtype}")

    if data.ndim < 3:
        raise ValueError(
            f"Expected at least 3D data (T, Y, X), got shape {data.shape}"
        )

    root = zarr.open_group(str(zarr_path), mode="w")
    zarr_array = root.create_array(
        group,
        shape=data.shape,
        dtype=data.dtype,
        chunks=(1, *data.shape[1:]),
    )

    print(f"Writing to {zarr_path}/{group} ...")
    for t in tqdm(range(data.shape[0]), desc="Writing timepoints"):
        zarr_array[t] = data[t]

    print(f"✓ Saved zarr to: {zarr_path}")
    print(f"  shape: {zarr_array.shape}  chunks: {zarr_array.chunks}")


def _convert_tiff_folder(folder: Path, zarr_path: Path, group: str) -> None:
    """Convert a folder of per-timepoint tiff files to zarr (sorted order)."""
    files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    # deduplicate while preserving order (glob may return .tif and .tiff separately)
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]

    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {folder}")

    print(f"Found {len(files)} tiff files in {folder}")
    example = tifffile.imread(str(files[0]))
    data_shape = (len(files), *example.shape)
    data_dtype = example.dtype
    print(f"  Per-frame shape: {example.shape}  dtype: {data_dtype}")
    print(f"  Full array shape: {data_shape}")

    root = zarr.open_group(str(zarr_path), mode="w")
    zarr_array = root.create_array(
        group,
        shape=data_shape,
        dtype=data_dtype,
        chunks=(1, *example.shape),
    )

    print(f"Writing to {zarr_path}/{group} ...")
    for t, f in enumerate(tqdm(files, desc="Writing timepoints")):
        frame = tifffile.imread(str(f))
        zarr_array[t] = frame

    print(f"✓ Saved zarr to: {zarr_path}")
    print(f"  shape: {zarr_array.shape}  chunks: {zarr_array.chunks}")
