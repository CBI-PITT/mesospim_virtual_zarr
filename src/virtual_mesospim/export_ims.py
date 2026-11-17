"""Stream locally composited virtual mosaic chunks into one Imaris file.

PyImarisWriter is imported only when exporting; serving and inspecting mosaics
do not require its native libraries. No HTTP server or intermediate Zarr is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path
import time

import numpy as np

from .config import color_to_rgb, load_mosaic_configs
from .engine import VirtualMosaic


DEFAULT_CHANNEL_COLORS = (
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.0, 0.7, 0.7),
    (1.0, 1.0, 1.0),
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _load_writer():
    try:
        from PyImarisWriter import PyImarisWriter as PW
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Export requires PyImarisWriter and its native libraries in this Python environment"
        ) from exc
    return PW


@dataclass(frozen=True)
class _Channel:
    mosaic: VirtualMosaic
    source_index: int
    name: str
    color: tuple[float, float, float]


def _source_channels(mosaic: VirtualMosaic) -> list[dict]:
    """OMERO display metadata is optional, even for source C axes."""
    ome = json.loads(mosaic.get_bytes("zarr.json"))["attributes"]["ome"]
    return ome.get("omero", {}).get("channels", [])


def _prepare_channels(mosaics: list[VirtualMosaic], level: int, time_index: int,
                      channel_names: list[str] | None,
                      channel_colors: list[tuple[float, float, float]] | None,
                      voxel_size_zyx_um: tuple[float, float, float] | None
                      ) -> tuple[list[_Channel], tuple[int, int, int], tuple[int, int, int],
                                 tuple[float, float, float]]:
    if not mosaics:
        raise ValueError("At least one mosaic is required")
    if level < 0 or time_index < 0:
        raise ValueError("level and time_index must be non-negative")
    first = mosaics[0]
    if level >= len(first.levels):
        raise ValueError(f"level={level} is unavailable")
    reference = first.levels[level]
    if first.dtype.kind != "u" or first.dtype.itemsize != 2:
        raise TypeError(f"PyImarisWriter export requires uint16 mosaics; got {first.dtype}")

    channels = []
    for mosaic in mosaics:
        if level >= len(mosaic.levels):
            raise ValueError(f"level={level} is unavailable for {mosaic.config.name}")
        lev = mosaic.levels[level]
        if (mosaic.dtype != first.dtype or lev.shape[-3:] != reference.shape[-3:]
                or lev.chunk_shape[-3:] != reference.chunk_shape[-3:]
                or not np.allclose(lev.scale[-3:], reference.scale[-3:], rtol=1e-7, atol=1e-9)):
            raise ValueError("All channels must have matching dtype, spatial shape, chunks and voxel scale")
        if "t" in mosaic.axis_names and time_index >= lev.shape[mosaic.axis_names.index("t")]:
            raise ValueError(f"time_index={time_index} is unavailable for {mosaic.config.name}")
        if "t" not in mosaic.axis_names and time_index != 0:
            raise ValueError(f"{mosaic.config.name} has no time axis; use time_index=0")
        count = lev.shape[mosaic.axis_names.index("c")] if "c" in mosaic.axis_names else 1
        metadata = _source_channels(mosaic)
        for source_index in range(count):
            display = metadata[source_index] if source_index < len(metadata) else {}
            name = (mosaic.config.channel or mosaic.config.name) if count == 1 else (
                f"{mosaic.config.channel or mosaic.config.name} {display.get('label', f'Ch{source_index}')}"
            )
            if count == 1 and mosaic.config.channel is None:
                name = display.get("label", name)
            color = (color_to_rgb(mosaic.config.color) if mosaic.config.color else
                     color_to_rgb(display["color"]) if "color" in display else
                     DEFAULT_CHANNEL_COLORS[len(channels) % len(DEFAULT_CHANNEL_COLORS)])
            channels.append(_Channel(mosaic, source_index, name, color))

    if channel_names is not None:
        if len(channel_names) != len(channels) or any(not name for name in channel_names):
            raise ValueError("--channel-names must provide one nonempty name per output channel")
        channels = [_Channel(ch.mosaic, ch.source_index, name, ch.color)
                    for ch, name in zip(channels, channel_names)]
    if channel_colors is not None:
        if len(channel_colors) != len(channels) or any(
            len(rgb) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in rgb)
            for rgb in channel_colors
        ):
            raise ValueError("--channel-colors must provide one valid RGB triple per output channel")
        channels = [_Channel(ch.mosaic, ch.source_index, ch.name, tuple(rgb))
                    for ch, rgb in zip(channels, channel_colors)]

    if voxel_size_zyx_um is None:
        units = [a.get("unit") for a in first.axes[-3:]]
        if any(unit not in ("micrometer", "micrometre", "um", "µm", "μm") for unit in units):
            raise ValueError("Spatial axes must be in micrometers; supply --voxel-size-zyx-um")
        voxel_size_zyx_um = tuple(reference.scale[-3:])
        for mosaic in mosaics[1:]:
            if [a.get("unit") for a in mosaic.axes[-3:]] != units:
                raise ValueError("Spatial units differ between channels; supply --voxel-size-zyx-um")
    if (len(voxel_size_zyx_um) != 3 or
            any(not math.isfinite(v) or v <= 0 for v in voxel_size_zyx_um)):
        raise ValueError("Voxel sizes must be three positive, finite values in Z Y X order")
    return channels, reference.shape[-3:], reference.chunk_shape[-3:], tuple(voxel_size_zyx_um)


def _read_block(channel: _Channel, level: int, time_index: int,
                bz: int, by: int, bx: int) -> np.ndarray:
    names = channel.mosaic.axis_names
    coords = ((time_index,) if "t" in names else ()) + (
        (channel.source_index,) if "c" in names else ()
    ) + (bz, by, bx)
    # compose_chunk returns full, zero-padded virtual chunks, including at edges.
    chunk = channel.mosaic.compose_chunk(level, coords)
    return np.ascontiguousarray(chunk.reshape(chunk.shape[-3:]), dtype=np.uint16)


def export_mosaics_to_ims(
    mosaics: list[VirtualMosaic], output_path: str | Path, *,
    level: int = 0, time_index: int = 0,
    voxel_size_zyx_um: tuple[float, float, float] | None = None,
    channel_names: list[str] | None = None,
    channel_colors: list[tuple[float, float, float]] | None = None,
    reader_threads: int = 16, writer_threads: int = 16,
    max_prefetched_blocks: int = 32,
) -> Path:
    """Write all mosaic channels into one IMS, with bounded in-flight blocks."""
    if min(reader_threads, writer_threads, max_prefetched_blocks) < 1:
        raise ValueError("Thread and prefetch counts must be positive")
    channels, (depth, height, width), (block_z, block_y, block_x), voxel_size = (
        _prepare_channels(mosaics, level, time_index, channel_names, channel_colors,
                          voxel_size_zyx_um)
    )
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".ims":
        output_path = output_path.with_suffix(".ims")
    working_path = output_path.with_suffix(".ims.part")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    PW = _load_writer()
    image_size = PW.ImageSize(x=width, y=height, z=depth, c=len(channels), t=1)
    sample_size = PW.ImageSize(x=1, y=1, z=1, c=1, t=1)
    block_size = PW.ImageSize(x=block_x, y=block_y, z=block_z, c=1, t=1)
    dimension_sequence = PW.DimensionSequence("x", "y", "z", "c", "t")
    options = PW.Options()
    options.mNumberOfThreads = writer_threads
    options.mCompressionAlgorithmType = PW.eCompressionAlgorithmGzipLevel2
    options.mEnableLogProgress = True

    class ProgressCallback(PW.CallbackClass):
        def __init__(self):
            super().__init__()
            self.previous_percent = -1

        def RecordProgress(self, progress, total_bytes_written):
            percent = int(progress * 100)
            if percent > self.previous_percent:
                self.previous_percent = percent
                _log(f"[writer] {percent:3d}% | {total_bytes_written / 1024**3:.3f} GiB written")

    blocks_z = math.ceil(depth / block_z)
    blocks_y = math.ceil(height / block_y)
    blocks_x = math.ceil(width / block_x)
    expected_blocks = len(channels) * blocks_z * blocks_y * blocks_x
    _log(f"Output shape CZYX: {len(channels)}, {depth}, {height}, {width}")
    _log(f"Writer block ZYX: {block_z}, {block_y}, {block_x}; {expected_blocks} blocks")
    _log(f"Channels: {[ch.name for ch in channels]}; voxel size ZYX (µm): {voxel_size}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if working_path.exists():
        _log(f"Removing incomplete output from a previous run: {working_path}")
        working_path.unlink()
    callback = ProgressCallback()
    converter = None
    finished = False
    start = time.perf_counter()
    try:
        converter = PW.ImageConverter(
            "uint16", image_size, sample_size, dimension_sequence, block_size,
            str(working_path), options, "MesoSPIM virtual mosaic to IMS", "1.0",
            callback,
        )
        coordinates = product(range(len(channels)), range(blocks_z), range(blocks_y), range(blocks_x))
        with ThreadPoolExecutor(max_workers=reader_threads, thread_name_prefix="mosaic-reader") as executor:
            pending = {}

            def submit_one() -> bool:
                try:
                    c, bz, by, bx = next(coordinates)
                except StopIteration:
                    return False
                pending[executor.submit(_read_block, channels[c], level, time_index, bz, by, bx)] = (c, bz, by, bx)
                return True

            for _ in range(max_prefetched_blocks):
                if not submit_one():
                    break
            copied = 0
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    c, bz, by, bx = pending.pop(future)
                    try:
                        block = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"Failed composing channel={c}, bz={bz}, by={by}, bx={bx}") from exc
                    block_index = PW.ImageSize(x=bx, y=by, z=bz, c=c, t=0)
                    if not converter.NeedCopyBlock(block_index):
                        raise RuntimeError(f"Writer rejected channel={c}, bz={bz}, by={by}, bx={bx}")
                    converter.CopyBlock(block, block_index)
                    copied += 1
                    if copied == 1 or copied % 50 == 0 or copied == expected_blocks:
                        _log(f"Copied {copied}/{expected_blocks} blocks in {time.perf_counter() - start:.1f}s")
                    submit_one()

        voxel_z, voxel_y, voxel_x = voxel_size
        extents = PW.ImageExtents(0.0, 0.0, 0.0,
                                  width * voxel_x, height * voxel_y, depth * voxel_z)
        parameters = PW.Parameters()
        color_infos = []
        for c, channel in enumerate(channels):
            parameters.set_channel_name(c, channel.name)
            color_info = PW.ColorInfo()
            color_info.set_base_color(PW.Color(*channel.color, 1.0))
            color_infos.append(color_info)
        _log("Calling Finish()")
        converter.Finish(extents, parameters, [datetime.now()], color_infos, True)
        finished = True
    finally:
        try:
            if converter is not None:
                converter.Destroy()
        finally:
            if not finished:
                working_path.unlink(missing_ok=True)

    if not working_path.is_file():
        raise RuntimeError(f"Writer finished but output is missing: {working_path}")
    working_path.replace(output_path)
    _log(f"Created {output_path} in {time.perf_counter() - start:.1f}s")
    return output_path


def export_manifest_to_ims(manifest_path: str | Path, output_path: str | Path, **kwargs) -> Path:
    mosaics = [VirtualMosaic(config) for config in load_mosaic_configs(manifest_path)]
    return export_mosaics_to_ims(mosaics, output_path, **kwargs)


def _parse_color(value: str) -> tuple[float, float, float]:
    try:
        rgb = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an r,g,b triple with values in [0, 1]") from exc
    if len(rgb) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in rgb):
        raise argparse.ArgumentTypeError("Use an r,g,b triple with values in [0, 1]")
    return rgb


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a virtual MesoSPIM mosaic to one multi-channel Imaris file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="JSON mosaic manifest (not a server URL)")
    parser.add_argument("output", type=Path, help="Output .ims file")
    parser.add_argument("--level", type=int, default=0, help="Virtual multiscale level to export")
    parser.add_argument("--time-index", type=int, default=0, help="Source time point to export")
    parser.add_argument("--voxel-size-zyx-um", type=float, nargs=3, metavar=("Z", "Y", "X"),
                        help="Override source spatial scales with Z Y X voxel sizes in micrometers")
    parser.add_argument("--channel-names", nargs="+", metavar="NAME",
                        help="Override names, one per output channel in manifest order")
    parser.add_argument("--channel-colors", nargs="+", type=_parse_color, metavar="R,G,B",
                        help="Override colors, one r,g,b triple per output channel")
    parser.add_argument("--reader-threads", type=int, default=16)
    parser.add_argument("--writer-threads", type=int, default=16)
    parser.add_argument("--max-prefetched-blocks", type=int, default=32)
    args = parser.parse_args(argv)
    if args.level < 0 or args.time_index < 0:
        parser.error("--level and --time-index must be non-negative")
    if min(args.reader_threads, args.writer_threads, args.max_prefetched_blocks) < 1:
        parser.error("Thread and prefetch counts must be positive")
    if args.voxel_size_zyx_um is not None and any(
        not math.isfinite(v) or v <= 0 for v in args.voxel_size_zyx_um
    ):
        parser.error("--voxel-size-zyx-um requires three positive, finite values")
    export_manifest_to_ims(
        args.manifest, args.output, level=args.level, time_index=args.time_index,
        voxel_size_zyx_um=args.voxel_size_zyx_um,
        channel_names=args.channel_names, channel_colors=args.channel_colors,
        reader_threads=args.reader_threads, writer_threads=args.writer_threads,
        max_prefetched_blocks=args.max_prefetched_blocks,
    )


if __name__ == "__main__":
    main()
