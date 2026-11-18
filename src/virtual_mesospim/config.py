"""Validated, JSON-serializable configuration. Nothing is written to tile stores."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional
import re


GAIN_MIN = 0.01
GAIN_MAX = 100.0
GAIN_FILE = "gain.json"


NAMED_COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (1.0, 1.0, 1.0),
    "grey": (1.0, 1.0, 1.0),
}


def color_to_rgb(color: str) -> tuple[float, float, float]:
    """Named color or #RGB/#RRGGBB hex to (r, g, b) floats in [0, 1]."""
    if color.lower() in NAMED_COLORS:
        return NAMED_COLORS[color.lower()]
    hex_part = color[1:] if color.startswith("#") else color
    if re.fullmatch(r"[0-9a-fA-F]{6}", hex_part):
        return tuple(int(hex_part[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    if re.fullmatch(r"[0-9a-fA-F]{3}", hex_part):
        return tuple(int(c * 2, 16) / 255.0 for c in hex_part)
    raise ValueError(f"Invalid color: {color!r}")


def _is_valid_color(color: str) -> bool:
    try:
        color_to_rgb(color)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class MosaicConfig:
    tiles: tuple[Path, ...]
    rows: int
    cols: int
    overlap_yx: tuple[float, float] = (0.10, 0.10)
    chunk_shape_zyx: tuple[int, int, int] = (32, 256, 256)
    blend: str = "feather"  # first, last, feather
    positions_zyx: Optional[tuple[tuple[int, int, int], ...]] = None
    row_direction: int = 1
    col_direction: int = 1
    gzip_level: int = 0  # 0: raw bytes; 1..9: gzip codec
    cache_chunks: int = 8  # bounded in-memory LRU, never written to disk
    max_concurrent_chunks: int = 4
    name: str = "MesoSPIM virtual mosaic"
    mount_path: str = "stitched.ome.zarr"
    channel: Optional[str] = None  # set for multi-channel manifests; mounted at /<mount_path>/<channel>
    color: Optional[str] = None  # optional display color for the Neuroglancer layer
    gains: Optional[tuple[float, ...]] = None  # per-tile multipliers from gain.json; tile order

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if len(self.tiles) != self.rows * self.cols:
            raise ValueError(f"Expected {self.rows * self.cols} tiles, got {len(self.tiles)}")
        if self.gains is not None:
            if len(self.gains) != len(self.tiles):
                raise ValueError(f"Expected {len(self.tiles)} gains, got {len(self.gains)}")
            for value in self.gains:
                if type(value) not in (int, float) or not math.isfinite(value) \
                        or not GAIN_MIN <= value <= GAIN_MAX:
                    raise ValueError(f"gains must be finite numbers in [{GAIN_MIN}, {GAIN_MAX}]")
        if len(self.overlap_yx) != 2 or any(not (0 <= v < 1) for v in self.overlap_yx):
            raise ValueError("overlap_yx must be two fractions in [0, 1)")
        if len(self.chunk_shape_zyx) != 3 or any(v < 1 for v in self.chunk_shape_zyx):
            raise ValueError("chunk_shape_zyx must be three positive integers")
        if self.blend not in ("first", "last", "feather"):
            raise ValueError("blend must be first, last, or feather")
        if self.row_direction not in (-1, 1) or self.col_direction not in (-1, 1):
            raise ValueError("row_direction and col_direction must be +1 or -1")
        if self.positions_zyx is not None:
            if len(self.positions_zyx) != len(self.tiles):
                raise ValueError("positions_zyx must contain one position per tile")
            for position in self.positions_zyx:
                if len(position) != 3 or any(type(i) is not int for i in position):
                    raise ValueError("positions_zyx must consist of integer [z, y, x] triplets")
        if not 0 <= self.gzip_level <= 9:
            raise ValueError("gzip_level must be 0..9")
        if self.cache_chunks < 0:
            raise ValueError("cache_chunks must be >= 0")
        if self.max_concurrent_chunks < 1:
            raise ValueError("max_concurrent_chunks must be >= 1")
        if not self.mount_path or "/" in self.mount_path or self.mount_path in (".", ".."):
            raise ValueError("mount_path must be a single URL path segment")
        if self.channel is not None and (
            not self.channel or "/" in self.channel or self.channel in (".", "..")
        ):
            raise ValueError("channel must be a single URL path segment")
        if self.color is not None and not _is_valid_color(self.color):
            raise ValueError("color must be a named color (e.g. green, magenta) or #RGB/#RRGGBB hex")

    @classmethod
    def from_json(cls, file_path: str | Path) -> "MosaicConfig":
        configs = load_mosaic_configs(file_path)
        if len(configs) != 1:
            raise ValueError("Multi-channel manifest; use load_mosaic_configs")
        return configs[0]


def _resolve_tiles(raw_tiles, manifest_path: Path) -> tuple[Path, ...]:
    # Allow relative paths in JSON, interpreted relative to the manifest.
    return tuple(
        (Path(p).expanduser() if Path(p).expanduser().is_absolute()
         else manifest_path.parent / Path(p).expanduser()).resolve()
        for p in raw_tiles
    )


def _prepare_common(cfg: dict) -> dict:
    for key in ("overlap_yx", "chunk_shape_zyx"):
        if key in cfg:
            cfg[key] = tuple(cfg[key])
    if cfg.get("positions_zyx") is not None:
        cfg["positions_zyx"] = tuple(tuple(v for v in position)
                                     for position in cfg["positions_zyx"])
    return cfg


def _validate_gains(raw_gains, tile_count: int, channel: str) -> tuple[float, ...]:
    if not isinstance(raw_gains, list) or len(raw_gains) != tile_count:
        raise ValueError(f"gain.json: expected {tile_count} gains for channel {channel!r}, "
                         f"got {len(raw_gains) if isinstance(raw_gains, list) else type(raw_gains).__name__}")
    for value in raw_gains:
        if type(value) not in (int, float) or not math.isfinite(value) \
                or not GAIN_MIN <= value <= GAIN_MAX:
            raise ValueError(f"gain.json: gains for channel {channel!r} must be finite "
                             f"numbers in [{GAIN_MIN}, {GAIN_MAX}]")
    return tuple(float(v) for v in raw_gains)


def _load_gain_file(path: Path, tiles: tuple[Path, ...], channel: Optional[str],
                    file_path: Path) -> Optional[tuple[float, ...]]:
    """Read auto-discovered gain.json next to the manifest; None when absent."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {GAIN_FILE} at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    acquisition = data.get("acquisition")
    if (acquisition is not None and tiles):
        tile_store = tiles[0].parent
        try:
            # The web app writes the acquisition folder relative to the
            # manifest, which also handles nested acquisitions.
            relative_store = tile_store.relative_to(file_path.parent).as_posix()
        except ValueError:
            relative_store = tile_store.name
        if relative_store != acquisition and tile_store.name != acquisition:
            raise ValueError(f"{path} acquisition {acquisition!r} does not match tile store "
                             f"{relative_store!r}")
    channels = data.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"{path} needs a non-empty channels list")
    if channel is None:
        if len(channels) != 1:
            raise ValueError(f"{path} must contain exactly one channels entry for a "
                             f"single-channel manifest")
        entry = channels[0]
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"{path} channels entries need a string name")
        return _validate_gains(entry.get("gains"), len(tiles), entry["name"])
    for entry in channels:
        if isinstance(entry, dict) and entry.get("name") == channel:
            return _validate_gains(entry.get("gains"), len(tiles), channel)
    raise ValueError(f"{path} has no gains for channel {channel!r}")


def load_mosaic_configs(file_path: str | Path) -> "list[MosaicConfig]":
    """Load one MosaicConfig per channel. Manifests with a top-level tiles list
    yield a single config; manifests with a channels list yield one config per
    channel sharing all geometry settings. A gain.json next to the manifest is
    auto-discovered and applied as per-tile brightness multipliers."""
    file_path = Path(file_path).resolve()
    cfg = json.loads(file_path.read_text(encoding="utf-8"))
    gain_path = file_path.parent / GAIN_FILE
    if "channels" in cfg:
        if "tiles" in cfg:
            raise ValueError("Provide either tiles (single channel) or channels, not both")
        channels = cfg.pop("channels")
        if not isinstance(channels, list) or not channels:
            raise ValueError("channels must be a non-empty list")
        entries = []
        for entry in channels:
            if not isinstance(entry, dict) or "name" not in entry or "tiles" not in entry:
                raise ValueError("Each channels entry needs name and tiles")
            name = entry["name"]
            if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
                raise ValueError(f"Invalid channel name: {name!r}")
            entry = dict(entry)
            name = entry.pop("name")
            tiles = _resolve_tiles(entry.pop("tiles"), file_path)
            color = entry.pop("color", None)
            merged = {**cfg, **entry}
            merged["tiles"] = tiles
            merged["channel"] = name
            merged["color"] = color
            merged["name"] = f"{merged['name']} {name}" if merged.get("name") else name
            merged["gains"] = _load_gain_file(gain_path, tiles, name, file_path)
            entries.append(MosaicConfig(**_prepare_common(merged)))
        if len({e.channel for e in entries}) != len(entries):
            raise ValueError("Channel names must be unique")
        return entries
    if "tiles" not in cfg:
        raise ValueError("Manifest needs a tiles list or a channels list")
    cfg["tiles"] = _resolve_tiles(cfg["tiles"], file_path)
    cfg["gains"] = _load_gain_file(gain_path, cfg["tiles"], None, file_path)
    return [MosaicConfig(**_prepare_common(cfg))]
