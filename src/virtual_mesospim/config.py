"""Validated, JSON-serializable configuration. Nothing is written to tile stores."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional


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

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if len(self.tiles) != self.rows * self.cols:
            raise ValueError(f"Expected {self.rows * self.cols} tiles, got {len(self.tiles)}")
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

    @classmethod
    def from_json(cls, file_path: str | Path) -> "MosaicConfig":
        file_path = Path(file_path).resolve()
        cfg = json.loads(file_path.read_text(encoding="utf-8"))
        # Allow relative paths in JSON, interpreted relative to the manifest.
        cfg["tiles"] = tuple(
            (Path(p).expanduser() if Path(p).expanduser().is_absolute()
             else file_path.parent / Path(p).expanduser()).resolve()
            for p in cfg["tiles"]
        )
        for name in ("overlap_yx", "chunk_shape_zyx"):
            if name in cfg:
                cfg[name] = tuple(cfg[name])
        if cfg.get("positions_zyx") is not None:
            cfg["positions_zyx"] = tuple(tuple(v for v in position)
                                          for position in cfg["positions_zyx"])
        return cls(**cfg)
