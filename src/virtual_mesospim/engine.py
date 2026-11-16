"""Virtual array geometry, compositing, and Zarr v3 key -> bytes translation.

Spatial mosaicing: column-major indices (tile = col * rows + row).
No source data or source metadata are altered, and no output chunks are stored.

This first implementation requires pure, axis-aligned *integer* translations.
The code intentionally does not claim to reproduce optical registration, rotation,
subpixel resampling, or physical stage coordinate parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
import json
import math
import re
from typing import Any

import numpy as np

from .config import MosaicConfig


ALLOWED_DTYPES = {
    "int8", "int16", "int32",
    "uint8", "uint16", "uint32", "uint64", "float32",
}


def _round_nearest(value: float) -> int:
    """Round halves away from zero; stable for positive and negative positions."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _json_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _get_axes(multiscale: dict[str, Any]) -> tuple[list[dict], list[str]]:
    axes = []
    for axis in multiscale.get("axes", []):
        axis = {"name": axis} if isinstance(axis, str) else dict(axis)
        name = axis.get("name")
        if name not in ("t", "c", "z", "y", "x"):
            raise ValueError(f"Unsupported axis {name!r}; expected (t,) (c,) z, y, x")
        expected_type = "space" if name in "zyx" else "time" if name == "t" else "channel"
        axis.setdefault("type", expected_type)
        if axis["type"] != expected_type:
            raise ValueError(f"Axis {name!r} should have type {expected_type!r}")
        axes.append(axis)
    names = [a["name"] for a in axes]
    if names[-3:] != ["z", "y", "x"] or names[:-3] not in ([], ["c"], ["t"], ["t", "c"]):
        raise ValueError(f"Unsupported axis order {names!r}; use ZYX, CZYX, TZYX, or TCZYX")
    return axes, names


def _transforms(dataset: dict[str, Any], ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """Read dataset transforms; reject unsupported affine/order semantics."""
    transforms = dataset.get("coordinateTransformations", [])
    scales = [x for x in transforms if x.get("type") == "scale"]
    translations = [x for x in transforms if x.get("type") == "translation"]
    if len(scales) != 1 or len(translations) > 1 or len(scales) + len(translations) != len(transforms):
        raise ValueError("Each source level must have exactly one scale and at most one translation")
    if translations and transforms.index(scales[0]) > transforms.index(translations[0]):
        raise ValueError("OME transforms must list scale before translation")
    scale = np.asarray(scales[0]["scale"], dtype=float)
    translation = (np.asarray(translations[0]["translation"], dtype=float)
                   if translations else np.zeros(ndim, dtype=float))
    if (scale.shape != (ndim,) or translation.shape != (ndim,)
            or not np.all(np.isfinite(scale)) or not np.all(np.isfinite(translation))
            or np.any(scale <= 0)):
        raise ValueError("Invalid source coordinate scale or translation")
    return scale, translation


@dataclass(frozen=True)
class Level:
    sources: tuple[Any, ...]
    tile_shape: tuple[int, ...]
    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    origins_zyx: tuple[tuple[int, int, int], ...]  # non-negative, normalized
    min_raw_zyx: tuple[int, int, int]
    scale: tuple[float, ...]
    translation: tuple[float, ...]
    feather_yx: tuple[float, float]


class VirtualMosaic:
    """Reads existing OME-Zarr v2 arrays and presents a Zarr v3 key namespace.

    The optional ``groups`` argument is for embedding/injection in tests.
    Normally the groups are opened from the configured source paths in r mode.
    """

    def __init__(self, config: MosaicConfig, *, groups: list[Any] | None = None):
        self.config = config
        if groups is None:
            import zarr
            groups = []
            for path in config.tiles:
                try:
                    # zarr-python 3 transparently reads legacy Zarr v2 arrays.
                    groups.append(zarr.open_group(str(path), mode="r", zarr_format=2))
                except Exception as exc:
                    raise ValueError(f"Cannot open tile as a Zarr v2 group: {path}") from exc
        if len(groups) != len(config.tiles):
            raise ValueError("Number of source groups does not match configured tiles")

        source_specs: list[tuple[Any, list, list, dict]] = []
        for i, group in enumerate(groups):
            root_attrs = dict(group.attrs)
            mss = root_attrs.get("multiscales")
            if not isinstance(mss, list) or len(mss) != 1:
                raise ValueError(f"Tile {i} needs exactly one OME-Zarr 0.4 multiscales entry")
            ms = mss[0]
            if ms.get("version", "0.4") != "0.4":
                raise ValueError(f"Tile {i} is not OME-Zarr 0.4 (version={ms.get('version')!r})")
            axes, names = _get_axes(ms)
            if not ms.get("datasets"):
                raise ValueError(f"Tile {i} multiscales.datasets is empty")
            if "coordinateTransformations" in ms:
                # Could be supported by compose of transforms; fail rather than lie.
                raise ValueError("Group-level coordinateTransformations are not supported in v0.1")
            dsets = ms["datasets"]
            arrays = [group[d["path"]] for d in dsets]
            source_specs.append((arrays, dsets, axes, root_attrs))
            if i == 0:
                self.axes, self.axis_names = axes, names
            elif names != self.axis_names or axes != self.axes:
                raise ValueError(f"Tile {i} has different OME axes/units than tile 0")

        self.dtype = np.dtype(source_specs[0][0][0].dtype)
        if self.dtype.name not in ALLOWED_DTYPES:
            raise ValueError(f"Unsupported dtype {self.dtype}; expected a Neuroglancer-supported numeric type")
        self.ndim = len(self.axis_names)
        nlevels = len(source_specs[0][1])
        if any(len(spec[1]) != nlevels for spec in source_specs):
            raise ValueError("All source tiles must have the same number of levels")

        base_spatial_shape = tuple(int(v) for v in source_specs[0][0][0].shape[-3:])
        if config.positions_zyx is not None:
            base_origins = tuple(tuple(x) for x in config.positions_zyx)
        else:
            sy = _round_nearest(base_spatial_shape[1] * (1 - config.overlap_yx[0]))
            sx = _round_nearest(base_spatial_shape[2] * (1 - config.overlap_yx[1]))
            if sy == 0 or sx == 0:
                raise ValueError("Calculated grid stride is zero; reduce overlap")
            base_origins = tuple(
                (0, (i % config.rows) * sy * config.row_direction,
                 (i // config.rows) * sx * config.col_direction)
                for i in range(len(config.tiles))
            )

        self.levels: list[Level] = []
        base_scale, _ = _transforms(source_specs[0][1][0], self.ndim)
        for k in range(nlevels):
            sources = tuple(spec[0][k] for spec in source_specs)
            tile_shape = tuple(int(v) for v in sources[0].shape)
            scale0, translation0 = _transforms(source_specs[0][1][k], self.ndim)
            for i, spec in enumerate(source_specs):
                source = spec[0][k]
                if tuple(source.shape) != tile_shape or np.dtype(source.dtype) != self.dtype:
                    raise ValueError(f"Tile {i}, level {k}: shape/dtype differs from tile 0")
                if len(source.shape) != self.ndim:
                    raise ValueError(f"Tile {i}, level {k}: array dimensionality differs from metadata")
                scale_i, _ = _transforms(spec[1][k], self.ndim)
                if not np.allclose(scale_i, scale0, rtol=1e-7, atol=1e-9):
                    raise ValueError(f"Tile {i}, level {k}: pixel scale differs from tile 0")

            # Coarse-level positions derive from base pixel coordinates in physical units.
            raw_origins = tuple(
                tuple(_round_nearest(base_origins[i][j] * base_scale[self.ndim - 3 + j]
                                     / scale0[self.ndim - 3 + j]) for j in range(3))
                for i in range(len(config.tiles))
            )
            lo = tuple(min(p[j] for p in raw_origins) for j in range(3))
            hi = tuple(max(p[j] + tile_shape[self.ndim - 3 + j] for p in raw_origins)
                       for j in range(3))
            output_shape = tile_shape[:-3] + tuple(hi[j] - lo[j] for j in range(3))
            origins = tuple(tuple(p[j] - lo[j] for j in range(3)) for p in raw_origins)
            # Physical translations are intentionally not serialized: the OME
            # coordinateTransformations below carry scale only, because the
            # Neuroglancer zarr driver silently stops requesting chunks when a
            # dataset transform contains a translation.
            chunk_shape = tuple([1] * (self.ndim - 3) + [
                min(config.chunk_shape_zyx[j], output_shape[self.ndim - 3 + j])
                for j in range(3)
            ])
            feather_yx = tuple(
                max(0.0, tile_shape[-2 + j] * config.overlap_yx[j]) for j in range(2)
            )
            self.levels.append(Level(sources, tile_shape, output_shape, chunk_shape,
                                     origins, lo, tuple(scale0), tuple(translation0),
                                     feather_yx))

        self._metadata = self._build_metadata(source_specs[0][3])
        # Thread-safe bounded LRU: only a few generated chunks are retained in RAM.
        if config.cache_chunks:
            self.get_bytes = lru_cache(maxsize=config.cache_chunks)(self._get_bytes_uncached)
        else:
            self.get_bytes = self._get_bytes_uncached

    def _build_metadata(self, first_attrs: dict) -> dict[str, bytes]:
        datasets = []
        result: dict[str, bytes] = {}
        for i, level in enumerate(self.levels):
            transformations = [{"type": "scale", "scale": list(level.scale)}]
            datasets.append({"path": str(i), "coordinateTransformations": transformations})
            serializer: dict[str, Any] = {"name": "bytes"}
            if self.dtype.itemsize > 1:
                serializer["configuration"] = {"endian": "little"}
            codecs = [serializer]
            if self.config.gzip_level:
                codecs.append({"name": "gzip", "configuration": {"level": self.config.gzip_level}})
            array_meta = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": list(level.shape),
                "data_type": self.dtype.name,
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(level.chunk_shape)}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0,
                "codecs": codecs,
                "dimension_names": self.axis_names,
                "attributes": {},
            }
            result[f"{i}/zarr.json"] = _json_bytes(array_meta)

        ome = {
            "version": "0.5",
            "multiscales": [{"name": self.config.name, "axes": self.axes, "datasets": datasets}],
        }
        omero = first_attrs.get("omero")
        n_channels = self.levels[0].shape[self.axis_names.index("c")] if "c" in self.axis_names else 0
        if (isinstance(omero, dict) and isinstance(omero.get("channels"), list)
                and len(omero["channels"]) == n_channels
                and all(isinstance(ch, dict) and isinstance(ch.get("color"), str)
                        and re.fullmatch(r"[0-9a-fA-F]{6}", ch["color"])
                        and isinstance(ch.get("window"), dict)
                        and all(k in ch["window"] for k in ("min", "max", "start", "end"))
                        for ch in omero["channels"])):
            # In OME-Zarr 0.5, OMERO display metadata lives under attributes.ome.
            ome["omero"] = omero
        root_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"ome": ome},
        }
        result["zarr.json"] = _json_bytes(root_meta)
        return result

    def describe(self) -> dict:
        return {
            "input_tiles": len(self.config.tiles),
            "axes": self.axis_names,
            "dtype": self.dtype.name,
            "levels": [
                {"level": i, "shape": list(lev.shape), "chunks": list(lev.chunk_shape),
                 "normalized_origins_zyx": [list(x) for x in lev.origins_zyx],
                 "voxel_scale": list(lev.scale)}
                for i, lev in enumerate(self.levels)
            ],
            "blend": self.config.blend,
            "compression": "gzip" if self.config.gzip_level else "bytes",
        }

    def _parse_chunk_key(self, key: str) -> tuple[int, tuple[int, ...]] | None:
        match = re.fullmatch(r"(0|[1-9]\d*)/c(?:/\d+)+", key)
        if match is None:
            return None
        level = int(match.group(1))
        if level >= len(self.levels):
            return None
        coords_s = key[len(match.group(1)) + 3:].split("/")
        if len(coords_s) != self.ndim or any(len(s) > 12 for s in coords_s):
            return None
        coords = tuple(int(s) for s in coords_s)
        lev = self.levels[level]
        if any(c * ch >= size for c, ch, size in zip(coords, lev.chunk_shape, lev.shape)):
            return None
        return level, coords

    def has_key(self, key: str) -> bool:
        if key in self._metadata:
            return True
        return self._parse_chunk_key(key) is not None

    def _get_bytes_uncached(self, key: str) -> bytes | None:
        if key in self._metadata:
            return self._metadata[key]
        parsed = self._parse_chunk_key(key)
        if parsed is None:
            return None
        level, coords = parsed
        chunk = self.compose_chunk(level, coords)
        # Zarr v3 bytes codec with C order and little-endian multi-byte elements.
        if self.dtype.itemsize > 1:
            little_endian_dtype = self.dtype.newbyteorder("<")
            chunk = np.asarray(chunk, dtype=little_endian_dtype, order="C")
        raw = np.ascontiguousarray(chunk).tobytes(order="C")
        if self.config.gzip_level:
            return gzip.compress(raw, compresslevel=self.config.gzip_level, mtime=0)
        return raw

    def compose_chunk(self, level_index: int, coords: tuple[int, ...]) -> np.ndarray:
        """Produce one full, zero-padded Zarr v3 chunk with no writes to disk."""
        if not 0 <= level_index < len(self.levels):
            raise IndexError(level_index)
        level = self.levels[level_index]
        if len(coords) != self.ndim or any(
            c < 0 or c * ch >= size
            for c, ch, size in zip(coords, level.chunk_shape, level.shape)
        ):
            raise IndexError(coords)

        out = np.zeros(level.chunk_shape, dtype=self.dtype)
        out_start = tuple(c * ch for c, ch in zip(coords, level.chunk_shape))
        # Request end is clipped to the valid mosaic extent; outside remains fill 0.
        out_end = tuple(min(a + b, n) for a, b, n in zip(out_start, level.chunk_shape, level.shape))
        blend = self.config.blend
        if blend == "feather":
            accum = np.zeros(level.chunk_shape, dtype=np.float64)
            weight_sum = np.zeros(level.chunk_shape, dtype=np.float64)
        else:
            covered = np.zeros(level.chunk_shape, dtype=bool) if blend == "first" else None

        for tile_index, (origin, source) in enumerate(zip(level.origins_zyx, level.sources)):
            tile_start = (0,) * (self.ndim - 3) + origin
            tile_end = tuple(tile_start[d] + level.tile_shape[d] for d in range(self.ndim))
            intersection_start = tuple(max(out_start[d], tile_start[d]) for d in range(self.ndim))
            intersection_end = tuple(min(out_end[d], tile_end[d]) for d in range(self.ndim))
            if any(b <= a for a, b in zip(intersection_start, intersection_end)):
                continue
            out_slice = tuple(slice(a - os, b - os) for a, b, os in
                              zip(intersection_start, intersection_end, out_start))
            src_slice = tuple(slice(a - ts, b - ts) for a, b, ts in
                              zip(intersection_start, intersection_end, tile_start))
            patch = np.asarray(source[src_slice])
            expected_shape = tuple(b-a for a, b in zip(intersection_start, intersection_end))
            if patch.shape != expected_shape:
                raise ValueError(f"Tile {tile_index} returned wrong patch shape: {patch.shape} != {expected_shape}")

            if blend == "last":
                out[out_slice] = patch
            elif blend == "first":
                covered_piece = covered[out_slice]
                np.copyto(out[out_slice], patch, where=~covered_piece)
                covered_piece[...] = True
            else:
                # Smoothly cross-fade y/x overlaps; normalize so outer borders
                # are never darkened where there is no neighboring tile.
                weights = np.ones(expected_shape, dtype=np.float64)
                for axis, width in ((self.ndim-2, level.feather_yx[0]),
                                    (self.ndim-1, level.feather_yx[1])):
                    if width <= 0:
                        continue
                    local_indices = np.arange(src_slice[axis].start,
                                              src_slice[axis].stop, dtype=np.float64)
                    n = level.tile_shape[axis]
                    ramp = np.minimum(1.0, np.minimum(
                        (local_indices + 0.5) / width,
                        (n - local_indices - 0.5) / width,
                    ))
                    ramp = np.maximum(ramp, 1e-4)
                    shape = [1] * self.ndim
                    shape[axis] = len(ramp)
                    weights *= ramp.reshape(shape)
                accum[out_slice] += patch.astype(np.float64, copy=False) * weights
                weight_sum[out_slice] += weights

        if blend == "feather":
            mask = weight_sum > 0
            if self.dtype.kind in "iu":
                values = np.rint(accum[mask] / weight_sum[mask])
                limits = np.iinfo(self.dtype)
                out[mask] = np.clip(values, limits.min, limits.max).astype(self.dtype)
            elif self.dtype.kind == "b":
                out[mask] = accum[mask] / weight_sum[mask] >= 0.5
            else:
                out[mask] = (accum[mask] / weight_sum[mask]).astype(self.dtype)
        return out
