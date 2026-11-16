"""Command-line entry point (no source stores are created or modified)."""
from __future__ import annotations

import argparse
import itertools
import json
from urllib.parse import quote

import numpy as np

from .config import MosaicConfig
from .engine import VirtualMosaic


NEUROGLANCER_APP = "https://neuroglancer-demo.appspot.com/"


def _view_hints(mosaic: VirtualMosaic) -> tuple[tuple[float, float] | None, list[float] | None]:
    """Display window and a data-centered position, estimated from the coarsest level."""
    try:
        index = len(mosaic.levels) - 1
        level = mosaic.levels[index]
        if level.shape[-3] * level.shape[-2] * level.shape[-1] > 64_000_000:
            return None, None
        ranges = [range(-(-size // chunk)) for size, chunk in zip(level.shape, level.chunk_shape)]
        full = np.empty(level.shape, dtype=mosaic.dtype)
        for coords in itertools.product(*ranges):
            chunk = mosaic.compose_chunk(index, coords)
            start = tuple(c * ch for c, ch in zip(coords, level.chunk_shape))
            stop = tuple(min(a + b, n) for a, b, n in zip(start, level.chunk_shape, level.shape))
            full[tuple(slice(a, b) for a, b in zip(start, stop))] = \
                chunk[tuple(slice(0, b - a) for a, b in zip(start, stop))]
        window = position = None
        values = full.ravel()
        nonzero = values[values > 0]
        if nonzero.size >= 0.01 * values.size:
            low, high = np.percentile(nonzero, (1, 99))
            if np.isfinite(low) and np.isfinite(high) and high > low:
                window = (float(low), float(high))
            bright = full > np.percentile(nonzero, 90)
            if bright.any():
                coords = np.nonzero(bright)[-3:]
                weights = full[bright].astype(np.float64)
                total = weights.sum()
                ratios = [mosaic.levels[0].shape[-3 + j] / full.shape[-3 + j] for j in range(3)]
                position = [float((c * weights).sum() / total) * r
                            for c, r in zip(coords, ratios)]
        return window, position
    except Exception:
        return None, None


def _neuroglancer_url(mosaic: VirtualMosaic, source: str) -> str:
    """Viewer URL for the hosted demo app with the mosaic layer preloaded."""
    level = mosaic.levels[0]
    names = mosaic.axis_names[-3:]
    scale_um = level.scale[-3:]
    shape = level.shape[-3:]
    state = {
        "dimensions": {name: [s * 1e-6, "m"] for name, s in zip(names, scale_um)},
        "position": [size / 2 for size in shape],
        "crossSectionScale": max(1.0, max(size * s for size, s in zip(shape, scale_um))
                                 / max(scale_um) / 800),
        "layers": [{"type": "image", "name": mosaic.config.name,
                    "source": source, "opacity": 1}],
        "selectedLayer": {"layer": mosaic.config.name},
        "layout": "4panel",
    }
    window, position = _view_hints(mosaic)
    if position is not None:
        state["position"] = position
    if window is not None:
        state["layers"][0]["shaderControls"] = {"normalized": {"range": list(window)}}
    return f"{NEUROGLANCER_APP}#!{quote(json.dumps(state, separators=(',', ':')), safe='')}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Serve an on-demand, read-only MesoSPIM OME-Zarr 0.5 mosaic")
    parser.add_argument("config", help="Path to JSON manifest")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 only behind access control")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--inspect", action="store_true", help="Validate manifest and print geometry; do not run server")
    args = parser.parse_args(argv)

    config = MosaicConfig.from_json(args.config)
    mosaic = VirtualMosaic(config)
    print(json.dumps(mosaic.describe(), indent=2))
    if args.inspect:
        return

    from .http import create_app
    import uvicorn
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    source = f"http://{host}:{args.port}/{config.mount_path}/|zarr3:"
    print(f"Virtual store: http://{host}:{args.port}/{config.mount_path}/")
    print(f"Neuroglancer link: {_neuroglancer_url(mosaic, source)}")
    print(f"Layer source: {source}")
    uvicorn.run(create_app(mosaic), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
