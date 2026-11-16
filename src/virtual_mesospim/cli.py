"""Command-line entry point (no source stores are created or modified)."""
from __future__ import annotations

import argparse
import itertools
import json
from urllib.parse import quote

import numpy as np

from .config import MosaicConfig, color_to_rgb, load_mosaic_configs
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


def _layer_source(mosaic: VirtualMosaic, host: str, port: int) -> str:
    base = f"http://{host}:{port}/{mosaic.config.mount_path}"
    if mosaic.config.channel:
        base += f"/{mosaic.config.channel}"
    return f"{base}/|zarr3:"


def _color_shader(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return ("#uicontrol invlerp normalized\n"
            "void main() { float v = normalized(); "
            f"emitRGB(vec3({r} * v, {g} * v, {b} * v)); }}")


def _neuroglancer_url(mosaics: list[VirtualMosaic], host: str, port: int) -> str:
    """Viewer URL for the hosted demo app with one image layer per channel."""
    first = mosaics[0]
    level = first.levels[0]
    names = first.axis_names[-3:]
    scale_um = level.scale[-3:]
    shape = level.shape[-3:]
    layers = []
    opacity = 0.7 if len(mosaics) > 1 else 1
    for mosaic in mosaics:
        layer = {"type": "image", "name": mosaic.config.name,
                 "source": _layer_source(mosaic, host, port), "opacity": opacity}
        if mosaic.config.color:
            layer["shader"] = _color_shader(color_to_rgb(mosaic.config.color))
        window, _ = _view_hints(mosaic)
        if window is not None:
            layer["shaderControls"] = {"normalized": {"range": list(window)}}
        layers.append(layer)
    _, position = _view_hints(first)
    state = {
        "dimensions": {name: [s * 1e-6, "m"] for name, s in zip(names, scale_um)},
        "position": position if position is not None else [size / 2 for size in shape],
        "crossSectionScale": max(1.0, max(size * s for size, s in zip(shape, scale_um))
                                 / max(scale_um) / 800),
        "layers": layers,
        "selectedLayer": {"layer": layers[0]["name"]},
        "layout": "4panel",
    }
    return f"{NEUROGLANCER_APP}#!{quote(json.dumps(state, separators=(',', ':')), safe='')}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Serve an on-demand, read-only MesoSPIM OME-Zarr 0.5 mosaic")
    parser.add_argument("config", help="Path to JSON manifest")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 only behind access control")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--inspect", action="store_true", help="Validate manifest and print geometry; do not run server")
    args = parser.parse_args(argv)

    configs = load_mosaic_configs(args.config)
    mosaics = [VirtualMosaic(config) for config in configs]
    if len(mosaics) == 1:
        print(json.dumps(mosaics[0].describe(), indent=2))
    else:
        print(json.dumps([mosaic.describe() for mosaic in mosaics], indent=2))
    if args.inspect:
        return

    from .http import create_app
    import uvicorn
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    mount = mosaics[0].config.mount_path
    print(f"Virtual store: http://{host}:{args.port}/{mount}/"
          + (f"  (channels: {', '.join(m.config.channel for m in mosaics if m.config.channel)})"
             if len(mosaics) > 1 else ""))
    print(f"Neuroglancer link: {_neuroglancer_url(mosaics, host, args.port)}")
    for mosaic in mosaics:
        label = mosaic.config.channel or mosaic.config.name
        print(f"Layer source ({label}): {_layer_source(mosaic, host, args.port)}")
    uvicorn.run(create_app(mosaics), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
