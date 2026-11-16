# MesoSPIM virtual OME-Zarr v3 mosaic (prototype v0.1)

This is a **read-only virtual image**, not an offline stitching/export tool. It reads
original **OME-Zarr 0.4 / Zarr v2 tile stores** and serves a new **OME-Zarr 0.5 /
Zarr v3** namespace over HTTP. It does **not** overwrite, modify, or duplicate
source Zarr tile files. Only requested output chunks are assembled in RAM.

It includes:

- `VirtualMosaic` — tile geometry, multiscale transforms and chunk compositing
- `VirtualZarrV3Store` — native read-only Zarr-Python 3 `Store` implementation
- FastAPI server — serves the exact same virtual keys to Neuroglancer via HTTP
- Col-major tile numbering: `tile_index = col * rows + row`
- Automatic grid offsets from fractional tile overlaps; manual integer positions
- `first`, `last`, or `feather` overlap policy; normalized feather blending
- Existing multiscale levels served on demand; no new downsampling computation
- `uint16` input preserved as `uint16` output, including for feather blending
- Full-size, zero-padded Zarr v3 edge chunks and standard default chunk keys
- Optional per-chunk gzip compression, in-memory LRU and limited HTTP concurrency
- Tests, including an optional real-Zarr v2 → v3 integration test

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

If using micromamba / conda, activate your environment, and run the same
`pip install -e '.[test]'` command. Tested logic is Python 3.10+ compatible.

**Note:** The distribution needs Zarr-Python **3.x** (which can still *read*
Zarr v2 stores), not the old Python package `zarr==2.x`.

## 2. Describe the source tiles

Edit `examples/mosaic.example.json` with your actual paths, which must point to
**individual tile OME-Zarr roots** containing `.zgroup` and `.zattrs` (usually
`Mag4_TileNN_...ome.zarr`), not to the enclosing directory.

The sample is a **3-row × 2-column** grid:

```text
      col 0     col 1
row 0 tile 0    tile 3
row 1 tile 1    tile 4
row 2 tile 2    tile 5
```

Each tile should contain one OME-Zarr multiscale image with levels, e.g. `0`,
`1`, `2`, etc. The actual source level names are read from OME metadata, and
are mapped to `0`, `1`, ... in the virtual store.

For a large MesoSPIM 4 × 3 grid, provide 12 paths in the order `Tile0` ...
`Tile11`, with `rows: 4, cols: 3`.

### Example: automatically create a manifest

```python
import json
import re
from pathlib import Path

folder = Path('/path/to/your/original/ome_zarr_tiles')
paths = list(folder.glob('Mag4_Tile*_Ch488*.ome.zarr'))
paths.sort(key=lambda p: int(re.search(r'Tile(\d+)', p.name).group(1)))
assert len(paths) == 12, f'Expected 12 tiles, found {len(paths)}'

config = {
    'tiles': [str(path.resolve()) for path in paths],
    'rows': 4,
    'cols': 3,
    'overlap_yx': [0.10, 0.10],
    'chunk_shape_zyx': [32, 256, 256],
    'blend': 'feather',
    'gzip_level': 0,
    'cache_chunks': 8,
    'max_concurrent_chunks': 4,
    'name': 'MesoSPIM Ch488 virtual mosaic',
    'mount_path': 'stitched.ome.zarr',
}
Path('my_mosaic.json').write_text(json.dumps(config, indent=2))
```

**For multi-channel acquisitions:** this version can serve `CZYX` or `TCZYX`
source arrays, preserving an existing channel dimension. If each wavelength is
in a *separate* OME-Zarr tile, run one virtual mosaic per channel/manifest; you
can add each mosaic as a separate Neuroglancer layer. Merging separate channel
stores into a newly synthesized channel dimension is not yet implemented.

## 3. Inspect geometry (does not load the tile image pixels)

```bash
mesospim-virtual-zarr my_mosaic.json --inspect
```

This reports axes, dtype, each virtual level's final shape, chunk shape,
physical scale and per-tile normalized voxel origins. Inspect those origins to
ensure `rows`, `cols`, ordering and orientation are correct.

## 4. Run the HTTP server

```bash
mesospim-virtual-zarr my_mosaic.json --host 127.0.0.1 --port 8080
```

Requests look like:

```text
GET http://127.0.0.1:8080/stitched.ome.zarr/zarr.json
GET http://127.0.0.1:8080/stitched.ome.zarr/0/zarr.json
GET http://127.0.0.1:8080/stitched.ome.zarr/0/c/0/0/0/0/0
GET http://127.0.0.1:8080/info
```

The server prints a **ready-to-open Neuroglancer link** with the mosaic layer
already configured, including a data-centered initial position, zoom level and
display window:

```text
Neuroglancer link: https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22...
```

Open this link in a desktop browser on the same machine as the server. The
first fetch from the hosted viewer to `127.0.0.1` triggers Chrome's
local-network permission prompt; click **Allow**. The virtual server answers
the required Private Network Access preflight automatically.

To add the layer manually instead, create a new image layer with source URL:

```text
http://127.0.0.1:8080/stitched.ome.zarr/|zarr3:
```

If a manually added layer stays empty grey, prefer the printed link: it sets a
position inside the volume, which the default view does not.

For a remote Neuroglancer served over HTTPS, publish the virtual mosaic behind
an HTTPS reverse proxy and configure CORS. See Neuroglancer's HTTP/CORS docs.
A remote Neuroglancer viewer cannot reach `127.0.0.1` on a remote HPC node:
use an SSH tunnel to the **user's local machine** or a properly configured
network-accessible HTTPS gateway.

**Security:** The demo server sends `Access-Control-Allow-Origin: *`. Bind to
`127.0.0.1` unless you have authentication/authorization and HTTPS in place.
Never expose private microscopy data to an unauthenticated public endpoint.

## 5. Read via Zarr-Python 3

You can also read the virtual view *inside Python*, no HTTP server required:

```python
import zarr
from virtual_mesospim import MosaicConfig, VirtualMosaic
from virtual_mesospim.store import VirtualZarrV3Store

cfg = MosaicConfig.from_json('my_mosaic.json')
mosaic = VirtualMosaic(cfg)
virtual_store = VirtualZarrV3Store(mosaic)

# Open a known array directly; virtual chunk-key enumeration is not supported.
array = zarr.open_array(store=virtual_store, path='0', mode='r', zarr_format=3)
print(array.shape, array.dtype, array.chunks)
small_patch = array[0, 0, 10:12, 100:200, 100:200]  # if input is TCZYX
```

The store is deliberately **non-listable**: enumerating billions of virtual
chunk paths would be expensive. Metadata and exact chunk requests work. Root
`group.keys()` and directory auto-completion are not available.

## 6. Positioning rules

The automatic grid computes the highest-resolution (level 0) tile origins as:

```python
index = col * rows + row
z0 = 0
y0 = row * round(tile_height * (1 - overlap_y)) * row_direction
x0 = col * round(tile_width * (1 - overlap_x)) * col_direction
```

The implementation uses consistent half-away-from-zero rounding. `row_direction`
and `col_direction` can be `+1` or `-1`; negative positions are internally
normalized to non-negative voxel positions. Positioning is **not**
inferred from MesoSPIM stage metadata.

Physical stage translations from the tile metadata are intentionally **not**
serialized into the virtual OME metadata: the Neuroglancer zarr driver stops
requesting chunks when a dataset transform contains a translation. Tile
placement remains exact in voxel space (see `--inspect`).

For registered but integer-aligned source tiles, override the grid with a
`positions_zyx` list containing one `[z, y, x]` **voxel** origin for each tile
(at the original highest-resolution scale). See `examples/manual_positions.example.json`.
Entries may be negative. At coarser levels the code converts the origins from
level 0 physical units to each level's pixel coordinates and rounds to the
nearest voxel. This means coarser-level mosaics may differ by ±1 voxel along
seams. Actual pixel resampling/subpixel registration is outside this version.

**Note:** This method keeps each tile's pixels as acquired. It does not
rotate/flip tiles, correct laser-dependent gain differences, estimate tile
transformations, or run an image registration algorithm.

## 7. Blending

- `first`: the lowest tile number wins in all overlaps (no extra floating-point
  image buffer, fastest).
- `last`: the highest tile number wins in overlaps (fastest).
- `feather`: weighted interpolation across y/x overlap strips; edge-normalized
  so single-tile borders do not darken. It uses floating-point arithmetic only
  while composing a chunk and rounds/clamps back into the input dtype.

Feather width is estimated from `overlap_yx` and each level's tile shape. If you
provide explicit positions whose overlaps vary significantly, this fixed
feather width may not be optimal. First try `blend: first` to verify geometry.

## 8. Chunking, memory and performance

Recommended starting point for `uint16`, `TCZYX` images:

```json
{
  "chunk_shape_zyx": [32, 256, 256],
  "gzip_level": 0,
  "cache_chunks": 8,
  "max_concurrent_chunks": 4
}
```

For `uint16`, one such chunk contains 4 MiB of uncompressed pixels. Feather
blending allocates two `float64` accumulator arrays, adding about 32 MiB for
these arrays (plus transient patches). At 4 concurrent requests, peaks can be
hundreds of MiB plus the original Zarr chunk read cache and tile data.
Adjust `max_concurrent_chunks` according to SLURM memory limits.

`gzip_level: 1-9` uses the standard Zarr v3 `gzip` codec. `0` means raw `bytes`
codec; raw is fast but uses greater network bandwidth. The source Zarr v2
compressor is decoded by Zarr-Python when reading input.

No chunks are persisted. `cache_chunks` is a bounded RAM-only LRU; restart the
server to clear it. Zero disables this cache. If you change a source tile while
server is running (not recommended), restart to avoid stale cached data.

**Important:** Each requested output chunk may cause one or more existing
compressed source chunks to be read and decoded. First visits to large
regions can therefore be slower than a fully materialized stitched Zarr.

## 9. Current limitations

1. Source OME-Zarr roots must have one `multiscales` entry and compatible
   level counts, source array shapes, axes, dtype and per-level pixel scales.
2. Accepted axes are `ZYX`, `CZYX`, `TZYX`, `TCZYX`, in that order.
3. Only integer axis-aligned tile translations are composited; no rotation,
   nonrigid warping or subpixel interpolation.
4. Every tile is expected to contain all listed multiscale levels; lower
   resolution levels are *not* generated on demand from level 0.
5. The output is a live virtual Zarr store accessible via the Python adapter
   or HTTP. It is **not** an offline, directly openable directory on disk.
6. No cross-process cache sharing. Run one `uvicorn` worker with the provided
   CLI; use a reverse proxy for network security if needed.
7. This version does not parse nonstandard input axes or source metadata
   containing group-level `coordinateTransformations`.

## 10. Testing

```bash
python -m pip install -e '.[test]'
pytest -q
```

The tests include a Zarr-python integration test (construct real Zarr v2 tile
arrays, then read the result through the v3 Store interface), geometry and
blend tests, multiscale metadata, edge padding, gzip and HTTP range support.

This is an initial implementation: verify on a *small* real MesoSPIM dataset
and inspect the seams/orientation before serving large whole-brain volumes.

## References

- https://ngff.openmicroscopy.org/0.5/
- https://zarr-specs.readthedocs.io/en/latest/v3/core/
- https://neuroglancer-docs.web.app/datasource/zarr/index.html
- https://neuroglancer-docs.web.app/datasource/http/index.html
- https://zarr.readthedocs.io/en/stable/user-guide/storage/
