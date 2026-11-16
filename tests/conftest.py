from pathlib import Path
import numpy as np
import pytest
from virtual_mesospim import MosaicConfig, VirtualMosaic


class FakeArray:
    def __init__(self, data):
        self.data = np.array(data, copy=True)
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def __getitem__(self, key):
        return self.data[key]


class FakeGroup:
    def __init__(self, arrays, attrs):
        self.arrays = arrays
        self.attrs = attrs

    def __getitem__(self, path):
        return self.arrays[path]


def make_groups(tile_values=(10, 20, 30, 40), shape=(1, 1, 2, 4, 4),
                multiscale=False, scales=None):
    if scales is None:
        scales = [1.0] * len(shape)
    axes = ([{'name': 't', 'type': 'time'}, {'name': 'c', 'type': 'channel'}]
            if len(shape) == 5 else [{'name': 'c', 'type': 'channel'}] if len(shape) == 4 else [])
    axes += [{'name': a, 'type': 'space', 'unit': 'micrometer'} for a in ('z', 'y', 'x')]
    source_groups = []
    for value in tile_values:
        arrays = {'0': FakeArray(np.full(shape, value, dtype=np.uint16))}
        datasets = [{'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': scales}]}]
        if multiscale:
            shape1 = shape[:-3] + tuple(max(1, s // 2) for s in shape[-3:])
            arrays['1'] = FakeArray(np.full(shape1, value, dtype=np.uint16))
            datasets.append({'path': '1', 'coordinateTransformations': [
                {'type': 'scale', 'scale': scales[:-3] + [s*2 for s in scales[-3:]]}
            ]})
        attrs = {'multiscales': [{'version': '0.4', 'axes': axes, 'datasets': datasets}]}
        source_groups.append(FakeGroup(arrays, attrs))
    return source_groups


def build(groups=None, **overrides):
    if groups is None:
        groups = make_groups()
    defaults = dict(tiles=tuple(Path(f'/nonexistent/Tile{i}.ome.zarr') for i in range(len(groups))),
                    rows=2, cols=2, overlap_yx=(0.25, 0.25), chunk_shape_zyx=(2, 4, 4),
                    blend='first', cache_chunks=0)
    defaults.update(overrides)
    return VirtualMosaic(MosaicConfig(**defaults), groups=groups)


@pytest.fixture
def mosaic():
    return build()
