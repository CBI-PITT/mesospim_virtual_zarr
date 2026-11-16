"""Integration test runs when zarr-python 3 is installed.

This environment-independent test constructs actual source OME-Zarr 0.4 tiles and
opens the virtual image through Zarr v3's public Store interface.
"""
import numpy as np
import pytest

zarr = pytest.importorskip('zarr')
from virtual_mesospim import MosaicConfig, VirtualMosaic
from virtual_mesospim.store import VirtualZarrV3Store


def test_real_zarr_v2_to_virtual_v3(tmp_path):
    paths = []
    for i, val in enumerate((10,20,30,40)):
        path = tmp_path / f'Tile{i}.ome.zarr'
        root = zarr.group(store=str(path), overwrite=True, zarr_format=2)
        root.create_array('0', data=np.full((1,1,2,4,4), val, dtype='uint16'), chunks=(1,1,2,2,2))
        root.attrs['multiscales'] = [{
            'version': '0.4',
            'axes': [{'name':'t','type':'time'}, {'name':'c','type':'channel'},
                     {'name':'z','type':'space','unit':'micrometer'},
                     {'name':'y','type':'space','unit':'micrometer'},
                     {'name':'x','type':'space','unit':'micrometer'}],
            'datasets': [{'path':'0','coordinateTransformations':[{'type':'scale','scale':[1,1,1,1,1]}]}]
        }]
        paths.append(path)

    mosaic = VirtualMosaic(MosaicConfig(tuple(paths), rows=2, cols=2,
                                      overlap_yx=(0.25,0.25),
                                      chunk_shape_zyx=(2,4,4), blend='first'))
    store = VirtualZarrV3Store(mosaic)
    virtual = zarr.open_array(store=store, path='0', mode='r', zarr_format=3)
    assert virtual.shape == (1,1,2,7,7)
    values = virtual[0,0,0]
    assert values[0,0] == 10
    assert values[6,0] == 20
    assert values[0,6] == 30
    assert values[6,6] == 40
    with pytest.raises((PermissionError, ValueError)):
        virtual[0,0,0,0,0] = 100
