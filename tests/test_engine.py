import gzip
import json

import numpy as np
import pytest

from conftest import make_groups, build


def test_col_major_order_and_overlap(mosaic):
    lev = mosaic.levels[0]
    assert lev.shape == (1, 1, 2, 7, 7)
    assert lev.origins_zyx == ((0, 0, 0), (0, 3, 0), (0, 0, 3), (0, 3, 3))
    # Request the entire 7x7 grid as a single full virtual chunk.
    m = build(chunk_shape_zyx=(2, 7, 7))
    image = m.compose_chunk(0, (0, 0, 0, 0, 0))[0, 0, 0]
    assert image[0, 0] == 10
    assert image[6, 0] == 20
    assert image[0, 6] == 30
    assert image[6, 6] == 40
    # In doubly overlapped center the first source wins.
    assert image[3, 3] == 10


def test_blend_modes():
    first = build(chunk_shape_zyx=(2, 7, 7), blend='first')
    last = build(chunk_shape_zyx=(2, 7, 7), blend='last')
    feather = build(chunk_shape_zyx=(2, 7, 7), blend='feather')
    coord = (0, 0, 0, 0, 0)
    a = first.compose_chunk(0, coord)[0, 0, 0]
    b = last.compose_chunk(0, coord)[0, 0, 0]
    c = feather.compose_chunk(0, coord)[0, 0, 0]
    assert a[3, 3] == 10
    assert b[3, 3] == 40
    assert c[3, 3] == 25
    assert c[0, 0] == 10 and c[6, 6] == 40  # normalization does not darken edges
    assert np.array_equal(c[1, :3], np.full(3, 10))


def test_zarr_v3_metadata():
    m = build()
    root = json.loads(m.get_bytes('zarr.json'))
    arr = json.loads(m.get_bytes('0/zarr.json'))
    assert root['zarr_format'] == 3
    assert root['node_type'] == 'group'
    assert root['attributes']['ome']['version'] == '0.5'
    ms = root['attributes']['ome']['multiscales'][0]
    assert [a['name'] for a in ms['axes']] == ['t', 'c', 'z', 'y', 'x']
    assert ms['datasets'][0]['path'] == '0'
    assert arr['zarr_format'] == 3
    assert arr['shape'] == [1, 1, 2, 7, 7]
    assert arr['dimension_names'] == ['t', 'c', 'z', 'y', 'x']
    assert arr['data_type'] == 'uint16'
    assert arr['codecs'] == [{'name': 'bytes', 'configuration': {'endian': 'little'}}]
    assert arr['chunk_key_encoding'] == {'name':'default','configuration':{'separator':'/'}}


def test_bytes_raw_and_edge_padding():
    m = build(chunk_shape_zyx=(2, 4, 4))
    key = '0/c/0/0/0/1/1'
    data = m.get_bytes(key)
    assert len(data) == 2 * 4 * 4 * 2
    block = np.frombuffer(data, dtype='<u2').reshape((2, 4, 4))
    assert block[0, 0, 0] == 40  # global (y=4,x=4) belongs to tile 3
    assert np.all(block[:, 3, :] == 0)
    assert np.all(block[:, :, 3] == 0)
    assert not m.has_key('0/c/0/0/0/2/0')
    assert m.get_bytes('99/zarr.json') is None
    assert m.get_bytes('../Tile0/.zarray') is None


def test_gzip_roundtrip():
    m = build(gzip_level=4)
    raw = build()
    key = '0/c/0/0/0/0/0'
    assert gzip.decompress(m.get_bytes(key)) == raw.get_bytes(key)
    codecs = json.loads(m.get_bytes('0/zarr.json'))['codecs']
    assert codecs[-1] == {'name':'gzip','configuration':{'level':4}}


def test_multiscale_geometry_and_spacing():
    groups = make_groups(shape=(1, 1, 4, 8, 8), multiscale=True,
                         scales=[1, 1, 2.0, 1.5, 1.0])
    m = build(groups, chunk_shape_zyx=(2,4,4))
    assert len(m.levels) == 2
    assert m.levels[0].shape == (1, 1, 4, 14, 14)
    assert m.levels[1].shape == (1, 1, 2, 7, 7)
    assert m.levels[1].origins_zyx == ((0,0,0),(0,3,0),(0,0,3),(0,3,3))
    ms = json.loads(m.get_bytes('zarr.json'))['attributes']['ome']['multiscales'][0]
    assert ms['datasets'][1]['coordinateTransformations'][0]['scale'] == [1,1,4,3,2]
    assert len(m.get_bytes('1/c/0/0/0/1/1')) == 2 * 4 * 4 * 2


def test_negative_directions_and_scale_only_transforms():
    m = build(row_direction=-1, col_direction=-1)
    lev = m.levels[0]
    assert lev.shape == (1,1,2,7,7)
    assert lev.origins_zyx == ((0,3,3),(0,0,3),(0,3,0),(0,0,0))
    # Neuroglancer's zarr driver fails on translations, so only scale is emitted.
    transforms = json.loads(m.get_bytes('zarr.json'))['attributes']['ome']['multiscales'][0]['datasets'][0]['coordinateTransformations']
    assert transforms == [{'type':'scale','scale':[1,1,1,1,1]}]


def test_explicit_z_positions_and_gaps():
    m = build(positions_zyx=((0,0,0), (1,8,0), (0,0,8), (1,8,8)),
              chunk_shape_zyx=(4,16,16))
    assert m.levels[0].shape == (1,1,3,12,12)
    chunk = m.compose_chunk(0, (0,0,0,0,0))
    assert chunk[0,0,0,6,6] == 0  # No tile intersects this gap
    assert chunk[0,0,1,10,10] == 40


def test_source_arrays_not_mutated():
    groups = make_groups()
    original = [[a.data.copy() for a in group.arrays.values()] for group in groups]
    m = build(groups, blend='feather', chunk_shape_zyx=(2,7,7))
    m.compose_chunk(0, (0,0,0,0,0))
    for group, snapshots in zip(groups, original):
        for arr, before in zip(group.arrays.values(), snapshots):
            np.testing.assert_array_equal(arr.data, before)


def test_reject_unsupported_dtype_and_shapes():
    groups = make_groups()
    groups[1].arrays['0'] = type(groups[1].arrays['0'])(np.zeros((1,1,2,5,4), dtype=np.uint16))
    with pytest.raises(ValueError, match='non-z shape'):
        build(groups)
    groups = make_groups()
    groups[1].arrays['0'] = type(groups[1].arrays['0'])(np.zeros((1,1,2,4,4), dtype=np.uint8))
    with pytest.raises(ValueError, match='dtype'):
        build(groups)
    with pytest.raises(ValueError, match='Expected 1 tiles'):
        build(rows=1, cols=1)


def test_gain_last_first_and_feather_blends():
    gains = (1.0, 2.0, 0.5, 1.0)
    last = build(blend='last', gains=gains)
    first = build(blend='first', gains=gains)
    feather = build(blend='feather', gains=gains)
    assert last.describe()['gains'] == [0.5, 2.0]
    chunk = last.compose_chunk(0, (0,0,0,0,0))
    assert chunk[0,0,0,0,0] == 10
    assert chunk[0,0,0,3,0] == 40
    assert chunk[0,0,0,0,3] == 15
    assert chunk[0,0,0,3,3] == 40
    chunk = first.compose_chunk(0, (0,0,0,0,0))
    assert chunk[0,0,0,3,0] == 10
    chunk = first.compose_chunk(0, (0,0,0,0,1))
    assert chunk[0,0,0,0,0] == 15
    chunk = feather.compose_chunk(0, (0,0,0,0,0))
    assert chunk[0,0,0,0,0] == 10
    assert chunk[0,0,0,3,0] == 25


def test_gain_clipped_to_dtype_range():
    groups = make_groups(tile_values=(60000, 60000, 60000, 60000))
    m = build(groups, blend='last', gains=(2.0, 1.0, 1.0, 1.0))
    chunk = m.compose_chunk(0, (0,0,0,0,0))
    assert chunk[0,0,0,0,0] == 65535
    assert chunk[0,0,0,3,0] == 60000


def test_gain_one_applied_is_identity():
    groups = make_groups()
    plain = build(groups, blend='last')
    gained = build(make_groups(), blend='last', gains=(1.0, 1.0, 1.0, 1.0))
    for coords in ((0,0,0,0,0), (0,0,0,0,1)):
        np.testing.assert_array_equal(plain.compose_chunk(0, coords),
                                      gained.compose_chunk(0, coords))


def test_allows_differing_z_extent():
    groups = make_groups(shape=(1, 1, 4, 4, 4), multiscale=True)
    groups[1].arrays['0'] = type(groups[1].arrays['0'])(np.full((1, 1, 3, 4, 4), 20, dtype=np.uint16))
    groups[1].arrays['1'] = type(groups[1].arrays['0'])(np.full((1, 1, 1, 2, 2), 20, dtype=np.uint16))
    m = build(groups, blend='last', chunk_shape_zyx=(2, 4, 4))
    assert m.levels[0].shape == (1, 1, 4, 7, 7)
    assert m.levels[1].shape == (1, 1, 2, 4, 4)
    chunk = m.compose_chunk(0, (0, 0, 1, 0, 0))
    assert chunk[0, 0, 0, 3, 0] == 20
    assert chunk[0, 0, 1, 3, 0] == 10
    assert chunk[0, 0, 1, 0, 0] == 10
    chunk = m.compose_chunk(1, (0, 0, 0, 0, 0))
    assert chunk[0, 0, 0, 2, 0] == 20
    assert chunk[0, 0, 1, 2, 0] == 0
