import asyncio
import json
from urllib.parse import unquote

import httpx
import pytest
import zarr

from virtual_mesospim.service import create_service_app, sign_token, verify_token


SECRET = 'test-secret'


def create_tile_dataset(group, name, shape, dtype, chunks):
    """zarr-python 2 create_dataset / zarr-python 3 create_array."""
    if hasattr(group, 'create_dataset'):
        return group.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)
    return group.create_array(name=name, shape=shape, dtype=dtype, chunks=chunks)


def build_dataset(root):
    """Real Zarr v2 tiles and a manifest in the layout the web app writes."""
    acquisition = root / 'scan.ome.zarr'
    acquisition.mkdir(parents=True)
    channels = []
    for channel in ('488', '561'):
        tiles = []
        for number in range(4):
            tile = acquisition / f'Mag8_Tile{number}_Ch{channel}_Sh0.ome.zarr'
            try:
                group = zarr.open_group(str(tile), mode='w', zarr_format=2)
            except TypeError:
                group = zarr.open_group(str(tile), mode='w')
            y, x = 12 * (number % 2), 12 * (number // 2)
            array = create_tile_dataset(group, '0', (2, 8, 8), 'uint16', (2, 4, 4))
            for z in range(2):
                array[z, :, :] = (number + 1) * 100 + z
            group.attrs['multiscales'] = [{
                'version': '0.4',
                'axes': [{'name': axis, 'type': 'space'} for axis in 'zyx'],
                'datasets': [{'path': '0', 'coordinateTransformations': [
                    {'type': 'scale', 'scale': [1.0, 1.0, 1.0]},
                    {'type': 'translation', 'translation': [0, y, x]},
                ]}],
            }]
            tiles.append(f'scan.ome.zarr/{tile.name}')
        channels.append({'name': f'Ch{channel}', 'tiles': tiles})
    manifest = {
        'channels': channels,
        'rows': 2, 'cols': 2,
        'overlap_yx': [0.25, 0.25],
        'chunk_shape_zyx': [2, 4, 4],
        'blend': 'first',
        'positions_zyx': [[0, 0, 0], [0, 12, 0], [0, 0, 12], [0, 12, 12]],
        'row_direction': 1, 'col_direction': 1,
        'gzip_level': 0, 'cache_chunks': 2, 'max_concurrent_chunks': 2,
        'name': 'Test mosaic', 'mount_path': 'stitched.ome.zarr',
    }
    (root / 'mosaic.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root


def test_verify_token_roundtrip(tmp_path):
    token = sign_token(SECRET, 7, str(tmp_path), 60)
    assert verify_token(token, SECRET, 7) == str(tmp_path)
    assert verify_token(token, SECRET, 8) is None
    assert verify_token(token, 'wrong-secret', 7) is None
    assert verify_token(token + 'x', SECRET, 7) is None
    assert verify_token(token.split('.')[0], SECRET, 7) is None
    assert verify_token('', SECRET, 7) is None
    assert verify_token(sign_token(SECRET, 7, str(tmp_path), -10), SECRET, 7) is None


def test_viewer_redirect_and_layers(tmp_path):
    root = build_dataset(tmp_path)
    app = create_service_app(SECRET, None, 4)
    token = sign_token(SECRET, 1, str(root), 3600)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.get('/datasets/1/viewer', params={'token': token})
            assert response.status_code == 302
            location = response.headers['location']
            # Regression: the bundled client is the python-integration build
            # and requires a /v/<token>/ path segment to initialize.
            assert location.startswith('/v/viewer/index.html#!')
            state = json.loads(unquote(location.split('#!', 1)[1]))
            assert [layer['name'] for layer in state['layers']] == \
                ['Test mosaic Ch488', 'Test mosaic Ch561']
            # Regression: layer sources must be absolute URLs carrying the
            # token in the path (the client corrupts query-string tokens when
            # it appends chunk keys to the namespace URL).
            assert state['layers'][0]['source'] == \
                f'http://test/t/{token}/datasets/1/stitched.ome.zarr/Ch488/|zarr3:'
            assert state['layers'][1]['source'] == \
                f'http://test/t/{token}/datasets/1/stitched.ome.zarr/Ch561/|zarr3:'
            assert state['layers'][0]['opacity'] == 0.7
            assert set(state['dimensions']) == {'z', 'y', 'x'}
            assert all(state['dimensions'][name][1] == 'm' for name in state['dimensions'])
            # Single-panel, zoomed-out view: the largest visible extent in
            # canonical voxels over the 200-pixel target.
            assert state['layout'] == 'xy'
            assert state['crossSectionScale'] == pytest.approx(20 / 200)
            # Position is in voxel coordinates: the middle z/y/x plane of the
            # 2 x 20 x 20 voxel volume.
            assert state['position'] == pytest.approx([1, 10, 10])

    asyncio.run(run())


def test_viewer_rejects_missing_or_bad_token(tmp_path):
    build_dataset(tmp_path)
    app = create_service_app(SECRET, None, 4)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            assert (await client.get('/datasets/1/viewer')).status_code == 403
            assert (await client.get('/datasets/1/viewer', params={'token': 'x.y'})).status_code == 403
            assert (await client.get('/datasets/1/viewer',
                                     params={'token': sign_token(SECRET, 2, str(tmp_path), 60)})
                    ).status_code == 403

    asyncio.run(run())


def test_chunk_serving_with_token(tmp_path):
    root = build_dataset(tmp_path)
    app = create_service_app(SECRET, None, 4)
    token = sign_token(SECRET, 1, str(root), 3600)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            # Path-token namespace, the format the viewer's layer sources use.
            base = f'/t/{token}/datasets/1/stitched.ome.zarr/Ch488'
            meta = await client.get(f'{base}/zarr.json')
            assert meta.status_code == 200
            assert meta.json()['zarr_format'] == 3
            chunk = await client.get(f'{base}/0/c/0/0/0')
            assert chunk.status_code == 200
            assert len(chunk.content) == 2 * 4 * 4 * 2
            part = await client.get(f'{base}/0/c/0/0/0', headers={'Range': 'bytes=2-7'})
            assert part.status_code == 206
            assert part.content == chunk.content[2:8]
            assert part.headers['content-range'] == f'bytes 2-7/{len(chunk.content)}'
            # Bad or corrupted token, unknown key, missing channel.
            assert (await client.get(f'/t/{token}x/datasets/1/stitched.ome.zarr/Ch488/zarr.json')
                    ).status_code == 403
            assert (await client.get(f'/t/{token}/datasets/9/stitched.ome.zarr/Ch488/zarr.json')
                    ).status_code == 403
            assert (await client.get(f'{base}/9/c/0/0/0')).status_code == 404
            assert (await client.get(f'/t/{token}/datasets/1/stitched.ome.zarr/Ch999/zarr.json')
                    ).status_code == 404
            assert (await client.get(f'/t/{token}/datasets/1/other.ome.zarr/zarr.json')
                    ).status_code == 404
            health = (await client.get('/healthz')).json()
            assert health['status'] == 'ok'

    asyncio.run(run())


def test_single_channel_top_level_tiles(tmp_path):
    build_dataset(tmp_path)
    manifest = {
        'tiles': [f'scan.ome.zarr/Mag8_Tile{n}_Ch488_Sh0.ome.zarr' for n in range(4)],
        'rows': 2, 'cols': 2,
        'overlap_yx': [0.25, 0.25],
        'chunk_shape_zyx': [2, 4, 4],
        'blend': 'first',
        'positions_zyx': [[0, 0, 0], [0, 12, 0], [0, 0, 12], [0, 12, 12]],
        'row_direction': 1, 'col_direction': 1,
        'gzip_level': 0, 'cache_chunks': 2, 'max_concurrent_chunks': 2,
        'name': 'Solo', 'mount_path': 'stitched.ome.zarr',
    }
    (tmp_path / 'mosaic.json').write_text(json.dumps(manifest), encoding='utf-8')
    app = create_service_app(SECRET, None, 4)
    token = sign_token(SECRET, 1, str(tmp_path), 3600)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            meta = await client.get('/datasets/1/stitched.ome.zarr/zarr.json',
                                    params={'token': token})
            assert meta.status_code == 200
            chunk = await client.get('/datasets/1/stitched.ome.zarr/0/c/0/0/0',
                                     params={'token': token})
            assert chunk.status_code == 200
            response = await client.get('/datasets/1/viewer', params={'token': token})
            state = json.loads(unquote(response.headers['location'].split('#!', 1)[1]))
            assert [layer['name'] for layer in state['layers']] == ['Solo']
            assert state['layers'][0]['source'] == \
                f'http://test/t/{token}/datasets/1/stitched.ome.zarr/|zarr3:'

    asyncio.run(run())


def test_manifest_change_invalidates_cache(tmp_path):
    root = build_dataset(tmp_path)
    app = create_service_app(SECRET, None, 4)
    token = sign_token(SECRET, 1, str(root), 3600)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.get('/datasets/1/viewer', params={'token': token})
            state = json.loads(unquote(response.headers['location'].split('#!', 1)[1]))
            assert state['layers'][0]['name'] == 'Test mosaic Ch488'

            manifest = json.loads((root / 'mosaic.json').read_text(encoding='utf-8'))
            manifest['name'] = 'Renamed'
            (root / 'mosaic.json').write_text(json.dumps(manifest), encoding='utf-8')

            response = await client.get('/datasets/1/viewer', params={'token': token})
            state = json.loads(unquote(response.headers['location'].split('#!', 1)[1]))
            assert state['layers'][0]['name'] == 'Renamed Ch488'

    asyncio.run(run())


def test_missing_manifest(tmp_path):
    root = tmp_path / 'empty'
    root.mkdir()
    app = create_service_app(SECRET, None, 4)
    token = sign_token(SECRET, 1, str(root), 3600)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.get('/datasets/1/viewer', params={'token': token})
            assert response.status_code == 500
            assert 'manifest' in response.json()['detail']

    asyncio.run(run())


def test_invalid_manifest(tmp_path):
    root = tmp_path / 'broken'
    root.mkdir()
    (root / 'mosaic.json').write_text('{"rows": 2}', encoding='utf-8')
    app = create_service_app(SECRET, None, 4)
    token = sign_token(SECRET, 1, str(root), 3600)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.get('/datasets/1/viewer', params={'token': token})
            assert response.status_code == 500

    asyncio.run(run())
