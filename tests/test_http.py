import asyncio
import httpx

from virtual_mesospim.http import create_app
from conftest import build


def test_http_zarr_get_head_range_cors():
    mosaic = build()
    app = create_app(mosaic)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            base = '/stitched.ome.zarr'
            metadata = await client.get(f'{base}/zarr.json', headers={'Origin': 'https://viewer.test'})
            assert metadata.status_code == 200
            assert metadata.json()['attributes']['ome']['version'] == '0.5'
            assert metadata.headers['access-control-allow-origin'] == '*'
            key = f'{base}/0/c/0/0/0/0/0'
            chunk = await client.get(key)
            assert chunk.status_code == 200
            assert len(chunk.content) == 2 * 4 * 4 * 2
            part = await client.get(key, headers={'Range': 'bytes=2-7'})
            assert part.status_code == 206
            assert part.content == chunk.content[2:8]
            assert part.headers['content-range'] == f'bytes 2-7/{len(chunk.content)}'
            suffix = await client.get(key, headers={'Range': 'bytes=-4'})
            assert suffix.status_code == 206
            assert suffix.content == chunk.content[-4:]
            head = await client.head(key)
            assert head.status_code == 200
            assert head.content == b''
            assert head.headers['content-length'] == str(len(chunk.content))
            assert (await client.get(f'{base}/0/c/0/0/0/3/0')).status_code == 404
            assert (await client.post(key)).status_code == 405
            assert (await client.get(key, headers={'Range': 'bytes=99999-'})).status_code == 416
            assert (await client.get('/healthz')).json()['status'] == 'ok'


def test_http_private_network_access_preflight():
    mosaic = build()
    app = create_app(mosaic)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            preflight = await client.options(
                '/stitched.ome.zarr/zarr.json',
                headers={'Origin': 'https://neuroglancer-demo.appspot.com',
                         'Access-Control-Request-Method': 'GET',
                         'Access-Control-Private-Network-Request': 'true'})
            assert preflight.status_code == 200
            assert preflight.headers['access-control-allow-private-network'] == 'true'
            assert preflight.headers['access-control-allow-origin'] == '*'
            chunk = await client.get('/stitched.ome.zarr/0/c/0/0/0/0/0',
                                     headers={'Origin': 'https://neuroglancer-demo.appspot.com',
                                              'Access-Control-Private-Network-Request': 'true'})
            assert chunk.status_code == 200
            assert chunk.headers['access-control-allow-private-network'] == 'true'

    asyncio.run(run())


def test_http_multichannel_mounting():
    m488 = build(channel='Ch488')
    m561 = build(channel='Ch561')
    app = create_app([m488, m561])

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            for ch in ('Ch488', 'Ch561'):
                meta = await client.get(f'/stitched.ome.zarr/{ch}/zarr.json')
                assert meta.status_code == 200
                assert meta.json()['attributes']['ome']['version'] == '0.5'
                chunk = await client.get(f'/stitched.ome.zarr/{ch}/0/c/0/0/0/0/0')
                assert chunk.status_code == 200
            # The parent namespace is not served when channels are mounted.
            assert (await client.get('/stitched.ome.zarr/zarr.json')).status_code == 404
            assert (await client.get('/stitched.ome.zarr/Ch488/zarr.json')).status_code == 200
            health = (await client.get('/healthz')).json()
            assert health['status'] == 'ok'
            assert [m['channel'] for m in health['mosaics']] == ['Ch488', 'Ch561']
            assert len((await client.get('/info')).json()) == 2
            preflight = await client.options(
                '/stitched.ome.zarr/Ch488/zarr.json',
                headers={'Origin': 'https://neuroglancer-demo.appspot.com',
                         'Access-Control-Request-Method': 'GET',
                         'Access-Control-Private-Network-Request': 'true'})
            assert preflight.status_code == 200
            assert preflight.headers['access-control-allow-private-network'] == 'true'

    asyncio.run(run())

    asyncio.run(run())
