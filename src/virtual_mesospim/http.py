"""Minimal read-only HTTP key-value gateway for Neuroglancer."""
from __future__ import annotations

import re
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .engine import VirtualMosaic


_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class PrivateNetworkAccessMiddleware:
    """Chrome Private Network Access: public HTTPS origins fetching loopback
    URLs get a CORS preflight carrying Access-Control-Private-Network-Request,
    and are blocked unless the response echoes
    Access-Control-Allow-Private-Network: true. Starlette's CORSMiddleware does
    not send that header, so this outer middleware adds it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                               for k, v in scope.get("headers", [])}
            if request_headers.get("access-control-private-network-request") == "true":
                async def send_wrapper(message):
                    if message["type"] == "http.response.start":
                        message = {**message, "headers": list(message.get("headers", []))
                                   + [(b"access-control-allow-private-network", b"true")]}
                    await send(message)
                await self.app(scope, receive, send_wrapper)
                return
        await self.app(scope, receive, send)


def create_app(mosaic: VirtualMosaic) -> FastAPI:
    """HTTP endpoints implement Zarr v3 keys at /<mount_path>/<key>.

    CORS is intentionally broad so a hosted Neuroglancer can fetch data.
    Do NOT expose this server publicly with private microscopy data unless
    appropriate authentication, HTTPS and CORS access control are configured.
    """
    app = FastAPI(title="Virtual MesoSPIM OME-Zarr 0.5", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["Range", "Content-Type"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )
    # Added last so it wraps (and can augment) CORSMiddleware preflight responses.
    app.add_middleware(PrivateNetworkAccessMiddleware)

    limiter = asyncio.Semaphore(mosaic.config.max_concurrent_chunks)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "levels": len(mosaic.levels)}

    @app.get("/info")
    def info():
        return JSONResponse(mosaic.describe())

    async def serve_key(key: str, request: Request):
        if not mosaic.has_key(key):
            raise HTTPException(status_code=404, detail="No such virtual Zarr key")
        try:
            async with limiter:
                content = await run_in_threadpool(mosaic.get_bytes, key)
        except Exception as exc:
            # Unexpected I/O/codec exceptions should not become 404s.
            raise HTTPException(status_code=500, detail="Source tile read or compositing failed") from exc
        if content is None:
            raise HTTPException(status_code=404, detail="No such virtual Zarr key")
        size = len(content)
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
        status = 200
        range_header = request.headers.get("range")
        if range_header:
            match = _RANGE.fullmatch(range_header)
            if match is None:
                raise HTTPException(status_code=416, detail="Only a single bytes=start-end range is supported",
                                    headers={"Content-Range": f"bytes */{size}"})
            if not match.group(1) and not match.group(2):
                raise HTTPException(status_code=416, detail="Empty byte range",
                                    headers={"Content-Range": f"bytes */{size}"})
            if match.group(1):
                start = int(match.group(1))
                stop = min(int(match.group(2)) + 1, size) if match.group(2) else size
            else:
                # Suffix form: bytes=-N
                start = max(0, size - int(match.group(2)))
                stop = size
            if start >= size or start >= stop:
                raise HTTPException(status_code=416, detail="Range not satisfiable",
                                    headers={"Content-Range": f"bytes */{size}"})
            content = content[start:stop]
            headers["Content-Range"] = f"bytes {start}-{stop-1}/{size}"
            status = 206
        if key.endswith("zarr.json"):
            media_type = "application/json"
        else:
            media_type = "application/octet-stream"
        return Response(content=b"" if request.method == "HEAD" else content,
                        status_code=status, media_type=media_type,
                        headers={**headers, "Content-Length": str(len(content))})

    # Register *only* the Zarr namespace. Arbitrary filesystem paths are never used.
    @app.api_route(f"/{mosaic.config.mount_path}/{{key:path}}", methods=["GET", "HEAD"])
    async def zarr_key(key: str, request: Request):
        return await serve_key(key, request)

    return app
