"""Zarr-Python 3's read-only async Store interface for the virtual mosaic.

Open with:

    store = VirtualZarrV3Store(mosaic)
    arr = zarr.open_array(store=store, path="0", mode="r", zarr_format=3)

For Neuroglancer, serve the same mosaic via virtual_mesospim.http.create_app.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from zarr.abc.store import Store, ByteRequest, OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype

from .engine import VirtualMosaic


def _range_bytes(data: bytes, byte_range: ByteRequest | None) -> bytes:
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start:byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset:]
    if isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix:]
    raise TypeError(f"Unknown Zarr byte range: {byte_range!r}")


class VirtualZarrV3Store(Store):
    """Read-only, non-listable virtual Zarr v3 store.

    Chunk keys are infinite in principle but bounded by each array's shape;
    enumerating all chunk keys would be prohibitively expensive. Zarr v3
    clients can open individual groups/arrays from their metadata without listing.
    """

    def __init__(self, mosaic: VirtualMosaic):
        super().__init__(read_only=True)
        self.mosaic = mosaic

    def __eq__(self, value: object) -> bool:
        return isinstance(value, VirtualZarrV3Store) and self.mosaic is value.mosaic

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return False

    @property
    def supports_consolidated_metadata(self) -> bool:
        return False

    def with_read_only(self, read_only: bool = True) -> "VirtualZarrV3Store":
        if not read_only:
            raise ValueError("VirtualZarrV3Store cannot be made writable")
        return VirtualZarrV3Store(self.mosaic)

    async def get(self, key: str, prototype: BufferPrototype,
                  byte_range: ByteRequest | None = None) -> Buffer | None:
        data = await asyncio.to_thread(self.mosaic.get_bytes, key)
        if data is None:
            return None
        return prototype.buffer.from_bytes(_range_bytes(data, byte_range))

    async def get_partial_values(self, prototype: BufferPrototype,
                                 key_ranges: Iterable[tuple[str, ByteRequest | None]]
                                 ) -> list[Buffer | None]:
        return await asyncio.gather(*(
            self.get(key, prototype=prototype, byte_range=byte_range)
            for key, byte_range in key_ranges
        ))

    async def exists(self, key: str) -> bool:
        return self.mosaic.has_key(key)

    async def set(self, key: str, value: Buffer) -> None:
        raise PermissionError("VirtualZarrV3Store is read-only")

    async def delete(self, key: str) -> None:
        raise PermissionError("VirtualZarrV3Store is read-only")

    async def set_partial_values(self, key_start_values) -> None:
        raise PermissionError("VirtualZarrV3Store is read-only")

    async def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("Listing all virtual chunks is intentionally unsupported")
        yield ""  # keep this an async generator

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("Listing all virtual chunks is intentionally unsupported")
        yield ""

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("Listing all virtual chunks is intentionally unsupported")
        yield ""
