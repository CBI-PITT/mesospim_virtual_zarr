"""Exercise stitched block export without needing proprietary writer libraries."""
import argparse
from collections import namedtuple
import json

import numpy as np
import pytest

from conftest import FakeArray, build, make_groups
from virtual_mesospim import export_ims


class FakeWriter:
    ImageSize = namedtuple("ImageSize", "x y z c t")
    ImageExtents = namedtuple("ImageExtents", "minx miny minz maxx maxy maxz")
    Color = namedtuple("Color", "r g b a")
    eCompressionAlgorithmGzipLevel2 = 2
    last = None

    class DimensionSequence:
        def __init__(self, *axes):
            self.axes = axes

    class CallbackClass:
        pass

    class Options:
        pass

    class Parameters:
        def __init__(self):
            self.names = {}

        def set_channel_name(self, index, name):
            self.names[index] = name

    class ColorInfo:
        def set_base_color(self, color):
            self.color = color

    class ImageConverter:
        def __init__(self, dtype, image_size, sample_size, sequence, block_size,
                     path, options, *args):
            assert dtype == "uint16"
            assert sequence.axes == ("x", "y", "z", "c", "t")
            self.image_size = image_size
            self.block_size = block_size
            self.path = path
            self.options = options
            self.blocks = {}
            self.destroyed = False
            FakeWriter.last = self

        def NeedCopyBlock(self, index):
            return True

        def CopyBlock(self, block, index):
            assert block.shape == (self.block_size.z, self.block_size.y, self.block_size.x)
            assert block.dtype == np.dtype("uint16") and block.flags.c_contiguous
            assert index not in self.blocks
            self.blocks[index] = block.copy()

        def Finish(self, extents, parameters, dates, colors, pyramid):
            self.extents = extents
            self.parameters = parameters
            self.colors = colors
            self.pyramid = pyramid
            assert len(dates) == 1
            from pathlib import Path
            Path(self.path).write_bytes(b"completed IMS")

        def Destroy(self):
            self.destroyed = True


@pytest.fixture
def fake_writer(monkeypatch):
    FakeWriter.last = None
    monkeypatch.setattr(export_ims, "_load_writer", lambda: FakeWriter)
    return FakeWriter


def _block(writer, c=0, z=0, y=0, x=0):
    return writer.blocks[FakeWriter.ImageSize(x=x, y=y, z=z, c=c, t=0)]


def test_manifest_exports_one_ims_with_blended_channels_and_edge_padding(tmp_path, fake_writer):
    zarr = pytest.importorskip("zarr")
    from numcodecs import Blosc

    channel_entries = []
    for name, color, values in (
        ("Ch488", "green", (10, 20, 30, 40)),
        ("Ch561", "#ff0000", (100, 200, 300, 400)),
    ):
        paths = []
        for i, value in enumerate(values):
            path = tmp_path / f"Tile{i}_{name}.ome.zarr"
            data = np.full((1, 1, 2, 4, 4), value, dtype="uint16")
            compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
            if zarr.__version__.split(".", 1)[0] == "2":
                root = zarr.open_group(str(path), mode="w")
                root.create_dataset("0", data=data, chunks=(1, 1, 2, 2, 2),
                                    compressor=compressor)
            else:
                root = zarr.open_group(str(path), mode="w", zarr_format=2)
                source = root.create_array("0", shape=data.shape, dtype="uint16",
                                           chunks=(1, 1, 2, 2, 2), compressor=compressor)
                source[:] = data
            root.attrs["multiscales"] = [{
                "version": "0.4",
                "axes": [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"}]
                + [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
                "datasets": [{"path": "0", "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1, 2, 1.5, 1]}
                ]}],
            }]
            paths.append(str(path))
        channel_entries.append({"name": name, "color": color, "tiles": paths})

    manifest = tmp_path / "mosaic.json"
    manifest.write_text(json.dumps({
        "channels": channel_entries, "rows": 2, "cols": 2,
        "positions_zyx": [[0, 0, 0], [0, 3, 0], [0, 0, 3], [0, 3, 3]],
        "overlap_yx": [0.25, 0.25], "chunk_shape_zyx": [2, 4, 4], "blend": "feather",
    }), encoding="utf-8")
    output = tmp_path / "result.ims"
    result = export_ims.export_manifest_to_ims(
        manifest, output, reader_threads=2, writer_threads=3, max_prefetched_blocks=2,
    )

    writer = fake_writer.last
    assert result == output and output.read_bytes() == b"completed IMS"
    assert not (tmp_path / "result.ims.part").exists()
    assert writer.destroyed and writer.pyramid is True
    assert writer.image_size == FakeWriter.ImageSize(x=7, y=7, z=2, c=2, t=1)
    assert writer.options.mNumberOfThreads == 3
    assert writer.extents == FakeWriter.ImageExtents(0, 0, 0, 7, 10.5, 4)
    assert writer.parameters.names == {0: "Ch488", 1: "Ch561"}
    assert writer.colors[0].color == FakeWriter.Color(0, 1, 0, 1)
    assert writer.colors[1].color == FakeWriter.Color(1, 0, 0, 1)
    assert len(writer.blocks) == 8
    assert _block(writer)[0, 3, 3] == 25
    assert _block(writer, c=1)[0, 3, 3] == 250
    assert _block(writer, y=1, x=1)[0, 2, 2] == 40
    assert np.all(_block(writer, y=1, x=1)[:, 3, :] == 0)
    assert np.all(_block(writer, y=1, x=1)[:, :, 3] == 0)


def test_time_level_and_source_c_axis(tmp_path, fake_writer):
    groups = make_groups(tile_values=(10,), shape=(2, 2, 4, 8, 8), multiscale=True,
                         scales=[1, 1, 2, 1.5, 1])
    groups[0].arrays["1"].data[1, 0] = 30
    groups[0].arrays["1"].data[1, 1] = 40
    mosaic = build(groups, rows=1, cols=1, blend="last")
    export_ims.export_mosaics_to_ims(
        [mosaic], tmp_path / "coarse.ims", level=1, time_index=1,
        channel_names=["GFP", "RFP"], channel_colors=[(0, 1, 0), (1, 0, 0)],
        reader_threads=1, max_prefetched_blocks=1,
    )
    writer = fake_writer.last
    assert writer.image_size == FakeWriter.ImageSize(x=4, y=4, z=2, c=2, t=1)
    assert writer.extents == FakeWriter.ImageExtents(0, 0, 0, 8, 12, 8)
    assert writer.parameters.names == {0: "GFP", 1: "RFP"}
    assert np.all(_block(writer, c=0) == 30)
    assert np.all(_block(writer, c=1) == 40)


def test_negative_z_shift_and_gap_are_written_to_correct_blocks(tmp_path, fake_writer):
    mosaic = build(make_groups(tile_values=(10, 20), shape=(2, 4, 4)),
                   rows=2, cols=1, positions_zyx=((-1, 0, 0), (2, 0, 0)))
    export_ims.export_mosaics_to_ims([mosaic], tmp_path / "shifted.ims")
    writer = fake_writer.last
    assert writer.image_size == FakeWriter.ImageSize(x=4, y=4, z=5, c=1, t=1)
    assert np.all(_block(writer, z=0) == 10)
    assert np.all(_block(writer, z=1)[0] == 0)
    assert np.all(_block(writer, z=1)[1] == 20)
    assert np.all(_block(writer, z=2)[0] == 20)
    assert np.all(_block(writer, z=2)[1] == 0)


def test_big_endian_uint16_is_converted_for_writer(tmp_path, fake_writer):
    groups = make_groups(tile_values=(10,), shape=(2, 4, 4))
    groups[0].arrays["0"] = FakeArray(np.full((2, 4, 4), 1000, dtype=">u2"))
    mosaic = build(groups, rows=1, cols=1)
    export_ims.export_mosaics_to_ims([mosaic], tmp_path / "endian.ims")
    assert np.all(_block(fake_writer.last) == 1000)


@pytest.mark.parametrize("shape", [(2, 4, 4), (1, 2, 4, 4)])
def test_export_without_time_or_channel_axis(tmp_path, fake_writer, shape):
    groups = make_groups(tile_values=(17,), shape=shape)
    mosaic = build(groups, rows=1, cols=1)
    export_ims.export_mosaics_to_ims([mosaic], tmp_path / "one.ims")
    assert fake_writer.last.image_size.c == 1
    assert np.all(_block(fake_writer.last) == 17)
    with pytest.raises(ValueError, match="no time axis"):
        export_ims.export_mosaics_to_ims([mosaic], tmp_path / "other.ims", time_index=1)


def test_rejects_incompatible_channels_and_existing_output(tmp_path, fake_writer):
    mosaic = build(make_groups(tile_values=(10,)), rows=1, cols=1)
    other = build(make_groups(tile_values=(10,), shape=(1, 1, 2, 8, 8)),
                  rows=1, cols=1)
    with pytest.raises(ValueError, match="matching dtype, spatial shape"):
        export_ims.export_mosaics_to_ims([mosaic, other], tmp_path / "bad.ims")
    with pytest.raises(ValueError, match="one nonempty name"):
        export_ims.export_mosaics_to_ims([mosaic], tmp_path / "bad.ims", channel_names=["a", "b"])
    output = tmp_path / "exists.ims"
    output.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        export_ims.export_mosaics_to_ims([mosaic], output)
    assert output.read_bytes() == b"do not overwrite"


def test_incomplete_output_from_previous_run_is_replaced(tmp_path, fake_writer):
    mosaic = build(make_groups(tile_values=(10,)), rows=1, cols=1)
    partial = tmp_path / "retry.ims.part"
    partial.write_bytes(b"incomplete IMS")

    output = export_ims.export_mosaics_to_ims([mosaic], tmp_path / "retry.ims")

    assert output.read_bytes() == b"completed IMS"
    assert not partial.exists()


def test_failed_copy_never_publishes_final_file(tmp_path, fake_writer, monkeypatch):
    def fail_copy(self, block, index):
        from pathlib import Path
        Path(self.path).write_bytes(b"incomplete IMS")
        raise RuntimeError("writer failed")

    monkeypatch.setattr(FakeWriter.ImageConverter, "CopyBlock", fail_copy)
    mosaic = build(make_groups(tile_values=(10,)), rows=1, cols=1)
    output = tmp_path / "failed.ims"
    with pytest.raises(RuntimeError, match="writer failed"):
        export_ims.export_mosaics_to_ims([mosaic], output, reader_threads=1)
    assert fake_writer.last.destroyed
    assert not output.exists()
    assert not (tmp_path / "failed.ims.part").exists()


def test_color_parser():
    assert export_ims._parse_color("0.1,0.5,1") == (0.1, 0.5, 1.0)
    for color in ("1,0", "0,1,nan", "1.2,0,0"):
        with pytest.raises(argparse.ArgumentTypeError):
            export_ims._parse_color(color)
