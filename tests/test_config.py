import json
from pathlib import Path

import pytest

from virtual_mesospim.config import MosaicConfig, load_mosaic_configs


def write_manifest(tmp_path, cfg):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_single_channel_manifest(tmp_path):
    tiles = [f"/data/Tile{i}_Ch488.ome.zarr" for i in range(4)]
    configs = load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))
    assert len(configs) == 1
    assert configs[0].channel is None
    assert configs[0].color is None
    assert configs[0].name == "MesoSPIM virtual mosaic"
    assert [str(p) for p in configs[0].tiles] == [str(Path(t)) for t in tiles]


def test_relative_tiles_resolved_against_manifest(tmp_path):
    tiles = ["Tile0.ome.zarr", "Tile1.ome.zarr"]
    configs = load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 1, "cols": 2}))
    assert configs[0].tiles[0] == (tmp_path / "Tile0.ome.zarr").resolve()


def test_multichannel_expansion(tmp_path):
    cfg = {
        "channels": [
            {"name": "Ch488", "color": "green", "tiles": [f"/d/Tile{i}_Ch488.ome.zarr" for i in range(4)]},
            {"name": "Ch561", "color": "#ff0000", "tiles": [f"/d/Tile{i}_Ch561.ome.zarr" for i in range(4)]},
        ],
        "rows": 2, "cols": 2, "overlap_yx": [0.035, 0.04],
        "positions_zyx": [[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "name": "My mosaic", "mount_path": "stitched.ome.zarr",
    }
    configs = load_mosaic_configs(write_manifest(tmp_path, cfg))
    assert [c.channel for c in configs] == ["Ch488", "Ch561"]
    assert [c.color for c in configs] == ["green", "#ff0000"]
    assert [c.name for c in configs] == ["My mosaic Ch488", "My mosaic Ch561"]
    assert configs[0].overlap_yx == (0.035, 0.04)
    assert configs[0].positions_zyx[1] == (1, 2, 3)
    assert configs[1].positions_zyx[3] == (7, 8, 9)
    assert all(len(c.tiles) == 4 for c in configs)
    assert configs[0].mount_path == "stitched.ome.zarr"


def test_multichannel_without_top_level_name(tmp_path):
    cfg = {
        "channels": [
            {"name": "Ch488", "tiles": [f"/d/Tile{i}.ome.zarr" for i in range(4)]},
            {"name": "Ch561", "tiles": [f"/d/Tile{i}.ome.zarr" for i in range(4)]},
        ],
        "rows": 2, "cols": 2,
    }
    configs = load_mosaic_configs(write_manifest(tmp_path, cfg))
    assert [c.name for c in configs] == ["Ch488", "Ch561"]
    assert [c.color for c in configs] == [None, None]


def test_rejects_tiles_and_channels_together(tmp_path):
    cfg = {"tiles": ["/d/a.ome.zarr"] * 4, "rows": 2, "cols": 2,
           "channels": [{"name": "c", "tiles": ["/d/a.ome.zarr"] * 4}]}
    with pytest.raises(ValueError, match="not both"):
        load_mosaic_configs(write_manifest(tmp_path, cfg))


def test_rejects_duplicate_channel_names(tmp_path):
    cfg = {"channels": [{"name": "c", "tiles": ["/d/a.ome.zarr"] * 4}] * 2, "rows": 2, "cols": 2}
    with pytest.raises(ValueError, match="unique"):
        load_mosaic_configs(write_manifest(tmp_path, cfg))


def test_rejects_invalid_channel_name(tmp_path):
    cfg = {"channels": [{"name": "a/b", "tiles": ["/d/a.ome.zarr"] * 4}], "rows": 2, "cols": 2}
    with pytest.raises(ValueError, match="channel name"):
        load_mosaic_configs(write_manifest(tmp_path, cfg))


def test_rejects_invalid_color(tmp_path):
    cfg = {"channels": [{"name": "c", "color": "notacolor", "tiles": ["/d/a.ome.zarr"] * 4}],
           "rows": 2, "cols": 2}
    with pytest.raises(ValueError, match="color"):
        load_mosaic_configs(write_manifest(tmp_path, cfg))


def test_rejects_mismatched_tile_counts(tmp_path):
    cfg = {"channels": [{"name": "a", "tiles": ["/d/a.ome.zarr"] * 4},
                        {"name": "b", "tiles": ["/d/a.ome.zarr"] * 3}],
           "rows": 2, "cols": 2}
    with pytest.raises(ValueError, match="Expected 4 tiles, got 3"):
        load_mosaic_configs(write_manifest(tmp_path, cfg))


def test_from_json_raises_on_multichannel(tmp_path):
    cfg = {"channels": [{"name": "a", "tiles": ["/d/a.ome.zarr"] * 4},
                        {"name": "b", "tiles": ["/d/b.ome.zarr"] * 4}],
           "rows": 2, "cols": 2}
    with pytest.raises(ValueError, match="load_mosaic_configs"):
        MosaicConfig.from_json(write_manifest(tmp_path, cfg))


def write_gain_file(tmp_path, data):
    path = tmp_path / "gain.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_gain_json_auto_discovery_single_channel(tmp_path):
    tiles = [f"/store/Tile{i}.ome.zarr" for i in range(4)]
    gains = [1.0, 1.33, 0.5, 2.0]
    write_gain_file(tmp_path, {"acquisition": "store",
                               "channels": [{"name": "Ch488", "gains": gains}]})
    configs = load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))
    assert configs[0].gains == (1.0, 1.33, 0.5, 2.0)


def test_gain_json_auto_discovery_multichannel(tmp_path):
    cfg = {"channels": [{"name": "Ch488", "tiles": ["/d/Tile0.ome.zarr"] * 4},
                        {"name": "Ch561", "tiles": ["/d/Tile0.ome.zarr"] * 4}],
           "rows": 2, "cols": 2}
    write_gain_file(tmp_path, {"acquisition": "d",
                               "channels": [{"name": "Ch488", "gains": [1.0] * 4},
                                            {"name": "Ch561", "gains": [1.33] * 4}]})
    configs = load_mosaic_configs(write_manifest(tmp_path, cfg))
    assert configs[0].gains == (1.0, 1.0, 1.0, 1.0)
    assert configs[1].gains == (1.33,) * 4


def test_no_gain_json_means_no_gains(tmp_path):
    tiles = [f"/data/Tile{i}.ome.zarr" for i in range(4)]
    configs = load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))
    assert configs[0].gains is None


def test_gain_json_wrong_count_rejected(tmp_path):
    tiles = [f"/data/Tile{i}.ome.zarr" for i in range(4)]
    write_gain_file(tmp_path, {"channels": [{"name": "Ch488", "gains": [1.0] * 3}]})
    with pytest.raises(ValueError, match="expected 4 gains"):
        load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))


def test_gain_json_out_of_range_rejected(tmp_path):
    tiles = [f"/data/Tile{i}.ome.zarr" for i in range(4)]
    write_gain_file(tmp_path, {"channels": [{"name": "Ch488", "gains": [200.0] + [1.0] * 3}]})
    with pytest.raises(ValueError, match=r"\[0\.01, 100"):
        load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))


def test_gain_json_multichannel_missing_channel_rejected(tmp_path):
    cfg = {"channels": [{"name": "Ch488", "tiles": ["/d/Tile0.ome.zarr"] * 4},
                        {"name": "Ch561", "tiles": ["/d/Tile0.ome.zarr"] * 4}],
           "rows": 2, "cols": 2}
    write_gain_file(tmp_path, {"channels": [{"name": "Ch488", "gains": [1.0] * 4}]})
    with pytest.raises(ValueError, match="no gains for channel 'Ch561'"):
        load_mosaic_configs(write_manifest(tmp_path, cfg))


def test_gain_json_multiple_entries_single_channel_rejected(tmp_path):
    tiles = [f"/data/Tile{i}.ome.zarr" for i in range(4)]
    write_gain_file(tmp_path, {"channels": [{"name": "Ch488", "gains": [1.0] * 4},
                                            {"name": "Ch561", "gains": [1.0] * 4}]})
    with pytest.raises(ValueError, match="exactly one channels entry"):
        load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))


def test_gain_json_acquisition_mismatch_rejected(tmp_path):
    tiles = [f"/store/Tile{i}.ome.zarr" for i in range(4)]
    write_gain_file(tmp_path, {"acquisition": "other_store",
                               "channels": [{"name": "Ch488", "gains": [1.0] * 4}]})
    with pytest.raises(ValueError, match="does not match tile store"):
        load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))


def test_gain_json_nested_acquisition_path(tmp_path):
    tiles = [f"acq/scan.ome.zarr/Tile{i}.ome.zarr" for i in range(4)]
    write_gain_file(tmp_path, {"acquisition": "acq/scan.ome.zarr",
                               "channels": [{"name": "Ch488", "gains": [1.0] * 4}]})
    configs = load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))
    assert configs[0].gains == (1.0,) * 4


def test_gain_json_invalid_json_rejected(tmp_path):
    tiles = [f"/data/Tile{i}.ome.zarr" for i in range(4)]
    (tmp_path / "gain.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read"):
        load_mosaic_configs(write_manifest(tmp_path, {"tiles": tiles, "rows": 2, "cols": 2}))
