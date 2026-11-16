import json
from urllib.parse import unquote

from conftest import build

from virtual_mesospim.cli import _neuroglancer_url


def _state(url):
    assert url.startswith("https://neuroglancer-demo.appspot.com/#!")
    return json.loads(unquote(url.split("#!", 1)[1]))


def test_neuroglancer_url_multichannel_layers():
    m488 = build(channel="Ch488", color="green", name="Test Ch488")
    m561 = build(channel="Ch561", color="#ff0000", name="Test Ch561")
    state = _state(_neuroglancer_url([m488, m561], "127.0.0.1", 8080))
    assert [layer["name"] for layer in state["layers"]] == ["Test Ch488", "Test Ch561"]
    assert state["layers"][0]["source"] == "http://127.0.0.1:8080/stitched.ome.zarr/Ch488/|zarr3:"
    assert state["layers"][1]["source"] == "http://127.0.0.1:8080/stitched.ome.zarr/Ch561/|zarr3:"
    assert "emitRGB" in state["layers"][0]["shader"]
    assert "1.0" in state["layers"][0]["shader"]
    assert "emitRGB" in state["layers"][1]["shader"]
    assert "shaderControls" in state["layers"][0]
    assert "shaderControls" in state["layers"][1]
    assert state["selectedLayer"]["layer"] == "Test Ch488"
    assert state["layout"] == "4panel"


def test_neuroglancer_url_single_channel_grayscale():
    mosaic = build()
    state = _state(_neuroglancer_url([mosaic], "127.0.0.1", 8080))
    assert len(state["layers"]) == 1
    assert "shader" not in state["layers"][0]
    assert state["layers"][0]["source"] == "http://127.0.0.1:8080/stitched.ome.zarr/|zarr3:"
    assert "shaderControls" in state["layers"][0]


def test_neuroglancer_url_position_inside_volume():
    mosaic = build()
    state = _state(_neuroglancer_url([mosaic], "127.0.0.1", 8080))
    z, y, x = state["position"]
    shape = mosaic.levels[0].shape[-3:]
    assert 0 <= z <= shape[0] and 0 <= y <= shape[1] and 0 <= x <= shape[2]
