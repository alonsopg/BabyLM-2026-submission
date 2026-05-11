from src.masking.sketches import SpaceSavingSketch


def test_space_saving_tracks_heavy_hitters():
    sketch = SpaceSavingSketch(capacity=3)
    for key in ["a", "a", "a", "b", "b", "c", "d", "a"]:
        sketch.update(key)
    top = dict(sketch.topk(2))
    assert "a" in top
    assert sketch.contains("a")
    assert sketch.estimate("a") >= 4
