import json

import numpy as np
import pytest
from PIL import Image

from film_agent import STYLES, analyze_photo, process_array, render_photo
from film_agent.cli import main


def scene() -> np.ndarray:
    # Smooth neutral area, warm light, bright region and textured foreground.
    xx = np.linspace(30, 230, 160, dtype=np.uint8)
    a = np.tile(xx, (100, 1))
    return np.stack([np.clip(a.astype(np.int16) + 12, 0, 255), a,
                     np.clip(a.astype(np.int16) - 15, 0, 255)], axis=2).astype(np.uint8)


@pytest.mark.parametrize("style", list(STYLES))
def test_every_style_has_correct_shape_and_range(style):
    original = scene()
    output, report = process_array(original, style=style, iso=3200)
    assert output.shape == original.shape and output.dtype == np.uint8
    assert output.min() >= 0 and output.max() <= 255
    assert report["analysis"]["iso"] == 3200
    assert not np.array_equal(output, original)


def test_bw_neutral_channels():
    result, _ = process_array(scene(), "trix-120")
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])


def test_seed_reproducibility_and_grain_control():
    base = scene()
    a, _ = process_array(base, "hp5-120", seed=111)
    b, _ = process_array(base, "hp5-120", seed=111)
    c, _ = process_array(base, "hp5-120", seed=222)
    clean, _ = process_array(base, "hp5-120", grain=0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, clean)


def test_adaptation_changes_dark_image_and_does_not_change_input():
    src = np.full((80, 80, 3), 35, dtype=np.uint8)
    snapshot = src.copy()
    automatic, report = process_array(src, "portra-400")
    manual, _ = process_array(src, "portra-400", auto=False)
    assert report["analysis"]["exposure_ev"] > 0
    assert np.mean(automatic) > np.mean(manual)
    assert np.array_equal(src, snapshot)


def test_iso_branches_and_invalid_args():
    assert analyze_photo(scene(), iso=6400)["denoise"] > analyze_photo(scene(), iso=100)["denoise"]
    with pytest.raises(ValueError):
        process_array(scene(), "nonexistent")
    with pytest.raises(ValueError):
        process_array(scene(), "hp5-120", grain=-1)


def test_cli_single_file_and_report(tmp_path):
    src = tmp_path / "input.jpg"
    out = tmp_path / "final.jpg"
    Image.fromarray(scene()).save(src)
    assert main([str(src), "--style", "gold-200", "-o", str(out), "--report"]) == 0
    assert out.exists() and out.with_suffix(".jpg.json").exists()
    report = json.loads(out.with_suffix(".jpg.json").read_text())
    assert report["style"] == "gold-200"
    with Image.open(out) as img:
        assert img.size == (160, 100)
        assert img.getexif().get(34853) is None  # Never copy location EXIF.


def test_cli_folder_keeps_relative_structure(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    (source / "sub").mkdir()
    Image.fromarray(scene()).save(source / "sub" / "a.png")
    assert main([str(source), "-o", str(source / "processed"), "--style", "ektar-100"]) == 0
    assert (source / "processed" / "sub" / "a.jpg").is_file()
    # Output nested in source must not be processed again on subsequent run.
    assert main([str(source), "-o", str(source / "processed")]) == 0


def test_one_file_cannot_overwrite_source(tmp_path):
    src = tmp_path / "photo.jpg"
    Image.fromarray(scene()).save(src)
    assert main([str(src), "-o", str(src)]) == 1
