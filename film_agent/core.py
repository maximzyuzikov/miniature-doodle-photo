"""Conservative, content-adaptive, non-generative film rendering.

No external AI service is used: analysis is deterministic and never invents image content.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

RAW_EXTENSIONS = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".raf", ".rw2", ".orf", ".pef"}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"}


@dataclass(frozen=True)
class FilmStyle:
    name: str
    description: str
    monochrome: bool
    contrast: float
    shadow_lift: float
    saturation: float
    warmth: float
    green: float
    grain: float
    grain_radius: float
    vignette: float
    bw_mix: tuple[float, float, float] = (0.24, 0.68, 0.08)


STYLES: dict[str, FilmStyle] = {
    "hp5-120": FilmStyle("Ilford HP5+ · 120", "Мягкие полутона, выразительное крупное зерно", True, 1.09, 0.022, 0, 0, 0, 0.056, 1.05, 0.09),
    "trix-120": FilmStyle("Kodak Tri-X · 120 pushed", "Контрастная документальная ч/б плёнка", True, 1.24, 0.007, 0, 0, 0, 0.068, 1.28, 0.15, (0.29, 0.61, 0.10)),
    "portra-400": FilmStyle("Kodak Portra 400 · 35mm", "Мягкий контраст, естественная кожа, деликатное зерно", False, 0.94, 0.018, 0.93, 0.017, -0.003, 0.014, 0.48, 0.06),
    "gold-200": FilmStyle("Kodak Gold 200 · 35mm", "Тёплый дневной цвет, умеренная насыщенность", False, 1.04, 0.018, 1.09, 0.034, -0.008, 0.017, 0.52, 0.09),
    "ektar-100": FilmStyle("Kodak Ektar 100 · 35mm", "Чистые насыщенные цвета и мелкое зерно", False, 1.12, 0.009, 1.16, 0.008, 0.006, 0.008, 0.35, 0.05),
}


def _register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError as exc:
        raise RuntimeError("Для HEIC/HEIF установите: pip install -e '.[heic]'") from exc
    register_heif_opener()


def load_photo(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read into 8-bit sRGB. Intentionally do not carry EXIF/GPS into output."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат: {p.suffix}")
    if suffix in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("Для RAW установите: pip install -e '.[raw]'") from exc
        try:
            with rawpy.imread(str(p)) as raw:
                iso = getattr(raw.metadata, "iso_speed", None)
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
        except Exception as exc:
            raise ValueError(f"Не удалось декодировать RAW: {p.name}: {exc}") from exc
        return rgb, {"iso": int(iso) if iso and iso > 0 else None, "format": "RAW", "raw": True}
    if suffix in {".heic", ".heif"}:
        _register_heif()
    with Image.open(p) as original:
        exif = original.getexif()
        iso = exif.get(34855) if exif else None
        img = ImageOps.exif_transpose(original)
        # Honor embedded camera/display profile before tonal operations.
        icc = img.info.get("icc_profile")
        if icc:
            try:
                src = ImageCms.ImageCmsProfile(BytesIO(icc))
                dst = ImageCms.createProfile("sRGB")
                img = ImageCms.profileToProfile(img.convert("RGB"), src, dst, outputMode="RGB")
            except (ValueError, OSError, ImageCms.PyCMSError):
                img = img.convert("RGB")
        else:
            img = img.convert("RGB")
        result = np.asarray(img, dtype=np.uint8).copy()
    try:
        iso = int(iso) if iso else None
    except (ValueError, TypeError, OverflowError):
        iso = None
    return result, {"iso": iso, "format": suffix.lstrip(".").upper(), "raw": False}


def analyze_photo(rgb: np.ndarray, iso: int | None = None) -> dict[str, Any]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Ожидается RGB uint8 с тремя каналами")
    if not min(rgb.shape[:2]) > 0:
        raise ValueError("Пустое изображение")
    # Analysis uses a small copy regardless of original resolution.
    h, w = rgb.shape[:2]
    ratio = min(1.0, 640 / max(h, w))
    small = cv2.resize(rgb, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA) if ratio < 1 else rgb
    pixels = small.astype(np.float32) / 255.0
    luminance = np.sum(pixels * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=2)
    p02, p50, p98 = np.percentile(luminance, [2, 50, 98])
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    neutral = (saturation < 0.13) & (luminance > 0.20) & (luminance < 0.86)
    # Only neutralish midtones provide weak WB evidence. Preserve intentional warm light.
    if neutral.mean() >= 0.045:
        means = pixels[neutral].mean(axis=0)
        target = float(means.mean())
        wb_gains = np.clip((target / np.maximum(means, 0.035)) ** 0.38, 0.89, 1.11)
    else:
        wb_gains = np.ones(3, dtype=np.float32)
    ev = float(np.clip(np.log2(0.40 / max(float(p50), 0.035)) * 0.58, -0.70, 0.90))
    if p98 > 0.93:
        ev = min(ev, 0.15)  # Avoid pushing large existing highlights into clipping.
    elif p98 > 0.84:
        ev = min(ev, 0.35)
    # Flat-region noise proxy: robust residual after a small median filter.
    gray8 = np.uint8(np.clip(luminance * 255, 0, 255))
    residual = np.abs(gray8.astype(np.float32) - cv2.medianBlur(gray8, 3).astype(np.float32))
    flat = cv2.Laplacian(gray8, cv2.CV_32F, ksize=3)
    region = (np.abs(flat) < 13) & (luminance > 0.10) & (luminance < 0.90)
    noise = float(np.median(residual[region])) if np.any(region) else 0.0
    iso = int(iso) if iso is not None and int(iso) > 0 else None
    # Very conservative cleaning: unknown ISO should not trigger aggressive smoothing.
    denoise = (0.18 if iso and iso >= 6400 else 0.12 if iso and iso >= 1600 else 0.06 if iso and iso >= 800 else 0.0)
    denoise = min(0.26, denoise + (0.04 if noise > 4.0 else 0.0))
    return {
        "dimensions": [w, h], "iso": iso,
        "luminance_p02": round(float(p02), 4), "luminance_p50": round(float(p50), 4),
        "luminance_p98": round(float(p98), 4),
        "exposure_ev": round(ev, 3),
        "wb_gains": [round(float(v), 4) for v in wb_gains],
        "noise_proxy": round(noise, 3), "denoise": round(denoise, 3),
    }


def _curve(rgb: np.ndarray, contrast: float, lift: float) -> np.ndarray:
    # Smooth pivoted contrast with a toe and shoulder that do not invert tonal order.
    v = np.clip(rgb, 0.0, 1.0)
    pivot = np.clip((v - 0.5) * contrast + 0.5, 0, 1)
    smooth = pivot * pivot * (3.0 - 2.0 * pivot)
    v = 0.78 * pivot + 0.22 * smooth
    return v + lift * (1.0 - v) ** 2


def _grain(h: int, w: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((h, w), dtype=np.float32)
    if radius > 0.42:
        raw = cv2.GaussianBlur(raw, (0, 0), radius, borderType=cv2.BORDER_REFLECT)
        raw /= max(float(raw.std()), 1e-5)
    return np.clip(raw, -3.0, 3.0)


def process_array(rgb: np.ndarray, style: str = "hp5-120", *, strength: float = 1.0,
                  grain: float = 1.0, seed: int = 42, iso: int | None = None,
                  auto: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a reproducible film interpretation, keeping original geometry."""
    if style not in STYLES:
        raise ValueError(f"Неизвестный стиль: {style}. Варианты: {', '.join(STYLES)}")
    if not (0.0 <= strength <= 2.0 and 0.0 <= grain <= 3.0):
        raise ValueError("strength должен быть 0..2, grain — 0..3")
    look = STYLES[style]
    info = analyze_photo(rgb, iso)
    x = rgb.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    if auto:
        wb = np.asarray(info["wb_gains"], dtype=np.float32)
        if not look.monochrome:
            x *= wb.reshape(1, 1, 3)
        exposure = 2.0 ** (-info["exposure_ev"] * 0.77)
        x = np.power(np.clip(x, 0, 1), exposure)
        if info["denoise"] > 0:
            # Blending a bilateral result preserves detail and avoids over-smoothing.
            img8 = np.uint8(np.round(np.clip(x, 0, 1) * 255))
            filtered = cv2.bilateralFilter(img8, d=5, sigmaColor=28, sigmaSpace=3)
            alpha = min(0.55, info["denoise"] * 2)
            x = (1 - alpha) * x + alpha * (filtered.astype(np.float32) / 255.0)
    intensity = float(strength)
    if look.monochrome:
        weights = np.asarray(look.bw_mix, dtype=np.float32)
        bw = np.sum(x * weights.reshape(1, 1, 3), axis=2, keepdims=True)
        bw = _curve(bw, 1 + (look.contrast - 1) * intensity, look.shadow_lift * intensity)
        x = np.repeat(bw, 3, axis=2)
    else:
        lum = np.sum(x * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=2, keepdims=True)
        x = lum + (x - lum) * (1 + (look.saturation - 1) * intensity)
        x = _curve(x, 1 + (look.contrast - 1) * intensity, look.shadow_lift * intensity)
        # Delicate warm/cool split: bright areas warmer than shadows.
        lum = np.mean(x, axis=2, keepdims=True)
        wshift = look.warmth * intensity * (0.6 + 0.4 * lum)
        x += np.concatenate((wshift, np.zeros_like(wshift), -wshift * 0.78), axis=2)
        x[:, :, 1] += look.green * intensity
    if look.vignette > 0 and intensity:
        yy, xx = np.ogrid[-1:1:complex(h), -1:1:complex(w)]
        vignette = np.maximum(0, (xx * xx + yy * yy) / 2)
        x *= (1 - look.vignette * min(intensity, 1.8) * vignette)[:, :, None]
    if grain > 0 and intensity:
        pattern = _grain(h, w, look.grain_radius, np.random.default_rng(seed))
        luma = np.mean(np.clip(x, 0, 1), axis=2)
        scale = look.grain * intensity * grain * (0.65 + 0.65 * (1 - luma))
        x += (pattern * scale)[:, :, None]
    out = np.uint8(np.round(np.clip(x, 0, 1) * 255))
    report = {
        "style": style, "style_name": look.name, "strength": strength, "grain": grain,
        "seed": seed, "auto": auto, "analysis": info,
        "output_clipped_black_fraction": round(float(np.mean(np.max(out, axis=2) == 0)), 4),
        "output_clipped_white_fraction": round(float(np.mean(np.min(out, axis=2) == 255)), 4),
        "note": "Художественная интерпретация, не измеренная спектральная эмуляция фотоплёнки.",
    }
    if report["output_clipped_white_fraction"] > 0.03:
        report["warning"] = "Заметны области белого клиппинга; проверьте входной файл и экспозицию."
    return out, report


def save_photo(rgb: np.ndarray, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb, "RGB")
    ext = target.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.save(target, quality=95, subsampling=0, optimize=True)
    elif ext in {".png", ".tif", ".tiff", ".webp"}:
        image.save(target)
    else:
        raise ValueError("Выход: .jpg, .png, .tif, .tiff или .webp")


def render_photo(input_path: str | Path, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    rgb, source = load_photo(input_path)
    if "iso" not in kwargs or kwargs["iso"] is None:
        kwargs["iso"] = source["iso"]
    processed, report = process_array(rgb, **kwargs)
    save_photo(processed, output_path)
    report["source"] = {"format": source["format"], "raw": source["raw"], "iso": source["iso"]}
    return report
