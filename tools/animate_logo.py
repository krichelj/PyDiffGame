"""Generate an animated GIF of the PyDiffGame logo.

A bright specular sweep slides across the mark and wordmark a few times,
damping into the v8 static "settled" styling on the final frame.  Reuses
the layer helpers from ``render_logo.py`` so the animated look stays
visually consistent with the static logo.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from tools.render_logo import (
    ROOT,
    SRC,
    _gradient_fill,
    _inner_highlight,
    _outer_glow,
    _split_mark_and_text,
    _stroke,
)


# All styling defaults below match the v8 static render so the animation
# settles into the exact look the README already ships.
STYLE = {
    "mark_top": (255, 95, 105),
    "mark_bottom": (170, 10, 25),
    "text_top": (8, 8, 12),
    "text_bottom": (0, 0, 5),
    "mark_fill_alpha": 230,
    "text_fill_alpha": 252,
    "mark_glow_alpha": 140,
    "text_glow_alpha": 80,
    "mark_glow_radius": 5.5,
    "text_glow_radius": 2.8,
    "highlight_alpha": 235,
    "stroke_alpha": 200,
    "settle_mark_gloss_alpha": 90,
    "settle_text_gloss_alpha": 70,
    "gloss_feather": 3.5,
}


def _sweep_gloss(
    mask: Image.Image,
    peak_y: float,
    band_width: float,
    alpha: int,
    feather: float = 3.5,
) -> Image.Image:
    """Symmetric specular band centred at ``peak_y`` (rather than the v8
    asymmetric top-bright falloff).  Animating ``peak_y`` makes a bright
    streak that slides across the shape.  At ``peak_y == bbox_top`` and
    ``band_width == 0.5 * bbox_height`` this collapses to the v8 gloss
    inside the visible mask, so the animation can settle into the static
    look without a discontinuous handoff.
    """
    a = np.asarray(mask, dtype=np.uint8)[..., 3]
    if a.sum() == 0 or band_width <= 0 or alpha <= 0:
        return Image.new("RGBA", mask.size, (0, 0, 0, 0))
    ys = np.arange(a.shape[0], dtype=np.float32)
    weight = np.clip(1.0 - np.abs(ys - peak_y) / band_width, 0.0, 1.0)
    falloff = np.broadcast_to(weight[:, None], a.shape)
    a_norm = a.astype(np.float32) / 255.0
    out_alpha = (a_norm * falloff * (alpha / 255.0) * 255.0).astype(np.uint8)
    out = np.full((*a.shape, 4), 255, dtype=np.uint8)
    out[..., 3] = out_alpha
    img = Image.fromarray(out, "RGBA")
    if feather > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=feather))
    return img


def _render_frame(
    src: Image.Image,
    mark_mask: Image.Image,
    text_mask: Image.Image,
    *,
    mark_peak_y: float,
    mark_gloss_alpha: int,
    mark_band_width: float,
    text_peak_y: float,
    text_gloss_alpha: int,
    text_band_width: float,
    background: str,
) -> Image.Image:
    w, h = src.size
    if background == "dark":
        canvas = Image.new("RGBA", (w, h), (8, 14, 28, 255))
    elif background == "light":
        canvas = Image.new("RGBA", (w, h), (245, 248, 255, 255))
    else:
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    s = STYLE
    # Mark
    mark_glow = _outer_glow(mark_mask, s["mark_bottom"], s["mark_glow_radius"], s["mark_glow_alpha"])
    mark_fill = _gradient_fill(mark_mask, s["mark_top"], s["mark_bottom"], s["mark_fill_alpha"])
    mark_gloss = _sweep_gloss(mark_mask, mark_peak_y, mark_band_width, mark_gloss_alpha, s["gloss_feather"])
    mark_stroke = _stroke(mark_mask, s["mark_bottom"], s["stroke_alpha"], width=1)
    mark_hi = _inner_highlight(mark_mask, offset=3, alpha=s["highlight_alpha"])

    canvas.alpha_composite(mark_glow)
    canvas.alpha_composite(mark_fill)
    canvas.alpha_composite(mark_gloss)
    canvas.alpha_composite(mark_stroke)
    canvas.alpha_composite(mark_hi)

    # Wordmark
    text_glow = _outer_glow(text_mask, s["text_bottom"], s["text_glow_radius"], s["text_glow_alpha"])
    text_fill = _gradient_fill(text_mask, s["text_top"], s["text_bottom"], s["text_fill_alpha"])
    text_gloss = _sweep_gloss(text_mask, text_peak_y, text_band_width, text_gloss_alpha, s["gloss_feather"])
    text_stroke = _stroke(text_mask, s["text_bottom"], max(s["stroke_alpha"] - 40, 0), width=1)
    text_hi = _inner_highlight(text_mask, offset=2, alpha=int(s["highlight_alpha"] * 0.75))

    canvas.alpha_composite(text_glow)
    canvas.alpha_composite(text_fill)
    canvas.alpha_composite(text_gloss)
    canvas.alpha_composite(text_stroke)
    canvas.alpha_composite(text_hi)

    return ImageEnhance.Sharpness(canvas).enhance(1.15)


def _peak_offset_frac(t: float, cycles: float, damping: float) -> float:
    """Damped sinusoid that ends exactly at 0 (peak at bbox top = settled).

    Starts at the upper half (peak above the shape), oscillates through the
    shape with shrinking amplitude, and decays to 0 at t=1.  Positive
    fractions push the peak below the bbox top; negative push it above.
    """
    amp = (1.0 - t) * math.exp(-damping * t)
    wave = math.sin(2 * math.pi * cycles * t - math.pi / 2)
    return amp * wave


def render_animation(
    out_path: Path,
    *,
    background: str = "transparent",
    frames: int = 48,
    frame_ms: int = 45,
    settle_ms: int = 3500,
    sweep_cycles: float = 2.0,
    sweep_damping: float = 2.1,
    sweep_amplitude: float = 1.0,
    sweep_gloss_alpha: int = 255,
    sweep_band_frac_text: float = 0.7,
    sweep_band_frac_mark: float = 0.55,
    quantize_colors: int = 128,
) -> Path:
    src = Image.open(SRC).convert("RGBA")
    mark_mask, text_mask = _split_mark_and_text(src)

    text_bbox = text_mask.getbbox()
    text_y0, text_y1 = text_bbox[1], text_bbox[3]
    text_h = text_y1 - text_y0
    text_band_w = text_h * sweep_band_frac_text

    mark_bbox = mark_mask.getbbox()
    mark_y0, mark_y1 = mark_bbox[1], mark_bbox[3]
    mark_h = mark_y1 - mark_y0
    mark_band_w = mark_h * sweep_band_frac_mark

    s = STYLE
    images: list[Image.Image] = []
    durations: list[int] = []

    for i in range(frames):
        t = i / max(frames - 1, 1)
        peak_frac = _peak_offset_frac(t, sweep_cycles, sweep_damping) * sweep_amplitude
        text_peak_y = text_y0 + peak_frac * text_h
        mark_peak_y = mark_y0 + peak_frac * mark_h

        sweep_strength = (1 - t) ** 1.6
        settle_strength = 1 - sweep_strength
        mark_gloss_alpha = int(
            sweep_gloss_alpha * sweep_strength + s["settle_mark_gloss_alpha"] * settle_strength
        )
        text_gloss_alpha = int(
            sweep_gloss_alpha * sweep_strength + s["settle_text_gloss_alpha"] * settle_strength
        )

        frame = _render_frame(
            src, mark_mask, text_mask,
            mark_peak_y=mark_peak_y,
            mark_gloss_alpha=mark_gloss_alpha,
            mark_band_width=mark_band_w,
            text_peak_y=text_peak_y,
            text_gloss_alpha=text_gloss_alpha,
            text_band_width=text_band_w,
            background=background,
        )
        images.append(frame)
        durations.append(frame_ms)

    # Long pause on the settled frame so the loop "rests" before re-running.
    # Append explicit duplicates instead of bumping the last duration: PIL's
    # GIF optimiser drops merged-duration metadata in some cases, but it does
    # honour the cumulative time across collapsed identical frames.
    settled_frame = images[-1]
    pause_copies = max(1, settle_ms // frame_ms)
    for _ in range(pause_copies):
        images.append(settled_frame)
        durations.append(frame_ms)

    save_kwargs = dict(
        save_all=True,
        append_images=[_to_palette(img, background, quantize_colors) for img in images[1:]],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    if background == "transparent":
        save_kwargs["transparency"] = 0
        save_kwargs["disposal"] = 2

    _to_palette(images[0], background, quantize_colors).save(out_path, **save_kwargs)
    return out_path


def _to_palette(img: Image.Image, background: str, colors: int) -> Image.Image:
    # PIL only allows Fast Octree (and libimagequant) on RGBA inputs, so the
    # transparent variant must use it; RGB backgrounds can use the higher
    # quality median-cut quantiser.
    if background == "transparent":
        return img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    return img.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--background", choices=["transparent", "dark", "light"], default="transparent")
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--frame-ms", type=int, default=45)
    p.add_argument("--settle-ms", type=int, default=3500)
    p.add_argument("--sweep-cycles", type=float, default=2.0)
    p.add_argument("--sweep-damping", type=float, default=2.1)
    p.add_argument("--sweep-amplitude", type=float, default=1.0)
    p.add_argument("--sweep-gloss-alpha", type=int, default=255)
    p.add_argument("--sweep-band-frac-text", type=float, default=0.7)
    p.add_argument("--sweep-band-frac-mark", type=float, default=0.55)
    p.add_argument("--quantize-colors", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    out = render_animation(
        a.out,
        background=a.background,
        frames=a.frames,
        frame_ms=a.frame_ms,
        settle_ms=a.settle_ms,
        sweep_cycles=a.sweep_cycles,
        sweep_damping=a.sweep_damping,
        sweep_amplitude=a.sweep_amplitude,
        sweep_gloss_alpha=a.sweep_gloss_alpha,
        sweep_band_frac_text=a.sweep_band_frac_text,
        sweep_band_frac_mark=a.sweep_band_frac_mark,
        quantize_colors=a.quantize_colors,
    )
    print(f"wrote {out}")
