"""Render a translucent, cool-toned variant of the PyDiffGame logo.

Reads the original ``images/logo.png`` (red glasses-shaped mark + black wordmark)
and produces a glassy, cool-blue translucent restyling.  The transformation is
deterministic so iterations can be compared side-by-side.

Knobs are exposed via CLI flags so a critic loop can tweak the look between
runs without editing the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "images" / "logo.png"


def _split_mark_and_text(src: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Split the original logo into (mark, wordmark) layers by colour.

    The original logo has a red mark (the abstract glasses/molecule) and a
    black wordmark "PYDIFFGAME".  We separate them so the new style can treat
    them differently (different hues, different translucency).
    """
    rgba = np.asarray(src.convert("RGBA"), dtype=np.int16)
    r, g, b, a = rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]
    # Red mark: red dominates and pixel is not transparent.
    red_mask = (r > 120) & (r - g > 40) & (r - b > 40) & (a > 32)
    # Black wordmark: low luminance, non-transparent.
    luma = (r + g + b) // 3
    black_mask = (luma < 90) & (a > 32) & ~red_mask

    mark = np.zeros_like(rgba, dtype=np.uint8)
    mark[red_mask] = [255, 255, 255, 255]

    text = np.zeros_like(rgba, dtype=np.uint8)
    text[black_mask] = [255, 255, 255, 255]

    return (
        Image.fromarray(mark, "RGBA"),
        Image.fromarray(text, "RGBA"),
    )


def _tint(mask: Image.Image, color: tuple[int, int, int], alpha: int) -> Image.Image:
    """Tint a white-on-transparent mask with ``color`` at ``alpha`` opacity."""
    arr = np.asarray(mask, dtype=np.uint8).copy()
    a_in = arr[..., 3].astype(np.float32) / 255.0
    out = np.zeros_like(arr)
    out[..., 0] = color[0]
    out[..., 1] = color[1]
    out[..., 2] = color[2]
    out[..., 3] = (a_in * alpha).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _vertical_gradient(
    size: tuple[int, int],
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
) -> Image.Image:
    """Vertical gradient from ``top`` colour at y=0 to ``bottom`` at y=h."""
    w, h = size
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    top_arr = np.array(top, dtype=np.float32)
    bot_arr = np.array(bottom, dtype=np.float32)
    line = top_arr * (1 - ys) + bot_arr * ys  # h x 3
    img = np.broadcast_to(line[:, None, :], (h, w, 3)).astype(np.uint8)
    return Image.fromarray(img, "RGB").convert("RGBA")


def _apply_gradient(mask: Image.Image, gradient: Image.Image, alpha: int) -> Image.Image:
    """Use ``mask`` (alpha) to cut the ``gradient`` out, scaled to ``alpha``."""
    arr = np.asarray(mask, dtype=np.uint8)
    grad = np.asarray(gradient.resize(mask.size), dtype=np.uint8).copy()
    a_in = arr[..., 3].astype(np.float32) / 255.0
    grad[..., 3] = (a_in * alpha).astype(np.uint8)
    return Image.fromarray(grad, "RGBA")


def _outer_glow(
    mask: Image.Image,
    color: tuple[int, int, int],
    radius: float,
    alpha: int,
) -> Image.Image:
    """Soft outer glow built by blurring a tinted copy of the mask."""
    glow = _tint(mask, color, alpha)
    return glow.filter(ImageFilter.GaussianBlur(radius=radius))


def _inner_highlight(mask: Image.Image, offset: int, alpha: int) -> Image.Image:
    """Top-edge inner highlight to suggest glass curvature."""
    base = np.asarray(mask, dtype=np.uint8)
    a = base[..., 3]
    shifted = np.zeros_like(a)
    shifted[offset:, :] = a[:-offset, :]
    inner = np.minimum(a, 255 - shifted)
    out = np.zeros_like(base)
    out[..., 0] = 255
    out[..., 1] = 255
    out[..., 2] = 255
    out[..., 3] = (inner.astype(np.float32) / 255.0 * alpha).astype(np.uint8)
    img = Image.fromarray(out, "RGBA").filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def _stroke(mask: Image.Image, color: tuple[int, int, int], alpha: int, width: int = 1) -> Image.Image:
    """Crisp outline around the shape (used for glass-edge feel)."""
    a = np.asarray(mask, dtype=np.uint8)[..., 3]
    eroded = Image.fromarray(a, "L").filter(ImageFilter.MinFilter(2 * width + 1))
    edge = ImageChops.subtract(Image.fromarray(a, "L"), eroded)
    out = np.zeros((*a.shape, 4), dtype=np.uint8)
    out[..., 0] = color[0]
    out[..., 1] = color[1]
    out[..., 2] = color[2]
    out[..., 3] = (np.asarray(edge, dtype=np.float32) / 255.0 * alpha).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def render(
    out_path: Path,
    *,
    mark_top: tuple[int, int, int] = (120, 220, 255),
    mark_bottom: tuple[int, int, int] = (40, 110, 220),
    text_top: tuple[int, int, int] = (210, 240, 255),
    text_bottom: tuple[int, int, int] = (90, 150, 220),
    mark_fill_alpha: int = 110,
    text_fill_alpha: int = 175,
    mark_glow_alpha: int = 180,
    text_glow_alpha: int = 90,
    mark_glow_radius: float = 14.0,
    text_glow_radius: float = 6.0,
    highlight_alpha: int = 150,
    stroke_alpha: int = 200,
    background: str = "transparent",
) -> Path:
    """Render the translucent cool-toned logo and save to ``out_path``."""
    src = Image.open(SRC).convert("RGBA")
    w, h = src.size
    mark_mask, text_mask = _split_mark_and_text(src)

    # Canvas (transparent or dark backdrop so glow reads in previews).
    if background == "dark":
        canvas = Image.new("RGBA", (w, h), (8, 14, 28, 255))
    elif background == "light":
        canvas = Image.new("RGBA", (w, h), (245, 248, 255, 255))
    else:
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # --- Mark (glasses) ---
    mark_grad = _vertical_gradient((w, h), mark_top, mark_bottom)
    mark_fill = _apply_gradient(mark_mask, mark_grad, mark_fill_alpha)
    mark_glow = _outer_glow(mark_mask, mark_top, mark_glow_radius, mark_glow_alpha)
    mark_stroke = _stroke(mark_mask, (220, 245, 255), stroke_alpha, width=1)
    mark_highlight = _inner_highlight(mark_mask, offset=3, alpha=highlight_alpha)

    canvas.alpha_composite(mark_glow)
    canvas.alpha_composite(mark_fill)
    canvas.alpha_composite(mark_stroke)
    canvas.alpha_composite(mark_highlight)

    # --- Wordmark ---
    text_grad = _vertical_gradient((w, h), text_top, text_bottom)
    text_fill = _apply_gradient(text_mask, text_grad, text_fill_alpha)
    text_glow = _outer_glow(text_mask, text_top, text_glow_radius, text_glow_alpha)
    text_stroke = _stroke(text_mask, (220, 245, 255), 140, width=1)
    text_highlight = _inner_highlight(text_mask, offset=2, alpha=110)

    canvas.alpha_composite(text_glow)
    canvas.alpha_composite(text_fill)
    canvas.alpha_composite(text_stroke)
    canvas.alpha_composite(text_highlight)

    # Light overall sharpen so glass edges stay crisp after blurs.
    sharpened = ImageEnhance.Sharpness(canvas).enhance(1.15)
    sharpened.save(out_path)
    return out_path


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--background", choices=["transparent", "dark", "light"], default="transparent")
    p.add_argument("--mark-top", nargs=3, type=int, default=[120, 220, 255])
    p.add_argument("--mark-bottom", nargs=3, type=int, default=[40, 110, 220])
    p.add_argument("--text-top", nargs=3, type=int, default=[210, 240, 255])
    p.add_argument("--text-bottom", nargs=3, type=int, default=[90, 150, 220])
    p.add_argument("--mark-fill-alpha", type=int, default=110)
    p.add_argument("--text-fill-alpha", type=int, default=175)
    p.add_argument("--mark-glow-alpha", type=int, default=180)
    p.add_argument("--text-glow-alpha", type=int, default=90)
    p.add_argument("--mark-glow-radius", type=float, default=14.0)
    p.add_argument("--text-glow-radius", type=float, default=6.0)
    p.add_argument("--highlight-alpha", type=int, default=150)
    p.add_argument("--stroke-alpha", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    render(
        args.out,
        background=args.background,
        mark_top=tuple(args.mark_top),
        mark_bottom=tuple(args.mark_bottom),
        text_top=tuple(args.text_top),
        text_bottom=tuple(args.text_bottom),
        mark_fill_alpha=args.mark_fill_alpha,
        text_fill_alpha=args.text_fill_alpha,
        mark_glow_alpha=args.mark_glow_alpha,
        text_glow_alpha=args.text_glow_alpha,
        mark_glow_radius=args.mark_glow_radius,
        text_glow_radius=args.text_glow_radius,
        highlight_alpha=args.highlight_alpha,
        stroke_alpha=args.stroke_alpha,
    )
    print(f"wrote {args.out}")
