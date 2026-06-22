"""Render a translucent, glossy-glass variant of the PyDiffGame logo.

Reads the original ``images/logo.png`` (red glasses-shaped mark + black
wordmark) and restyles each layer as polished glass: a vertical tonal gradient
that keeps the original palette (red mark, near-black wordmark), a soft
matched-colour outer glow, a top-edge inner highlight, a broad top-half gloss
band ("glaring" specular), and a crisp edge stroke.  The transformation is
deterministic so iterations can be compared side-by-side.

Knobs are exposed via CLI flags so iterations can tune the look without
editing the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parent.parent
# Source is the immutable original (red mark + black wordmark) committed
# alongside this script.  The renderer reads it and writes a styled copy
# (typically over images/logo.png) so the source survives restyles.
SRC = ROOT / "images" / "logo_source.png"


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


def _gradient_fill(
    mask: Image.Image,
    top_color: tuple[int, int, int],
    bottom_color: tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    """Fill ``mask`` with a vertical gradient anchored to the mask's bbox.

    Pixels above the bbox get ``top_color``, pixels below get ``bottom_color``,
    and within the bbox the colour interpolates linearly top→bottom.  The
    output alpha is the mask alpha scaled by ``alpha``.
    """
    w, h = mask.size
    a = np.asarray(mask, dtype=np.uint8)[..., 3]
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y0, y1 = bbox[1], bbox[3]
    ys = np.arange(h, dtype=np.float32)
    t = np.clip((ys - y0) / max(y1 - y0, 1), 0.0, 1.0)
    top_arr = np.array(top_color, dtype=np.float32)
    bot_arr = np.array(bottom_color, dtype=np.float32)
    line = top_arr * (1 - t)[:, None] + bot_arr * t[:, None]
    rgb = np.broadcast_to(line[:, None, :], (h, w, 3)).astype(np.uint8)
    a_norm = a.astype(np.float32) / 255.0
    out_alpha = (a_norm * (alpha / 255.0) * 255.0).astype(np.uint8)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = out_alpha
    return Image.fromarray(out, "RGBA")


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


def _gloss_band(
    mask: Image.Image,
    height_frac: float,
    alpha: int,
    feather: float,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Broad top-half specular gloss clipped to the shape (the "glare").

    Builds a vertical falloff that is fully opaque at the top of the mask's
    bounding box and fades to zero ``height_frac`` of the way down it, then
    multiplies that falloff by the shape's own alpha so the gloss only
    appears on the shape's surface.  A small Gaussian blur softens the edge.
    """
    a = np.asarray(mask, dtype=np.uint8)[..., 3]
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("RGBA", mask.size, (0, 0, 0, 0))
    y0, y1 = bbox[1], bbox[3]
    h = max(y1 - y0, 1)
    band_h = max(int(h * height_frac), 1)

    falloff = np.zeros_like(a, dtype=np.float32)
    ys = np.arange(a.shape[0], dtype=np.float32)
    weight = np.clip(1.0 - (ys - y0) / band_h, 0.0, 1.0)
    falloff[:] = weight[:, None]

    a_norm = a.astype(np.float32) / 255.0
    out_alpha = (a_norm * falloff * (alpha / 255.0) * 255.0).astype(np.uint8)

    out = np.zeros((*a.shape, 4), dtype=np.uint8)
    out[..., 0] = color[0]
    out[..., 1] = color[1]
    out[..., 2] = color[2]
    out[..., 3] = out_alpha
    img = Image.fromarray(out, "RGBA")
    if feather > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=feather))
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
    mark_top: tuple[int, int, int] = (255, 95, 105),
    mark_bottom: tuple[int, int, int] = (170, 10, 25),
    text_top: tuple[int, int, int] = (75, 75, 82),
    text_bottom: tuple[int, int, int] = (5, 5, 12),
    mark_fill_alpha: int = 230,
    text_fill_alpha: int = 245,
    mark_glow_alpha: int = 140,
    text_glow_alpha: int = 80,
    mark_glow_radius: float = 5.5,
    text_glow_radius: float = 2.8,
    highlight_alpha: int = 235,
    stroke_alpha: int = 200,
    mark_gloss_alpha: int = 90,
    text_gloss_alpha: int = 150,
    gloss_height: float = 0.5,
    gloss_feather: float = 3.5,
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
    # Order: outer glow (rim light) -> bbox-anchored gradient fill -> top
    # gloss band (the broad specular glare) -> rim edge stroke -> crisp
    # top-edge inner highlight on top.
    mark_fill = _gradient_fill(mark_mask, mark_top, mark_bottom, mark_fill_alpha)
    mark_glow = _outer_glow(mark_mask, mark_bottom, mark_glow_radius, mark_glow_alpha)
    mark_gloss = _gloss_band(mark_mask, gloss_height, mark_gloss_alpha, gloss_feather)
    mark_stroke = _stroke(mark_mask, mark_bottom, stroke_alpha, width=1)
    mark_highlight = _inner_highlight(mark_mask, offset=3, alpha=highlight_alpha)

    canvas.alpha_composite(mark_glow)
    canvas.alpha_composite(mark_fill)
    canvas.alpha_composite(mark_gloss)
    canvas.alpha_composite(mark_stroke)
    canvas.alpha_composite(mark_highlight)

    # --- Wordmark ---
    text_fill = _gradient_fill(text_mask, text_top, text_bottom, text_fill_alpha)
    text_glow = _outer_glow(text_mask, text_bottom, text_glow_radius, text_glow_alpha)
    text_gloss = _gloss_band(text_mask, gloss_height, text_gloss_alpha, gloss_feather)
    text_stroke = _stroke(text_mask, text_bottom, max(stroke_alpha - 40, 0), width=1)
    text_highlight = _inner_highlight(text_mask, offset=2, alpha=int(highlight_alpha * 0.75))

    canvas.alpha_composite(text_glow)
    canvas.alpha_composite(text_fill)
    canvas.alpha_composite(text_gloss)
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
    p.add_argument("--mark-top", nargs=3, type=int, default=[255, 95, 105])
    p.add_argument("--mark-bottom", nargs=3, type=int, default=[170, 10, 25])
    p.add_argument("--text-top", nargs=3, type=int, default=[75, 75, 82])
    p.add_argument("--text-bottom", nargs=3, type=int, default=[5, 5, 12])
    p.add_argument("--mark-fill-alpha", type=int, default=230)
    p.add_argument("--text-fill-alpha", type=int, default=245)
    p.add_argument("--mark-glow-alpha", type=int, default=140)
    p.add_argument("--text-glow-alpha", type=int, default=80)
    p.add_argument("--mark-glow-radius", type=float, default=5.5)
    p.add_argument("--text-glow-radius", type=float, default=2.8)
    p.add_argument("--highlight-alpha", type=int, default=235)
    p.add_argument("--stroke-alpha", type=int, default=200)
    p.add_argument("--mark-gloss-alpha", type=int, default=90)
    p.add_argument("--text-gloss-alpha", type=int, default=150)
    p.add_argument("--gloss-height", type=float, default=0.5)
    p.add_argument("--gloss-feather", type=float, default=3.5)
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
        mark_gloss_alpha=args.mark_gloss_alpha,
        text_gloss_alpha=args.text_gloss_alpha,
        gloss_height=args.gloss_height,
        gloss_feather=args.gloss_feather,
    )
    print(f"wrote {args.out}")
