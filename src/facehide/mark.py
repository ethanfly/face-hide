from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

STATUS_IDLE = "idle"
STATUS_WATCHING = "watching"
STATUS_DEV = "dev"
STATUS_ALERT = "alert"

_TOP = (86, 142, 255)
_BOT = (35, 82, 208)
_RING = (186, 216, 255)
_WELL = (11, 17, 30)
_FACE = (232, 240, 255)
_BAR = (255, 139, 123)
_BAR_HI = (255, 198, 188)
_SHEEN = (255, 255, 255)
_DOT = {
    STATUS_WATCHING: (98, 216, 154, 255),
    STATUS_DEV: (243, 193, 107, 255),
    STATUS_ALERT: (255, 139, 123, 255),
}

_cache: dict[tuple[int, str], Image.Image] = {}


def render_mark(size: int, status: str = STATUS_IDLE) -> Image.Image:
    key = (int(size), status)
    cached = _cache.get(key)
    if cached is not None:
        return cached.copy()
    image = _render_mark(int(size), status)
    _cache[key] = image
    return image.copy()


def save_ico(path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    images = [render_mark(size) for size in ICON_SIZES]
    largest = images[-1]
    largest.save(
        dest,
        format="ICO",
        append_images=images[:-1],
        sizes=[image.size for image in images],
    )
    return dest


def _render_mark(size: int, status: str) -> Image.Image:
    sample = 4 if size <= 64 else 2 if size <= 128 else 1
    canvas = _paint(size * sample, size)
    if status in _DOT:
        _badge(canvas, status, size * sample)
    if sample > 1:
        canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    return canvas


def _paint(canvas: int, logical: int) -> Image.Image:
    s = canvas
    tile = _vertical_gradient(s, _TOP, _BOT)
    mask = _rounded_mask(s, max(2, round(s * 0.22)))
    tile.putalpha(mask)

    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out = Image.alpha_composite(out, tile)
    if logical >= 24:
        out = Image.alpha_composite(out, _sheen(s, mask))

    layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ring = s * 0.145
    well = s * 0.175
    draw.ellipse((ring, ring, s - ring, s - ring), fill=_RING + (230,))
    draw.ellipse((well, well, s - well, s - well), fill=_WELL + (255,))

    face = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    fd = ImageDraw.Draw(face)
    hx, hy = s * 0.50, s * 0.455
    hrx, hry = s * 0.188, s * 0.228
    fd.ellipse((hx - hrx, hy - hry, hx + hrx, hy + hry), fill=_FACE + (255,))
    if logical >= 28:
        fd.rectangle((s * 0.445, s * 0.62, s * 0.555, s * 0.74), fill=_FACE + (255,))
        sx, sy = s * 0.50, s * 0.86
        srx, sry = s * 0.30, s * 0.18
        fd.ellipse((sx - srx, sy - sry, sx + srx, sy + sry), fill=_FACE + (255,))

    well_mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(well_mask).ellipse((well, well, s - well, s - well), fill=255)
    face.putalpha(ImageChops.multiply(face.getchannel("A"), well_mask))
    layer = Image.alpha_composite(layer, face)

    bd = ImageDraw.Draw(layer)
    bar_l, bar_r = s * 0.205, s * 0.795
    if logical < 24:
        bar_t, bar_b = s * 0.38, s * 0.52
        radius = max(2, s * 0.055)
    else:
        bar_t, bar_b = s * 0.385, s * 0.495
        radius = max(2, s * 0.04)
    bd.rounded_rectangle((bar_l, bar_t, bar_r, bar_b), radius=radius, fill=_BAR + (255,))
    if logical >= 40:
        hi = max(1, s * 0.016)
        bd.rounded_rectangle(
            (bar_l + s * 0.03, bar_t + s * 0.012, bar_r - s * 0.03, bar_t + s * 0.012 + hi),
            radius=hi,
            fill=_BAR_HI + (150,),
        )

    out = Image.alpha_composite(out, layer)
    return out


def _vertical_gradient(size: int, top: tuple[int, int, int], bot: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(image)
    last = max(1, size - 1)
    for y in range(size):
        t = y / last
        color = (
            int(top[0] + (bot[0] - top[0]) * t),
            int(top[1] + (bot[1] - top[1]) * t),
            int(top[2] + (bot[2] - top[2]) * t),
            255,
        )
        draw.line((0, y, size, y), fill=color)
    return image


def _rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def _sheen(size: int, mask: Image.Image) -> Image.Image:
    sheen = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(sheen).ellipse(
        (-size * 0.15, -size * 0.78, size * 1.15, size * 0.52),
        fill=_SHEEN + (40,),
    )
    alpha = ImageChops.multiply(sheen.split()[-1], mask)
    sheen.putalpha(alpha)
    return sheen


def _badge(image: Image.Image, status: str, canvas: int) -> None:
    color = _DOT[status]
    draw = ImageDraw.Draw(image)
    r = max(3, canvas * 0.13)
    cx = canvas - canvas * 0.20
    cy = canvas - canvas * 0.20
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 255))
    inner = r * 0.68
    draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=color)
