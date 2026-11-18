"""Generate the profile's illustrative text-refinement animation.

Run from the repository root:
    uv run --with "pillow>=10.1" python assets/generate-text-diffusion.py

This is a deterministic graphic, NOT output from a diffusion model.
Inspired by the changing-word canvas on DeepMind's DiffusionGemma page:
https://deepmind.google/models/gemma/diffusiongemma

macOS uses Arial/Andale Mono to match the original preview. Linux and Windows
use available system fonts; Pillow's bundled font is the final fallback.
"""

import math
import random
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
W, H = 1200, 360
BG = (13, 17, 23)
FG = (233, 237, 244)
MUTED = (132, 141, 158)
ACCENT = (186, 177, 249)
TOKEN_BG = (28, 30, 42)


@lru_cache
def font(size, mono=False):
    candidates = [
        Path("/System/Library/Fonts/Supplemental")
        / ("Andale Mono.ttf" if mono else "Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"),
        Path("C:/Windows/Fonts") / ("consola.ttf" if mono else "arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


# Fixed slots prevent layout shifts while candidates update in parallel.
SLOTS = [
    ("Giacomo", 45, 81, 329, 76, 68),
    ("Pasin", 392, 81, 237, 76, 68),
    ("Machine", 47, 182, 228, 49, 39),
    ("Intelligence", 293, 182, 303, 49, 39),
    ("ETH", 47, 261, 91, 34, 26),
    ("Zürich", 156, 261, 116, 34, 26),
]
CANDIDATE_WORDS = [
    ["Vector", "Latent", "Random", "Tensor", "Kernel", "Signal", "Sample"],
    ["prior", "noise", "token", "score", "state", "space", "drift"],
    ["Learning", "Discrete", "Language", "Sampling", "Neural", "Parallel"],
    ["Representation", "Generation", "Embedding", "Distribution", "Attention"],
    ["ML", "AI", "NLP", "GPU", "ETH"],
    ["latent", "tokens", "sample", "vector", "decode", "Zürich"],
]
LOCK_TIMES = [.62, .31, .49, .84, .20, .72]


def mix(a, b, k):
    return tuple(int(x * (1 - k) + y * k) for x, y in zip(a, b))


def render(frame):
    # 1.8 seconds of refinement, 1.2 seconds readable, .4 seconds reset.
    if frame < 36:
        progress = frame / 35
    elif frame < 60:
        progress = 1.
    else:
        progress = max(0, 1 - (frame - 60) / 7)
    visible = {i for i, lock in enumerate(LOCK_TIMES) if progress >= lock}
    rng = random.Random(330 + frame // 2)
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((47, 25), "TEXT DIFFUSION", font=font(13, True), fill=MUTED)
    phase = "RESOLVED" if progress >= 1 else "REFINING"
    draw.text((1005, 25), phase, font=font(12, True), fill=ACCENT)
    draw.line((47, 55, 1153, 55), fill=(43, 48, 59), width=1)

    for index, (word, x, y, width, height, size) in enumerate(SLOTS):
        locked = index in visible
        candidate = word if locked else rng.choice(CANDIDATE_WORDS[index])
        age = max(0, progress - LOCK_TIMES[index])
        settle = min(1, age / .16) if locked else 0
        draw.rounded_rectangle(
            (x - 6, y, x + width, y + height),
            radius=height // 3,
            fill=mix(TOKEN_BG, BG, settle),
        )
        text_font = font(size)
        while draw.textlength(candidate, font=text_font) > width - 14 and size > 12:
            size -= 1
            text_font = font(size)
        box = draw.textbbox((0, 0), candidate, font=text_font)
        color = mix(ACCENT, FG, settle) if locked else mix(MUTED, ACCENT, .24)
        drift = 0 if locked else math.sin(frame * 1.2 + index * 2) * 2
        ty = y + (height - (box[3] - box[1])) / 2 - box[1] + drift
        draw.text((x, ty), candidate, font=text_font, fill=color)
        if locked and settle < 1:
            draw.line(
                (x, y + height + 3, x + width * (1 - settle), y + height + 3),
                fill=ACCENT,
                width=2,
            )

    # Parallel slot states, not measured confidence scores.
    for index in range(len(SLOTS)):
        x = 1005 + index * 25
        draw.rounded_rectangle(
            (x, 276, x + 15, 282),
            radius=2,
            fill=ACCENT if index in visible else (49, 47, 69),
        )
    draw.line((47, 326, 1153, 326), fill=(43, 48, 59), width=1)
    draw.line((47, 326, 47 + 1106 * progress, 326), fill=ACCENT, width=1)
    return image


def main():
    frames = [render(frame) for frame in range(68)]
    palette_source = Image.new("RGB", (W, H * 2))
    palette_source.paste(frames[0], (0, 0))
    palette_source.paste(frames[48], (0, H))
    palette = palette_source.quantize(colors=96)
    indexed = [
        image.quantize(palette=palette, dither=Image.Dither.NONE)
        for image in frames
    ]
    indexed[0].save(
        OUT / "text-diffusion.gif",
        save_all=True,
        append_images=indexed[1:],
        duration=[50] * len(indexed),
        loop=0,
        optimize=True,
        disposal=1,
    )
    frames[48].save(OUT / "text-diffusion-still.png")
    print("Rendered text-diffusion.gif: 20 fps, 3.4-second loop.")


if __name__ == "__main__":
    main()
