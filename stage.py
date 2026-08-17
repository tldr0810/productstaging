#!/usr/bin/env python3
"""Open-source product staging CLI.

The product RGB layer is always sourced from the input image. Optional models
are used to predict a mask and a background only; they never get to redraw the
product. This makes the fallback useful on a laptop while keeping the model
path faithful to the same compositing contract.
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - reported by the CLI
    raise SystemExit("numpy is required; run `pip install -r requirements.txt`") from exc


DEFAULT_CUTOUT_MODEL = os.getenv("STAGING_CUTOUT_MODEL", "birefnet-general")
DEFAULT_CUTOUT_EDGE = os.getenv("STAGING_CUTOUT_EDGE", "preserve")
DEFAULT_SCENE_MODEL = os.getenv("STAGING_SCENE_MODEL", "stabilityai/sd-turbo")
CUTOUT_EDGE_MODES = ("preserve", "decontaminate", "alpha_matting", "vitmatte")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _soft_mask(mask: Image.Image, blur: float = 1.25) -> Image.Image:
    """Return a feathered alpha mask while keeping its binary boundary available."""
    return mask.convert("L").filter(ImageFilter.GaussianBlur(blur))


def _fallback_mask(image: Image.Image) -> Image.Image:
    """Estimate a foreground mask for the common white/simple-background shot.

    A border colour estimate plus a flood-like colour distance keeps similarly
    coloured product pixels safer than a global brightness threshold. It is a
    deliberately conservative fallback; rembg is preferred for production use.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate(
        [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0
    )
    bg = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - bg, axis=2)
    brightness = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    # Most product photos use a near-white sweep. Keep saturated/dark objects.
    near_white = (brightness > 238) & (saturation < 24)
    foreground = (distance > 26) | ~near_white
    # Remove only connected-ish empty border pixels; avoid erasing white products.
    foreground[:2, :] = False
    foreground[-2:, :] = False
    foreground[:, :2] = False
    foreground[:, -2:] = False
    alpha = Image.fromarray((foreground.astype(np.uint8) * 255))
    alpha = alpha.filter(ImageFilter.MedianFilter(3))
    return alpha


@lru_cache(maxsize=8)
def _rembg_session(model: str) -> Any:
    """Load one rembg/ONNX session per model and reuse it across requests."""
    from rembg import new_session  # type: ignore

    return new_session(model)


def _rembg_cutout(original: Image.Image, model: str, edge_mode: str) -> tuple[Image.Image, Image.Image]:
    from rembg import remove  # type: ignore

    session = _rembg_session(model)
    source = _png_bytes(original)
    if edge_mode == "preserve":
        mask_bytes = remove(source, session=session, only_mask=True)
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L").resize(original.size, Image.Resampling.LANCZOS)
        cutout = original.copy()
        cutout.putalpha(mask)
        return cutout, mask

    options: dict[str, Any] = {"session": session}
    if edge_mode == "decontaminate":
        options["decontaminate"] = True
    elif edge_mode == "alpha_matting":
        options["alpha_matting"] = True
    elif edge_mode == "vitmatte":
        options["vitmatte"] = True
    result = remove(source, **options)
    processed = Image.open(io.BytesIO(result)).convert("RGBA").resize(original.size, Image.Resampling.LANCZOS)
    mask = processed.getchannel("A")
    # Alpha matting and ViTMatte improve coverage. Reapply source RGB so only
    # the edge mode that explicitly decontaminates is allowed to change colour.
    if edge_mode != "decontaminate":
        cutout = original.copy()
        cutout.putalpha(mask)
    else:
        cutout = processed
    return cutout, mask


def cutout_product(
    image: Image.Image,
    model: str = DEFAULT_CUTOUT_MODEL,
    edge_mode: str = DEFAULT_CUTOUT_EDGE,
) -> tuple[Image.Image, Image.Image, str]:
    """Return (RGBA cutout, binary mask, backend name).

    rembg is loaded lazily so the CLI remains runnable without ONNX/CUDA. The
    original RGB data is reapplied after inference unless the caller explicitly
    chooses ``decontaminate`` for coloured edge-halo removal. A requested model
    falls back to u2net before the deterministic simple-background mask.
    """
    original = image.convert("RGBA")
    if edge_mode not in CUTOUT_EDGE_MODES:
        raise ValueError(f"cutout edge mode must be one of: {', '.join(CUTOUT_EDGE_MODES)}")
    models = [model] if model == "u2net" else [model, "u2net"]
    edge_attempts = [(candidate, edge_mode) for candidate in models]
    if edge_mode != "preserve":
        # Missing optional matting dependencies should not discard a usable
        # model mask; retry the same model in strict RGB-preserving mode.
        edge_attempts = [(model, edge_mode), (model, "preserve")]
        if model != "u2net":
            edge_attempts.extend([("u2net", edge_mode), ("u2net", "preserve")])
    errors: list[str] = []
    for candidate, candidate_edge in edge_attempts:
        try:
            cutout, mask = _rembg_cutout(original, candidate, candidate_edge)
            backend = f"rembg:{candidate}:{candidate_edge}"
            return cutout, mask.point(lambda value: 255 if value >= 128 else 0, mode="L"), backend
        except Exception as exc:  # noqa: BLE001 - model availability is optional
            errors.append(f"{candidate}/{candidate_edge}: {exc}")

    print(f"warning: rembg unavailable ({'; '.join(errors)}); using simple-background mask", file=sys.stderr)
    mask = _fallback_mask(original)
    cutout = original.copy()
    cutout.putalpha(mask)
    backend = "fallback-mask"

    binary = mask.point(lambda value: 255 if value >= 128 else 0, mode="L")
    return cutout, binary, backend


def _scene_size(product: Image.Image, max_side: int = 1024) -> tuple[int, int]:
    width, height = product.size
    scale = min(1.0, max_side / max(width, height))
    # Diffusion pipelines require dimensions divisible by 8. Invalid aspect
    # sizes otherwise throw and silently send the request to the fallback.
    scaled_width = max(512, int(width * scale))
    scaled_height = max(512, int(height * scale))
    return (scaled_width // 8) * 8, (scaled_height // 8) * 8


def _model_prompt(prompt: str) -> str:
    """Turn the user's scene description into a background-only prompt."""
    description = " ".join(prompt.split())
    return (
        "High-end commercial product photography background, empty foreground "
        f"reserved for the photographed product, {description}. "
        "Natural perspective, realistic materials, coherent soft lighting, "
        "shallow depth of field, no extra product, no people, no text, no logo."
    )


def _prompt_kind(prompt: str) -> str:
    value = prompt.lower()
    for kind, words in {
        "wood": ("wood", "rustic", "table", "desk"),
        "marble": ("marble", "counter", "kitchen"),
        "outdoor": ("outdoor", "garden", "patio", "grass", "sun"),
        "studio": ("studio", "seamless", "catalog"),
    }.items():
        if any(word in value for word in words):
            return kind
    return "wood"


def _fallback_scene(prompt: str, size: tuple[int, int], seed: int = 7) -> Image.Image:
    """Create a deterministic textured scene when diffusers is not installed."""
    random.seed(seed)
    width, height = size
    kind = _prompt_kind(prompt)
    if kind == "marble":
        top, bottom = (219, 221, 218), (152, 157, 155)
    elif kind == "outdoor":
        top, bottom = (171, 207, 220), (85, 125, 67)
    elif kind == "studio":
        top, bottom = (242, 242, 238), (205, 207, 204)
    else:
        top, bottom = (198, 157, 107), (100, 66, 39)
    scene = Image.new("RGB", size)
    pixels = scene.load()
    horizon = int(height * (0.55 if kind == "outdoor" else 0.48))
    for y in range(height):
        # Slightly brighter upper half gives a soft photographic light falloff.
        if y < horizon:
            q = y / max(horizon, 1) * 0.75
        else:
            q = 0.75 + (y - horizon) / max(height - horizon, 1) * 0.25
        colour = tuple(int(top[i] * (1 - q) + bottom[i] * q) for i in range(3))
        for x in range(width):
            pixels[x, y] = colour
    draw = ImageDraw.Draw(scene, "RGBA")
    if kind == "outdoor":
        draw.rectangle((0, horizon, width, height), fill=(46, 83, 34, 35))
        for _ in range(90):
            x = random.randrange(width)
            y = random.randrange(horizon, height)
            draw.ellipse((x, y, x + random.randrange(2, 8), y + random.randrange(2, 8)), fill=(213, 225, 136, 45))
        draw.line((0, horizon, width, horizon), fill=(248, 246, 223, 100), width=3)
    elif kind == "marble":
        for _ in range(24):
            x = random.randrange(-width, width)
            draw.line((x, horizon, x + random.randrange(-240, 240), height), fill=(255, 255, 255, 36), width=random.randrange(2, 7))
    elif kind == "wood":
        draw.line((0, horizon, width, horizon), fill=(67, 42, 25, 110), width=4)
        for _ in range(35):
            y = random.randrange(horizon, height)
            draw.line((0, y, width, y + random.randrange(-8, 9)), fill=(52, 29, 16, 35), width=random.randrange(1, 4))
    else:
        draw.line((0, horizon, width, horizon), fill=(255, 255, 255, 80), width=3)
    # A broad, blurred key light keeps the fallback from looking flat.
    light = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(light).ellipse((width * 0.02, -height * 0.3, width * 0.72, height * 0.55), fill=(255, 239, 204, 72))
    return Image.alpha_composite(scene.convert("RGBA"), light.filter(ImageFilter.GaussianBlur(width * 0.12))).convert("RGB")


def generate_scene(prompt: str, size: tuple[int, int], model_id: str = DEFAULT_SCENE_MODEL,
                   device: str = "auto", seed: int = 7) -> tuple[Image.Image, str]:
    """Generate a scene with diffusers, falling back to a local textured scene."""
    try:
        import torch  # type: ignore
        from diffusers import AutoPipelineForText2Image  # type: ignore

        target = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if target == "auto":
            target = "cpu"
        dtype = torch.float16 if target == "cuda" else torch.float32
        generator = torch.Generator(device=target).manual_seed(seed)
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(target)
        is_turbo = "turbo" in model_id.lower()
        options = {
            "width": size[0],
            "height": size[1],
            "num_inference_steps": 4 if is_turbo else 30,
            "guidance_scale": 0.0 if is_turbo else 7.0,
            "generator": generator,
        }
        if not is_turbo:
            options["negative_prompt"] = "blurry, low quality, duplicate product, extra objects, text, watermark"
        result = pipe(_model_prompt(prompt), **options)
        return result.images[0].convert("RGB"), f"diffusers:{model_id}"
    except Exception as exc:  # noqa: BLE001 - optional heavyweight backend
        print(f"warning: scene model unavailable ({exc}); using local textured scene", file=sys.stderr)
        return _fallback_scene(prompt, size, seed), "fallback-scene"


def _product_box(alpha: Image.Image) -> tuple[int, int, int, int]:
    box = alpha.getbbox()
    if not box:
        raise ValueError("no foreground was detected; use a simpler background or install rembg")
    return box


def _shadow_layer(size: tuple[int, int], box: tuple[int, int, int, int], alpha: Image.Image,
                  prompt: str) -> Image.Image:
    width, height = size
    left, top, right, bottom = box
    product_width = max(1, right - left)
    kind = _prompt_kind(prompt)
    # Put the contact patch just below the detected base and bias it away from light.
    light_from_left = "left" in prompt.lower() or "morning" in prompt.lower()
    offset = int(product_width * (0.035 if light_from_left else -0.02))
    shadow = Image.new("L", size, 0)
    patch = Image.new("L", (product_width, max(8, int(product_width * 0.18))), 0)
    ImageDraw.Draw(patch).ellipse((0, 0, patch.width, patch.height), fill=135 if kind != "studio" else 85)
    patch = patch.filter(ImageFilter.GaussianBlur(max(4, int(product_width * 0.04))))
    shadow.paste(patch, (left + offset, min(height - patch.height // 2, bottom - patch.height // 3)))
    # A very soft secondary occlusion follows the product footprint.
    footprint = alpha.filter(ImageFilter.GaussianBlur(max(5, int(product_width * 0.07))))
    footprint = ImageChops.offset(footprint, offset, max(0, int(product_width * 0.025)))
    shadow = ImageChops.lighter(shadow, footprint.point(lambda value: int(value * 0.22)))
    layer = Image.new("RGBA", size, (18, 13, 10, 0))
    layer.putalpha(shadow)
    return layer


def _edge_tint(cutout: Image.Image, scene: Image.Image, alpha: Image.Image, box: tuple[int, int, int, int], prompt: str) -> Image.Image:
    """Tint a 1-2px edge ring only; opaque interior pixels are left untouched."""
    rgba = np.asarray(cutout.convert("RGBA"), dtype=np.uint8).copy()
    a = np.asarray(alpha, dtype=np.uint8)
    edge = np.asarray(alpha.filter(ImageFilter.MaxFilter(5)), dtype=np.int16) - np.asarray(alpha.filter(ImageFilter.MinFilter(5)), dtype=np.int16)
    edge = np.clip(edge, 0, 255).astype(np.float32) / 255.0
    scene_rgb = np.asarray(scene.convert("RGB"), dtype=np.float32)
    warm = any(word in prompt.lower() for word in ("warm", "morning", "sunset", "rustic"))
    target = scene_rgb * (0.11 if warm else 0.08)
    if warm:
        target[..., 0] += 13
    else:
        target[..., 2] += 7
    blend = np.clip(edge * 0.28, 0, 0.28)[..., None]
    rgba[..., :3] = np.clip(rgba[..., :3] * (1 - blend) + target * blend, 0, 255).astype(np.uint8)
    rgba[..., 3] = a
    return Image.fromarray(rgba)


def composite_product(scene: Image.Image, cutout: Image.Image, mask: Image.Image, prompt: str,
                      product_scale: float = 0.52) -> Image.Image:
    scene = scene.convert("RGBA")
    cutout = cutout.convert("RGBA")
    box = _product_box(mask)
    cropped = cutout.crop(box)
    cropped_mask = mask.crop(box)
    target_height = max(1, int(scene.height * product_scale))
    scale = target_height / max(1, cropped.height)
    new_size = (max(1, int(cropped.width * scale)), target_height)
    # Geometric scaling is the only product-wide transform; no generative redraw occurs.
    cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
    cropped_mask = cropped_mask.resize(new_size, Image.Resampling.LANCZOS)
    feathered = _soft_mask(cropped_mask, max(0.65, min(1.6, new_size[0] / 700)))
    x = max(0, (scene.width - cropped.width) // 2)
    y = max(0, int(scene.height * 0.55) - cropped.height // 2)
    full_alpha = Image.new("L", scene.size, 0)
    full_alpha.paste(feathered, (x, y))
    resized_cutout = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    resized_cutout.paste(cropped, (x, y), cropped)
    tinted = _edge_tint(resized_cutout, scene, full_alpha, (x, y, x + cropped.width, y + cropped.height), prompt)
    output = Image.alpha_composite(scene, _shadow_layer(scene.size, (x, y, x + cropped.width, y + cropped.height), full_alpha, prompt))
    output.alpha_composite(tinted, (0, 0))
    return output.convert("RGB")


def comparison_image(original: Image.Image, staged: Image.Image) -> Image.Image:
    """Create a shareable before/after board with identical framing."""
    width = max(original.width, staged.width)
    height = max(original.height, staged.height)
    board = Image.new("RGB", (width * 2, height + 54), (244, 242, 238))
    font = ImageFont.load_default()
    left = ImageOps.contain(original.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    right = ImageOps.contain(staged.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    board.paste(left, ((width - left.width) // 2, 42 + (height - left.height) // 2))
    board.paste(right, (width + (width - right.width) // 2, 42 + (height - right.height) // 2))
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "ORIGINAL", fill=(36, 38, 42), font=font)
    draw.text((width + 18, 14), "STAGED", fill=(36, 38, 42), font=font)
    return board


def stage(input_path: Path, prompt: str, output_path: Path, cutout_model: str = DEFAULT_CUTOUT_MODEL,
          scene_model: str = DEFAULT_SCENE_MODEL, device: str = "auto", seed: int = 7,
          product_scale: float = 0.52, cutout_edge: str = DEFAULT_CUTOUT_EDGE) -> dict[str, Path | str]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("output path must differ from input path so the source photo is preserved")
    original = Image.open(input_path).convert("RGBA")
    cutout, binary_mask, cutout_backend = cutout_product(original, cutout_model, cutout_edge)
    scene, scene_backend = generate_scene(prompt, _scene_size(original), scene_model, device, seed)
    staged = composite_product(scene, cutout, binary_mask, prompt, product_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix(".png")
    cutout_path = output_path.with_name(f"{output_path.stem}.cutout.png")
    mask_path = output_path.with_name(f"{output_path.stem}.mask.png")
    comparison_path = output_path.with_name(f"{output_path.stem}.comparison.png")
    cutout.save(cutout_path)
    binary_mask.save(mask_path)
    staged.save(output_path)
    comparison_image(original, staged).save(comparison_path)
    return {"output": output_path, "cutout": cutout_path, "mask": mask_path,
            "comparison": comparison_path, "cutout_backend": cutout_backend,
            "scene_backend": scene_backend}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage a product photo in an AI or local lifestyle scene.")
    parser.add_argument("--input", required=True, type=Path, help="Product photo (PNG/JPEG/WebP).")
    parser.add_argument("--prompt", required=True, help="Scene description, e.g. 'on a marble kitchen counter'.")
    parser.add_argument("--output", required=True, type=Path, help="Staged output PNG path.")
    parser.add_argument("--cutout-model", default=DEFAULT_CUTOUT_MODEL,
                        help=f"rembg model name (default: {DEFAULT_CUTOUT_MODEL}).")
    parser.add_argument("--cutout-edge", choices=CUTOUT_EDGE_MODES, default=DEFAULT_CUTOUT_EDGE,
                        help="edge handling: preserve, decontaminate, alpha_matting, or vitmatte.")
    parser.add_argument("--scene-model", default=DEFAULT_SCENE_MODEL, help=f"diffusers model id (default: {DEFAULT_SCENE_MODEL}).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--product-scale", type=float, default=0.52, help="Product height as a fraction of scene height.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2
    if not 0.1 <= args.product_scale <= 1.0:
        print("error: --product-scale must be between 0.1 and 1.0", file=sys.stderr)
        return 2
    try:
        result = stage(args.input, args.prompt, args.output, args.cutout_model, args.scene_model,
                       args.device, args.seed, args.product_scale, args.cutout_edge)
    except Exception as exc:  # noqa: BLE001 - friendly CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"staged: {result['output']}")
    print(f"cutout: {result['cutout']} ({result['cutout_backend']})")
    print(f"mask: {result['mask']}")
    print(f"comparison: {result['comparison']}")
    print(f"scene: {result['scene_backend']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
