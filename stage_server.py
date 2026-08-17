#!/usr/bin/env python3
"""Tiny HTTP adapter for stage.py, suitable for a single-user MVP service."""

from __future__ import annotations

import base64
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any

from PIL import Image

from stage import composite_product, comparison_image, cutout_product, generate_scene, _scene_size


MAX_IMAGE_BYTES = 12 * 1024 * 1024
STAGING_AUTH_TOKEN = os.getenv("STAGING_AUTH_TOKEN", "").strip()


def encode_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run_stage(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = payload.get("image")
    prompt = payload.get("prompt")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image is required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image is too large; maximum is 12 MB")
    original = Image.open(BytesIO(raw)).convert("RGBA")
    if max(original.size) > 4096:
        original.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
    seed = int(payload.get("seed", 7))
    scale = float(payload.get("productScale", 0.52))
    if not 0.1 <= scale <= 1.0:
        raise ValueError("productScale must be between 0.1 and 1.0")
    cutout, binary_mask, cutout_backend = cutout_product(original, str(payload.get("cutoutModel", "u2net")))
    scene, scene_backend = generate_scene(
        prompt.strip(), _scene_size(original), str(payload.get("sceneModel", os.getenv("STAGING_SCENE_MODEL", "stabilityai/sd-turbo"))),
        str(payload.get("device", "auto")), seed,
    )
    staged = composite_product(scene, cutout, binary_mask, prompt, scale)
    return {
        "staged": encode_png(staged),
        "cutout": encode_png(cutout),
        "mask": encode_png(binary_mask),
        "comparison": encode_png(comparison_image(original, staged)),
        "cutoutBackend": cutout_backend,
        "sceneBackend": scene_backend,
    }


class StageHandler(BaseHTTPRequestHandler):
    server_version = "ProductStaging/0.1"

    def authorized(self) -> bool:
        """Require a shared bearer token when this service is deployed publicly."""
        if not STAGING_AUTH_TOKEN:
            return True
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {STAGING_AUTH_TOKEN}"
        return hmac.compare_digest(supplied, expected)

    def send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "service": "product-staging-python"})
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/stage":
            self.send_json(404, {"error": "not_found"})
            return
        if not self.authorized():
            self.send_json(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_IMAGE_BYTES * 2:
                raise ValueError("request is too large")
            payload = json.loads(self.rfile.read(length))
            result = run_stage(payload)
            self.send_json(200, result)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - HTTP boundary
            print(f"[stage] failed: {exc}")
            self.send_json(500, {"error": "staging_failed"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[stage] {format % args}")


if __name__ == "__main__":
    host = os.getenv("STAGE_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8787"))
    print(f"Product staging backend listening on {host}:{port}")
    ThreadingHTTPServer((host, port), StageHandler).serve_forever()
