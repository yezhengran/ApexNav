#!/usr/bin/env python3
"""Exercise all four VLM HTTP services with a small synthetic image."""

import base64
import argparse
import time

import cv2
import numpy as np
import requests

from vlm.server_wrapper import MODEL_ENDPOINTS, api_headers, model_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--service", choices=list(MODEL_ENDPOINTS))
    args = parser.parse_args()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(image, (48, 48), (176, 176), (200, 120, 80), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    encoded_image = base64.b64encode(encoded).decode("ascii")
    requests_to_make = [
        ("blip2", 12182, "blip2itm", {"txt": "a colored square"}),
        ("yolo", 12184, "yolov7", {
            "agnostic_nms": True, "conf_thres": 0.3, "iou_thres": 0.5,
        }),
        ("sam", 12183, "mobile_sam", {"bbox": [48, 48, 176, 176]}),
        ("dino", 12181, "gdino", {
            "caption": "chair .", "box_threshold": 0.4, "text_threshold": 0.25,
        }),
    ]
    session = requests.Session()
    session.trust_env = False
    session.headers.update(api_headers())
    for name, port, route, extra in requests_to_make:
        if args.service and name != args.service:
            continue
        endpoint = model_url(name)
        health_url = endpoint.rsplit("/", 1)[0] + "/healthz"
        health = session.get(health_url, timeout=(5, 5))
        health.raise_for_status()
        assert health.json() == {"status": "ready", "model": route}
        if args.health_only:
            print(f"[OK] {name} API ready", flush=True)
            continue
        started = time.monotonic()
        response = session.post(
            endpoint,
            json={"image": encoded_image, **extra}, timeout=(5, 120),
        )
        response.raise_for_status()
        payload = response.json()
        if name == "blip2":
            assert np.isfinite(payload["response"])
            assert 0 <= payload["itm score"] <= 1
        elif name == "sam":
            assert len(base64.b64decode(payload["cropped_mask"])) == 224 * 224
        else:
            assert len(payload["boxes"]) == len(payload["logits"]) == len(payload["phrases"])
        print(f"[OK] {name} HTTP inference ({time.monotonic() - started:.2f}s)", flush=True)
    print("APEXNAV_VLM_HEALTH_OK" if args.health_only else "APEXNAV_VLM_HTTP_OK")


if __name__ == "__main__":
    main()
