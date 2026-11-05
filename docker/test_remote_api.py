#!/usr/bin/env python3
"""Protocol tests using a local mock HTTP server; no model or GPU is loaded."""

import base64
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import numpy as np
import requests

from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.detector.yolov7 import YOLOv7Client
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.segmentor.sam import MobileSAMClient
from vlm.Labels import detector_category
from vlm.server_wrapper import (
    host_model, model_url, release_idle_cuda_cache, send_request, str_to_image,
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.headers.get("Authorization") != "Bearer test-token":
            self.send_response(401)
            self.end_headers()
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "image" in payload:
            image = str_to_image(payload["image"])
            assert image.shape == (32, 32, 3)
        route = self.path.rsplit("/", 1)[-1]
        if route == "blip2itm":
            response = {"response": 0.5, "itm score": 0.75}
        elif route == "mobile_sam":
            mask = np.ones((32, 32), dtype=bool)
            response = {"cropped_mask": base64.b64encode(mask).decode("ascii")}
        elif route == "slow":
            time.sleep(1.2)
            response = {"response": "slow-ok"}
        else:
            response = {"boxes": [[0.1, 0.1, 0.9, 0.9]], "logits": [0.8], "phrases": ["chair"]}
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RemoteAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/proxy"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "APEXNAV_GDINO_URL": self.base + "/gdino",
            "APEXNAV_YOLO_URL": self.base + "/yolov7",
            "APEXNAV_SAM_URL": self.base + "/mobile_sam",
            "APEXNAV_BLIP2_URL": self.base + "/blip2itm",
            "APEXNAV_VLM_API_KEY": "test-token",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)

    def test_clients_use_configured_urls_and_protocol(self):
        dino = GroundingDINOClient()
        yolo = YOLOv7Client()
        sam = MobileSAMClient()
        blip2 = BLIP2ITMClient()
        self.assertEqual(dino.url, self.base + "/gdino")
        self.assertEqual(dino.predict(self.image).phrases, ["chair"])
        self.assertEqual(yolo.predict(self.image).phrases, ["chair"])
        self.assertEqual(sam.segment_bbox(self.image, [0, 0, 31, 31]).shape, (32, 32))
        self.assertEqual(blip2.cosine(self.image, "chair"), 0.5)
        self.assertEqual(blip2.itm_score(self.image, "chair"), 0.75)

    def test_explicit_url_has_priority(self):
        self.assertEqual(model_url("sam", url=self.base + "/custom/"), self.base + "/custom")

    def test_shared_remote_host_and_legacy_port(self):
        with patch.dict(os.environ, {"APEXNAV_VLM_HOST": "gpu.example"}):
            os.environ.pop("APEXNAV_SAM_URL")
            self.assertEqual(model_url("sam", port=13000), "http://gpu.example:13000/mobile_sam")

    def test_server_health_and_bearer_authentication(self):
        class EchoModel:
            def process_payload(self, payload):
                return payload

        with patch.dict(os.environ, {"APEXNAV_VLM_SERVER_TOKEN": "test-token"}):
            with patch("flask.Flask.run", autospec=True) as run:
                host_model(EchoModel(), "echo", port=13000)
                app = run.call_args.args[0]
            client = app.test_client()
            self.assertEqual(client.get("/healthz").status_code, 401)
            headers = {"Authorization": "Bearer test-token"}
            health = client.get("/healthz", headers=headers)
            self.assertEqual(health.json, {"status": "ready", "model": "echo"})
            self.assertEqual(client.post("/echo", json={"ok": True}, headers=headers).json, {"ok": True})

    def test_remote_inference_can_take_longer_than_one_second(self):
        self.assertEqual(send_request(self.base + "/slow")["response"], "slow-ok")

    def test_authentication_error_is_reported(self):
        with patch.dict(os.environ, {"APEXNAV_VLM_API_KEY": "wrong"}):
            with self.assertRaises(requests.HTTPError):
                send_request(self.base + "/gdino", image=self.image)

    def test_hm3d_labels_do_not_require_mp3d_files(self):
        self.assertEqual(detector_category("sofa"), "couch")
        self.assertEqual(detector_category("plant"), "potted plant")
        self.assertEqual(detector_category("tv_monitor"), "tv")

    def test_cuda_cache_release_requires_opt_in_and_initialized_context(self):
        fake_torch = Mock()
        fake_torch.cuda.is_initialized.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch.dict(os.environ, {"APEXNAV_VLM_RELEASE_IDLE_CACHE": "0"}):
                release_idle_cuda_cache()
                fake_torch.cuda.empty_cache.assert_not_called()
            with patch.dict(os.environ, {"APEXNAV_VLM_RELEASE_IDLE_CACHE": "1"}):
                release_idle_cuda_cache()
                fake_torch.cuda.empty_cache.assert_called_once_with()
                fake_torch.cuda.empty_cache.reset_mock()
                fake_torch.cuda.is_initialized.return_value = False
                release_idle_cuda_cache()
                fake_torch.cuda.empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
