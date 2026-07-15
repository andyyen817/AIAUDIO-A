import hashlib
import importlib
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


def tiny_wav() -> bytes:
    pcm = b"\x00\x00" * 160
    data_size = len(pcm)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data"
        + struct.pack("<I", data_size)
        + pcm
    )


class FakeOssBucket:
    def __init__(self) -> None:
        self.objects: dict[str, int] = {}

    def sign_url(self, method: str, key: str, expires: int, headers: dict | None = None) -> str:
        return f"https://oss.example.test/{key}?method={method}&expires={expires}"

    def get_object_meta(self, key: str):
        if key not in self.objects:
            raise RuntimeError("object not found")
        return SimpleNamespace(headers={"Content-Length": str(self.objects[key])})


class UploadApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {
                "INCOMING_DIR": str(Path(self.temp_dir.name) / "incoming"),
                "RECORDINGS_DIR": str(Path(self.temp_dir.name) / "recordings"),
                "DATABASE_REQUIRED": "false",
                "UPLOAD_TOKEN": "test-token",
            },
            clear=False,
        )
        self.env.start()
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("POSTGRES_URL", None)
        import server.main

        self.module = importlib.reload(server.main)
        self.module.on_startup()
        self.client = TestClient(self.module.app)

    def tearDown(self) -> None:
        self.client.close()
        self.env.stop()
        self.temp_dir.cleanup()

    def upload(self, audio: bytes, sha256: str | None = None):
        digest = sha256 or hashlib.sha256(audio).hexdigest()
        return self.client.post(
            "/api/v1/recordings",
            headers={"Authorization": "Bearer test-token"},
            data={
                "original_name": "20260710120000.wav",
                "source": "local",
                "recording_time": "2026-07-10T12:00:00+08:00",
                "duration_ms": "10",
                "audio_sha256": digest,
                "app_version": "0.2.0",
                "device_id": "test-device",
            },
            files={
                "audio_file": ("20260710120000.wav", audio, "audio/wav"),
                "transcript_file": ("20260710120000.txt", "测试文本\n".encode(), "text/plain"),
            },
        )

    def test_upload_and_duplicate_are_idempotent(self) -> None:
        audio = tiny_wav()
        first = self.upload(audio)
        second = self.upload(audio)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["recording_id"], second.json()["recording_id"])
        self.assertEqual(len(list(Path(self.temp_dir.name).rglob("*.wav"))), 1)
        self.assertEqual(len(list(Path(self.temp_dir.name).rglob("*.txt"))), 1)

    def test_invalid_hash_is_rejected_and_parts_are_removed(self) -> None:
        response = self.upload(tiny_wav(), sha256="0" * 64)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(list(Path(self.temp_dir.name).rglob("*.part")))

    def test_missing_token_is_rejected(self) -> None:
        response = self.client.post("/api/v1/recordings")
        self.assertEqual(response.status_code, 403)

    def test_airec_endpoint_remains_compatible(self) -> None:
        response = self.client.post(
            "/api/airec/upload",
            data={"fileName": "20260710120000.wav", "sn": "DEVICE_1"},
            files={"file": ("20260710120000.wav", tiny_wav(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.text, "ok\n")

    def test_direct_upload_init_and_complete(self) -> None:
        bucket = FakeOssBucket()
        audio = tiny_wav()
        transcript = "direct upload transcript\n".encode()
        payload = {
            "original_name": "20260710120000.wav",
            "source": "local",
            "recording_time": "2026-07-10T12:00:00+08:00",
            "duration_ms": 10,
            "audio_sha256": hashlib.sha256(audio).hexdigest(),
            "app_version": "0.2.0",
            "device_id": "test-device",
            "audio_size_bytes": len(audio),
            "transcript_size_bytes": len(transcript),
        }

        with patch.dict(os.environ, {"ALIYUN_OSS_BUCKET": "lightasr"}, clear=False), \
                patch.object(self.module, "oss_enabled", return_value=True), \
                patch.object(self.module, "oss_bucket", return_value=bucket):
            self.module.oss_status = "ready"
            init_response = self.client.post(
                "/api/v1/recordings/direct-upload/init",
                headers={"Authorization": "Bearer test-token"},
                json=payload,
            )
            self.assertEqual(init_response.status_code, 200, init_response.text)
            init_json = init_response.json()
            self.assertEqual(init_json["upload_status"], "pending_direct_upload")
            self.assertEqual(init_json["audio"]["method"], "PUT")
            self.assertIn("Content-Type", init_json["audio"]["headers"])
            bucket.objects[init_json["audio"]["oss_key"]] = len(audio)
            bucket.objects[init_json["transcript"]["oss_key"]] = len(transcript)

            complete_response = self.client.post(
                "/api/v1/recordings/direct-upload/complete",
                headers={"Authorization": "Bearer test-token"},
                json={**payload, "recording_id": init_json["recording_id"]},
            )
            self.assertEqual(complete_response.status_code, 200, complete_response.text)
            self.assertEqual(complete_response.json()["recording_id"], init_json["recording_id"])


if __name__ == "__main__":
    unittest.main()
