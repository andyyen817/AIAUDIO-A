import hashlib
import logging
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import oss2
except ImportError:  # OSS is optional for local file-only tests.
    oss2 = None

try:
    import psycopg
except ImportError:  # Local file-only tests do not require PostgreSQL.
    psycopg = None

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("lightasr.upload")

FILE_NAME_RE = re.compile(r"^\d{14}\.wav$", re.IGNORECASE)
SN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_AUDIO_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_TRANSCRIPT_BYTES = 10 * 1024 * 1024
DEFAULT_SIGNED_URL_EXPIRES_SECONDS = 15 * 60

app = FastAPI(title="LightASR Upload Receiver", version="0.2.0")
database_status = "disabled"
database_error: Optional[str] = None
oss_status = "disabled"
oss_error: Optional[str] = None


class DirectUploadInitRequest(BaseModel):
    original_name: str
    source: str
    recording_time: Optional[str] = None
    duration_ms: Optional[int] = None
    audio_sha256: str
    app_version: Optional[str] = None
    device_id: str
    audio_size_bytes: int
    transcript_size_bytes: int


class DirectUploadCompleteRequest(DirectUploadInitRequest):
    recording_id: str


def incoming_root() -> Path:
    return Path(os.getenv("INCOMING_DIR", "/data/lightasr/incoming")).resolve()


def analyzed_recordings_root() -> Path:
    configured = os.getenv("RECORDINGS_DIR")
    if configured:
        return Path(configured).resolve()
    return (incoming_root().parent / "recordings").resolve()


def database_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")


def database_required() -> bool:
    return os.getenv("DATABASE_REQUIRED", "false").lower() in {"1", "true", "yes"}


def oss_enabled() -> bool:
    return bool(os.getenv("ALIYUN_OSS_BUCKET", "").strip())


def oss_required() -> bool:
    return os.getenv("OSS_REQUIRED", "false").lower() in {"1", "true", "yes"}


def oss_prefix() -> str:
    return os.getenv("ALIYUN_OSS_PREFIX", "lightasr").strip().strip("/")


def oss_bucket():
    if not oss_enabled():
        return None
    if oss2 is None:
        raise RuntimeError("ALIYUN_OSS_BUCKET is configured but oss2 is not installed")
    endpoint = os.getenv("ALIYUN_OSS_ENDPOINT", "").strip()
    bucket_name = os.getenv("ALIYUN_OSS_BUCKET", "").strip()
    access_key_id = os.getenv("ALIYUN_OSS_ACCESS_KEY_ID", "").strip()
    access_key_secret = os.getenv("ALIYUN_OSS_ACCESS_KEY_SECRET", "").strip()
    if not endpoint or not bucket_name or not access_key_id or not access_key_secret:
        raise RuntimeError(
            "ALIYUN_OSS_ENDPOINT, ALIYUN_OSS_BUCKET, "
            "ALIYUN_OSS_ACCESS_KEY_ID, and ALIYUN_OSS_ACCESS_KEY_SECRET are required"
        )
    auth = oss2.Auth(access_key_id, access_key_secret)
    return oss2.Bucket(auth, endpoint, bucket_name)


def build_oss_key(*parts: str) -> str:
    clean_parts = [part.strip("/").replace("\\", "/") for part in parts if part.strip("/")]
    prefix = oss_prefix()
    if prefix:
        clean_parts.insert(0, prefix)
    return "/".join(clean_parts)


def init_oss() -> None:
    global oss_status, oss_error
    if not oss_enabled():
        oss_status = "disabled"
        oss_error = None
        logger.info("oss disabled: ALIYUN_OSS_BUCKET is not configured")
        return
    try:
        bucket = oss_bucket()
        assert bucket is not None
        bucket.get_bucket_info()
        oss_status = "ready"
        oss_error = None
        logger.info("oss ready bucket=%s prefix=%s", os.getenv("ALIYUN_OSS_BUCKET"), oss_prefix())
    except Exception as exc:
        oss_status = "error"
        oss_error = str(exc)
        logger.exception("oss initialization failed: %s", exc)
        if oss_required():
            raise


def max_audio_bytes() -> int:
    return int(os.getenv("MAX_AUDIO_BYTES", str(DEFAULT_MAX_AUDIO_BYTES)))


def max_transcript_bytes() -> int:
    return int(os.getenv("MAX_TRANSCRIPT_BYTES", str(DEFAULT_MAX_TRANSCRIPT_BYTES)))


def signed_url_expires_seconds() -> int:
    return int(os.getenv("OSS_SIGNED_URL_EXPIRES_SECONDS", str(DEFAULT_SIGNED_URL_EXPIRES_SECONDS)))


def is_allowed_sn(sn: str) -> bool:
    allow_all = os.getenv("ALLOW_ALL_SN", "true").lower() in {"1", "true", "yes"}
    if allow_all:
        return True
    allowed = {
        item.strip()
        for item in os.getenv("AUTHORIZED_SN", "").split(",")
        if item.strip()
    }
    return sn in allowed


def validate_upload_token(authorization: Optional[str]) -> None:
    configured_token = os.getenv("UPLOAD_TOKEN", "").strip()
    if not configured_token:
        logger.warning("UPLOAD_TOKEN is not configured; Android upload API is unauthenticated")
        return
    scheme, _, supplied_token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(supplied_token, configured_token):
        raise HTTPException(status_code=403, detail="invalid upload token")


def validate_upload_fields(file_name: Optional[str], sn: Optional[str]) -> tuple[str, str]:
    if not file_name:
        raise HTTPException(status_code=400, detail="fileName is required")
    if not sn:
        raise HTTPException(status_code=400, detail="sn is required")
    if not FILE_NAME_RE.fullmatch(file_name):
        raise HTTPException(status_code=400, detail="fileName must match yyyyMMddHHmmss.wav")
    if not SN_RE.fullmatch(sn):
        raise HTTPException(status_code=400, detail="sn contains invalid characters")
    if not is_allowed_sn(sn):
        raise HTTPException(status_code=403, detail="sn is not authorized")
    return file_name, sn


def validate_recording_fields(
    original_name: Optional[str],
    device_id: Optional[str],
    audio_sha256: Optional[str],
    source: Optional[str],
) -> tuple[str, str, str, str]:
    name = Path(original_name or "").name
    if not name or name != original_name or not name.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="original_name must be a safe .wav file name")
    if not device_id or not DEVICE_ID_RE.fullmatch(device_id):
        raise HTTPException(status_code=400, detail="device_id contains invalid characters")
    if not audio_sha256 or not SHA256_RE.fullmatch(audio_sha256):
        raise HTTPException(status_code=400, detail="audio_sha256 must contain 64 hex characters")
    normalized_source = (source or "").lower()
    if normalized_source not in {"local", "airec", "shared"}:
        raise HTTPException(status_code=400, detail="source must be local, airec, or shared")
    return name, device_id, audio_sha256.lower(), normalized_source


def ensure_inside_root(root: Path, candidate: Path) -> None:
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid save path") from exc


def require_oss_bucket():
    if not oss_enabled():
        raise HTTPException(status_code=503, detail="oss is not configured")
    if oss_status != "ready":
        raise HTTPException(status_code=503, detail=f"oss is not ready: {oss_error or oss_status}")
    bucket = oss_bucket()
    assert bucket is not None
    return bucket


def validate_recording_sizes(audio_size_bytes: int, transcript_size_bytes: int) -> None:
    if audio_size_bytes <= 44:
        raise HTTPException(status_code=400, detail="audio_size_bytes must be a non-empty wav size")
    if audio_size_bytes > max_audio_bytes():
        raise HTTPException(status_code=413, detail="audio file is too large")
    if transcript_size_bytes <= 0:
        raise HTTPException(status_code=400, detail="transcript_size_bytes must be positive")
    if transcript_size_bytes > max_transcript_bytes():
        raise HTTPException(status_code=413, detail="transcript file is too large")


def recording_storage(
    safe_name: str,
    safe_device_id: str,
    recording_id: str,
) -> tuple[Path, Path, str, str, str, str]:
    root = analyzed_recordings_root()
    target_dir = (root / safe_device_id / recording_id).resolve()
    audio_path = (target_dir / safe_name).resolve()
    transcript_name = f"{Path(safe_name).stem}.txt"
    transcript_path = (target_dir / transcript_name).resolve()
    for path in (target_dir, audio_path, transcript_path):
        ensure_inside_root(root, path)
    audio_oss_key = build_oss_key("recordings", safe_device_id, recording_id, safe_name)
    transcript_oss_key = build_oss_key("recordings", safe_device_id, recording_id, transcript_name)
    bucket_name = os.getenv("ALIYUN_OSS_BUCKET", "").strip()
    audio_storage_path = f"oss://{bucket_name}/{audio_oss_key}" if oss_enabled() else str(audio_path)
    transcript_storage_path = (
        f"oss://{bucket_name}/{transcript_oss_key}" if oss_enabled() else str(transcript_path)
    )
    return audio_path, transcript_path, audio_oss_key, transcript_oss_key, audio_storage_path, transcript_storage_path


def signed_put_target(bucket, oss_key: str, content_type: str) -> dict:
    headers = {"Content-Type": content_type}
    url = bucket.sign_url("PUT", oss_key, signed_url_expires_seconds(), headers=headers)
    return {
        "method": "PUT",
        "url": url,
        "oss_key": oss_key,
        "content_type": content_type,
        "headers": headers,
    }


def oss_object_size(bucket, oss_key: str) -> Optional[int]:
    meta = bucket.get_object_meta(oss_key)
    headers = getattr(meta, "headers", {}) or {}
    value = None
    for key in ("Content-Length", "content-length"):
        if key in headers:
            value = headers[key]
            break
    if value is None and hasattr(meta, "content_length"):
        value = getattr(meta, "content_length")
    return int(value) if value is not None else None


def ensure_oss_object(bucket, oss_key: str, expected_size: int, label: str) -> None:
    try:
        actual_size = oss_object_size(bucket, oss_key)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"{label} is not available in oss") from exc
    if actual_size is not None and actual_size != expected_size:
        raise HTTPException(status_code=409, detail=f"{label} size does not match uploaded object")


def require_database_driver() -> None:
    if database_url() and psycopg is None:
        raise RuntimeError("DATABASE_URL is configured but psycopg is not installed")


def init_database() -> None:
    global database_status, database_error
    url = database_url()
    if not url:
        database_status = "disabled"
        database_error = None
        logger.info("database disabled: DATABASE_URL is not configured")
        return
    try:
        require_database_driver()
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS airec_uploads (
                        id BIGSERIAL PRIMARY KEY,
                        sn TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        saved_path TEXT NOT NULL,
                        size_bytes BIGINT NOT NULL,
                        status TEXT NOT NULL,
                        upload_time TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE (sn, file_name)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lightasr_recordings (
                        recording_id TEXT PRIMARY KEY,
                        device_id TEXT NOT NULL,
                        audio_sha256 TEXT NOT NULL,
                        original_name TEXT NOT NULL,
                        source TEXT NOT NULL,
                        recording_time TIMESTAMPTZ,
                        duration_ms BIGINT,
                        app_version TEXT,
                        audio_path TEXT NOT NULL,
                        transcript_path TEXT NOT NULL,
                        audio_size_bytes BIGINT NOT NULL,
                        transcript_size_bytes BIGINT NOT NULL,
                        upload_status TEXT NOT NULL,
                        analysis_status TEXT NOT NULL DEFAULT 'not_requested',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE (device_id, audio_sha256)
                    )
                    """
                )
            conn.commit()
        database_status = "ready"
        database_error = None
        logger.info("database ready")
    except Exception as exc:
        database_status = "error"
        database_error = str(exc)
        logger.exception("database initialization failed: %s", exc)
        if database_required():
            raise


def record_airec_upload(sn: str, file_name: str, saved_path: Path, size_bytes: int) -> None:
    url = database_url()
    if not url or database_status != "ready":
        return
    try:
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO airec_uploads
                        (sn, file_name, saved_path, size_bytes, status, upload_time)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sn, file_name) DO NOTHING
                    """,
                    (sn, file_name, str(saved_path), size_bytes, "uploaded", datetime.now(timezone.utc)),
                )
            conn.commit()
    except Exception as exc:
        logger.exception("database record failed: sn=%s fileName=%s error=%s", sn, file_name, exc)


def upsert_recording(
    *,
    recording_id: str,
    device_id: str,
    audio_sha256: str,
    original_name: str,
    source: str,
    recording_time: Optional[datetime],
    duration_ms: Optional[int],
    app_version: Optional[str],
    audio_path: str | Path,
    transcript_path: str | Path,
    audio_size_bytes: int,
    transcript_size_bytes: int,
) -> None:
    url = database_url()
    if not url:
        if database_required():
            raise RuntimeError("DATABASE_URL is required")
        return
    if database_status != "ready":
        raise RuntimeError(f"database is not ready: {database_error or database_status}")
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lightasr_recordings (
                    recording_id, device_id, audio_sha256, original_name, source,
                    recording_time, duration_ms, app_version, audio_path, transcript_path,
                    audio_size_bytes, transcript_size_bytes, upload_status, analysis_status,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, 'uploaded', 'not_requested', now(), now()
                )
                ON CONFLICT (device_id, audio_sha256) DO UPDATE SET
                    audio_path = EXCLUDED.audio_path,
                    transcript_path = EXCLUDED.transcript_path,
                    audio_size_bytes = EXCLUDED.audio_size_bytes,
                    transcript_size_bytes = EXCLUDED.transcript_size_bytes,
                    upload_status = 'uploaded',
                    updated_at = now()
                """,
                (
                    recording_id, device_id, audio_sha256, original_name, source,
                    recording_time, duration_ms, app_version, str(audio_path), str(transcript_path),
                    audio_size_bytes, transcript_size_bytes,
                ),
            )
        conn.commit()


def upload_file_to_oss(local_path: Path, oss_key: str, content_type: str) -> None:
    if not oss_enabled():
        return
    if oss_status != "ready":
        raise RuntimeError(f"oss is not ready: {oss_error or oss_status}")
    bucket = oss_bucket()
    assert bucket is not None
    headers = {"Content-Type": content_type}
    bucket.put_object_from_file(oss_key, str(local_path), headers=headers)


def recording_id_for(device_id: str, audio_sha256: str) -> str:
    digest = hashlib.sha256(f"{device_id}:{audio_sha256}".encode("utf-8")).hexdigest()
    return f"rec_{digest[:32]}"


def parse_recording_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="recording_time must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise HTTPException(status_code=400, detail="recording_time must include timezone")
    return parsed


async def stream_upload_to_file(
    upload: UploadFile,
    target: Path,
    limit_bytes: int,
    hash_audio: bool = False,
) -> tuple[int, Optional[str]]:
    size_bytes = 0
    digest = hashlib.sha256() if hash_audio else None
    with target.open("wb") as output:
        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > limit_bytes:
                raise HTTPException(status_code=413, detail="uploaded file is too large")
            output.write(chunk)
            if digest is not None:
                digest.update(chunk)
        output.flush()
        os.fsync(output.fileno())
    return size_bytes, digest.hexdigest() if digest is not None else None


def validate_wav_file(path: Path) -> None:
    with path.open("rb") as input_file:
        header = input_file.read(12)
    if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise HTTPException(status_code=400, detail="file is not a RIFF/WAVE wav")
    if path.stat().st_size <= 44:
        raise HTTPException(status_code=400, detail="wav file is empty")


@app.on_event("startup")
def on_startup() -> None:
    incoming_root().mkdir(parents=True, exist_ok=True)
    analyzed_recordings_root().mkdir(parents=True, exist_ok=True)
    init_database()
    init_oss()
    logger.info(
        "LightASR upload receiver started incomingDir=%s recordingsDir=%s oss=%s",
        incoming_root(), analyzed_recordings_root(), oss_status,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("validation failed path=%s error=%s", request.url.path, exc)
    return PlainTextResponse("bad request", status_code=400)


@app.get("/", response_class=PlainTextResponse)
def index() -> str:
    return "LightASR upload receiver is running\n"


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok\n"


@app.get("/ready")
def ready() -> JSONResponse:
    ready_state = database_status != "error" and (
        not database_required() or database_status == "ready"
    ) and oss_status != "error" and (not oss_required() or oss_status == "ready")
    return JSONResponse(
        {
            "ready": ready_state,
            "database": database_status,
            "oss": oss_status,
            "storage": str(analyzed_recordings_root()),
        },
        status_code=200 if ready_state else 503,
    )


@app.post("/api/airec/upload", response_class=PlainTextResponse)
async def upload_airec(
    file: Optional[UploadFile] = File(default=None),
    fileName: Optional[str] = Form(default=None),
    sn: Optional[str] = Form(default=None),
) -> str:
    if file is None:
        raise HTTPException(status_code=400, detail="file is required")
    file_name, device_sn = validate_upload_fields(fileName, sn)
    logger.info("upload start: sn=%s fileName=%s", device_sn, file_name)
    root = incoming_root()
    device_dir = (root / device_sn).resolve()
    final_path = (device_dir / file_name).resolve()
    part_path = (device_dir / f"{file_name}.part").resolve()
    ensure_inside_root(root, final_path)
    ensure_inside_root(root, part_path)
    device_dir.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        logger.info("duplicate ignored: sn=%s fileName=%s", device_sn, file_name)
        return "ok\n"
    part_path.unlink(missing_ok=True)
    try:
        size_bytes, _ = await stream_upload_to_file(file, part_path, max_audio_bytes())
        validate_wav_file(part_path)
        part_path.replace(final_path)
        record_airec_upload(device_sn, file_name, final_path, size_bytes)
        logger.info(
            "upload success: sn=%s fileName=%s sizeBytes=%s savedPath=%s",
            device_sn, file_name, size_bytes, final_path,
        )
        return "ok\n"
    except HTTPException:
        part_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        part_path.unlink(missing_ok=True)
        logger.exception("save failed: sn=%s fileName=%s error=%s", device_sn, file_name, exc)
        raise HTTPException(status_code=500, detail="file save failed") from exc


@app.post("/api/v1/recordings")
async def upload_analyzed_recording(
    audio_file: Optional[UploadFile] = File(default=None),
    transcript_file: Optional[UploadFile] = File(default=None),
    original_name: Optional[str] = Form(default=None),
    source: Optional[str] = Form(default=None),
    recording_time: Optional[str] = Form(default=None),
    duration_ms: Optional[int] = Form(default=None),
    audio_sha256: Optional[str] = Form(default=None),
    app_version: Optional[str] = Form(default=None),
    device_id: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
) -> dict[str, str]:
    validate_upload_token(authorization)
    if audio_file is None:
        raise HTTPException(status_code=400, detail="audio_file is required")
    if transcript_file is None:
        raise HTTPException(status_code=400, detail="transcript_file is required")
    safe_name, safe_device_id, expected_sha256, safe_source = validate_recording_fields(
        original_name, device_id, audio_sha256, source,
    )
    parsed_recording_time = parse_recording_time(recording_time)
    if duration_ms is not None and duration_ms < 0:
        raise HTTPException(status_code=400, detail="duration_ms must be non-negative")

    recording_id = recording_id_for(safe_device_id, expected_sha256)
    root = analyzed_recordings_root()
    (
        audio_path,
        transcript_path,
        audio_oss_key,
        transcript_oss_key,
        audio_storage_path,
        transcript_storage_path,
    ) = recording_storage(
        safe_name=safe_name,
        safe_device_id=safe_device_id,
        recording_id=recording_id,
    )
    target_dir = audio_path.parent
    audio_part = Path(f"{audio_path}.part")
    transcript_part = Path(f"{transcript_path}.part")
    for path in (audio_part, transcript_part):
        ensure_inside_root(root, path)
    target_dir.mkdir(parents=True, exist_ok=True)

    if audio_path.exists() and transcript_path.exists():
        upsert_recording(
            recording_id=recording_id, device_id=safe_device_id, audio_sha256=expected_sha256,
            original_name=safe_name, source=safe_source, recording_time=parsed_recording_time,
            duration_ms=duration_ms, app_version=app_version,
            audio_path=audio_storage_path, transcript_path=transcript_storage_path,
            audio_size_bytes=audio_path.stat().st_size,
            transcript_size_bytes=transcript_path.stat().st_size,
        )
        logger.info("recording duplicate ignored recordingId=%s", recording_id)
        return {"recording_id": recording_id, "upload_status": "uploaded"}

    audio_part.unlink(missing_ok=True)
    transcript_part.unlink(missing_ok=True)
    try:
        audio_size, actual_sha256 = await stream_upload_to_file(
            audio_file, audio_part, max_audio_bytes(), hash_audio=True,
        )
        transcript_size, _ = await stream_upload_to_file(
            transcript_file, transcript_part, max_transcript_bytes(),
        )
        validate_wav_file(audio_part)
        if actual_sha256 != expected_sha256:
            raise HTTPException(status_code=400, detail="audio_sha256 does not match uploaded audio")
        if transcript_size <= 0:
            raise HTTPException(status_code=400, detail="transcript_file is empty")
        transcript_part.read_text(encoding="utf-8")
        audio_part.replace(audio_path)
        transcript_part.replace(transcript_path)
        upload_file_to_oss(audio_path, audio_oss_key, "audio/wav")
        upload_file_to_oss(transcript_path, transcript_oss_key, "text/plain; charset=utf-8")
        upsert_recording(
            recording_id=recording_id, device_id=safe_device_id, audio_sha256=expected_sha256,
            original_name=safe_name, source=safe_source, recording_time=parsed_recording_time,
            duration_ms=duration_ms, app_version=app_version,
            audio_path=audio_storage_path, transcript_path=transcript_storage_path,
            audio_size_bytes=audio_size,
            transcript_size_bytes=transcript_size,
        )
        logger.info(
            "recording upload success recordingId=%s deviceId=%s audioBytes=%s transcriptBytes=%s oss=%s",
            recording_id, safe_device_id, audio_size, transcript_size, oss_enabled(),
        )
        return {"recording_id": recording_id, "upload_status": "uploaded"}
    except HTTPException:
        audio_part.unlink(missing_ok=True)
        transcript_part.unlink(missing_ok=True)
        raise
    except UnicodeDecodeError as exc:
        audio_part.unlink(missing_ok=True)
        transcript_part.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="transcript_file must be UTF-8 text") from exc
    except Exception as exc:
        audio_part.unlink(missing_ok=True)
        transcript_part.unlink(missing_ok=True)
        logger.exception("recording upload failed recordingId=%s error=%s", recording_id, exc)
        raise HTTPException(status_code=500, detail="recording upload failed") from exc


@app.post("/api/v1/recordings/direct-upload/init")
def init_direct_recording_upload(
    request: DirectUploadInitRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    validate_upload_token(authorization)
    bucket = require_oss_bucket()
    safe_name, safe_device_id, expected_sha256, safe_source = validate_recording_fields(
        request.original_name, request.device_id, request.audio_sha256, request.source,
    )
    validate_recording_sizes(request.audio_size_bytes, request.transcript_size_bytes)
    if request.duration_ms is not None and request.duration_ms < 0:
        raise HTTPException(status_code=400, detail="duration_ms must be non-negative")
    parse_recording_time(request.recording_time)

    recording_id = recording_id_for(safe_device_id, expected_sha256)
    _, _, audio_oss_key, transcript_oss_key, _, _ = recording_storage(
        safe_name=safe_name,
        safe_device_id=safe_device_id,
        recording_id=recording_id,
    )
    expires_in = signed_url_expires_seconds()
    logger.info(
        "direct upload initialized recordingId=%s deviceId=%s audioBytes=%s transcriptBytes=%s source=%s",
        recording_id, safe_device_id, request.audio_size_bytes, request.transcript_size_bytes, safe_source,
    )
    return {
        "recording_id": recording_id,
        "upload_status": "pending_direct_upload",
        "expires_in_seconds": expires_in,
        "audio": signed_put_target(bucket, audio_oss_key, "audio/wav"),
        "transcript": signed_put_target(bucket, transcript_oss_key, "text/plain; charset=utf-8"),
    }


@app.post("/api/v1/recordings/direct-upload/complete")
def complete_direct_recording_upload(
    request: DirectUploadCompleteRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, str]:
    validate_upload_token(authorization)
    bucket = require_oss_bucket()
    safe_name, safe_device_id, expected_sha256, safe_source = validate_recording_fields(
        request.original_name, request.device_id, request.audio_sha256, request.source,
    )
    validate_recording_sizes(request.audio_size_bytes, request.transcript_size_bytes)
    parsed_recording_time = parse_recording_time(request.recording_time)
    if request.duration_ms is not None and request.duration_ms < 0:
        raise HTTPException(status_code=400, detail="duration_ms must be non-negative")

    recording_id = recording_id_for(safe_device_id, expected_sha256)
    if request.recording_id != recording_id:
        raise HTTPException(status_code=400, detail="recording_id does not match upload metadata")
    (
        _audio_path,
        _transcript_path,
        audio_oss_key,
        transcript_oss_key,
        audio_storage_path,
        transcript_storage_path,
    ) = recording_storage(
        safe_name=safe_name,
        safe_device_id=safe_device_id,
        recording_id=recording_id,
    )
    ensure_oss_object(bucket, audio_oss_key, request.audio_size_bytes, "audio_file")
    ensure_oss_object(bucket, transcript_oss_key, request.transcript_size_bytes, "transcript_file")
    upsert_recording(
        recording_id=recording_id, device_id=safe_device_id, audio_sha256=expected_sha256,
        original_name=safe_name, source=safe_source, recording_time=parsed_recording_time,
        duration_ms=request.duration_ms, app_version=request.app_version,
        audio_path=audio_storage_path, transcript_path=transcript_storage_path,
        audio_size_bytes=request.audio_size_bytes, transcript_size_bytes=request.transcript_size_bytes,
    )
    logger.info(
        "direct recording upload complete recordingId=%s deviceId=%s audioBytes=%s transcriptBytes=%s",
        recording_id, safe_device_id, request.audio_size_bytes, request.transcript_size_bytes,
    )
    return {"recording_id": recording_id, "upload_status": "uploaded"}


@app.get("/api/v1/recordings/{recording_id}")
def get_recording(recording_id: str, authorization: Optional[str] = Header(default=None)) -> dict:
    validate_upload_token(authorization)
    if not re.fullmatch(r"rec_[a-f0-9]{32}", recording_id):
        raise HTTPException(status_code=404, detail="recording not found")
    root = analyzed_recordings_root()
    matches = list(root.glob(f"*/{recording_id}"))
    if not matches:
        raise HTTPException(status_code=404, detail="recording not found")
    target_dir = matches[0]
    audio_files = [path for path in target_dir.glob("*.wav") if path.is_file()]
    transcript_files = [path for path in target_dir.glob("*.txt") if path.is_file()]
    if not audio_files or not transcript_files:
        raise HTTPException(status_code=404, detail="recording not found")
    return {
        "recording_id": recording_id,
        "upload_status": "uploaded",
        "analysis_status": "not_requested",
        "audio_size_bytes": audio_files[0].stat().st_size,
        "transcript_size_bytes": transcript_files[0].stat().st_size,
    }
