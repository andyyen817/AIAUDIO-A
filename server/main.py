import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("lightasr.upload")

FILE_NAME_RE = re.compile(r"^\d{14}\.wav$", re.IGNORECASE)
SN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
CHUNK_SIZE = 1024 * 1024

app = FastAPI(title="LightASR AIREC Upload Receiver", version="0.1.0")


def incoming_root() -> Path:
    return Path(os.getenv("INCOMING_DIR", "/data/lightasr/incoming")).resolve()


def database_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")


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


def ensure_inside_root(root: Path, candidate: Path) -> None:
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid save path") from exc


def init_database() -> None:
    url = database_url()
    if not url:
        logger.info("database disabled: DATABASE_URL is not configured")
        return

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
        conn.commit()
    logger.info("database ready")


def record_upload(sn: str, file_name: str, saved_path: Path, size_bytes: int) -> None:
    url = database_url()
    if not url:
        return

    try:
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO airec_uploads
                        (sn, file_name, saved_path, size_bytes, status, upload_time)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sn, file_name) DO NOTHING
                    """,
                    (
                        sn,
                        file_name,
                        str(saved_path),
                        size_bytes,
                        "uploaded",
                        datetime.now(timezone.utc),
                    ),
                )
            conn.commit()
    except Exception as exc:
        logger.exception("database record failed: sn=%s fileName=%s error=%s", sn, file_name, exc)


@app.on_event("startup")
def on_startup() -> None:
    incoming_root().mkdir(parents=True, exist_ok=True)
    init_database()
    logger.info("LightASR upload receiver started incomingDir=%s", incoming_root())


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
        size_bytes = final_path.stat().st_size
        logger.info(
            "duplicate ignored: sn=%s fileName=%s sizeBytes=%s savedPath=%s",
            device_sn,
            file_name,
            size_bytes,
            final_path,
        )
        return "ok\n"

    if part_path.exists():
        part_path.unlink()

    try:
        header = await file.read(12)
        if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise HTTPException(status_code=400, detail="file is not a RIFF/WAVE wav")

        size_bytes = 0
        with part_path.open("wb") as out:
            out.write(header)
            size_bytes += len(header)
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                size_bytes += len(chunk)

        if size_bytes <= 44:
            part_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="wav file is empty")

        part_path.replace(final_path)
        record_upload(device_sn, file_name, final_path, size_bytes)

        logger.info(
            "upload success: sn=%s fileName=%s sizeBytes=%s savedPath=%s",
            device_sn,
            file_name,
            size_bytes,
            final_path,
        )
        return "ok\n"
    except HTTPException:
        raise
    except Exception as exc:
        part_path.unlink(missing_ok=True)
        logger.exception("save failed: sn=%s fileName=%s error=%s", device_sn, file_name, exc)
        raise HTTPException(status_code=500, detail="file save failed") from exc
