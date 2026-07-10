# LightASR Android

LightASR is an Android proof-of-concept app for local audio transcription and
recording intake. It is focused on offline WAV transcription on Android devices.

## Current Features

- Local WAV file selection.
- Android share import for WAV audio from other apps.
- AIREC LAN receiving mode:
  - Embedded HTTP server in the Android app.
  - `POST /api/airec/upload`
  - multipart fields: `file`, `fileName`, `sn`
  - saves uploaded files under the app-specific `Incoming/{sn}/{fileName}` area.
- Local sherpa-onnx / Paraformer Chinese ASR.
- Long WAV streaming and chunked processing to avoid loading full recordings into memory.
- Silero VAD SpeechGate before ASR, used to skip no-speech chunks.
- 15-second target transcript segments with overlap de-duplication in the output layer.
- Absolute timestamp TXT output.
- AIREC-style file names such as `20260522120423.wav` are parsed as recording
  start time `2026-05-22 12:04:23.000`.
- Persistent transcription task history, checkpoint files, manual resume, and
  foreground notifications for long local jobs.
- Manual upload of completed WAV + TXT results to the LightASR server endpoint.
- Voiceprint MVP UI and local employee/sample management are present for testing.

## Not Yet Complete

- MP3/M4A/AAC decoding is not implemented; the current import path is WAV-first.
- Automatic cloud-side ASR and business analysis are not implemented in V1.
- Production-grade speaker diarization for 10-hour recordings is not complete.
- Voiceprint identification is still experimental and must be validated with a
  real Android-side speaker embedding runtime before production use.

## Project Layout

```text
app/                 Android application module
sherpa_onnx/         Local Android library module wrapping sherpa-onnx APIs
vendor/              Required sherpa-onnx Kotlin API sources
gradle/              Gradle wrapper and version catalog
scripts/             Local setup helpers
```

## Required Runtime Assets

The app currently includes required runtime assets under `app/src/main/assets/`:

- `sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx`
- `sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt`
- `models/vad/silero_vad.onnx`
- `models/voiceprint/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`
- `test.wav` as a tiny bundled test fixture

These assets are required for the current APK behavior and are intentionally
kept in source control unless the project later moves models to Git LFS,
release assets, or external download storage.

## Local Configuration

Do not commit `local.properties`. If needed, copy:

```text
local.properties.example -> local.properties
```

and edit `sdk.dir` for your local Android SDK path.

Use JDK 17 for local builds.

## Build

```powershell
.\gradlew.bat :app:assembleDebug
```

The debug package uses the `.vadtest` application id suffix.

## Zeabur Deployment

This repository also contains a minimal server entrypoint for Zeabur. It is
separate from the Android APK. Zeabur should build the repository root
`Dockerfile`, which runs a FastAPI upload receiver on port `8080`.

Endpoints:

```text
GET  /health
GET  /ready
POST /api/airec/upload
POST /api/v1/recordings
GET  /api/v1/recordings/{recording_id}
```

`POST /api/airec/upload` accepts AIREC-style multipart form data:

```text
file      WAV file, required
fileName  yyyyMMddHHmmss.wav, required
sn        device serial number, required
```

Files are saved under:

```text
${INCOMING_DIR}/{sn}/{fileName}
```

Default:

```text
/data/lightasr/incoming
```

Recommended Zeabur environment variables:

```text
PORT=8080
INCOMING_DIR=/data/lightasr/incoming
RECORDINGS_DIR=/data/lightasr/recordings
DATABASE_URL=<rotated Zeabur PostgreSQL connection string>
DATABASE_REQUIRED=true
UPLOAD_TOKEN=<long random token>
ALLOW_ALL_SN=true
MAX_AUDIO_BYTES=2147483648
MAX_TRANSCRIPT_BYTES=10485760
```

For production, set `ALLOW_ALL_SN=false` and configure:

```text
AUTHORIZED_SN=DEVICE_SN_123456,DEVICE_SN_789
```

Local server test:

```powershell
docker build -t lightasr-upload .
docker run --rm -p 8080:8080 lightasr-upload
```

Upload test:

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/airec/upload `
  -F "file=@C:\path\to\20260522120423.wav;type=audio/wav" `
  -F "fileName=20260522120423.wav" `
  -F "sn=DEVICE_SN_123456"
```

The server returns HTTP 200 with `ok` when the upload is saved or when the same
`sn + fileName` has already been uploaded.

`POST /api/v1/recordings` is used by the Android app after local recognition.
It uploads the original WAV plus the generated UTF-8 TXT transcript. If
`UPLOAD_TOKEN` is configured on the server, the Android app must send the same
token as a Bearer token.

## Notes

Generated APKs, build directories, downloaded archives, real recordings,
incoming uploads, generated transcripts, logs, local databases, local SDK paths,
and signing keys are intentionally excluded from Git.
