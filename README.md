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
- Voiceprint MVP UI and local employee/sample management are present for testing.

## Not Yet Complete

- MP3/M4A/AAC decoding is not implemented; the current import path is WAV-first.
- Cloud/server upload is not implemented in this Android project.
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

## Notes

Generated APKs, build directories, downloaded archives, real recordings,
incoming uploads, generated transcripts, logs, local databases, local SDK paths,
and signing keys are intentionally excluded from Git.
