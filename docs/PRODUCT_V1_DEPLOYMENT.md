# LightASR 0.2.0 First Release Deployment

## Product boundary

Version 0.2.0 provides local PCM WAV and AIREC import, streaming VAD + ASR,
foreground long-task notification, per-chunk checkpoint, task history, and manual
WAV + TXT upload.

Voiceprint remains experimental and is not written into the official transcript.
Server-side business analysis is not included.

## Zeabur environment

Configure these values in the Zeabur service dashboard:

    PORT=8080
    INCOMING_DIR=/data/lightasr/incoming
    RECORDINGS_DIR=/data/lightasr/recordings
    DATABASE_URL=<new PostgreSQL connection string>
    DATABASE_REQUIRED=true
    UPLOAD_TOKEN=<long random token>
    ALLOW_ALL_SN=true
    MAX_AUDIO_BYTES=2147483648
    MAX_TRANSCRIPT_BYTES=10485760

For Alibaba Cloud OSS storage, also configure:

    ALIYUN_OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
    ALIYUN_OSS_BUCKET=<bucket name>
    ALIYUN_OSS_ACCESS_KEY_ID=<RAM access key id>
    ALIYUN_OSS_ACCESS_KEY_SECRET=<RAM access key secret>
    ALIYUN_OSS_PREFIX=lightasr
    OSS_REQUIRED=true

Use the regional endpoint, not the bucket domain. For a cn-hangzhou bucket, use
https://oss-cn-hangzhou.aliyuncs.com.

For Alibaba Cloud RDS PostgreSQL, set DATABASE_URL to the public or private RDS
connection string:

    DATABASE_URL=postgresql://<user>:<password>@<host>:5432/<database>
    DATABASE_REQUIRED=true

If the app still runs on Zeabur, use the RDS public endpoint and configure the
RDS IP whitelist accordingly. If the app runs inside Alibaba Cloud VPC, prefer
the RDS internal endpoint.

Mount a persistent volume at /data/lightasr. The upload is not production-ready
until a WAV and TXT remain available after a Zeabur redeploy.

Health checks are GET /health and GET /ready. The second endpoint also reports
database readiness and the recording storage path.

## Android server configuration

In the App server section, enter the public HTTPS URL and the same token configured
as Zeabur UPLOAD_TOKEN. The token is kept in App-private SharedPreferences and is
not committed to Git.

## Release signing

Put these values in ~/.gradle/gradle.properties or CI secrets:

    LIGHTASR_KEYSTORE_FILE=E:/secure/lightasr-release.jks
    LIGHTASR_KEYSTORE_PASSWORD=...
    LIGHTASR_KEY_ALIAS=lightasr
    LIGHTASR_KEY_PASSWORD=...

Build with .\gradlew.bat :app:assembleRelease --no-daemon --console=plain.
Without all four values the artifact is unsigned and must not be distributed.

## Server tests

Run .\.venv\Scripts\python.exe -m unittest server.tests.test_upload_api -v.

## Manual acceptance

1. A short WAV completes and creates a TXT.
2. A one-hour WAV continues while the screen is locked.
3. Cancel works from the foreground notification.
4. Reopening an interrupted task shows 已中断，可继续.
5. Resume does not repeat checkpointed chunks.
6. WAV + TXT upload returns a recording ID.
7. Repeating the upload returns the same recording ID.
8. Zeabur redeploy does not remove uploaded files.
9. The release APK installs and upgrades on an arm64 phone.
