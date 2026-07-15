package com.threemountain.lightasr

import android.content.Context
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

private const val UPLOAD_CONNECT_TIMEOUT_MS = 30_000
private const val UPLOAD_READ_TIMEOUT_MS = 120_000
private const val UPLOAD_STALL_TIMEOUT_MS = 45_000L
private const val UPLOAD_FLUSH_EVERY_BYTES = 256L * 1024L

data class LightAsrServerConfig(val baseUrl: String, val uploadToken: String)

object LightAsrServerConfigRepository {
    private const val PREFS = "lightasr_server_config"
    private const val KEY_URL = "base_url"
    private const val KEY_TOKEN = "upload_token"
    const val DEFAULT_URL = "https://aiaudio-a.zeabur.app"

    fun load(context: Context): LightAsrServerConfig {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return LightAsrServerConfig(
            baseUrl = prefs.getString(KEY_URL, DEFAULT_URL)?.trim().orEmpty().ifBlank { DEFAULT_URL },
            uploadToken = prefs.getString(KEY_TOKEN, "")?.trim().orEmpty(),
        )
    }

    fun save(context: Context, baseUrl: String, uploadToken: String) {
        val normalized = baseUrl.trim().trimEnd('/')
        require(normalized.startsWith("https://") || normalized.startsWith("http://")) {
            "服务器地址必须以 http:// 或 https:// 开头"
        }
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(KEY_URL, normalized).putString(KEY_TOKEN, uploadToken.trim()).commit()
    }
}

data class RecordingUploadResponse(val recordingId: String, val uploadStatus: String)

private data class DirectUploadTarget(
    val method: String,
    val url: String,
    val contentType: String,
    val headers: Map<String, String>,
)

private data class DirectUploadInitResponse(
    val recordingId: String,
    val audio: DirectUploadTarget,
    val transcript: DirectUploadTarget,
)

private class UploadHttpException(val statusCode: Int, message: String) : RuntimeException(message)

class RecordingUploadClient(private val config: LightAsrServerConfig) {
    fun upload(
        audioFile: File,
        transcriptFile: File,
        originalName: String,
        source: String,
        recordingTimeIso: String?,
        durationMs: Long,
        audioSha256: String,
        appVersion: String,
        deviceId: String,
        onStage: (String) -> Unit = {},
        onProgress: (Int) -> Unit,
    ): RecordingUploadResponse {
        require(audioFile.isFile && audioFile.length() > 44L) { "原始 WAV 不存在或为空" }
        require(transcriptFile.isFile && transcriptFile.length() > 0L) { "TXT 不存在或为空" }

        val serverRoot = serverRootUrl(config.baseUrl)
        onStage("连接服务器中：$serverRoot")
        verifyServerHealth(serverRoot)

        val metadata = uploadMetadata(
            audioFile = audioFile,
            transcriptFile = transcriptFile,
            originalName = originalName,
            source = source,
            recordingTimeIso = recordingTimeIso,
            durationMs = durationMs,
            audioSha256 = audioSha256,
            appVersion = appVersion,
            deviceId = deviceId,
        )
        return try {
            uploadDirectToOss(metadata, audioFile, transcriptFile, appVersion, onStage, onProgress)
        } catch (e: UploadHttpException) {
            if (e.statusCode == 404 || e.statusCode == 405) {
                onStage("服务器未启用直传，改用兼容上传")
                uploadViaServer(metadata, audioFile, transcriptFile, originalName, appVersion, onStage, onProgress)
            } else {
                throw e
            }
        }
    }

    private fun uploadDirectToOss(
        metadata: JSONObject,
        audioFile: File,
        transcriptFile: File,
        appVersion: String,
        onStage: (String) -> Unit,
        onProgress: (Int) -> Unit,
    ): RecordingUploadResponse {
        onStage("请求 OSS 直传地址")
        val initJson = postJson(directUploadInitEndpoint(config.baseUrl), metadata, appVersion)
        val init = parseDirectUploadInit(initJson)
        val totalFileBytes = audioFile.length() + transcriptFile.length()
        var sentFileBytes = 0L

        onStage("直传音频到 OSS：${formatBytes(audioFile.length())}")
        sentFileBytes += putFileToSignedUrl(
            target = init.audio,
            file = audioFile,
            totalFileBytes = totalFileBytes,
            alreadySent = sentFileBytes,
            onProgress = onProgress,
        )
        onStage("直传 TXT 到 OSS：${formatBytes(transcriptFile.length())}")
        sentFileBytes += putFileToSignedUrl(
            target = init.transcript,
            file = transcriptFile,
            totalFileBytes = totalFileBytes,
            alreadySent = sentFileBytes,
            onProgress = onProgress,
        )

        onProgress(99)
        onStage("通知服务器登记上传结果")
        val completePayload = JSONObject(metadata.toString()).put("recording_id", init.recordingId)
        val completeJson = postJson(directUploadCompleteEndpoint(config.baseUrl), completePayload, appVersion)
        onStage("上传成功")
        return RecordingUploadResponse(
            recordingId = completeJson.optString("recording_id", init.recordingId),
            uploadStatus = completeJson.optString("upload_status", "uploaded"),
        )
    }

    private fun uploadViaServer(
        metadata: JSONObject,
        audioFile: File,
        transcriptFile: File,
        originalName: String,
        appVersion: String,
        onStage: (String) -> Unit,
        onProgress: (Int) -> Unit,
    ): RecordingUploadResponse {
        val endpoint = uploadEndpoint(config.baseUrl)
        val boundary = "----LightASR${UUID.randomUUID().toString().replace("-", "")}"
        val textParts = buildList {
            add(textPartBytes(boundary, "original_name", metadata.getString("original_name")))
            add(textPartBytes(boundary, "source", metadata.getString("source")))
            metadata.optString("recording_time").takeIf { it.isNotBlank() }?.let {
                add(textPartBytes(boundary, "recording_time", it))
            }
            add(textPartBytes(boundary, "duration_ms", metadata.getLong("duration_ms").toString()))
            add(textPartBytes(boundary, "audio_sha256", metadata.getString("audio_sha256")))
            add(textPartBytes(boundary, "app_version", metadata.getString("app_version")))
            add(textPartBytes(boundary, "device_id", metadata.getString("device_id")))
        }
        val audioHeader = fileHeaderBytes(boundary, "audio_file", originalName, "audio/wav")
        val transcriptHeader = fileHeaderBytes(
            boundary,
            "transcript_file",
            originalName.substringBeforeLast('.', originalName) + ".txt",
            "text/plain; charset=utf-8",
        )
        val partBreak = "\r\n".toByteArray(StandardCharsets.UTF_8)
        val closing = "--$boundary--\r\n".toByteArray(StandardCharsets.UTF_8)
        val totalMultipartBytes = textParts.sumOf { it.size.toLong() } +
            audioHeader.size.toLong() + audioFile.length() + partBreak.size.toLong() +
            transcriptHeader.size.toLong() + transcriptFile.length() + partBreak.size.toLong() +
            closing.size.toLong()

        val connection = URL(endpoint).openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.doOutput = true
        connection.useCaches = false
        connection.connectTimeout = UPLOAD_CONNECT_TIMEOUT_MS
        connection.readTimeout = UPLOAD_READ_TIMEOUT_MS
        connection.setFixedLengthStreamingMode(totalMultipartBytes)
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
        connection.setRequestProperty("Accept", "application/json")
        connection.setRequestProperty("User-Agent", "LightASR-Android/$appVersion")
        connection.setRequestProperty("Connection", "close")
        addAuthorization(connection)

        val watchdog = UploadWatchdog(connection)
        watchdog.start()
        try {
            val totalFileBytes = audioFile.length() + transcriptFile.length()
            var sentFileBytes = 0L
            onStage("准备上传数据：${formatBytes(totalFileBytes)}")
            watchdog.markActivity()
            BufferedOutputStream(connection.outputStream, 64 * 1024).use { output ->
                watchdog.markActivity()
                textParts.forEach { output.write(it) }
                output.flush()
                watchdog.markActivity()

                onStage("上传音频文件：${formatBytes(audioFile.length())}")
                sentFileBytes += writeFilePart(
                    output, audioHeader, partBreak, audioFile,
                    totalFileBytes, sentFileBytes, onProgress, watchdog::markActivity,
                )
                onStage("上传 TXT 文件：${formatBytes(transcriptFile.length())}")
                sentFileBytes += writeFilePart(
                    output, transcriptHeader, partBreak, transcriptFile,
                    totalFileBytes, sentFileBytes, onProgress, watchdog::markActivity,
                )
                output.write(closing)
                output.flush()
                watchdog.markActivity()
            }
            onProgress(99)
            onStage("等待服务器确认")
            watchdog.markActivity()

            val json = readJsonResponse(connection, watchdog::markActivity)
            onStage("上传成功")
            return RecordingUploadResponse(
                recordingId = json.getString("recording_id"),
                uploadStatus = json.optString("upload_status", "uploaded"),
            )
        } catch (t: Throwable) {
            watchdog.throwIfAborted()
            throw t
        } finally {
            watchdog.stop()
            connection.disconnect()
        }
    }

    private fun putFileToSignedUrl(
        target: DirectUploadTarget,
        file: File,
        totalFileBytes: Long,
        alreadySent: Long,
        onProgress: (Int) -> Unit,
    ): Long {
        val connection = URL(target.url).openConnection() as HttpURLConnection
        connection.requestMethod = target.method.ifBlank { "PUT" }
        connection.doOutput = true
        connection.useCaches = false
        connection.connectTimeout = UPLOAD_CONNECT_TIMEOUT_MS
        connection.readTimeout = UPLOAD_READ_TIMEOUT_MS
        connection.setFixedLengthStreamingMode(file.length())
        connection.setRequestProperty("Connection", "close")
        target.headers.forEach { (name, value) -> connection.setRequestProperty(name, value) }
        if (target.headers.keys.none { it.equals("Content-Type", ignoreCase = true) }) {
            connection.setRequestProperty("Content-Type", target.contentType)
        }

        val watchdog = UploadWatchdog(connection)
        watchdog.start()
        try {
            BufferedOutputStream(connection.outputStream, 64 * 1024).use { output ->
                writeRawFile(output, file, totalFileBytes, alreadySent, onProgress, watchdog::markActivity)
                output.flush()
                watchdog.markActivity()
            }
            val status = connection.responseCode
            watchdog.markActivity()
            if (status !in 200..299) {
                val responseText = connection.errorStream
                    ?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }.orEmpty()
                throw UploadHttpException(status, "OSS 上传失败：HTTP $status ${responseText.take(300)}")
            }
            return file.length()
        } catch (t: Throwable) {
            watchdog.throwIfAborted()
            throw t
        } finally {
            watchdog.stop()
            connection.disconnect()
        }
    }

    private fun postJson(endpoint: String, payload: JSONObject, appVersion: String): JSONObject {
        val body = payload.toString().toByteArray(StandardCharsets.UTF_8)
        val connection = URL(endpoint).openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.doOutput = true
        connection.useCaches = false
        connection.connectTimeout = UPLOAD_CONNECT_TIMEOUT_MS
        connection.readTimeout = UPLOAD_READ_TIMEOUT_MS
        connection.setFixedLengthStreamingMode(body.size)
        connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
        connection.setRequestProperty("Accept", "application/json")
        connection.setRequestProperty("User-Agent", "LightASR-Android/$appVersion")
        connection.setRequestProperty("Connection", "close")
        addAuthorization(connection)
        try {
            connection.outputStream.use { it.write(body) }
            return readJsonResponse(connection) {}
        } finally {
            connection.disconnect()
        }
    }

    private fun readJsonResponse(connection: HttpURLConnection, onActivity: () -> Unit): JSONObject {
        val status = connection.responseCode
        onActivity()
        val responseText = (if (status in 200..299) connection.inputStream else connection.errorStream)
            ?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }.orEmpty()
        if (status !in 200..299) {
            val detail = runCatching { JSONObject(responseText).optString("detail") }.getOrNull()
            throw UploadHttpException(
                status,
                "服务器返回 HTTP $status：${detail?.takeIf { it.isNotBlank() } ?: responseText.take(300)}",
            )
        }
        return JSONObject(responseText)
    }

    private fun verifyServerHealth(serverRoot: String) {
        val healthUrl = "$serverRoot/health"
        val connection = URL(healthUrl).openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 10_000
        connection.useCaches = false
        try {
            val status = connection.responseCode
            if (status !in 200..299) {
                throw UploadHttpException(status, "服务器健康检查失败：HTTP $status")
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun uploadMetadata(
        audioFile: File,
        transcriptFile: File,
        originalName: String,
        source: String,
        recordingTimeIso: String?,
        durationMs: Long,
        audioSha256: String,
        appVersion: String,
        deviceId: String,
    ): JSONObject {
        return JSONObject().apply {
            put("original_name", originalName)
            put("source", source)
            if (!recordingTimeIso.isNullOrBlank()) put("recording_time", recordingTimeIso)
            put("duration_ms", durationMs)
            put("audio_sha256", audioSha256)
            put("app_version", appVersion)
            put("device_id", deviceId)
            put("audio_size_bytes", audioFile.length())
            put("transcript_size_bytes", transcriptFile.length())
        }
    }

    private fun parseDirectUploadInit(json: JSONObject): DirectUploadInitResponse {
        return DirectUploadInitResponse(
            recordingId = json.getString("recording_id"),
            audio = parseDirectUploadTarget(json.getJSONObject("audio")),
            transcript = parseDirectUploadTarget(json.getJSONObject("transcript")),
        )
    }

    private fun parseDirectUploadTarget(json: JSONObject): DirectUploadTarget {
        val headers = linkedMapOf<String, String>()
        json.optJSONObject("headers")?.let { headerJson ->
            val keys = headerJson.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                headers[key] = headerJson.getString(key)
            }
        }
        return DirectUploadTarget(
            method = json.optString("method", "PUT"),
            url = json.getString("url"),
            contentType = json.optString("content_type", "application/octet-stream"),
            headers = headers,
        )
    }

    private fun addAuthorization(connection: HttpURLConnection) {
        if (config.uploadToken.isNotBlank()) {
            connection.setRequestProperty("Authorization", "Bearer ${config.uploadToken}")
        }
    }

    private fun textPartBytes(boundary: String, name: String, value: String): ByteArray {
        return buildString {
            append("--").append(boundary).append("\r\n")
            append("Content-Disposition: form-data; name=\"").append(name).append("\"\r\n")
            append("Content-Type: text/plain; charset=utf-8\r\n\r\n")
            append(value).append("\r\n")
        }.toByteArray(StandardCharsets.UTF_8)
    }

    private fun fileHeaderBytes(
        boundary: String,
        fieldName: String,
        fileName: String,
        contentType: String,
    ): ByteArray {
        val safeFileName = File(fileName).name.replace("\"", "_")
        return buildString {
            append("--").append(boundary).append("\r\n")
            append("Content-Disposition: form-data; name=\"").append(fieldName)
                .append("\"; filename=\"").append(safeFileName).append("\"\r\n")
            append("Content-Type: ").append(contentType).append("\r\n\r\n")
        }.toByteArray(StandardCharsets.UTF_8)
    }

    private fun writeFilePart(
        output: BufferedOutputStream,
        header: ByteArray,
        partBreak: ByteArray,
        file: File,
        totalFileBytes: Long,
        alreadySent: Long,
        onProgress: (Int) -> Unit,
        onBytesWritten: () -> Unit,
    ): Long {
        output.write(header)
        output.flush()
        onBytesWritten()
        val fileSent = writeRawFile(output, file, totalFileBytes, alreadySent, onProgress, onBytesWritten)
        output.write(partBreak)
        output.flush()
        onBytesWritten()
        return fileSent
    }

    private fun writeRawFile(
        output: BufferedOutputStream,
        file: File,
        totalFileBytes: Long,
        alreadySent: Long,
        onProgress: (Int) -> Unit,
        onBytesWritten: () -> Unit,
    ): Long {
        var fileSent = 0L
        var lastFlushAt = 0L
        val buffer = ByteArray(64 * 1024)
        FileInputStream(file).use { input ->
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                output.write(buffer, 0, count)
                fileSent += count
                onBytesWritten()
                if (fileSent - lastFlushAt >= UPLOAD_FLUSH_EVERY_BYTES) {
                    output.flush()
                    lastFlushAt = fileSent
                    onBytesWritten()
                }
                val progress = if (totalFileBytes > 0L) {
                    (((alreadySent + fileSent) * 100L) / totalFileBytes).toInt()
                } else 0
                onProgress(progress.coerceIn(0, 99))
            }
        }
        return fileSent
    }

    private fun uploadEndpoint(baseUrl: String): String {
        val normalized = baseUrl.trim().trimEnd('/')
        return when {
            normalized.endsWith("/api/v1/recordings", ignoreCase = true) -> normalized
            normalized.endsWith("/api/v1", ignoreCase = true) -> "$normalized/recordings"
            else -> "$normalized/api/v1/recordings"
        }
    }

    private fun directUploadInitEndpoint(baseUrl: String): String {
        val normalized = baseUrl.trim().trimEnd('/')
        return when {
            normalized.endsWith("/api/v1", ignoreCase = true) -> "$normalized/recordings/direct-upload/init"
            else -> "$normalized/api/v1/recordings/direct-upload/init"
        }
    }

    private fun directUploadCompleteEndpoint(baseUrl: String): String {
        val normalized = baseUrl.trim().trimEnd('/')
        return when {
            normalized.endsWith("/api/v1", ignoreCase = true) -> "$normalized/recordings/direct-upload/complete"
            else -> "$normalized/api/v1/recordings/direct-upload/complete"
        }
    }

    private fun serverRootUrl(baseUrl: String): String {
        val url = URL(baseUrl.trim())
        val port = if (url.port >= 0) ":${url.port}" else ""
        return "${url.protocol}://${url.host}$port"
    }

    private fun formatBytes(bytes: Long): String {
        if (bytes < 1024L) return "$bytes B"
        val kb = bytes / 1024.0
        if (kb < 1024.0) return "%.1f KB".format(kb)
        return "%.2f MB".format(kb / 1024.0)
    }

    private class UploadWatchdog(private val connection: HttpURLConnection) {
        private val lastActivityAt = AtomicLong(System.currentTimeMillis())
        private val finished = AtomicBoolean(false)
        private val aborted = AtomicBoolean(false)
        private var threadRef: Thread? = null

        fun start() {
            threadRef = thread(start = true, name = "lightasr-upload-watchdog") {
                while (!finished.get()) {
                    val idleMs = System.currentTimeMillis() - lastActivityAt.get()
                    if (idleMs > UPLOAD_STALL_TIMEOUT_MS) {
                        aborted.set(true)
                        connection.disconnect()
                        break
                    }
                    try {
                        Thread.sleep(2_000L)
                    } catch (_: InterruptedException) {
                        break
                    }
                }
            }
        }

        fun markActivity() {
            lastActivityAt.set(System.currentTimeMillis())
        }

        fun throwIfAborted() {
            if (aborted.get()) {
                throw SocketTimeoutException(
                    "上传超过 ${UPLOAD_STALL_TIMEOUT_MS / 1000} 秒没有网络进展，已中断。请检查手机网络或稍后重试。",
                )
            }
        }

        fun stop() {
            finished.set(true)
            threadRef?.interrupt()
        }
    }
}
