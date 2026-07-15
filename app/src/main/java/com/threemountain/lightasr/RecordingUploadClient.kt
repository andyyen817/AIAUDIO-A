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
        val endpoint = uploadEndpoint(config.baseUrl)
        onStage("连接服务器中：$serverRoot")
        verifyServerHealth(serverRoot)

        val boundary = "----LightASR${UUID.randomUUID().toString().replace("-", "")}"
        val textParts = buildList {
            add(textPartBytes(boundary, "original_name", originalName))
            add(textPartBytes(boundary, "source", source))
            if (!recordingTimeIso.isNullOrBlank()) add(textPartBytes(boundary, "recording_time", recordingTimeIso))
            add(textPartBytes(boundary, "duration_ms", durationMs.toString()))
            add(textPartBytes(boundary, "audio_sha256", audioSha256))
            add(textPartBytes(boundary, "app_version", appVersion))
            add(textPartBytes(boundary, "device_id", deviceId))
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
        if (config.uploadToken.isNotBlank()) {
            connection.setRequestProperty("Authorization", "Bearer ${config.uploadToken}")
        }

        val lastUploadActivityAt = AtomicLong(System.currentTimeMillis())
        val uploadFinished = AtomicBoolean(false)
        val abortedByWatchdog = AtomicBoolean(false)
        val markUploadActivity = { lastUploadActivityAt.set(System.currentTimeMillis()) }
        val watchdog = thread(start = true, name = "lightasr-upload-watchdog") {
            while (!uploadFinished.get()) {
                val idleMs = System.currentTimeMillis() - lastUploadActivityAt.get()
                if (idleMs > UPLOAD_STALL_TIMEOUT_MS) {
                    abortedByWatchdog.set(true)
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

        try {
            val totalFileBytes = audioFile.length() + transcriptFile.length()
            var sentFileBytes = 0L
            onStage("准备上传数据：${formatBytes(totalFileBytes)}")
            markUploadActivity()
            BufferedOutputStream(connection.outputStream, 64 * 1024).use { output ->
                markUploadActivity()
                textParts.forEach { output.write(it) }
                output.flush()
                markUploadActivity()

                onStage("上传音频文件：${formatBytes(audioFile.length())}")
                sentFileBytes += writeFilePart(
                    output, audioHeader, partBreak, audioFile,
                    totalFileBytes, sentFileBytes, onProgress, markUploadActivity,
                )
                onStage("上传 TXT 文件：${formatBytes(transcriptFile.length())}")
                sentFileBytes += writeFilePart(
                    output, transcriptHeader, partBreak, transcriptFile,
                    totalFileBytes, sentFileBytes, onProgress, markUploadActivity,
                )
                output.write(closing)
                output.flush()
                markUploadActivity()
            }
            onProgress(99)
            onStage("等待服务器确认")
            markUploadActivity()

            val status = connection.responseCode
            markUploadActivity()
            val responseText = (if (status in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (status !in 200..299) {
                val detail = runCatching { JSONObject(responseText).optString("detail") }.getOrNull()
                error("服务器返回 HTTP $status：${detail?.takeIf { it.isNotBlank() } ?: responseText.take(300)}")
            }
            val json = JSONObject(responseText)
            onStage("上传成功")
            return RecordingUploadResponse(
                recordingId = json.getString("recording_id"),
                uploadStatus = json.optString("upload_status", "uploaded"),
            )
        } catch (t: Throwable) {
            if (abortedByWatchdog.get()) {
                throw SocketTimeoutException("上传超过 ${UPLOAD_STALL_TIMEOUT_MS / 1000} 秒没有网络进展，已中断。请检查手机网络或稍后重试。")
            }
            throw t
        } finally {
            uploadFinished.set(true)
            watchdog.interrupt()
            connection.disconnect()
        }
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
                error("服务器健康检查失败：HTTP $status")
            }
        } finally {
            connection.disconnect()
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
        output.write(partBreak)
        output.flush()
        onBytesWritten()
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
}
