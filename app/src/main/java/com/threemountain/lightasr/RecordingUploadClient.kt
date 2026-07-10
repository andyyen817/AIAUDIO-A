package com.threemountain.lightasr

import android.content.Context
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.UUID

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
        onProgress: (Int) -> Unit,
    ): RecordingUploadResponse {
        require(audioFile.isFile && audioFile.length() > 44L) { "原始 WAV 不存在或为空" }
        require(transcriptFile.isFile && transcriptFile.length() > 0L) { "TXT 不存在或为空" }

        val endpoint = config.baseUrl.trimEnd('/') + "/api/v1/recordings"
        val boundary = "----LightASR${UUID.randomUUID().toString().replace("-", "")}"
        val connection = URL(endpoint).openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.doOutput = true
        connection.useCaches = false
        connection.connectTimeout = 30_000
        connection.readTimeout = 120_000
        connection.setChunkedStreamingMode(64 * 1024)
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
        connection.setRequestProperty("Accept", "application/json")
        connection.setRequestProperty("User-Agent", "LightASR-Android/$appVersion")
        if (config.uploadToken.isNotBlank()) {
            connection.setRequestProperty("Authorization", "Bearer ${config.uploadToken}")
        }

        val totalFileBytes = audioFile.length() + transcriptFile.length()
        var sentFileBytes = 0L
        BufferedOutputStream(connection.outputStream, 64 * 1024).use { output ->
            writeTextPart(output, boundary, "original_name", originalName)
            writeTextPart(output, boundary, "source", source)
            if (!recordingTimeIso.isNullOrBlank()) writeTextPart(output, boundary, "recording_time", recordingTimeIso)
            writeTextPart(output, boundary, "duration_ms", durationMs.toString())
            writeTextPart(output, boundary, "audio_sha256", audioSha256)
            writeTextPart(output, boundary, "app_version", appVersion)
            writeTextPart(output, boundary, "device_id", deviceId)

            sentFileBytes += writeFilePart(
                output, boundary, "audio_file", originalName, "audio/wav", audioFile,
                totalFileBytes, sentFileBytes, onProgress,
            )
            sentFileBytes += writeFilePart(
                output, boundary, "transcript_file",
                originalName.substringBeforeLast('.', originalName) + ".txt",
                "text/plain; charset=utf-8", transcriptFile,
                totalFileBytes, sentFileBytes, onProgress,
            )
            output.write("--$boundary--\r\n".toByteArray(StandardCharsets.UTF_8))
            output.flush()
        }

        val status = connection.responseCode
        val responseText = (if (status in 200..299) connection.inputStream else connection.errorStream)
            ?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }.orEmpty()
        connection.disconnect()
        if (status !in 200..299) {
            val detail = runCatching { JSONObject(responseText).optString("detail") }.getOrNull()
            error("服务器返回 HTTP $status：${detail?.takeIf { it.isNotBlank() } ?: responseText.take(300)}")
        }
        val json = JSONObject(responseText)
        return RecordingUploadResponse(
            recordingId = json.getString("recording_id"),
            uploadStatus = json.optString("upload_status", "uploaded"),
        )
    }

    private fun writeTextPart(output: BufferedOutputStream, boundary: String, name: String, value: String) {
        val text = buildString {
            append("--").append(boundary).append("\r\n")
            append("Content-Disposition: form-data; name=\"").append(name).append("\"\r\n")
            append("Content-Type: text/plain; charset=utf-8\r\n\r\n")
            append(value).append("\r\n")
        }
        output.write(text.toByteArray(StandardCharsets.UTF_8))
    }

    private fun writeFilePart(
        output: BufferedOutputStream,
        boundary: String,
        fieldName: String,
        fileName: String,
        contentType: String,
        file: File,
        totalFileBytes: Long,
        alreadySent: Long,
        onProgress: (Int) -> Unit,
    ): Long {
        val safeFileName = File(fileName).name.replace("\"", "_")
        val header = buildString {
            append("--").append(boundary).append("\r\n")
            append("Content-Disposition: form-data; name=\"").append(fieldName)
                .append("\"; filename=\"").append(safeFileName).append("\"\r\n")
            append("Content-Type: ").append(contentType).append("\r\n\r\n")
        }
        output.write(header.toByteArray(StandardCharsets.UTF_8))
        var fileSent = 0L
        val buffer = ByteArray(64 * 1024)
        FileInputStream(file).use { input ->
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                output.write(buffer, 0, count)
                fileSent += count
                val progress = if (totalFileBytes > 0L) {
                    (((alreadySent + fileSent) * 100L) / totalFileBytes).toInt()
                } else 0
                onProgress(progress.coerceIn(0, 99))
            }
        }
        output.write("\r\n".toByteArray(StandardCharsets.UTF_8))
        return fileSent
    }
}
