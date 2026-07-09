package com.threemountain.lightasr

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.net.BindException
import java.net.Inet4Address
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.Executors
import kotlin.concurrent.thread
import kotlin.math.min

private const val AIREC_TAG = "LightASR-AIREC"
private const val AIREC_PORT = 18080
private const val AIREC_PATH = "/api/airec/upload"
private const val AIREC_CHUNK_SIZE = 64 * 1024
private const val AIREC_FIELD_LIMIT = 64 * 1024
private const val AIREC_NOTIFICATION_ID = 18080
private const val AIREC_CHANNEL_ID = "lightasr_airec_receiver"

data class AirecUploadRecord(
    val sn: String,
    val fileName: String,
    val savedPath: String,
    val fileSize: Long,
    val uploadTimeMs: Long,
    val status: String,
)

object AirecUploadRepository {
    private const val PREFS = "airec_upload_records"
    private const val KEY_RECORDS = "records"
    private const val MAX_RECORDS = 100

    @Synchronized
    fun addIfAbsent(context: Context, record: AirecUploadRecord): Boolean {
        val records = list(context).toMutableList()
        if (records.any { it.sn == record.sn && it.fileName == record.fileName }) {
            return false
        }
        records.add(0, record)
        save(context, records.take(MAX_RECORDS))
        return true
    }

    @Synchronized
    fun list(context: Context): List<AirecUploadRecord> {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val raw = prefs.getString(KEY_RECORDS, "[]") ?: "[]"
        val array = JSONArray(raw)
        return buildList {
            for (i in 0 until array.length()) {
                val item = array.optJSONObject(i) ?: continue
                add(
                    AirecUploadRecord(
                        sn = item.optString("sn"),
                        fileName = item.optString("fileName"),
                        savedPath = item.optString("savedPath"),
                        fileSize = item.optLong("fileSize"),
                        uploadTimeMs = item.optLong("uploadTimeMs"),
                        status = item.optString("status", "pending_transcription"),
                    )
                )
            }
        }
    }

    private fun save(context: Context, records: List<AirecUploadRecord>) {
        val array = JSONArray()
        records.forEach { record ->
            array.put(
                JSONObject()
                    .put("sn", record.sn)
                    .put("fileName", record.fileName)
                    .put("savedPath", record.savedPath)
                    .put("fileSize", record.fileSize)
                    .put("uploadTimeMs", record.uploadTimeMs)
                    .put("status", record.status)
            )
        }
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_RECORDS, array.toString())
            .apply()
    }
}

object AirecNetwork {
    fun localIpv4Address(): String? {
        val interfaces = NetworkInterface.getNetworkInterfaces()?.toList().orEmpty()
        for (networkInterface in interfaces) {
            if (!networkInterface.isUp || networkInterface.isLoopback) continue
            val addresses = networkInterface.inetAddresses.toList()
            for (address in addresses) {
                if (address is Inet4Address &&
                    !address.isLoopbackAddress &&
                    !address.hostAddress.startsWith("169.254.")
                ) {
                    return address.hostAddress
                }
            }
        }
        return null
    }

    fun uploadUrl(): String {
        val ip = localIpv4Address() ?: "127.0.0.1"
        return "http://$ip:$AIREC_PORT$AIREC_PATH"
    }
}

class AirecReceiverService : Service() {
    private var server: AirecUploadServer? = null

    override fun onCreate() {
        super.onCreate()
        ensureNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopReceiver()
                stopSelf()
                return START_NOT_STICKY
            }
            else -> startReceiver()
        }
        return START_STICKY
    }

    override fun onDestroy() {
        stopReceiver()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startReceiver() {
        if (isRunning) {
            broadcastState()
            return
        }

        startForeground(AIREC_NOTIFICATION_ID, buildNotification("AIREC 接收模式运行中"))

        try {
            val uploadServer = AirecUploadServer(
                context = applicationContext,
                port = AIREC_PORT,
                onRecordSaved = {
                    sendBroadcast(Intent(ACTION_UPLOAD_RECEIVED).setPackage(packageName))
                    broadcastState()
                },
                onError = { message ->
                    lastError = message
                    broadcastState()
                },
            )
            uploadServer.start()
            server = uploadServer
            isRunning = true
            lastError = null
            Log.i(AIREC_TAG, "AIREC receiver started url=${AirecNetwork.uploadUrl()}")
            broadcastState()
        } catch (bind: BindException) {
            lastError = "端口 $AIREC_PORT 已被占用"
            Log.e(AIREC_TAG, lastError ?: "bind failed", bind)
            broadcastState()
            stopForegroundCompat()
            stopSelf()
        } catch (t: Throwable) {
            lastError = "接收模式启动失败：${t.message ?: t::class.java.simpleName}"
            Log.e(AIREC_TAG, lastError ?: "start failed", t)
            broadcastState()
            stopForegroundCompat()
            stopSelf()
        }
    }

    private fun stopReceiver() {
        server?.stop()
        server = null
        isRunning = false
        Log.i(AIREC_TAG, "AIREC receiver stopped")
        broadcastState()
        stopForegroundCompat()
    }

    private fun buildNotification(text: String): Notification {
        val openIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, AIREC_CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }

        return builder
            .setSmallIcon(android.R.drawable.stat_sys_upload_done)
            .setContentTitle("LightASR AIREC 接收模式")
            .setContentText(text)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun ensureNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        val channel = NotificationChannel(
            AIREC_CHANNEL_ID,
            "LightASR AIREC Receiver",
            NotificationManager.IMPORTANCE_LOW,
        )
        manager.createNotificationChannel(channel)
    }

    private fun stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
    }

    private fun broadcastState() {
        sendBroadcast(Intent(ACTION_STATE_CHANGED).setPackage(packageName))
    }

    companion object {
        const val ACTION_START = "com.threemountain.lightasr.AIREC_START"
        const val ACTION_STOP = "com.threemountain.lightasr.AIREC_STOP"
        const val ACTION_STATE_CHANGED = "com.threemountain.lightasr.AIREC_STATE_CHANGED"
        const val ACTION_UPLOAD_RECEIVED = "com.threemountain.lightasr.AIREC_UPLOAD_RECEIVED"

        @Volatile
        var isRunning: Boolean = false
            private set

        @Volatile
        var lastError: String? = null
            private set
    }
}

private class AirecUploadServer(
    private val context: Context,
    private val port: Int,
    private val onRecordSaved: (AirecUploadRecord) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val executor = Executors.newCachedThreadPool()
    private val saveLock = Any()
    @Volatile
    private var running = false
    private var serverSocket: ServerSocket? = null

    fun start() {
        val socket = ServerSocket()
        socket.reuseAddress = true
        socket.bind(InetSocketAddress(InetAddress.getByName("0.0.0.0"), port))
        serverSocket = socket
        running = true
        thread(name = "airec-http-accept") {
            while (running) {
                try {
                    val client = socket.accept()
                    executor.execute { handleClient(client) }
                } catch (t: Throwable) {
                    if (running) {
                        Log.e(AIREC_TAG, "accept failed: ${t.message}", t)
                        onError("接收连接失败：${t.message ?: t::class.java.simpleName}")
                    }
                }
            }
        }
    }

    fun stop() {
        running = false
        try {
            serverSocket?.close()
        } catch (_: Throwable) {
        }
        executor.shutdownNow()
    }

    private fun handleClient(socket: Socket) {
        socket.use { client ->
            client.soTimeout = 30 * 60 * 1000
            val input = BufferedInputStream(client.getInputStream(), AIREC_CHUNK_SIZE)
            val output = BufferedOutputStream(client.getOutputStream())
            try {
                val requestLine = readAsciiLine(input, 8192)
                    ?: throw AirecHttpError(400, "empty request")
                val requestParts = requestLine.split(" ")
                if (requestParts.size < 2) throw AirecHttpError(400, "bad request")
                val method = requestParts[0].uppercase(Locale.US)
                val path = requestParts[1].substringBefore("?")
                if (method != "POST") throw AirecHttpError(405, "method not allowed")
                if (path != AIREC_PATH) {
                    throw AirecHttpError(404, "not found")
                }

                val headers = readHttpHeaders(input)
                val contentLength = headers["content-length"]?.toLongOrNull()
                    ?: throw AirecHttpError(411, "content-length required")
                val contentType = headers["content-type"].orEmpty()
                val boundary = parseBoundary(contentType)
                    ?: throw AirecHttpError(400, "multipart boundary missing")

                Log.i(AIREC_TAG, "upload start contentLength=$contentLength path=$path")
                val record = receiveMultipart(BodyInput(input, contentLength), boundary)
                sendResponse(output, 200, "ok")

                if (record != null) {
                    Log.i(
                        AIREC_TAG,
                        "upload success: sn=${record.sn}, fileName=${record.fileName}, " +
                            "sizeBytes=${record.fileSize}, savedPath=${record.savedPath}",
                    )
                    onRecordSaved(record)
                }
            } catch (e: AirecHttpError) {
                Log.w(AIREC_TAG, "validation failed: ${e.message}")
                sendResponse(output, e.statusCode, e.message ?: "error")
            } catch (t: Throwable) {
                Log.e(AIREC_TAG, "save failed: ${t.message}", t)
                sendResponse(output, 500, "save failed")
                onError("上传保存失败：${t.message ?: t::class.java.simpleName}")
            } finally {
                output.flush()
            }
        }
    }

    private fun receiveMultipart(body: BodyInput, boundary: String): AirecUploadRecord? {
        val firstBoundary = body.readLine(8192) ?: throw AirecHttpError(400, "missing multipart body")
        if (firstBoundary != "--$boundary") {
            throw AirecHttpError(400, "bad multipart boundary")
        }

        val fields = mutableMapOf<String, String>()
        var fileState: FilePartState? = null
        var done = false
        val boundaryPattern = "\r\n--$boundary".toByteArray(StandardCharsets.ISO_8859_1)

        try {
            while (!done) {
                val partHeaders = body.readPartHeaders()
                val disposition = parseContentDisposition(partHeaders["content-disposition"].orEmpty())
                val name = disposition["name"] ?: throw AirecHttpError(400, "multipart field name missing")

                if (name == "file") {
                    if (fileState != null) throw AirecHttpError(400, "重复 file 字段")
                    val state = FilePartState(createStagingFile())
                    fileState = state
                    state.use {
                        readPartBodyUntilBoundary(body, boundaryPattern) { bytes, offset, length ->
                            it.write(bytes, offset, length)
                        }
                        it.finish()
                    }
                } else {
                    val fieldOut = ByteArrayOutputStream()
                    readPartBodyUntilBoundary(body, boundaryPattern) { bytes, offset, length ->
                        if (fieldOut.size() + length > AIREC_FIELD_LIMIT) {
                            throw AirecHttpError(400, "字段过长")
                        }
                        fieldOut.write(bytes, offset, length)
                    }
                    fields[name] = fieldOut.toString(StandardCharsets.UTF_8.name()).trim()
                }

                val suffix = body.readLine(1024) ?: throw AirecHttpError(400, "bad multipart ending")
                done = when (suffix) {
                    "" -> false
                    "--" -> true
                    else -> throw AirecHttpError(400, "bad multipart suffix")
                }
            }

            val fileName = fields["fileName"].orEmpty()
            val sn = fields["sn"].orEmpty()
            val file = fileState ?: throw AirecHttpError(400, "缺少 file")
            validateFileName(fileName)
            validateSn(sn)

            return moveUploadIntoIncoming(file, sn, fileName)
        } catch (t: Throwable) {
            fileState?.deleteQuietly()
            throw t
        }
    }

    private fun moveUploadIntoIncoming(
        file: FilePartState,
        sn: String,
        fileName: String,
    ): AirecUploadRecord? {
        synchronized(saveLock) {
            val root = File(context.getExternalFilesDir(null) ?: context.filesDir, "Incoming").canonicalFile
            val snDir = File(root, sn).canonicalFile
            val finalFile = File(snDir, fileName).canonicalFile
            val partFile = File(snDir, "$fileName.part").canonicalFile
            requireInside(root, snDir)
            requireInside(snDir, finalFile)
            requireInside(snDir, partFile)

            if (finalFile.exists()) {
                file.deleteQuietly()
                Log.i(AIREC_TAG, "duplicate ignored: sn=$sn, fileName=$fileName")
                return null
            }

            if (!snDir.exists() && !snDir.mkdirs()) {
                throw AirecHttpError(500, "无法创建 Incoming 目录")
            }
            if (partFile.exists() && !partFile.delete()) {
                throw AirecHttpError(500, "无法清理旧 .part 文件")
            }
            if (!file.stagingFile.renameTo(partFile)) {
                throw AirecHttpError(500, "无法写入临时文件")
            }
            if (!partFile.renameTo(finalFile)) {
                partFile.delete()
                throw AirecHttpError(500, "无法保存上传文件")
            }

            val record = AirecUploadRecord(
                sn = sn,
                fileName = fileName,
                savedPath = finalFile.absolutePath,
                fileSize = finalFile.length(),
                uploadTimeMs = System.currentTimeMillis(),
                status = "pending_transcription",
            )
            AirecUploadRepository.addIfAbsent(context, record)
            return record
        }
    }

    private fun createStagingFile(): File {
        val root = File(context.getExternalFilesDir(null) ?: context.filesDir, "Incoming")
        val stagingDir = File(root, ".staging")
        if (!stagingDir.exists() && !stagingDir.mkdirs()) {
            throw AirecHttpError(500, "无法创建上传临时目录")
        }
        return File(stagingDir, "${System.currentTimeMillis()}-${UUID.randomUUID()}.part")
    }
}

private class FilePartState(
    val stagingFile: File,
) : Closeable {
    private val output = FileOutputStream(stagingFile)
    private val header = ByteArrayOutputStream(12)
    var sizeBytes: Long = 0L
        private set

    fun write(bytes: ByteArray, offset: Int, length: Int) {
        if (length <= 0) return
        if (header.size() < 12) {
            val headerNeed = min(12 - header.size(), length)
            header.write(bytes, offset, headerNeed)
        }
        output.write(bytes, offset, length)
        sizeBytes += length.toLong()
    }

    fun finish() {
        output.fd.sync()
        close()
        val headerBytes = header.toByteArray()
        if (
            headerBytes.size < 12 ||
            headerBytes.copyOfRange(0, 4).decodeToString() != "RIFF" ||
            headerBytes.copyOfRange(8, 12).decodeToString() != "WAVE"
        ) {
            throw AirecHttpError(400, "file 不是 RIFF/WAVE WAV 文件")
        }
        if (sizeBytes <= 12L) {
            throw AirecHttpError(400, "file 内容为空")
        }
    }

    override fun close() {
        output.close()
    }

    fun deleteQuietly() {
        try {
            close()
        } catch (_: Throwable) {
        }
        try {
            stagingFile.delete()
        } catch (_: Throwable) {
        }
    }
}

private class AirecHttpError(
    val statusCode: Int,
    override val message: String,
) : Exception(message)

private class BodyInput(
    private val source: InputStream,
    private var remaining: Long,
) {
    private var pushback = ByteArray(0)
    private var pushbackOffset = 0

    fun read(buffer: ByteArray, offset: Int, length: Int): Int {
        if (length == 0) return 0
        var copied = 0
        if (pushbackOffset < pushback.size) {
            val count = min(length, pushback.size - pushbackOffset)
            System.arraycopy(pushback, pushbackOffset, buffer, offset, count)
            pushbackOffset += count
            copied += count
            if (pushbackOffset >= pushback.size) {
                pushback = ByteArray(0)
                pushbackOffset = 0
            }
        }
        if (copied == length) return copied
        if (remaining <= 0L) return if (copied > 0) copied else -1

        val toRead = min((length - copied).toLong(), remaining).toInt()
        val read = source.read(buffer, offset + copied, toRead)
        if (read > 0) {
            remaining -= read.toLong()
            copied += read
        }
        return if (copied > 0) copied else read
    }

    fun read(): Int {
        val one = ByteArray(1)
        val count = read(one, 0, 1)
        return if (count == 1) one[0].toInt() and 0xFF else -1
    }

    fun unread(bytes: ByteArray) {
        if (bytes.isEmpty()) return
        val rest = if (pushbackOffset < pushback.size) {
            pushback.copyOfRange(pushbackOffset, pushback.size)
        } else {
            ByteArray(0)
        }
        pushback = bytes + rest
        pushbackOffset = 0
    }

    fun readLine(maxBytes: Int): String? {
        val out = ByteArrayOutputStream()
        while (out.size() <= maxBytes) {
            val b = read()
            if (b < 0) {
                return if (out.size() == 0) null else out.toString(StandardCharsets.ISO_8859_1.name())
            }
            if (b == '\n'.code) {
                return out.toString(StandardCharsets.ISO_8859_1.name()).trimEnd('\r')
            }
            out.write(b)
        }
        throw AirecHttpError(400, "line too long")
    }

    fun readPartHeaders(): Map<String, String> {
        val headers = mutableMapOf<String, String>()
        while (true) {
            val line = readLine(8192) ?: throw AirecHttpError(400, "part header missing")
            if (line.isEmpty()) return headers
            val split = line.indexOf(':')
            if (split <= 0) throw AirecHttpError(400, "bad part header")
            headers[line.substring(0, split).lowercase(Locale.US)] = line.substring(split + 1).trim()
        }
    }
}

private fun readPartBodyUntilBoundary(
    body: BodyInput,
    boundaryPattern: ByteArray,
    onBytes: (ByteArray, Int, Int) -> Unit,
) {
    val buffer = ByteArray(AIREC_CHUNK_SIZE)
    var tail = ByteArray(0)
    while (true) {
        val read = body.read(buffer, 0, buffer.size)
        if (read < 0) throw AirecHttpError(400, "multipart boundary not found")

        val data = ByteArray(tail.size + read)
        System.arraycopy(tail, 0, data, 0, tail.size)
        System.arraycopy(buffer, 0, data, tail.size, read)

        val boundaryIndex = data.indexOfBytes(boundaryPattern)
        if (boundaryIndex >= 0) {
            if (boundaryIndex > 0) onBytes(data, 0, boundaryIndex)
            val afterStart = boundaryIndex + boundaryPattern.size
            if (afterStart < data.size) {
                body.unread(data.copyOfRange(afterStart, data.size))
            }
            return
        }

        val keep = min(boundaryPattern.size - 1, data.size)
        val writeLength = data.size - keep
        if (writeLength > 0) onBytes(data, 0, writeLength)
        tail = data.copyOfRange(writeLength, data.size)
    }
}

private fun ByteArray.indexOfBytes(pattern: ByteArray): Int {
    if (pattern.isEmpty() || this.size < pattern.size) return -1
    for (i in 0..(this.size - pattern.size)) {
        var matched = true
        for (j in pattern.indices) {
            if (this[i + j] != pattern[j]) {
                matched = false
                break
            }
        }
        if (matched) return i
    }
    return -1
}

private fun readHttpHeaders(input: InputStream): Map<String, String> {
    val headers = mutableMapOf<String, String>()
    while (true) {
        val line = readAsciiLine(input, 8192) ?: throw AirecHttpError(400, "headers missing")
        if (line.isEmpty()) return headers
        val split = line.indexOf(':')
        if (split <= 0) throw AirecHttpError(400, "bad header")
        headers[line.substring(0, split).lowercase(Locale.US)] = line.substring(split + 1).trim()
    }
}

private fun readAsciiLine(input: InputStream, maxBytes: Int): String? {
    val out = ByteArrayOutputStream()
    while (out.size() <= maxBytes) {
        val b = input.read()
        if (b < 0) {
            return if (out.size() == 0) null else out.toString(StandardCharsets.ISO_8859_1.name())
        }
        if (b == '\n'.code) {
            return out.toString(StandardCharsets.ISO_8859_1.name()).trimEnd('\r')
        }
        out.write(b)
    }
    throw AirecHttpError(400, "line too long")
}

private fun parseBoundary(contentType: String): String? {
    return contentType
        .split(";")
        .map { it.trim() }
        .firstOrNull { it.startsWith("boundary=", ignoreCase = true) }
        ?.substringAfter("=")
        ?.trim()
        ?.trim('"')
        ?.takeIf { it.isNotBlank() }
}

private fun parseContentDisposition(value: String): Map<String, String> {
    val result = mutableMapOf<String, String>()
    value.split(";").forEach { part ->
        val item = part.trim()
        val split = item.indexOf('=')
        if (split > 0) {
            val key = item.substring(0, split).trim().lowercase(Locale.US)
            val raw = item.substring(split + 1).trim()
            result[key] = raw.trim('"')
        }
    }
    return result
}

private fun validateFileName(fileName: String) {
    if (fileName.isBlank()) throw AirecHttpError(400, "缺少 fileName")
    if (fileName.contains("/") || fileName.contains("\\")) {
        throw AirecHttpError(400, "fileName 不允许包含路径")
    }
    if (!Regex("""^\d{14}\.wav$""").matches(fileName)) {
        throw AirecHttpError(400, "fileName 必须符合 yyyyMMddHHmmss.wav")
    }
}

private fun validateSn(sn: String) {
    if (sn.isBlank()) throw AirecHttpError(400, "缺少 sn")
    if (!Regex("""^[A-Za-z0-9_-]+$""").matches(sn)) {
        throw AirecHttpError(400, "sn 只能包含字母、数字、下划线和中划线")
    }
}

private fun requireInside(root: File, child: File) {
    val rootPath = root.canonicalPath
    val childPath = child.canonicalPath
    if (childPath != rootPath && !childPath.startsWith("$rootPath${File.separator}")) {
        throw AirecHttpError(400, "非法保存路径")
    }
}

private fun sendResponse(output: OutputStream, statusCode: Int, body: String) {
    val reason = when (statusCode) {
        200 -> "OK"
        400 -> "Bad Request"
        403 -> "Forbidden"
        404 -> "Not Found"
        405 -> "Method Not Allowed"
        411 -> "Length Required"
        500 -> "Internal Server Error"
        else -> "Error"
    }
    val bodyBytes = body.toByteArray(StandardCharsets.UTF_8)
    val header = "HTTP/1.1 $statusCode $reason\r\n" +
        "Connection: close\r\n" +
        "Content-Type: text/plain; charset=utf-8\r\n" +
        "Content-Length: ${bodyBytes.size}\r\n" +
        "\r\n"
    output.write(header.toByteArray(StandardCharsets.ISO_8859_1))
    output.write(bodyBytes)
}

fun formatAirecUploadTime(timeMs: Long): String {
    return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(timeMs))
}
