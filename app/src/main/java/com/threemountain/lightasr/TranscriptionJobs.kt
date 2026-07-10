package com.threemountain.lightasr

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

enum class TranscriptionJobStatus {
    PREPARING, TRANSCRIBING, INTERRUPTED, LOCAL_COMPLETED, FAILED, CANCELED,
    UPLOADING, UPLOADED, UPLOAD_FAILED,
}

data class TranscriptionJob(
    val id: String,
    val sourceName: String,
    val sourceType: String,
    val originalUri: String,
    val audioPath: String? = null,
    val transcriptPath: String? = null,
    val audioSha256: String? = null,
    val durationMs: Long = 0L,
    val totalChunks: Int = 0,
    val nextChunkIndex: Int = 0,
    val progressPercent: Int = 0,
    val status: TranscriptionJobStatus = TranscriptionJobStatus.PREPARING,
    val serverRecordingId: String? = null,
    val errorMessage: String? = null,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
)

data class TranscriptionCheckpoint(
    val nextRangeIndex: Int,
    val speechDurationMs: Long,
    val skippedNoSpeechDurationMs: Long,
    val speechChunks: Int,
    val skippedChunks: Int,
    val asrProcessedDurationMs: Long,
    val segments: List<RecognitionSegment>,
)

object TranscriptionJobRepository {
    private const val PREFS = "transcription_jobs_v1"
    private const val KEY = "jobs"
    private const val MAX_JOBS = 100

    @Synchronized
    fun create(context: Context, sourceName: String, sourceType: String, originalUri: String): TranscriptionJob {
        val job = TranscriptionJob(
            id = UUID.randomUUID().toString(),
            sourceName = sourceName,
            sourceType = sourceType,
            originalUri = originalUri,
        )
        save(context, (list(context) + job).sortedByDescending { it.updatedAt }.take(MAX_JOBS))
        return job
    }

    @Synchronized
    fun get(context: Context, jobId: String): TranscriptionJob? =
        list(context).firstOrNull { it.id == jobId }

    @Synchronized
    fun list(context: Context): List<TranscriptionJob> {
        val raw = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, "[]") ?: "[]"
        val array = runCatching { JSONArray(raw) }.getOrElse { JSONArray() }
        return buildList {
            for (index in 0 until array.length()) {
                val item = array.optJSONObject(index) ?: continue
                runCatching { add(item.toJob()) }
            }
        }.sortedByDescending { it.updatedAt }
    }

    @Synchronized
    fun update(
        context: Context,
        jobId: String,
        transform: (TranscriptionJob) -> TranscriptionJob,
    ): TranscriptionJob? {
        val jobs = list(context).toMutableList()
        val index = jobs.indexOfFirst { it.id == jobId }
        if (index < 0) return null
        val updated = transform(jobs[index]).copy(updatedAt = System.currentTimeMillis())
        jobs[index] = updated
        save(context, jobs.sortedByDescending { it.updatedAt }.take(MAX_JOBS))
        return updated
    }

    @Synchronized
    fun markInterruptedJobs(context: Context) {
        val updated = list(context).map { job ->
            if (job.status in setOf(
                    TranscriptionJobStatus.PREPARING,
                    TranscriptionJobStatus.TRANSCRIBING,
                    TranscriptionJobStatus.UPLOADING,
                )
            ) {
                job.copy(
                    status = TranscriptionJobStatus.INTERRUPTED,
                    errorMessage = "任务被系统中断，可以从已保存进度继续。",
                    updatedAt = System.currentTimeMillis(),
                )
            } else job
        }
        save(context, updated)
    }

    private fun save(context: Context, jobs: List<TranscriptionJob>) {
        val array = JSONArray()
        jobs.forEach { array.put(it.toJson()) }
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(KEY, array.toString()).commit()
    }

    private fun TranscriptionJob.toJson() = JSONObject()
        .put("id", id).put("sourceName", sourceName).put("sourceType", sourceType)
        .put("originalUri", originalUri).put("audioPath", audioPath)
        .put("transcriptPath", transcriptPath).put("audioSha256", audioSha256)
        .put("durationMs", durationMs).put("totalChunks", totalChunks)
        .put("nextChunkIndex", nextChunkIndex).put("progressPercent", progressPercent)
        .put("status", status.name).put("serverRecordingId", serverRecordingId)
        .put("errorMessage", errorMessage).put("createdAt", createdAt).put("updatedAt", updatedAt)

    private fun JSONObject.toJob() = TranscriptionJob(
        id = getString("id"),
        sourceName = optString("sourceName", "unknown.wav"),
        sourceType = optString("sourceType", "local"),
        originalUri = optString("originalUri"),
        audioPath = nullable("audioPath"),
        transcriptPath = nullable("transcriptPath"),
        audioSha256 = nullable("audioSha256"),
        durationMs = optLong("durationMs"),
        totalChunks = optInt("totalChunks"),
        nextChunkIndex = optInt("nextChunkIndex"),
        progressPercent = optInt("progressPercent"),
        status = runCatching { TranscriptionJobStatus.valueOf(optString("status")) }
            .getOrDefault(TranscriptionJobStatus.FAILED),
        serverRecordingId = nullable("serverRecordingId"),
        errorMessage = nullable("errorMessage"),
        createdAt = optLong("createdAt"),
        updatedAt = optLong("updatedAt"),
    )

    private fun JSONObject.nullable(key: String): String? =
        if (!has(key) || isNull(key)) null else optString(key).takeIf { it.isNotBlank() && it != "null" }
}

object TranscriptionCheckpointStore {
    private const val STATE = "checkpoint.json"
    private const val SEGMENTS = "segments.jsonl"

    @Synchronized
    fun load(context: Context, jobId: String): TranscriptionCheckpoint? {
        val dir = jobDirectory(context, jobId)
        val stateFile = File(dir, STATE)
        if (!stateFile.exists()) return null
        val state = runCatching { JSONObject(stateFile.readText(Charsets.UTF_8)) }.getOrNull() ?: return null
        val byRange = linkedMapOf<Int, RecognitionSegment>()
        File(dir, SEGMENTS).takeIf { it.exists() }?.forEachLine(Charsets.UTF_8) { line ->
            runCatching {
                val item = JSONObject(line)
                byRange[item.getInt("range")] = item.getJSONObject("segment").toSegment()
            }
        }
        return TranscriptionCheckpoint(
            nextRangeIndex = state.optInt("nextRangeIndex"),
            speechDurationMs = state.optLong("speechDurationMs"),
            skippedNoSpeechDurationMs = state.optLong("skippedNoSpeechDurationMs"),
            speechChunks = state.optInt("speechChunks"),
            skippedChunks = state.optInt("skippedChunks"),
            asrProcessedDurationMs = state.optLong("asrProcessedDurationMs"),
            segments = byRange.toSortedMap().values.toList(),
        )
    }

    @Synchronized
    fun saveAfterChunk(
        context: Context,
        jobId: String,
        rangePosition: Int,
        nextRangeIndex: Int,
        speechDurationMs: Long,
        skippedNoSpeechDurationMs: Long,
        speechChunks: Int,
        skippedChunks: Int,
        asrProcessedDurationMs: Long,
        segment: RecognitionSegment?,
    ) {
        val dir = jobDirectory(context, jobId)
        check(dir.exists() || dir.mkdirs()) { "无法创建任务检查点目录" }
        if (segment != null) {
            val line = JSONObject().put("range", rangePosition).put("segment", segment.toJson()).toString() + "\n"
            FileOutputStream(File(dir, SEGMENTS), true).use {
                it.write(line.toByteArray(Charsets.UTF_8))
                it.flush()
                it.fd.sync()
            }
        }
        val state = JSONObject()
            .put("nextRangeIndex", nextRangeIndex)
            .put("speechDurationMs", speechDurationMs)
            .put("skippedNoSpeechDurationMs", skippedNoSpeechDurationMs)
            .put("speechChunks", speechChunks)
            .put("skippedChunks", skippedChunks)
            .put("asrProcessedDurationMs", asrProcessedDurationMs)
        atomicWrite(File(dir, STATE), state.toString())
    }

    fun jobDirectory(context: Context, jobId: String): File =
        File(context.filesDir, "transcription_jobs/$jobId")

    private fun RecognitionSegment.toJson() = JSONObject()
        .put("index", index).put("startMs", startMs).put("endMs", endMs)
        .put("text", text).put("cutMode", cutMode).put("startUs", startUs).put("endUs", endUs)

    private fun JSONObject.toSegment() = RecognitionSegment(
        index = optInt("index"), startMs = optLong("startMs"), endMs = optLong("endMs"),
        text = optString("text"), cutMode = optString("cutMode", "checkpoint"),
        startUs = optLong("startUs"), endUs = optLong("endUs"),
    )

    private fun atomicWrite(target: File, content: String) {
        val parent = target.parentFile ?: error("invalid checkpoint path")
        check(parent.exists() || parent.mkdirs()) { "无法创建检查点目录" }
        val temp = File(parent, target.name + ".tmp")
        FileOutputStream(temp).use {
            it.write(content.toByteArray(Charsets.UTF_8))
            it.flush()
            it.fd.sync()
        }
        if (target.exists()) target.delete()
        if (!temp.renameTo(target)) {
            temp.copyTo(target, overwrite = true)
            temp.delete()
        }
    }
}
