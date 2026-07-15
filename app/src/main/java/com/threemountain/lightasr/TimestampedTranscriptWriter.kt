package com.threemountain.lightasr

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

enum class TimestampSource {
    ASR_TOKEN_TIMESTAMP,
    ASR_SEGMENT_TIMESTAMP,
    VAD_BOUNDARY,
    ESTIMATED,
}

data class RecordingTimeContext(
    val recordingStartEpochMs: Long,
    val timezoneId: String,
    val source: String = "fallback-default",
)

data class TimestampedTranscriptSegment(
    val relativeStartUs: Long,
    val relativeEndUs: Long,
    val absoluteStartEpochMs: Long,
    val absoluteEndEpochMs: Long,
    val speaker: String,
    val text: String,
    val timestampSource: TimestampSource,
)

object TimestampedTranscriptWriter {
    fun buildTxt(
        sourceFileName: String,
        recordingTimeContext: RecordingTimeContext,
        durationSec: Double,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int,
        segmentationMode: String,
        stats: TranscriptionStats,
        transcriptSegments: List<TimestampedTranscriptSegment>,
    ): String {
        val formatter = absoluteFormatter(recordingTimeContext.timezoneId)
        val recordingStart = formatter.format(Date(recordingTimeContext.recordingStartEpochMs))

        return buildString {
            appendLine("录音文件：$sourceFileName")
            appendLine("录音开始时间：$recordingStart")
            appendLine("录音开始时间来源：${recordingTimeContext.source}")
            appendLine("时区：${recordingTimeContext.timezoneId}")
            appendLine("识别模式：local-asr-natural-utterance-vad")
            appendLine("输出模式：relative-time-table-one-natural-utterance-per-row")
            appendLine("时间戳来源：${timestampSourceSummary(transcriptSegments)}")
            appendLine("音频时长：${"%.2f".format(Locale.US, durationSec)} 秒")
            appendLine("采样率：$sampleRate Hz")
            appendLine("声道数：$channels")
            appendLine("位深：$bitsPerSample-bit")
            appendLine("切分方式：$segmentationMode")
            appendLine()
            appendLine("识别统计：")
            appendLine("总时长：${formatRelativeTimestamp(stats.totalDurationMs * 1000L)}")
            appendLine("检测到的人声时长：${formatRelativeTimestamp(stats.speechDurationMs * 1000L)}")
            appendLine("跳过的无人声时长：${formatRelativeTimestamp(stats.skippedNoSpeechDurationMs * 1000L)}")
            appendLine("自然话语数量：${stats.totalChunks}")
            appendLine("送入 ASR 的话语数量：${stats.speechChunks}")
            appendLine("跳过的话语数量：${stats.skippedChunks}")
            appendLine("ASR 实际处理时长：${formatRelativeTimestamp(stats.asrProcessedDurationMs * 1000L)}")
            appendLine()
            appendLine("| 开始时间 | 结束时间 | 说话人 | 内容 |")
            appendLine("|---|---|---|---|")

            if (transcriptSegments.isEmpty()) {
                appendLine("|  |  | unknown | 未检测到有效人声，已跳过识别。 |")
            } else {
                transcriptSegments.forEach { segment ->
                    appendLine(
                        "| ${formatRelativeTimestamp(segment.relativeStartUs)} " +
                            "| ${formatRelativeTimestamp(segment.relativeEndUs)} " +
                            "| ${escapeTableCell(segment.speaker)} " +
                            "| ${escapeTableCell(segment.text)} |"
                    )
                }
            }
        }
    }

    fun buildDisplayText(
        recordingTimeContext: RecordingTimeContext,
        transcriptSegments: List<TimestampedTranscriptSegment>,
    ): String {
        if (transcriptSegments.isEmpty()) return ""

        val formatter = absoluteFormatter(recordingTimeContext.timezoneId)
        return transcriptSegments.joinToString("\n") { segment ->
            "[${formatter.format(Date(segment.absoluteStartEpochMs))} - " +
                "${formatter.format(Date(segment.absoluteEndEpochMs))}] " +
                segment.text
        }
    }

    private fun timestampSourceSummary(segments: List<TimestampedTranscriptSegment>): String {
        if (segments.isEmpty()) return "NONE"
        return segments
            .map { it.timestampSource }
            .distinct()
            .joinToString(",") { it.name }
    }

    private fun absoluteFormatter(timezoneId: String): SimpleDateFormat {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).apply {
            timeZone = TimeZone.getTimeZone(timezoneId)
        }
    }

    private fun escapeTableCell(value: String): String {
        return value
            .replace("|", "\\|")
            .replace("\r", " ")
            .replace("\n", " ")
            .filter { !it.isISOControl() || it == '\t' }
            .trim()
    }

    private fun formatRelativeTimestamp(us: Long): String {
        val totalMs = us / 1000L
        val ms = totalMs % 1000L
        val totalSeconds = totalMs / 1000L
        val s = totalSeconds % 60L
        val totalMinutes = totalSeconds / 60L
        val m = totalMinutes % 60L
        val h = totalMinutes / 60L
        return "%02d:%02d:%02d.%03d".format(Locale.US, h, m, s, ms)
    }
}
