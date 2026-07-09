package com.threemountain.lightasr

import android.util.Log
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

private const val FINE_VAD_TAG = "LightASR-FineVAD"

data class SpeechRegion(
    val startSample: Long,
    val endSample: Long,
    val startUs: Long,
    val endUs: Long,
) {
    val durationUs: Long = endUs - startUs
}

data class FineGrainedVadConfig(
    val minSpeechMs: Long = 300L,
    val minSilenceMs: Long = 1_200L,
    val speechPadMs: Long = 800L,
    val mergeGapMs: Long = 1_500L,
    val minRecognitionWindowMs: Long = 5_000L,
    val maxSegmentMs: Long = 15_000L,
)

class FineGrainedVadSegmenter(
    private val config: FineGrainedVadConfig = FineGrainedVadConfig(),
) {
    fun segment(
        chunkStartUs: Long,
        chunkEndUs: Long,
        sourceSampleRate: Int,
        vadResult: VadResult,
    ): List<SpeechRegion> {
        require(sourceSampleRate > 0) { "invalid sourceSampleRate=$sourceSampleRate" }
        if (!vadResult.hasSpeech || chunkEndUs <= chunkStartUs) return emptyList()

        val sourceSegments = vadResult.rawSpeechSegments
            .ifEmpty { vadResult.speechSegments }
            .filter { it.endMs > it.startMs }

        val chunkStartMs = chunkStartUs / 1000L
        val chunkEndMs = chunkEndUs / 1000L
        val padded = sourceSegments
            .map { segment ->
                SpeechSegment(
                    startMs = max(chunkStartMs, segment.startMs - config.speechPadMs),
                    endMs = min(chunkEndMs, segment.endMs + config.speechPadMs),
                )
            }
            .filter { it.endMs - it.startMs >= config.minSpeechMs }

        val merged = mergeCloseSegments(padded)
        val contextExpanded = mergeCloseSegments(
            expandShortSegmentsForRecognition(
                segments = merged,
                chunkStartMs = chunkStartMs,
                chunkEndMs = chunkEndMs,
            )
        )
        val split = splitLongSegments(contextExpanded)
        val regions = split
            .mapNotNull { segment ->
                val startUs = (segment.startMs * 1000L).coerceIn(chunkStartUs, chunkEndUs)
                val endUs = (segment.endMs * 1000L).coerceIn(chunkStartUs, chunkEndUs)
                if (endUs <= startUs) {
                    null
                } else {
                    SpeechRegion(
                        startSample = usToSampleIndex(startUs - chunkStartUs, sourceSampleRate),
                        endSample = usToSampleIndex(endUs - chunkStartUs, sourceSampleRate),
                        startUs = startUs,
                        endUs = endUs,
                    )
                }
            }
            .filter { it.endSample > it.startSample }

        Log.i(
            FINE_VAD_TAG,
            "fine vad chunkStartUs=$chunkStartUs, durationUs=${chunkEndUs - chunkStartUs}, " +
                "raw=${sourceSegments.size}, padded=${padded.size}, merged=${merged.size}, " +
                "contextExpanded=${contextExpanded.size}, " +
                "regions=${regions.size}, minSpeechMs=${config.minSpeechMs}, " +
                "minSilenceMs=${config.minSilenceMs}, padMs=${config.speechPadMs}, " +
                "mergeGapMs=${config.mergeGapMs}, minRecognitionWindowMs=${config.minRecognitionWindowMs}, " +
                "maxSegmentMs=${config.maxSegmentMs}"
        )
        regions.forEachIndexed { index, region ->
            Log.i(
                FINE_VAD_TAG,
                "fine region ${index + 1}/${regions.size}, localStartUs=${region.startUs - chunkStartUs}, " +
                    "localEndUs=${region.endUs - chunkStartUs}, globalStartUs=${region.startUs}, " +
                    "globalEndUs=${region.endUs}, durationSec=${"%.3f".format(Locale.US, region.durationUs / 1_000_000.0)}"
            )
        }

        return regions
    }

    private fun expandShortSegmentsForRecognition(
        segments: List<SpeechSegment>,
        chunkStartMs: Long,
        chunkEndMs: Long,
    ): List<SpeechSegment> {
        if (segments.isEmpty()) return emptyList()

        val minWindowMs = config.minRecognitionWindowMs.coerceAtLeast(config.minSpeechMs)
        return segments.map { segment ->
            val durationMs = segment.endMs - segment.startMs
            if (durationMs >= minWindowMs) {
                segment
            } else {
                val missingMs = minWindowMs - durationMs
                var startMs = (segment.startMs - missingMs / 2L).coerceAtLeast(chunkStartMs)
                var endMs = (segment.endMs + missingMs - missingMs / 2L).coerceAtMost(chunkEndMs)

                val stillMissingMs = minWindowMs - (endMs - startMs)
                if (stillMissingMs > 0L && startMs == chunkStartMs) {
                    endMs = (endMs + stillMissingMs).coerceAtMost(chunkEndMs)
                } else if (stillMissingMs > 0L && endMs == chunkEndMs) {
                    startMs = (startMs - stillMissingMs).coerceAtLeast(chunkStartMs)
                }

                SpeechSegment(startMs = startMs, endMs = endMs)
            }
        }
    }

    private fun mergeCloseSegments(segments: List<SpeechSegment>): List<SpeechSegment> {
        if (segments.isEmpty()) return emptyList()

        val merged = mutableListOf<SpeechSegment>()
        var current = segments.sortedBy { it.startMs }.first()

        for (next in segments.sortedBy { it.startMs }.drop(1)) {
            val gapMs = next.startMs - current.endMs
            current = if (gapMs <= config.mergeGapMs || gapMs < config.minSilenceMs) {
                SpeechSegment(
                    startMs = current.startMs,
                    endMs = max(current.endMs, next.endMs),
                )
            } else {
                merged += current
                next
            }
        }
        merged += current
        return merged
    }

    private fun splitLongSegments(segments: List<SpeechSegment>): List<SpeechSegment> {
        if (segments.isEmpty()) return emptyList()

        val maxSegmentMs = config.maxSegmentMs.coerceAtLeast(config.minSpeechMs)
        val result = mutableListOf<SpeechSegment>()
        for (segment in segments) {
            var startMs = segment.startMs
            while (segment.endMs - startMs > maxSegmentMs) {
                val endMs = startMs + maxSegmentMs
                result += SpeechSegment(startMs = startMs, endMs = endMs)
                startMs = endMs
            }
            if (segment.endMs > startMs) {
                result += SpeechSegment(startMs = startMs, endMs = segment.endMs)
            }
        }
        return result
    }

    private fun usToSampleIndex(relativeUs: Long, sampleRate: Int): Long {
        return ((relativeUs.coerceAtLeast(0L) * sampleRate.toLong()) + 500_000L) / 1_000_000L
    }
}
