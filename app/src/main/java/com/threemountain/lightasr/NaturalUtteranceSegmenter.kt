package com.threemountain.lightasr

import kotlin.math.max
import kotlin.math.min

data class NaturalUtteranceConfig(
    val mergeGapMs: Long = 450L,
    val speechPaddingMs: Long = 220L,
    val minSpeechMs: Long = 180L,
)

/**
 * Turns raw VAD speech islands into non-overlapping, natural utterance ranges.
 * Short pauses inside one phrase are retained; longer pauses start a new TXT row.
 */
object NaturalUtteranceSegmenter {
    fun assemble(
        rawSegments: List<SpeechSegment>,
        audioDurationMs: Long,
        config: NaturalUtteranceConfig = NaturalUtteranceConfig(),
    ): List<SpeechSegment> {
        if (audioDurationMs <= 0L || rawSegments.isEmpty()) return emptyList()

        val sanitized = rawSegments
            .map { segment ->
                SpeechSegment(
                    startMs = segment.startMs.coerceIn(0L, audioDurationMs),
                    endMs = segment.endMs.coerceIn(0L, audioDurationMs),
                )
            }
            .filter { it.endMs > it.startMs }
            .sortedBy { it.startMs }
        if (sanitized.isEmpty()) return emptyList()

        val merged = mutableListOf<SpeechSegment>()
        var current = sanitized.first()
        for (next in sanitized.drop(1)) {
            current = if (next.startMs <= current.endMs + config.mergeGapMs) {
                SpeechSegment(current.startMs, max(current.endMs, next.endMs))
            } else {
                merged += current
                next
            }
        }
        merged += current

        val padded = merged
            .filter { it.endMs - it.startMs >= config.minSpeechMs }
            .map { segment ->
                SpeechSegment(
                    startMs = max(0L, segment.startMs - config.speechPaddingMs),
                    endMs = min(audioDurationMs, segment.endMs + config.speechPaddingMs),
                )
            }
            .toMutableList()

        // Padding must not make two independent utterances overlap. If it does,
        // split the remaining silence at its midpoint so overlap-only dedup is safe.
        for (index in 1 until padded.size) {
            val previous = padded[index - 1]
            val next = padded[index]
            if (next.startMs < previous.endMs) {
                val midpoint = (previous.endMs + next.startMs) / 2L
                padded[index - 1] = previous.copy(endMs = midpoint)
                padded[index] = next.copy(startMs = midpoint)
            }
        }

        return padded.filter { it.endMs > it.startMs }
    }
}
