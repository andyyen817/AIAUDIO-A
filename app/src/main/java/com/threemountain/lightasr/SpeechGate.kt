package com.threemountain.lightasr

import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import java.util.Locale
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong

private const val SPEECH_GATE_TAG = "LightASR-SpeechGate"

data class SpeechSegment(
    val startMs: Long,
    val endMs: Long,
)

data class VadResult(
    val hasSpeech: Boolean,
    val speechSegments: List<SpeechSegment>,
    val totalSpeechMs: Long,
    val speechRatio: Float,
    val rawSpeechSegments: List<SpeechSegment> = speechSegments,
)

data class SpeechGateConfig(
    val modelAssetPath: String = "models/vad/silero_vad.onnx",
    val targetSampleRate: Int = 16000,
    val threshold: Float = 0.5f,
    val minSpeechMs: Long = 600L,
    val minSpeechRatio: Float = 0.03f,
    val speechPadMs: Long = 300L,
    val mergeGapMs: Long = 500L,
    val modelMinSilenceDurationSec: Float = 0.25f,
    val modelMinSpeechDurationSec: Float = 0.25f,
    val modelMaxSpeechDurationSec: Float = 15.0f,
    val modelWindowSize: Int = 512,
)

class SpeechGate(
    assetManager: AssetManager,
    private val config: SpeechGateConfig = SpeechGateConfig(),
) {
    private val vad: Vad

    init {
        val vadConfig = VadModelConfig(
            sileroVadModelConfig = SileroVadModelConfig(
                model = config.modelAssetPath,
                threshold = config.threshold,
                minSilenceDuration = config.modelMinSilenceDurationSec,
                minSpeechDuration = config.modelMinSpeechDurationSec,
                windowSize = config.modelWindowSize,
                maxSpeechDuration = config.modelMaxSpeechDurationSec,
            ),
            sampleRate = config.targetSampleRate,
            numThreads = 1,
            provider = "cpu",
            debug = false,
        )

        vad = Vad(assetManager = assetManager, config = vadConfig)
        runInitSmokeTest()
        Log.i(
            SPEECH_GATE_TAG,
            "VAD initialized model=${config.modelAssetPath}, sampleRate=${config.targetSampleRate}, " +
                "threshold=${config.threshold}, minSpeechMs=${config.minSpeechMs}, " +
                "minSpeechRatio=${config.minSpeechRatio}, speechPadMs=${config.speechPadMs}, " +
                "mergeGapMs=${config.mergeGapMs}, modelMinSilenceSec=${config.modelMinSilenceDurationSec}, " +
                "modelMinSpeechSec=${config.modelMinSpeechDurationSec}, " +
                "modelMaxSpeechSec=${config.modelMaxSpeechDurationSec}"
        )
    }

    @Synchronized
    fun analyze(
        chunkStartMs: Long,
        samples: FloatArray,
        sampleRate: Int,
    ): VadResult {
        require(sampleRate > 0) { "invalid VAD input sampleRate=$sampleRate" }
        if (samples.isEmpty()) {
            return VadResult(false, emptyList(), 0L, 0.0f)
        }

        val chunkDurationMs = samplesToMs(samples.size, sampleRate).coerceAtLeast(1L)
        val vadSamples = if (sampleRate == config.targetSampleRate) {
            samples
        } else {
            resampleLinear(samples, sampleRate, config.targetSampleRate)
        }

        val rawSegments = mutableListOf<SpeechSegment>()
        vad.reset()
        val frame = FloatArray(config.modelWindowSize)
        var sampleOffset = 0
        while (sampleOffset < vadSamples.size) {
            val frameEnd = min(sampleOffset + config.modelWindowSize, vadSamples.size)
            frame.fill(0.0f)
            vadSamples.copyInto(
                destination = frame,
                destinationOffset = 0,
                startIndex = sampleOffset,
                endIndex = frameEnd,
            )
            vad.acceptWaveform(frame)
            drainDetectedSegments(rawSegments, chunkStartMs, chunkDurationMs)
            sampleOffset = frameEnd
        }
        vad.flush()
        drainDetectedSegments(rawSegments, chunkStartMs, chunkDurationMs)
        vad.clear()

        val paddedAndMerged = mergeSpeechSegments(
            rawSegments
                .map { segment ->
                    SpeechSegment(
                        startMs = (segment.startMs - config.speechPadMs).coerceAtLeast(chunkStartMs),
                        endMs = (segment.endMs + config.speechPadMs).coerceAtMost(chunkStartMs + chunkDurationMs),
                    )
                }
                .filter { it.endMs > it.startMs },
            mergeGapMs = config.mergeGapMs,
        )
        val totalSpeechMs = paddedAndMerged.sumOf { it.endMs - it.startMs }
        val speechRatio = totalSpeechMs.toFloat() / chunkDurationMs.toFloat()
        val hasSpeech = totalSpeechMs >= config.minSpeechMs || speechRatio >= config.minSpeechRatio

        Log.i(
            SPEECH_GATE_TAG,
            "VAD chunk startMs=$chunkStartMs, durationMs=$chunkDurationMs, " +
                "hasSpeech=$hasSpeech, totalSpeechMs=$totalSpeechMs, " +
                "speechRatio=${"%.4f".format(Locale.US, speechRatio)}, " +
                "rawSegments=${rawSegments.size}, mergedSegments=${paddedAndMerged.size}"
        )

        return VadResult(
            hasSpeech = hasSpeech,
            speechSegments = if (hasSpeech) paddedAndMerged else emptyList(),
            totalSpeechMs = if (hasSpeech) totalSpeechMs else 0L,
            speechRatio = speechRatio,
            rawSpeechSegments = if (hasSpeech) rawSegments else emptyList(),
        )
    }

    fun release() {
        vad.release()
    }

    private fun runInitSmokeTest() {
        vad.reset()
        vad.acceptWaveform(FloatArray(config.modelWindowSize))
        vad.flush()
        vad.clear()
        vad.reset()
    }

    private fun drainDetectedSegments(
        output: MutableList<SpeechSegment>,
        chunkStartMs: Long,
        chunkDurationMs: Long,
    ) {
        while (!vad.empty()) {
            val segment = vad.front()
            val relativeStartMs = samplesToMs(segment.start, config.targetSampleRate)
            val relativeEndMs = samplesToMs(
                segment.start + segment.samples.size,
                config.targetSampleRate,
            )
            if (relativeEndMs > relativeStartMs) {
                output += SpeechSegment(
                    startMs = (chunkStartMs + relativeStartMs).coerceAtLeast(chunkStartMs),
                    endMs = (chunkStartMs + relativeEndMs)
                        .coerceAtMost(chunkStartMs + chunkDurationMs),
                )
            }
            vad.pop()
        }
    }

    private fun mergeSpeechSegments(
        segments: List<SpeechSegment>,
        mergeGapMs: Long,
    ): List<SpeechSegment> {
        if (segments.isEmpty()) return emptyList()

        val sorted = segments.sortedBy { it.startMs }
        val merged = mutableListOf<SpeechSegment>()
        var current = sorted.first()

        for (next in sorted.drop(1)) {
            current = if (next.startMs - current.endMs <= mergeGapMs) {
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

    private fun samplesToMs(sampleCount: Int, sampleRate: Int): Long {
        return ((sampleCount.toDouble() * 1000.0) / sampleRate.toDouble()).roundToLong()
    }

    private fun resampleLinear(
        source: FloatArray,
        sourceRate: Int,
        targetRate: Int,
    ): FloatArray {
        if (source.isEmpty()) return FloatArray(0)
        val targetSize = ((source.size.toDouble() * targetRate.toDouble()) / sourceRate.toDouble())
            .roundToLong()
            .coerceAtLeast(1L)
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()
        val ratio = sourceRate.toDouble() / targetRate.toDouble()
        return FloatArray(targetSize) { index ->
            val sourcePosition = index.toDouble() * ratio
            val left = floor(sourcePosition).toInt().coerceIn(0, source.lastIndex)
            val right = min(left + 1, source.lastIndex)
            val fraction = sourcePosition - left.toDouble()
            (source[left].toDouble() * (1.0 - fraction) + source[right].toDouble() * fraction).toFloat()
        }
    }
}
