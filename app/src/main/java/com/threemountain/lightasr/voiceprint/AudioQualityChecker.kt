package com.threemountain.lightasr.voiceprint

import kotlin.math.sqrt

object AudioQualityChecker {
    fun checkForEnrollment(
        audio: FloatArray,
        sampleRate: Int,
        config: VoiceprintConfig,
    ): AudioQualityResult {
        return check(
            audio = audio,
            sampleRate = sampleRate,
            minDurationSec = config.minEnrollSampleSec,
            maxDurationSec = config.maxEnrollSampleSec,
            targetSampleRate = config.targetSampleRate,
        )
    }

    fun checkForIdentification(
        audio: FloatArray,
        sampleRate: Int,
        config: VoiceprintConfig,
    ): AudioQualityResult {
        return check(
            audio = audio,
            sampleRate = sampleRate,
            minDurationSec = config.minIdentifySpeechSec,
            maxDurationSec = null,
            targetSampleRate = config.targetSampleRate,
        )
    }

    private fun check(
        audio: FloatArray,
        sampleRate: Int,
        minDurationSec: Double,
        maxDurationSec: Double?,
        targetSampleRate: Int,
    ): AudioQualityResult {
        val durationSec = if (sampleRate > 0) audio.size.toDouble() / sampleRate.toDouble() else 0.0
        val rms = calculateRms(audio)
        val clippingRatio = calculateClippingRatio(audio)
        val reason = when {
            sampleRate != targetSampleRate -> "invalid_sample_rate"
            durationSec < minDurationSec -> "audio_too_short"
            maxDurationSec != null && durationSec > maxDurationSec -> "audio_too_long"
            rms <= 0.005 -> "audio_too_quiet"
            clippingRatio >= 0.05 -> "audio_clipping_too_high"
            else -> "ok"
        }

        return AudioQualityResult(
            ok = reason == "ok",
            reason = reason,
            durationSec = durationSec,
            rms = rms,
            clippingRatio = clippingRatio,
        )
    }

    private fun calculateRms(audio: FloatArray): Double {
        if (audio.isEmpty()) return 0.0
        var sumSquares = 0.0
        for (sample in audio) {
            val value = sample.toDouble()
            sumSquares += value * value
        }
        return sqrt(sumSquares / audio.size.toDouble())
    }

    private fun calculateClippingRatio(audio: FloatArray): Double {
        if (audio.isEmpty()) return 0.0
        val clipped = audio.count { it >= 0.999f || it <= -0.999f }
        return clipped.toDouble() / audio.size.toDouble()
    }
}
