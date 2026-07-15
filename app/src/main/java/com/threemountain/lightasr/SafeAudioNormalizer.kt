package com.threemountain.lightasr

import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

data class AudioNormalizationConfig(
    val targetRmsDb: Double = -20.0,
    val maxGainDb: Double = 12.0,
    val maxAttenuationDb: Double = 6.0,
    val peakCeiling: Double = 0.95,
)

data class AudioNormalizationResult(
    val samples: FloatArray,
    val inputRmsDb: Double,
    val outputRmsDb: Double,
    val inputPeak: Double,
    val outputPeak: Double,
    val gain: Double,
    val removedDcOffset: Double,
    val clippedSamples: Int,
    val applied: Boolean,
)

object SafeAudioNormalizer {
    fun normalize(
        input: FloatArray,
        config: AudioNormalizationConfig = AudioNormalizationConfig(),
    ): AudioNormalizationResult {
        if (input.isEmpty()) {
            return AudioNormalizationResult(
                samples = input,
                inputRmsDb = -120.0,
                outputRmsDb = -120.0,
                inputPeak = 0.0,
                outputPeak = 0.0,
                gain = 1.0,
                removedDcOffset = 0.0,
                clippedSamples = 0,
                applied = false,
            )
        }

        val mean = input.sumOf { it.toDouble() } / input.size.toDouble()
        var sumSquares = 0.0
        var inputPeak = 0.0
        for (sample in input) {
            val centered = sample.toDouble() - mean
            sumSquares += centered * centered
            inputPeak = maxOf(inputPeak, abs(centered))
        }

        val inputRms = sqrt(sumSquares / input.size.toDouble())
        if (!inputRms.isFinite() || inputRms < 1e-9 || inputPeak < 1e-9) {
            return AudioNormalizationResult(
                samples = input.copyOf(),
                inputRmsDb = amplitudeToDb(inputRms),
                outputRmsDb = amplitudeToDb(inputRms),
                inputPeak = inputPeak,
                outputPeak = inputPeak,
                gain = 1.0,
                removedDcOffset = 0.0,
                clippedSamples = 0,
                applied = false,
            )
        }

        val targetRms = 10.0.pow(config.targetRmsDb / 20.0)
        val maxGain = 10.0.pow(config.maxGainDb / 20.0)
        val minGain = 10.0.pow(-config.maxAttenuationDb / 20.0)
        val rmsGain = (targetRms / inputRms).coerceIn(minGain, maxGain)
        val peakLimitedGain = config.peakCeiling / inputPeak
        val gain = min(rmsGain, peakLimitedGain).coerceAtLeast(0.0)

        val output = FloatArray(input.size)
        var outputSumSquares = 0.0
        var outputPeak = 0.0
        var clippedSamples = 0
        for (index in input.indices) {
            val scaled = (input[index].toDouble() - mean) * gain
            val limited = scaled.coerceIn(-config.peakCeiling, config.peakCeiling)
            if (limited != scaled) clippedSamples += 1
            output[index] = limited.toFloat()
            outputSumSquares += limited * limited
            outputPeak = maxOf(outputPeak, abs(limited))
        }

        val outputRms = sqrt(outputSumSquares / output.size.toDouble())
        return AudioNormalizationResult(
            samples = output,
            inputRmsDb = amplitudeToDb(inputRms),
            outputRmsDb = amplitudeToDb(outputRms),
            inputPeak = inputPeak,
            outputPeak = outputPeak,
            gain = gain,
            removedDcOffset = mean,
            clippedSamples = clippedSamples,
            applied = abs(gain - 1.0) >= 0.01 || abs(mean) >= 0.0001,
        )
    }

    private fun amplitudeToDb(value: Double): Double {
        return if (!value.isFinite() || value <= 1e-6) -120.0 else 20.0 * log10(value)
    }
}
