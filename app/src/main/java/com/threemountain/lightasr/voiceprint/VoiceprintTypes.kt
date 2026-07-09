package com.threemountain.lightasr.voiceprint

import java.io.File

data class Employee(
    val id: Long,
    val name: String,
    val storeName: String?,
    val enrolled: Boolean,
    val sampleCount: Int,
    val createdAt: Long,
    val updatedAt: Long,
)

data class VoiceSample(
    val id: Long,
    val employeeId: Long,
    val filePath: String,
    val durationSec: Double,
    val sampleRate: Int,
    val createdAt: Long,
)

data class EmployeeVoiceprint(
    val employeeId: Long,
    val employeeName: String,
    val storeName: String?,
    val embedding: FloatArray,
    val embeddingDim: Int,
    val modelVersion: String,
    val sampleCount: Int,
    val enrolledAt: Long,
)

data class VoiceprintConfig(
    // 临时默认值，需要用真机采集样本重新校准；不同模型、麦克风和录音环境下阈值会变化。
    val threshold: Float = 0.65f,
    // 临时默认值，需要结合 top1/top2 实测分布调整，不能作为最终生产参数。
    val margin: Float = 0.08f,
    val minEnrollSampleSec: Double = 5.0,
    val maxEnrollSampleSec: Double = 15.0,
    val minIdentifySpeechSec: Double = 2.0,
    val targetSampleRate: Int = 16000,
    val requiredEnrollSamples: Int = 2,
)

data class VoiceprintMatchResult(
    val employeeId: Long?,
    val employeeName: String?,
    val storeName: String?,
    val top1Score: Float,
    val top2Score: Float?,
    val scoreMargin: Float?,
    val accepted: Boolean,
    val reason: String,
)

data class AudioQualityResult(
    val ok: Boolean,
    val reason: String,
    val durationSec: Double,
    val rms: Double,
    val clippingRatio: Double,
)

data class WavInfo(
    val audioFormat: Int,
    val channels: Int,
    val sampleRate: Int,
    val byteRate: Int,
    val blockAlign: Int,
    val bitsPerSample: Int,
    val dataStart: Long,
    val dataSize: Long,
) {
    val bytesPerSample: Int = bitsPerSample / 8
    val totalFrames: Long = if (blockAlign > 0) dataSize / blockAlign else 0L
    val durationSec: Double = if (sampleRate > 0) {
        totalFrames.toDouble() / sampleRate.toDouble()
    } else {
        0.0
    }
}

data class VoiceprintAudio(
    val samples: FloatArray,
    val sampleRate: Int,
) {
    val durationSec: Double = if (sampleRate > 0) {
        samples.size.toDouble() / sampleRate.toDouble()
    } else {
        0.0
    }
}

data class VoiceprintRuntimeCheck(
    val supported: Boolean,
    val reason: String,
)

internal fun File.requireWavExtension() {
    require(name.lowercase().endsWith(".wav")) {
        "当前声纹样本暂只支持标准 PCM WAV，后续再接 MediaExtractor 转码。"
    }
}
