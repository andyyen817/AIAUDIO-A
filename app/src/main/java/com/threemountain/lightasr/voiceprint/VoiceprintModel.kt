package com.threemountain.lightasr.voiceprint

interface VoiceprintModel {
    val modelVersion: String
    val embeddingDim: Int?

    fun isAvailable(): Boolean

    fun extractEmbedding(
        audio: FloatArray,
        sampleRate: Int,
    ): FloatArray
}

class UnavailableVoiceprintModel(
    private val unavailableReason: String = "当前 Android runtime 暂不支持 CAM++ embedding。",
) : VoiceprintModel {
    override val modelVersion: String = "unavailable"
    override val embeddingDim: Int? = null

    override fun isAvailable(): Boolean = false

    override fun extractEmbedding(audio: FloatArray, sampleRate: Int): FloatArray {
        throw IllegalStateException("voiceprint model not available: $unavailableReason")
    }

    fun reason(): String = unavailableReason
}

class FunAsrCamppVoiceprintModel : VoiceprintModel {
    override val modelVersion: String = "funasr-campp-android-unwired"
    override val embeddingDim: Int? = null

    override fun isAvailable(): Boolean = false

    override fun extractEmbedding(audio: FloatArray, sampleRate: Int): FloatArray {
        throw IllegalStateException("voiceprint model not available: FunASR CAM++ Android binding is not wired")
    }
}

class OnnxCamppVoiceprintModel(
    private val modelPath: String,
) : VoiceprintModel {
    override val modelVersion: String = "onnx-campp-unwired:$modelPath"
    override val embeddingDim: Int? = null

    override fun isAvailable(): Boolean = false

    override fun extractEmbedding(audio: FloatArray, sampleRate: Int): FloatArray {
        throw IllegalStateException("voiceprint model not available: CAM++ ONNX frontend/preprocess is not implemented")
    }
}
