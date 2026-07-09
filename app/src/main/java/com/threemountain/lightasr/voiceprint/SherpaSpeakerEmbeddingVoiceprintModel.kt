package com.threemountain.lightasr.voiceprint

import android.util.Log
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractor
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig
import java.io.File

private const val SHERPA_VOICEPRINT_TAG = "LightASR-Voiceprint"

class SherpaSpeakerEmbeddingVoiceprintModel(
    private val modelPath: String,
    private val numThreads: Int = 2,
) : VoiceprintModel {
    private val lock = Any()
    private var extractor: SpeakerEmbeddingExtractor? = null
    private var initError: String? = null
    private var cachedEmbeddingDim: Int? = null

    override val modelVersion: String = "sherpa-onnx-3dspeaker:${File(modelPath).name}"

    override val embeddingDim: Int?
        get() = getExtractorOrNull()?.let { extractor ->
            cachedEmbeddingDim ?: extractor.dim().also { cachedEmbeddingDim = it }
        }

    override fun isAvailable(): Boolean = getExtractorOrNull() != null

    override fun extractEmbedding(audio: FloatArray, sampleRate: Int): FloatArray {
        val extractor = getExtractorOrNull()
            ?: error("voiceprint model not available: ${initError ?: "sherpa speaker embedding extractor init failed"}")
        require(audio.isNotEmpty()) { "voiceprint audio is empty" }
        require(sampleRate > 0) { "invalid sample rate: $sampleRate" }

        val stream = extractor.createStream()
        try {
            stream.acceptWaveform(audio, sampleRate)
            stream.inputFinished()
            if (!extractor.isReady(stream)) {
                error("voiceprint audio is too short for speaker embedding extractor")
            }

            val embedding = extractor.compute(stream)
            require(embedding.isNotEmpty()) { "speaker embedding is empty" }
            Log.i(
                SHERPA_VOICEPRINT_TAG,
                "sherpa speaker embedding computed dim=${embedding.size} sampleRate=$sampleRate samples=${audio.size}"
            )
            return embedding
        } finally {
            stream.release()
        }
    }

    fun unavailableReason(): String? = initError

    fun release() {
        synchronized(lock) {
            extractor?.release()
            extractor = null
            cachedEmbeddingDim = null
        }
    }

    private fun getExtractorOrNull(): SpeakerEmbeddingExtractor? {
        synchronized(lock) {
            extractor?.let { return it }
            initError?.let { return null }

            return try {
                val created = SpeakerEmbeddingExtractor(
                    assetManager = null,
                    config = SpeakerEmbeddingExtractorConfig(
                        model = modelPath,
                        numThreads = numThreads,
                        debug = false,
                        provider = "cpu",
                    )
                )
                val dim = created.dim()
                require(dim > 0) { "invalid speaker embedding dim: $dim" }
                cachedEmbeddingDim = dim
                extractor = created
                Log.i(
                    SHERPA_VOICEPRINT_TAG,
                    "sherpa speaker embedding extractor ready model=$modelPath dim=$dim"
                )
                created
            } catch (t: Throwable) {
                initError = t.message ?: t::class.java.simpleName
                Log.w(
                    SHERPA_VOICEPRINT_TAG,
                    "sherpa speaker embedding extractor init failed model=$modelPath reason=$initError"
                )
                null
            }
        }
    }
}
