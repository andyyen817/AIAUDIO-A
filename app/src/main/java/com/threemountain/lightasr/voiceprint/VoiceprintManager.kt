package com.threemountain.lightasr.voiceprint

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import java.io.File
import kotlin.math.sqrt

private const val VOICEPRINT_TAG = "LightASR-Voiceprint"

class VoiceprintManager(
    private val employeeRepository: EmployeeRepository,
    private val voiceSampleRepository: VoiceSampleRepository,
    private val voiceprintRepository: VoiceprintRepository,
    private val model: VoiceprintModel,
    val config: VoiceprintConfig = VoiceprintConfig(),
) {
    val modelVersion: String = model.modelVersion

    fun createEmployee(name: String, storeName: String?): Employee {
        val employee = employeeRepository.createEmployee(name, storeName)
        Log.i(VOICEPRINT_TAG, "employee created id=${employee.id} name=${employee.name} store=${employee.storeName ?: ""}")
        return employee
    }

    fun listEmployees(): List<Employee> = employeeRepository.listEmployees()

    fun getEmployee(employeeId: Long): Employee? = employeeRepository.getEmployee(employeeId)

    fun deleteEmployee(employeeId: Long) {
        voiceSampleRepository.deleteSamplesForEmployee(employeeId)
        voiceprintRepository.deleteVoiceprint(employeeId)
        employeeRepository.deleteEmployee(employeeId)
        Log.i(VOICEPRINT_TAG, "employee deleted id=$employeeId")
    }

    fun addVoiceSample(employeeId: Long, audioFile: File): VoiceSample {
        val sample = voiceSampleRepository.addVoiceSample(employeeId, audioFile)
        Log.i(
            VOICEPRINT_TAG,
            "voice sample added employee=$employeeId duration=${"%.2f".format(sample.durationSec)} " +
                "sampleRate=${sample.sampleRate} sampleCount=${voiceSampleRepository.listSamples(employeeId).size}"
        )
        return sample
    }

    fun enrollEmployee(employeeId: Long): EmployeeVoiceprint {
        if (!model.isAvailable()) {
            Log.i(VOICEPRINT_TAG, "voiceprint model not available")
            throw IllegalStateException("voiceprint model not available")
        }

        val employee = employeeRepository.getEmployee(employeeId)
            ?: error("员工不存在：$employeeId")
        val samples = voiceSampleRepository.listSamples(employeeId)
        if (samples.isEmpty()) {
            error("员工没有上传任何样本，无法注册")
        }
        if (samples.size < config.requiredEnrollSamples) {
            error("有效语音样本不足，请重新录制")
        }

        Log.i(VOICEPRINT_TAG, "voiceprint enroll start employee=$employeeId samples=${samples.size}")
        val embeddings = mutableListOf<FloatArray>()
        for (sample in samples) {
            val audio = VoiceprintWavReader.readAsTargetMonoFloat(
                wavFile = File(sample.filePath),
                targetSampleRate = config.targetSampleRate,
            )
            val quality = AudioQualityChecker.checkForEnrollment(audio.samples, audio.sampleRate, config)
            if (!quality.ok) {
                Log.i(
                    VOICEPRINT_TAG,
                    "voice sample skipped id=${sample.id} reason=${quality.reason} " +
                        "duration=${"%.2f".format(quality.durationSec)} rms=${"%.5f".format(quality.rms)}"
                )
                continue
            }
            val embedding = model.extractEmbedding(audio.samples, audio.sampleRate)
            Log.i(VOICEPRINT_TAG, "voiceprint embedding extracted dim=${embedding.size}")
            embeddings += embedding
        }

        if (embeddings.size < config.requiredEnrollSamples) {
            error("有效语音样本不足，请重新录制")
        }

        val averaged = SpeakerMatcher.averageEmbeddings(embeddings)
        val voiceprint = EmployeeVoiceprint(
            employeeId = employee.id,
            employeeName = employee.name,
            storeName = employee.storeName,
            embedding = averaged,
            embeddingDim = averaged.size,
            modelVersion = model.modelVersion,
            sampleCount = embeddings.size,
            enrolledAt = System.currentTimeMillis(),
        )
        voiceprintRepository.saveVoiceprint(voiceprint)
        employeeRepository.markEnrolled(employeeId, true)
        Log.i(VOICEPRINT_TAG, "voiceprint template saved employee=$employeeId")
        return voiceprint
    }

    fun identifySpeaker(audioFile: File): VoiceprintMatchResult {
        if (!model.isAvailable()) {
            Log.i(VOICEPRINT_TAG, "identify speaker skipped because model is not available")
            return unknown("model_not_available")
        }

        val expectedDim = model.embeddingDim
        val enrolled = voiceprintRepository.listVoiceprints(
            expectedModelVersion = model.modelVersion,
            expectedEmbeddingDim = expectedDim,
        )
        if (enrolled.isEmpty()) {
            return unknown("no_enrolled_voiceprints")
        }

        return try {
            Log.i(VOICEPRINT_TAG, "identify speaker start file=${audioFile.absolutePath}")
            val audio = VoiceprintWavReader.readAsTargetMonoFloat(audioFile, config.targetSampleRate)
            val quality = AudioQualityChecker.checkForIdentification(audio.samples, audio.sampleRate, config)
            if (!quality.ok) {
                return unknown(if (quality.reason == "audio_too_short") "audio_too_short" else quality.reason)
            }
            val embedding = model.extractEmbedding(audio.samples, audio.sampleRate)
            val result = SpeakerMatcher.matchSpeaker(embedding, enrolled, config)
            Log.i(
                VOICEPRINT_TAG,
                "match top1=${result.employeeName ?: "unknown"} score=${result.top1Score} " +
                    "top2=${result.top2Score} accepted=${result.accepted} reason=${result.reason}"
            )
            result
        } catch (t: Throwable) {
            Log.i(VOICEPRINT_TAG, "identify speaker failed: ${t.message ?: t::class.java.simpleName}")
            unknown("embedding_failed")
        }
    }

    fun runtimeCheck(): VoiceprintRuntimeCheck {
        return VoiceprintRuntimeCheck(
            supported = model.isAvailable(),
            reason = if (model.isAvailable()) {
                "voiceprint model available: ${model.modelVersion}"
            } else {
                "声纹模型未就绪：当前 Android runtime 暂不支持 CAM++ embedding。"
            },
        )
    }

    fun runMatcherSelfCheck(): String {
        val same = SpeakerMatcher.cosineSimilarity(floatArrayOf(1.0f, 0.0f), floatArrayOf(1.0f, 0.0f))
        val orthogonal = SpeakerMatcher.cosineSimilarity(floatArrayOf(1.0f, 0.0f), floatArrayOf(0.0f, 1.0f))
        val averaged = SpeakerMatcher.averageEmbeddings(
            listOf(floatArrayOf(1.0f, 0.0f), floatArrayOf(1.0f, 0.0f))
        )
        val averagedNorm = SpeakerMatcher.cosineSimilarity(averaged, averaged)
        val noTemplates = SpeakerMatcher.matchSpeaker(
            queryEmbedding = floatArrayOf(1.0f, 0.0f),
            enrolled = emptyList(),
            config = config,
        )
        val lowScore = SpeakerMatcher.matchSpeaker(
            queryEmbedding = floatArrayOf(1.0f, 0.0f),
            enrolled = listOf(
                EmployeeVoiceprint(1L, "低分", null, floatArrayOf(0.0f, 1.0f), 2, "self-test", 1, 0L)
            ),
            config = config,
        )
        val ambiguous = SpeakerMatcher.matchSpeaker(
            queryEmbedding = floatArrayOf(1.0f, 0.0f),
            enrolled = listOf(
                EmployeeVoiceprint(1L, "A", null, floatArrayOf(1.0f, 0.0f), 2, "self-test", 1, 0L),
                EmployeeVoiceprint(2L, "B", null, floatArrayOf(0.99f, 0.01f), 2, "self-test", 1, 0L),
            ),
            config = config,
        )
        return """
SpeakerMatcher 自检：
cosine([1,0],[1,0])=${"%.3f".format(same)}
cosine([1,0],[0,1])=${"%.3f".format(orthogonal)}
averageEmbeddings normCheck=${"%.3f".format(averagedNorm)}
noTemplates reason=${noTemplates.reason}
lowScore reason=${lowScore.reason}
ambiguous reason=${ambiguous.reason}
""".trim()
    }

    private fun unknown(reason: String): VoiceprintMatchResult {
        return VoiceprintMatchResult(
            employeeId = null,
            employeeName = null,
            storeName = null,
            top1Score = 0.0f,
            top2Score = null,
            scoreMargin = null,
            accepted = false,
            reason = reason,
        )
    }

    companion object {
        fun create(context: Context, assets: AssetManager): VoiceprintManager {
            val storage = VoiceprintStorage(context)
            val employeeRepository = EmployeeRepository(storage)
            val sampleRepository = VoiceSampleRepository(storage, employeeRepository)
            val voiceprintRepository = VoiceprintRepository(storage)
            val model = buildVoiceprintModel(context, assets)
            return VoiceprintManager(
                employeeRepository = employeeRepository,
                voiceSampleRepository = sampleRepository,
                voiceprintRepository = voiceprintRepository,
                model = model,
            )
        }

        private fun buildVoiceprintModel(context: Context, assets: AssetManager): VoiceprintModel {
            val sherpaModelPath = findFirstExistingAsset(
                assets,
                listOf(
                    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                    "models/voiceprint/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                )
            )
            if (sherpaModelPath != null) {
                return try {
                    Log.i(VOICEPRINT_TAG, "voiceprint model asset found path=$sherpaModelPath")
                    val modelFile = copyAssetToFilesDir(
                        context = context,
                        assets = assets,
                        assetPath = sherpaModelPath,
                        relativePath = "models/voiceprint/${sherpaModelPath.substringAfterLast('/')}",
                    )
                    val model = SherpaSpeakerEmbeddingVoiceprintModel(
                        modelPath = modelFile.absolutePath,
                    )
                    if (!model.isAvailable()) {
                        val reason = model.unavailableReason() ?: "speaker embedding extractor init failed"
                        model.release()
                        UnavailableVoiceprintModel("声纹模型初始化失败：$reason")
                    } else {
                        val smokeEmbeddingDim = runEmbeddingSmokeTest(context, assets, model)
                        Log.i(
                            VOICEPRINT_TAG,
                            "voiceprint model available model=${model.modelVersion} embeddingDim=$smokeEmbeddingDim"
                        )
                        model
                    }
                } catch (t: Throwable) {
                    Log.w(
                        VOICEPRINT_TAG,
                        "voiceprint model setup failed: ${t.message ?: t::class.java.simpleName}"
                    )
                    UnavailableVoiceprintModel("声纹模型初始化失败：${t.message ?: t::class.java.simpleName}")
                }
            }

            val hasCamppModel = assetFileExists(assets, "models/voiceprint/campplus.onnx")
            val hasCamppConfig = assetFileExists(assets, "models/voiceprint/config.json")
            return if (hasCamppModel && hasCamppConfig) {
                UnavailableVoiceprintModel(
                    "检测到 CAM++ 资源占位，但当前 App 尚未实现 CAM++ ONNX 前处理和推理绑定；" +
                        "建议先接入 sherpa-onnx 3D-Speaker 模型。"
                )
            } else {
                UnavailableVoiceprintModel(
                    "未发现声纹模型。请放置 assets/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx " +
                        "或 assets/models/voiceprint/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx。"
                )
            }
        }

        private fun findFirstExistingAsset(
            assets: AssetManager,
            paths: List<String>,
        ): String? {
            return paths.firstOrNull { path -> assetFileExists(assets, path) }
        }

        private fun copyAssetToFilesDir(
            context: Context,
            assets: AssetManager,
            assetPath: String,
            relativePath: String,
        ): File {
            val targetFile = File(context.filesDir, relativePath)
            val parent = targetFile.parentFile ?: error("invalid target path: ${targetFile.absolutePath}")
            if (!parent.exists() && !parent.mkdirs()) {
                error("无法创建模型目录：${parent.absolutePath}")
            }

            val assetSize = assets.open(assetPath).use { input -> input.available().toLong() }
            if (targetFile.exists() && targetFile.length() == assetSize && targetFile.length() > 0L) {
                Log.i(
                    VOICEPRINT_TAG,
                    "voiceprint model copy skipped target=${targetFile.absolutePath} size=${targetFile.length()}"
                )
                return targetFile
            }

            val tempFile = File(parent, "${targetFile.name}.tmp")
            assets.open(assetPath).use { input ->
                tempFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            require(tempFile.length() > 0L) { "模型复制失败：目标临时文件为空" }

            if (targetFile.exists() && !targetFile.delete()) {
                error("无法覆盖旧模型文件：${targetFile.absolutePath}")
            }
            if (!tempFile.renameTo(targetFile)) {
                tempFile.copyTo(targetFile, overwrite = true)
                tempFile.delete()
            }

            Log.i(
                VOICEPRINT_TAG,
                "voiceprint model copied asset=$assetPath target=${targetFile.absolutePath} size=${targetFile.length()}"
            )
            return targetFile
        }

        private fun runEmbeddingSmokeTest(
            context: Context,
            assets: AssetManager,
            model: SherpaSpeakerEmbeddingVoiceprintModel,
        ): Int {
            val smokeWav = copyAssetToFilesDir(
                context = context,
                assets = assets,
                assetPath = "test.wav",
                relativePath = "models/voiceprint/smoke_test.wav",
            )
            val audio = VoiceprintWavReader.readAsTargetMonoFloat(
                wavFile = smokeWav,
                targetSampleRate = 16000,
            )
            val embedding = model.extractEmbedding(audio.samples, audio.sampleRate)
            require(embedding.isNotEmpty()) { "smoke test embedding 为空" }
            require(embedding.all { !it.isNaN() && !it.isInfinite() }) {
                "smoke test embedding 包含 NaN 或 Inf"
            }

            val norm = sqrt(embedding.fold(0.0) { acc, value ->
                acc + value.toDouble() * value.toDouble()
            })
            require(norm > 1e-8) { "smoke test embedding norm 异常：$norm" }
            SpeakerMatcher.l2Normalize(embedding)

            Log.i(
                VOICEPRINT_TAG,
                "voiceprint embedding smoke test ok dim=${embedding.size} norm=$norm " +
                    "audioDuration=${"%.2f".format(audio.durationSec)}"
            )
            return embedding.size
        }

        private fun assetFileExists(assets: AssetManager, path: String): Boolean {
            val dir = path.substringBeforeLast('/', "")
            val fileName = path.substringAfterLast('/')
            return assets.list(dir)?.contains(fileName) == true
        }
    }
}
