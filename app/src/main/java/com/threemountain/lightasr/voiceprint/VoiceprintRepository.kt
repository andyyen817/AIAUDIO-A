package com.threemountain.lightasr.voiceprint

import org.json.JSONObject

class VoiceprintRepository(
    private val storage: VoiceprintStorage,
) {
    @Synchronized
    fun saveVoiceprint(voiceprint: EmployeeVoiceprint) {
        storage.ensureDirectories()
        storage.templateFile(voiceprint.employeeId)
            .writeText(voiceprint.toJson().toString(2), Charsets.UTF_8)
    }

    @Synchronized
    fun getVoiceprint(employeeId: Long): EmployeeVoiceprint? {
        val file = storage.templateFile(employeeId)
        if (!file.exists()) return null
        return JSONObject(file.readText(Charsets.UTF_8)).toVoiceprint()
    }

    @Synchronized
    fun listVoiceprints(
        expectedModelVersion: String? = null,
        expectedEmbeddingDim: Int? = null,
    ): List<EmployeeVoiceprint> {
        if (!storage.templatesDir.exists()) return emptyList()

        return storage.templatesDir
            .listFiles { file -> file.isFile && file.extension == "json" }
            .orEmpty()
            .mapNotNull { file ->
                runCatching { JSONObject(file.readText(Charsets.UTF_8)).toVoiceprint() }.getOrNull()
            }
            .filter { template ->
                val modelOk = expectedModelVersion == null || template.modelVersion == expectedModelVersion
                val dimOk = expectedEmbeddingDim == null || template.embeddingDim == expectedEmbeddingDim
                modelOk && dimOk
            }
    }

    @Synchronized
    fun deleteVoiceprint(employeeId: Long) {
        storage.templateFile(employeeId).delete()
    }

    private fun EmployeeVoiceprint.toJson(): JSONObject {
        return JSONObject()
            .put("employeeId", employeeId)
            .put("employeeName", employeeName)
            .put("storeName", storeName)
            .put("embedding", storage.encodeFloatArray(embedding))
            .put("embeddingDim", embeddingDim)
            .put("modelVersion", modelVersion)
            .put("sampleCount", sampleCount)
            .put("enrolledAt", enrolledAt)
    }

    private fun JSONObject.toVoiceprint(): EmployeeVoiceprint {
        val dim = getInt("embeddingDim")
        return EmployeeVoiceprint(
            employeeId = getLong("employeeId"),
            employeeName = getString("employeeName"),
            storeName = optString("storeName", "").ifBlank { null },
            embedding = storage.decodeFloatArray(getString("embedding"), dim),
            embeddingDim = dim,
            modelVersion = getString("modelVersion"),
            sampleCount = optInt("sampleCount", 0),
            enrolledAt = optLong("enrolledAt", 0L),
        )
    }
}
