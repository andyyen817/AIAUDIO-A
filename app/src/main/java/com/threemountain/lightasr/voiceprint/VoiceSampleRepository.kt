package com.threemountain.lightasr.voiceprint

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class VoiceSampleRepository(
    private val storage: VoiceprintStorage,
    private val employeeRepository: EmployeeRepository,
) {
    @Synchronized
    fun addVoiceSample(employeeId: Long, audioFile: File): VoiceSample {
        employeeRepository.getEmployee(employeeId)
            ?: error("员工不存在：$employeeId")
        audioFile.requireWavExtension()

        val wavInfo = VoiceprintWavReader.parseHeader(audioFile)
        val samples = readSamples().toMutableList()
        val now = System.currentTimeMillis()
        val sampleId = (samples.maxOfOrNull { it.id } ?: 0L) + 1L
        val target = File(storage.employeeSampleDir(employeeId), "sample_$sampleId.wav")
        audioFile.copyTo(target, overwrite = true)
        require(target.length() > 0L) { "声纹样本保存失败：文件为空" }

        val sample = VoiceSample(
            id = sampleId,
            employeeId = employeeId,
            filePath = target.absolutePath,
            durationSec = wavInfo.durationSec,
            sampleRate = wavInfo.sampleRate,
            createdAt = now,
        )
        samples += sample
        writeSamples(samples)
        employeeRepository.updateSampleCount(employeeId, samples.count { it.employeeId == employeeId })
        return sample
    }

    @Synchronized
    fun listSamples(employeeId: Long): List<VoiceSample> {
        return readSamples().filter { it.employeeId == employeeId }
    }

    @Synchronized
    fun deleteSamplesForEmployee(employeeId: Long) {
        val remaining = readSamples().filterNot { sample ->
            if (sample.employeeId == employeeId) {
                File(sample.filePath).delete()
                true
            } else {
                false
            }
        }
        File(storage.samplesDir, "employee_$employeeId").deleteRecursively()
        writeSamples(remaining)
        employeeRepository.updateSampleCount(employeeId, 0)
    }

    private fun readSamples(): List<VoiceSample> {
        if (!storage.samplesFile.exists()) return emptyList()
        val content = storage.samplesFile.readText(Charsets.UTF_8).ifBlank { "[]" }
        val array = JSONArray(content)
        return List(array.length()) { index ->
            array.getJSONObject(index).toVoiceSample()
        }
    }

    private fun writeSamples(samples: List<VoiceSample>) {
        storage.ensureDirectories()
        val array = JSONArray()
        samples.forEach { array.put(it.toJson()) }
        storage.samplesFile.writeText(array.toString(2), Charsets.UTF_8)
    }

    private fun VoiceSample.toJson(): JSONObject {
        return JSONObject()
            .put("id", id)
            .put("employeeId", employeeId)
            .put("filePath", filePath)
            .put("durationSec", durationSec)
            .put("sampleRate", sampleRate)
            .put("createdAt", createdAt)
    }

    private fun JSONObject.toVoiceSample(): VoiceSample {
        return VoiceSample(
            id = getLong("id"),
            employeeId = getLong("employeeId"),
            filePath = getString("filePath"),
            durationSec = getDouble("durationSec"),
            sampleRate = getInt("sampleRate"),
            createdAt = optLong("createdAt", 0L),
        )
    }
}
