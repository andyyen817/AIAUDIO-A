package com.threemountain.lightasr.voiceprint

import android.content.Context
import android.util.Base64
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VoiceprintStorage(context: Context) {
    val rootDir: File = File(context.filesDir, "voiceprint")
    val samplesDir: File = File(rootDir, "samples")
    val templatesDir: File = File(rootDir, "templates")
    val employeesFile: File = File(rootDir, "employees.json")
    val samplesFile: File = File(rootDir, "samples.json")

    init {
        ensureDirectories()
    }

    fun ensureDirectories() {
        listOf(rootDir, samplesDir, templatesDir).forEach { dir ->
            if (!dir.exists() && !dir.mkdirs()) {
                error("无法创建声纹存储目录：${dir.absolutePath}")
            }
        }
    }

    fun employeeSampleDir(employeeId: Long): File {
        val dir = File(samplesDir, "employee_$employeeId")
        if (!dir.exists() && !dir.mkdirs()) {
            error("无法创建员工声纹样本目录：${dir.absolutePath}")
        }
        return dir
    }

    fun templateFile(employeeId: Long): File = File(templatesDir, "employee_$employeeId.json")

    fun encodeFloatArray(vector: FloatArray): String {
        val buffer = ByteBuffer.allocate(vector.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        vector.forEach { buffer.putFloat(it) }
        return Base64.encodeToString(buffer.array(), Base64.NO_WRAP)
    }

    fun decodeFloatArray(encoded: String, expectedDim: Int): FloatArray {
        val bytes = Base64.decode(encoded, Base64.NO_WRAP)
        require(bytes.size == expectedDim * 4) {
            "embedding 维度不匹配：bytes=${bytes.size}, dim=$expectedDim"
        }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(expectedDim) { buffer.getFloat() }
    }
}
