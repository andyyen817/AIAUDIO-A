package com.threemountain.lightasr.voiceprint

import java.io.EOFException
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.floor
import kotlin.math.min

object VoiceprintWavReader {
    fun parseHeader(wavFile: File): WavInfo {
        wavFile.requireWavExtension()

        RandomAccessFile(wavFile, "r").use { input ->
            if (input.length() < 44L) {
                error("WAV 文件太小，无法读取 header")
            }

            if (input.readFourCc() != "RIFF") {
                error("不是标准 WAV 文件：缺少 RIFF 标记")
            }
            input.readUInt32Le()
            if (input.readFourCc() != "WAVE") {
                error("不是标准 WAV 文件：缺少 WAVE 标记")
            }

            var audioFormat: Int? = null
            var channels: Int? = null
            var sampleRate: Int? = null
            var byteRate: Int? = null
            var blockAlign: Int? = null
            var bitsPerSample: Int? = null
            var dataStart: Long? = null
            var dataSize: Long? = null

            while (input.filePointer + 8L <= input.length()) {
                val chunkId = input.readFourCc()
                val chunkSize = input.readUInt32Le()
                val chunkDataStart = input.filePointer
                val nextChunk = chunkDataStart + chunkSize + (chunkSize % 2L)
                if (nextChunk > input.length() + 1L) {
                    error("WAV 文件结构异常：chunk=$chunkId 大小超出文件范围")
                }

                when (chunkId) {
                    "fmt " -> {
                        if (chunkSize < 16L) error("WAV fmt chunk 异常：长度不足")
                        audioFormat = input.readUInt16Le()
                        channels = input.readUInt16Le()
                        sampleRate = input.readUInt32Le().toInt()
                        byteRate = input.readUInt32Le().toInt()
                        blockAlign = input.readUInt16Le()
                        bitsPerSample = input.readUInt16Le()
                    }
                    "data" -> {
                        dataStart = chunkDataStart
                        dataSize = chunkSize
                    }
                }

                input.seek(nextChunk.coerceAtMost(input.length()))
                if (audioFormat != null && dataStart != null) break
            }

            val info = WavInfo(
                audioFormat = audioFormat ?: error("WAV 文件缺少 fmt chunk"),
                channels = channels ?: error("WAV 文件缺少声道信息"),
                sampleRate = sampleRate ?: error("WAV 文件缺少采样率信息"),
                byteRate = byteRate ?: error("WAV 文件缺少 byteRate 信息"),
                blockAlign = blockAlign ?: error("WAV 文件缺少 blockAlign 信息"),
                bitsPerSample = bitsPerSample ?: error("WAV 文件缺少位深信息"),
                dataStart = dataStart ?: error("WAV 文件缺少 data chunk"),
                dataSize = dataSize ?: error("WAV 文件 data chunk 为空"),
            )
            validate(info)
            return info
        }
    }

    fun readAsTargetMonoFloat(
        wavFile: File,
        targetSampleRate: Int,
    ): VoiceprintAudio {
        val info = parseHeader(wavFile)
        val sourceSamples = readAllMonoFloat(wavFile, info)
        val targetSamples = if (info.sampleRate == targetSampleRate) {
            sourceSamples
        } else {
            resampleLinear(sourceSamples, info.sampleRate, targetSampleRate)
        }
        return VoiceprintAudio(samples = targetSamples, sampleRate = targetSampleRate)
    }

    private fun validate(info: WavInfo) {
        if (info.audioFormat != 1) {
            error("当前声纹样本暂只支持标准 PCM WAV，后续再接 MediaExtractor 转码。")
        }
        require(info.channels > 0) { "WAV 声道数无效：${info.channels}" }
        require(info.sampleRate > 0) { "WAV 采样率无效：${info.sampleRate}" }
        require(info.bitsPerSample in listOf(16, 24, 32)) {
            "当前声纹样本支持 16/24/32-bit PCM WAV，当前位深：${info.bitsPerSample}"
        }
        val expectedAlign = info.channels * (info.bitsPerSample / 8)
        require(info.blockAlign == expectedAlign) {
            "WAV blockAlign 异常：当前 ${info.blockAlign}，预期 $expectedAlign"
        }
        require(info.dataSize > 0L) { "WAV data chunk 为空" }
        require(info.totalFrames <= Int.MAX_VALUE) {
            "声纹 MVP 暂不一次性读取超长样本，请选择 5-15 秒 WAV 样本"
        }
    }

    private fun readAllMonoFloat(wavFile: File, info: WavInfo): FloatArray {
        val byteCountLong = info.totalFrames * info.blockAlign.toLong()
        require(byteCountLong <= Int.MAX_VALUE) {
            "声纹 MVP 暂不一次性读取超长样本，请选择 5-15 秒 WAV 样本"
        }
        val byteCount = byteCountLong.toInt()
        val bytes = ByteArray(byteCount)
        RandomAccessFile(wavFile, "r").use { input ->
            input.seek(info.dataStart)
            input.readFully(bytes)
        }

        val samples = FloatArray(info.totalFrames.toInt())
        val bytesPerSample = info.bytesPerSample
        for (frameIndex in samples.indices) {
            val frameOffset = frameIndex * info.blockAlign
            var sum = 0.0
            for (channel in 0 until info.channels) {
                val sampleOffset = frameOffset + channel * bytesPerSample
                sum += decodePcmSample(bytes, sampleOffset, info.bitsPerSample).toDouble()
            }
            samples[frameIndex] = (sum / info.channels.toDouble()).toFloat()
        }
        return samples
    }

    private fun decodePcmSample(bytes: ByteArray, offset: Int, bitsPerSample: Int): Float {
        return when (bitsPerSample) {
            16 -> {
                val value = ((bytes[offset].toInt() and 0xFF) or (bytes[offset + 1].toInt() shl 8)).toShort()
                (value.toFloat() / 32768.0f).coerceIn(-1.0f, 1.0f)
            }
            24 -> {
                var value = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16)
                if ((value and 0x800000) != 0) value = value or -0x1000000
                (value.toFloat() / 8388608.0f).coerceIn(-1.0f, 1.0f)
            }
            32 -> {
                val value = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    (bytes[offset + 3].toInt() shl 24)
                (value.toFloat() / 2147483648.0f).coerceIn(-1.0f, 1.0f)
            }
            else -> error("不支持的 PCM 位深：$bitsPerSample")
        }
    }

    private fun resampleLinear(
        source: FloatArray,
        sourceRate: Int,
        targetRate: Int,
    ): FloatArray {
        if (source.isEmpty()) return FloatArray(0)
        require(sourceRate > 0 && targetRate > 0) { "采样率无效：$sourceRate -> $targetRate" }
        val targetSize = ((source.size.toDouble() * targetRate.toDouble()) / sourceRate.toDouble()).toInt()
            .coerceAtLeast(1)
        val ratio = sourceRate.toDouble() / targetRate.toDouble()
        return FloatArray(targetSize) { index ->
            val sourcePosition = index.toDouble() * ratio
            val left = floor(sourcePosition).toInt().coerceIn(0, source.lastIndex)
            val right = min(left + 1, source.lastIndex)
            val fraction = sourcePosition - left.toDouble()
            (source[left].toDouble() * (1.0 - fraction) + source[right].toDouble() * fraction).toFloat()
        }
    }

    private fun RandomAccessFile.readFourCc(): String {
        val bytes = ByteArray(4)
        try {
            readFully(bytes)
        } catch (e: EOFException) {
            error("WAV header 读取失败：文件提前结束")
        }
        return String(bytes, Charsets.US_ASCII)
    }

    private fun RandomAccessFile.readUInt16Le(): Int {
        val b0 = read()
        val b1 = read()
        if (b0 < 0 || b1 < 0) error("WAV header 读取失败：文件提前结束")
        return (b0 and 0xFF) or ((b1 and 0xFF) shl 8)
    }

    private fun RandomAccessFile.readUInt32Le(): Long {
        val b0 = read()
        val b1 = read()
        val b2 = read()
        val b3 = read()
        if (b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0) {
            error("WAV header 读取失败：文件提前结束")
        }
        return ((b0 and 0xFF).toLong()) or
            ((b1 and 0xFF).toLong() shl 8) or
            ((b2 and 0xFF).toLong() shl 16) or
            ((b3 and 0xFF).toLong() shl 24)
    }
}
