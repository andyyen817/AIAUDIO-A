package com.threemountain.lightasr.voiceprint

import kotlin.math.sqrt

object SpeakerMatcher {
    fun l2Normalize(vector: FloatArray): FloatArray {
        var sumSquares = 0.0
        for (value in vector) {
            val doubleValue = value.toDouble()
            sumSquares += doubleValue * doubleValue
        }

        val norm = sqrt(sumSquares)
        require(norm >= 1e-8) { "zero embedding norm" }

        return FloatArray(vector.size) { index ->
            (vector[index].toDouble() / norm).toFloat()
        }
    }

    fun averageEmbeddings(embeddings: List<FloatArray>): FloatArray {
        require(embeddings.isNotEmpty()) { "embedding 列表为空" }
        val dim = embeddings.first().size
        require(dim > 0) { "embedding 维度为空" }
        require(embeddings.all { it.size == dim }) { "embedding 维度不一致" }

        val sum = FloatArray(dim)
        embeddings.forEach { embedding ->
            val normalized = l2Normalize(embedding)
            for (index in normalized.indices) {
                sum[index] += normalized[index]
            }
        }

        val averaged = FloatArray(dim) { index -> sum[index] / embeddings.size.toFloat() }
        return l2Normalize(averaged)
    }

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "embedding 维度不一致：${a.size} vs ${b.size}" }
        val normalizedA = l2Normalize(a)
        val normalizedB = l2Normalize(b)
        var dot = 0.0f
        for (index in normalizedA.indices) {
            dot += normalizedA[index] * normalizedB[index]
        }
        return dot
    }

    fun matchSpeaker(
        queryEmbedding: FloatArray,
        enrolled: List<EmployeeVoiceprint>,
        config: VoiceprintConfig,
    ): VoiceprintMatchResult {
        if (enrolled.isEmpty()) {
            return VoiceprintMatchResult(
                employeeId = null,
                employeeName = null,
                storeName = null,
                top1Score = 0.0f,
                top2Score = null,
                scoreMargin = null,
                accepted = false,
                reason = "no_enrolled_voiceprints",
            )
        }

        val normalizedQuery = l2Normalize(queryEmbedding)
        val ranked = enrolled
            .map { voiceprint -> voiceprint to cosineSimilarity(normalizedQuery, voiceprint.embedding) }
            .sortedByDescending { it.second }

        val (top1, top1Score) = ranked.first()
        val top2Score = ranked.getOrNull(1)?.second
        val scoreMargin = top2Score?.let { top1Score - it }
        val thresholdOk = top1Score >= config.threshold
        val marginOk = top2Score == null || (scoreMargin != null && scoreMargin >= config.margin)
        val accepted = thresholdOk && marginOk
        val reason = when {
            accepted -> "accepted"
            !thresholdOk -> "below_threshold"
            else -> "ambiguous_speaker"
        }

        return VoiceprintMatchResult(
            employeeId = if (accepted) top1.employeeId else null,
            employeeName = if (accepted) top1.employeeName else null,
            storeName = if (accepted) top1.storeName else null,
            top1Score = top1Score,
            top2Score = top2Score,
            scoreMargin = scoreMargin,
            accepted = accepted,
            reason = reason,
        )
    }
}
