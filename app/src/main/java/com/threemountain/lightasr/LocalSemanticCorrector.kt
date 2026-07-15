package com.threemountain.lightasr

data class LocalSemanticRule(
    val observed: String,
    val canonical: String,
    val category: String,
)

data class AppliedSemanticRule(
    val observed: String,
    val canonical: String,
    val category: String,
    val occurrences: Int,
)

data class LocalSemanticCorrectionResult(
    val originalText: String,
    val correctedText: String,
    val appliedRules: List<AppliedSemanticRule>,
) {
    val changed: Boolean
        get() = originalText != correctedText
}

class LocalSemanticCorrector private constructor(
    rules: List<LocalSemanticRule>,
) {
    private val orderedRules = rules
        .distinctBy { it.observed }
        .sortedByDescending { it.observed.length }

    val ruleCount: Int
        get() = orderedRules.size

    fun correct(text: String): LocalSemanticCorrectionResult {
        var corrected = text
        val appliedRules = mutableListOf<AppliedSemanticRule>()

        for (rule in orderedRules) {
            val occurrences = corrected.countNonOverlapping(rule.observed)
            if (occurrences == 0) continue
            corrected = corrected.replace(rule.observed, rule.canonical)
            appliedRules += AppliedSemanticRule(
                observed = rule.observed,
                canonical = rule.canonical,
                category = rule.category,
                occurrences = occurrences,
            )
        }

        return LocalSemanticCorrectionResult(
            originalText = text,
            correctedText = corrected,
            appliedRules = appliedRules,
        )
    }

    companion object {
        fun empty(): LocalSemanticCorrector = LocalSemanticCorrector(emptyList())

        fun fromTsv(content: String): LocalSemanticCorrector {
            val rules = content.lineSequence()
                .map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("#") }
                .mapNotNull { line ->
                    val fields = line.split('\t')
                    if (fields.size < 2) return@mapNotNull null
                    val observed = fields[0].trim()
                    val canonical = fields[1].trim()
                    val category = fields.getOrNull(2)?.trim().orEmpty().ifBlank { "domain" }
                    if (observed.isEmpty() || canonical.isEmpty() || observed == canonical) {
                        null
                    } else {
                        LocalSemanticRule(observed, canonical, category)
                    }
                }
                .toList()
            return LocalSemanticCorrector(rules)
        }
    }
}

private fun String.countNonOverlapping(target: String): Int {
    if (target.isEmpty()) return 0
    var count = 0
    var startIndex = 0
    while (startIndex <= length - target.length) {
        val matchIndex = indexOf(target, startIndex)
        if (matchIndex < 0) break
        count += 1
        startIndex = matchIndex + target.length
    }
    return count
}
