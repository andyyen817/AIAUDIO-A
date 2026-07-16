package com.threemountain.lightasr

import android.Manifest
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.OpenableColumns
import android.provider.Settings
import android.text.InputType
import android.text.method.ScrollingMovementMethod
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineParaformerModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.getFeatureConfig
import com.threemountain.lightasr.voiceprint.Employee
import com.threemountain.lightasr.voiceprint.VoiceSample
import com.threemountain.lightasr.voiceprint.VoiceprintManager
import com.threemountain.lightasr.voiceprint.VoiceprintMatchResult
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.roundToLong
import kotlin.math.sqrt

private const val TAG = "LightASR"
private const val MODEL_DIR = "sherpa-onnx-paraformer-zh-2024-03-09"
private const val VAD_MODEL_ASSET = "models/vad/silero_vad.onnx"
private const val DOMAIN_CORRECTIONS_ASSET = "models/text/domain_corrections.tsv"
private const val TEST_WAV = "test.wav"
private const val NO_SPEECH_MESSAGE = "未检测到有效人声，已跳过识别。"
private const val FIXED_SEGMENT_SECONDS = 5.0
private const val CHUNKED_SEGMENT_MS = 5_000L
private const val CHUNKED_OVERLAP_MS = 800L
private const val VAD_BOUNDARY_SEARCH_RADIUS_MS = 1_000L
private const val VAD_BOUNDARY_MIN_SILENCE_MS = 300L
private const val VAD_BOUNDARY_FRAME_MS = 20L
private const val VAD_BOUNDARY_NOISE_MARGIN_DB = 8.0
private const val VAD_FRAME_MS = 30L
private const val VAD_HOP_MS = 10L
private const val VAD_MIN_SPEECH_MS = 300L
private const val VAD_MIN_SILENCE_MS = 500L
private const val VAD_SPEECH_PADDING_MS = 200L
private const val VAD_MAX_SEGMENT_MS = 5_000L
private const val VAD_MERGE_GAP_MS = 300L
private const val VAD_MIN_SEGMENT_MS = 500L
private const val NATURAL_VAD_SCAN_WINDOW_MS = 30_000L
private const val NATURAL_VAD_SCAN_OVERLAP_MS = 1_000L
private const val NATURAL_UTTERANCE_MERGE_GAP_MS = 450L
private const val NATURAL_UTTERANCE_PADDING_MS = 220L
private const val NATURAL_UTTERANCE_MIN_SPEECH_MS = 180L
private const val SHARED_IMPORT_BUFFER_SIZE = 64 * 1024
private const val SHARED_IMPORT_MAX_NAME_LENGTH = 120
private const val SHARED_IMPORT_RETRY_COUNT = 8
private const val SHARED_IMPORT_RETRY_DELAY_MS = 500L
private const val DEFAULT_RECORDING_START_TEXT = "2026-07-09 10:00:00.000"
private const val DEFAULT_RECORDING_TIMEZONE_ID = "Asia/Shanghai"
private const val DEDUP_MAX_CHECK_CHARS = 320
private const val DEDUP_ACCUMULATED_TAIL_CHARS = 2_000
private const val DEDUP_MIN_OVERLAP_CHARS = 3
private const val DEDUP_FUZZY_MIN_OVERLAP_CHARS = 8
private const val DEDUP_FUZZY_THRESHOLD = 0.82
private const val TRANSCRIPT_MAX_SENTENCE_NORM_CHARS = 32
private const val STATE_PENDING_TXT_EXPORT_PATH = "pending_txt_export_path"

private val ENERGY_VAD_CONFIG = EnergyVadConfig(
    frameMs = VAD_FRAME_MS,
    hopMs = VAD_HOP_MS,
    minSpeechMs = VAD_MIN_SPEECH_MS,
    minSilenceMs = VAD_MIN_SILENCE_MS,
    speechPaddingMs = VAD_SPEECH_PADDING_MS,
    maxSegmentMs = VAD_MAX_SEGMENT_MS,
    mergeGapMs = VAD_MERGE_GAP_MS,
    minSegmentMs = VAD_MIN_SEGMENT_MS,
)

data class SelectedAudioFile(
    val uri: Uri,
    val displayName: String?,
    val mimeType: String?,
    val sizeBytes: Long?,
)

enum class SharedImportStatus {
    IMPORTING,
    RECEIVED,
    PENDING_TRANSCRIPTION,
    FAILED,
}

private enum class LightAsrPage {
    HOME,
    RESULT,
    JOB_DETAIL,
    VOICEPRINT,
}

data class SharedAudioRecord(
    val fileName: String,
    val mimeType: String?,
    val fileSizeBytes: Long,
    val originalUri: String,
    val savedPath: String,
    val receivedAt: Long,
    val status: SharedImportStatus,
)

private data class SharedImportResult(
    val record: SharedAudioRecord,
    val selectedAudioFile: SelectedAudioFile,
)

data class RecognitionOutput(
    val text: String,
    val durationSec: Double,
    val segments: List<RecognitionSegment>,
    val timestampedSentences: List<TimestampedSentence>,
    val transcriptSegments: List<TimestampedTranscriptSegment>,
    val mergedText: String,
    val segmentationMode: String,
    val fallbackToFixedSegments: Boolean,
    val rawSegmentCount: Int,
    val mergedSegmentCount: Int,
    val sampleRate: Int,
    val channels: Int,
    val bitsPerSample: Int,
    val stats: TranscriptionStats,
    val recordingTimeContext: RecordingTimeContext,
)

data class RecognitionSegment(
    val index: Int,
    val startMs: Long,
    val endMs: Long,
    val text: String,
    val cutMode: String = "fixed-fallback",
    val startUs: Long = startMs * 1000L,
    val endUs: Long = endMs * 1000L,
)

data class TimestampedSentence(
    val startSec: Double,
    val endSec: Double,
    val text: String,
    val sourceSegmentIndex: Int,
    val timeMode: String,
)

private data class EnergyVadConfig(
    val frameMs: Long,
    val hopMs: Long,
    val minSpeechMs: Long,
    val minSilenceMs: Long,
    val speechPaddingMs: Long,
    val maxSegmentMs: Long,
    val mergeGapMs: Long,
    val minSegmentMs: Long,
)

private data class SegmentRange(
    val startMs: Long,
    val endMs: Long,
    val index: Int = 0,
    val cutMode: String = "fixed-fallback",
    val targetEndMs: Long? = null,
    val adjustedEndMs: Long? = null,
)

private data class SegmentationResult(
    val mode: String,
    val fallbackToFixedSegments: Boolean,
    val rawSegmentCount: Int,
    val mergedSegmentCount: Int,
    val segments: List<SegmentRange>,
)

private data class WavHeader(
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
    val durationMs: Long = if (sampleRate > 0) {
        ((totalFrames.toDouble() * 1000.0) / sampleRate.toDouble()).roundToLong()
    } else {
        0L
    }
}

private data class ChunkedAudio(
    val startMs: Long,
    val endMs: Long,
    val samples: FloatArray,
    val startUs: Long = startMs * 1000L,
    val endUs: Long = endMs * 1000L,
)

data class TranscriptionStats(
    val totalDurationMs: Long,
    val speechDurationMs: Long,
    val skippedNoSpeechDurationMs: Long,
    val totalChunks: Int,
    val speechChunks: Int,
    val skippedChunks: Int,
    val asrProcessedDurationMs: Long,
) {
    val hasAnySpeech: Boolean = speechChunks > 0
}

private data class ChunkRecognitionResult(
    val segments: List<RecognitionSegment>,
    val stats: TranscriptionStats,
)

private data class ResultPageState(
    val displayName: String,
    val output: RecognitionOutput,
    val txtFile: File?,
    val txtSaveError: String?,
    val jobId: String?,
    val audioSizeBytes: Long?,
)

private data class SilenceCandidate(
    val startMs: Long,
    val endMs: Long,
) {
    val centerMs: Long = startMs + (endMs - startMs) / 2L
    val durationMs: Long = endMs - startMs
}

class MainActivity : ComponentActivity() {
    private val audioFilePicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            onAudioFileSelected(uri)
        }
    }

    private val txtDocumentCreator = registerForActivityResult(
        ActivityResultContracts.CreateDocument("text/plain")
    ) { uri ->
        finishTxtExport(uri)
    }

    private lateinit var pickAudioButton: Button
    private lateinit var transcribeSelectedButton: Button
    private lateinit var copyButton: Button
    private lateinit var startAirecReceiverButton: Button
    private lateinit var stopAirecReceiverButton: Button
    private lateinit var copyAirecUrlButton: Button
    private lateinit var airecReceiverStatusView: TextView
    private lateinit var airecUploadRecordsView: TextView
    private lateinit var airecUploadRecordActions: LinearLayout
    private lateinit var sharedImportRecordsView: TextView
    private lateinit var jobHistoryActions: LinearLayout
    private lateinit var serverUrlInput: EditText
    private lateinit var serverTokenInput: EditText
    private lateinit var saveServerConfigButton: Button
    private lateinit var employeeNameInput: EditText
    private lateinit var employeeStoreInput: EditText
    private lateinit var voiceprintEmployeeView: TextView
    private lateinit var createEmployeeButton: Button
    private lateinit var nextEmployeeButton: Button
    private lateinit var addVoiceSampleButton: Button
    private lateinit var enrollVoiceprintButton: Button
    private lateinit var identifySpeakerButton: Button
    private lateinit var listEmployeesButton: Button
    private lateinit var deleteEmployeeButton: Button
    private lateinit var statusView: TextView
    private lateinit var selectedAudioView: TextView
    private lateinit var resultView: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var mainScrollView: ScrollView
    private lateinit var pageRoot: LinearLayout

    private lateinit var voiceprintManager: VoiceprintManager
    private var recognizer: OfflineRecognizer? = null
    private var speechGate: SpeechGate? = null
    private var lastOutput: String = ""
    private var selectedAudioFile: SelectedAudioFile? = null
    private var selectedVoiceprintEmployeeId: Long? = null
    private var activeTranscriptionJobId: String? = null
    private var latestResultPageState: ResultPageState? = null
    private var activeJobDetailId: String? = null
    private var currentPage: LightAsrPage = LightAsrPage.HOME
    private var pendingTxtExportPath: String? = null
    private val localSemanticCorrector: LocalSemanticCorrector by lazy {
        runCatching {
            val content = assets.open(DOMAIN_CORRECTIONS_ASSET)
                .bufferedReader(Charsets.UTF_8)
                .use { it.readText() }
            LocalSemanticCorrector.fromTsv(content).also {
                Log.i(TAG, "local semantic corrector ready rules=${it.ruleCount}")
            }
        }.getOrElse { error ->
            Log.e(TAG, "local semantic corrector disabled asset=$DOMAIN_CORRECTIONS_ASSET", error)
            LocalSemanticCorrector.empty()
        }
    }
    private val sharedAudioRecords = mutableListOf<SharedAudioRecord>()
    private var airecReceiverRegistered = false
    private val airecReceiverUpdates = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            activeTranscriptionJobId = TranscriptionForegroundService.activeJobId
            refreshAirecReceiverUi()
            refreshJobHistoryUi()
            refreshResultPageIfVisible()
            refreshJobDetailPageIfVisible()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pendingTxtExportPath = savedInstanceState?.getString(STATE_PENDING_TXT_EXPORT_PATH)
        voiceprintManager = VoiceprintManager.create(this, assets)
        if (!TranscriptionForegroundService.isRunning) {
            TranscriptionJobRepository.markInterruptedJobs(this)
        }
        activeTranscriptionJobId = TranscriptionForegroundService.activeJobId
        buildUi()
        onBackPressedDispatcher.addCallback(
            this,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    handleAppBack()
                }
            },
        )
        refreshAirecReceiverUi()
        refreshSharedImportRecordsUi()
        refreshJobHistoryUi()
        loadServerConfigUi()
        updateVoiceprintEmployeeView()
        initRecognizerAsync()
        handleShareIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleShareIntent(intent)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        pendingTxtExportPath?.let { outState.putString(STATE_PENDING_TXT_EXPORT_PATH, it) }
        super.onSaveInstanceState(outState)
    }

    private fun buildUi() {
        mainScrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#F6F8FC"))
        }
        pageRoot = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(24), dp(16), dp(20))
        }

        pickAudioButton = Button(this).apply {
            text = "选择音频开始分析"
            stylePrimaryButton(this)
            setOnClickListener { pickAudioFile() }
        }

        selectedAudioView = TextView(this).apply {
            text = "未选择文件"
            textSize = 14f
            setInfoBoxStyle(this, Color.parseColor("#F4F7FB"))
        }

        transcribeSelectedButton = Button(this).apply {
            text = "识别所选音频"
            isEnabled = false
            stylePrimaryButton(this)
            setOnClickListener { transcribeSelectedAudio() }
        }

        copyButton = Button(this).apply {
            text = "复制结果"
            isEnabled = false
            styleSecondaryButton(this)
            setOnClickListener { copyResult() }
        }

        airecReceiverStatusView = TextView(this).apply {
            textSize = 14f
            setInfoBoxStyle(this, Color.parseColor("#F4F7FB"))
        }

        startAirecReceiverButton = Button(this).apply {
            text = "开启接收模式"
            stylePrimaryButton(this)
            setOnClickListener { startAirecReceiver() }
        }

        stopAirecReceiverButton = Button(this).apply {
            text = "停止接收模式"
            styleSecondaryButton(this)
            setOnClickListener { stopAirecReceiver() }
        }

        copyAirecUrlButton = Button(this).apply {
            text = "复制上传地址"
            styleSecondaryButton(this)
            setOnClickListener { copyAirecUploadUrl() }
        }

        airecUploadRecordsView = TextView(this).apply {
            textSize = 14f
            setInfoBoxStyle(this, Color.parseColor("#F7FAF8"))
        }

        airecUploadRecordActions = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        sharedImportRecordsView = TextView(this).apply {
            textSize = 14f
            setInfoBoxStyle(this, Color.parseColor("#F4F7FB"))
        }

        jobHistoryActions = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        val serverConfig = LightAsrServerConfigRepository.load(this)
        serverUrlInput = EditText(this).apply {
            setHint("服务器地址")
            setSingleLine(true)
            setText(serverConfig.baseUrl)
            setInputStyle(this)
        }
        serverTokenInput = EditText(this).apply {
            setHint("上传 Token（由管理员提供）")
            setSingleLine(true)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            setText(serverConfig.uploadToken)
            setInputStyle(this)
        }
        saveServerConfigButton = Button(this).apply {
            text = "保存服务器设置"
            styleSecondaryButton(this)
            setOnClickListener { saveServerConfig() }
        }

        employeeNameInput = EditText(this).apply {
            setHint("员工姓名，例如：张三")
            setSingleLine(true)
            setInputStyle(this)
        }

        employeeStoreInput = EditText(this).apply {
            setHint("门店，可选")
            setSingleLine(true)
            setInputStyle(this)
        }

        voiceprintEmployeeView = TextView(this).apply {
            textSize = 14f
            setInfoBoxStyle(this, Color.parseColor("#FFF7E8"), Color.parseColor("#F0D6A8"))
        }

        createEmployeeButton = Button(this).apply {
            text = "创建员工"
            stylePrimaryButton(this)
            setOnClickListener { createVoiceprintEmployee() }
        }

        nextEmployeeButton = Button(this).apply {
            text = "选择下一个员工"
            styleSecondaryButton(this)
            setOnClickListener { selectNextVoiceprintEmployee() }
        }

        addVoiceSampleButton = Button(this).apply {
            text = "添加所选音频为声纹样本"
            isEnabled = false
            styleSecondaryButton(this)
            setOnClickListener { addVoiceprintSampleFromSelected() }
        }

        enrollVoiceprintButton = Button(this).apply {
            text = "执行声纹注册"
            isEnabled = false
            styleSecondaryButton(this)
            setOnClickListener { enrollSelectedVoiceprintEmployee() }
        }

        identifySpeakerButton = Button(this).apply {
            text = "识别所选音频说话人"
            isEnabled = false
            styleSecondaryButton(this)
            setOnClickListener { identifySelectedSpeaker() }
        }

        listEmployeesButton = Button(this).apply {
            text = "查看员工列表"
            styleSecondaryButton(this)
            setOnClickListener { showVoiceprintEmployees() }
        }

        deleteEmployeeButton = Button(this).apply {
            text = "删除当前员工"
            isEnabled = false
            styleSecondaryButton(this)
            setOnClickListener { deleteSelectedVoiceprintEmployee() }
        }

        progressBar = ProgressBar(this).apply {
            visibility = View.GONE
        }

        statusView = TextView(this).apply {
            text = "正在初始化模型..."
            textSize = 14f
            setInfoBoxStyle(this, Color.parseColor("#F4F7FB"))
        }

        resultView = TextView(this).apply {
            text = ""
            textSize = 15f
            movementMethod = ScrollingMovementMethod()
            setInfoBoxStyle(this, Color.parseColor("#F7FAF8"))
        }

        mainScrollView.addView(pageRoot)
        setContentView(mainScrollView)
        showHomePage()
    }

    private fun showHomePage() {
        currentPage = LightAsrPage.HOME
        activeJobDetailId = null
        pageRoot.contentDescription = LightAsrPage.HOME.name
        pageRoot.removeAllViews()
        pageRoot.addView(buildHomeHeader(), matchWrap())
        pageRoot.addView(buildStatusCard(), sectionWrap())
        pageRoot.addView(buildStartAnalysisCard(), sectionWrap())
        pageRoot.addView(buildRecentTasksCard(), sectionWrap())
        pageRoot.addView(buildAirecReceiverCard(), sectionWrap())
        pageRoot.addView(buildVoiceprintEntryCard(), sectionWrap())
        pageRoot.addView(
            buildSection(
                "服务器设置",
                "本地分析完成后，可手动上传 WAV 和 TXT 到 Zeabur 后端",
                serverUrlInput,
                serverTokenInput,
                saveServerConfigButton,
            ),
            sectionWrap(),
        )
        scrollCurrentPageToTop()
    }

    private fun showVoiceprintPage() {
        currentPage = LightAsrPage.VOICEPRINT
        activeJobDetailId = null
        pageRoot.contentDescription = LightAsrPage.VOICEPRINT.name
        pageRoot.removeAllViews()
        pageRoot.addView(buildVoiceprintHeader(), matchWrap())
        pageRoot.addView(
            buildSection(
                "员工声纹实验功能",
                "用于短音频注册和识别验证；未接入真实模型时只提示未就绪",
                buildVoiceprintMarkerText(),
                selectedAudioView,
                pickAudioButton,
                employeeNameInput,
                employeeStoreInput,
                voiceprintEmployeeView,
                buildButtonRow(createEmployeeButton, nextEmployeeButton),
                buildButtonRow(addVoiceSampleButton, enrollVoiceprintButton),
                identifySpeakerButton,
                buildButtonRow(listEmployeesButton, deleteEmployeeButton),
            ),
            sectionWrap(),
        )
        pageRoot.addView(
            buildSection(
                "声纹运行状态",
                null,
                progressBar,
                statusView,
                resultView,
                copyButton,
            ),
            sectionWrap(),
        )
        scrollCurrentPageToTop()
    }

    private fun showResultPage(state: ResultPageState, scrollToTop: Boolean = true) {
        currentPage = LightAsrPage.RESULT
        activeJobDetailId = null
        latestResultPageState = state
        pageRoot.contentDescription = LightAsrPage.RESULT.name
        pageRoot.removeAllViews()
        pageRoot.addView(buildResultHeader(), matchWrap())
        pageRoot.addView(buildResultCompleteCard(state), sectionWrap())
        pageRoot.addView(buildTxtFileCard(state), sectionWrap())
        pageRoot.addView(buildUploadServerCard(state), sectionWrap())
        pageRoot.addView(buildResultStatsCard(state.output.stats, state.output.segments.size), sectionWrap())
        pageRoot.addView(buildTranscriptPreviewCard(state), sectionWrap())
        if (scrollToTop) scrollCurrentPageToTop()
    }

    private fun showJobDetailPage(jobId: String, scrollToTop: Boolean = true) {
        val job = TranscriptionJobRepository.get(this, jobId) ?: run {
            setStatus("任务记录不存在")
            showHomePage()
            return
        }
        currentPage = LightAsrPage.JOB_DETAIL
        activeJobDetailId = job.id
        pageRoot.contentDescription = LightAsrPage.JOB_DETAIL.name
        pageRoot.removeAllViews()
        pageRoot.addView(buildJobDetailHeader(), matchWrap())
        pageRoot.addView(buildJobDetailInfoCard(job), sectionWrap())
        pageRoot.addView(buildJobUploadServerCard(job), sectionWrap())
        pageRoot.addView(buildJobTranscriptCard(job), sectionWrap())
        if (scrollToTop) scrollCurrentPageToTop()
    }

    private fun handleAppBack() {
        val page = visiblePage()
        if (page != LightAsrPage.HOME) {
            goHomeFromSubpage("system back from $page")
            return
        }
        Log.i(TAG, "finish app from home back")
        finish()
    }

    @Deprecated("Use OnBackPressedDispatcher on newer Android versions")
    override fun onBackPressed() {
        handleAppBack()
    }

    private fun goHomeFromSubpage(reason: String) {
        Log.i(TAG, "navigate home: $reason")
        showHomePage()
    }

    private fun visiblePage(): LightAsrPage {
        val tagPage = pageRoot.contentDescription
            ?.toString()
            ?.let { value -> LightAsrPage.values().firstOrNull { it.name == value } }
        return tagPage ?: currentPage
    }

    private fun scrollCurrentPageToTop() {
        if (::mainScrollView.isInitialized) {
            mainScrollView.post { mainScrollView.scrollTo(0, 0) }
        }
    }

    private fun buildResultHeader(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(buildTopBackButton("result header back"), LinearLayout.LayoutParams(dp(56), dp(48)))
            addView(
                TextView(this@MainActivity).apply {
                    text = "识别结果"
                    textSize = 28f
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(Color.parseColor("#111827"))
                },
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
            )
            addView(View(this@MainActivity), LinearLayout.LayoutParams(dp(48), dp(48)))
        }
    }

    private fun buildJobDetailHeader(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(buildTopBackButton("job detail back"), LinearLayout.LayoutParams(dp(56), dp(48)))
            addView(
                TextView(this@MainActivity).apply {
                    text = "任务详情"
                    textSize = 28f
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(Color.parseColor("#111827"))
                },
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
            )
            addView(View(this@MainActivity), LinearLayout.LayoutParams(dp(48), dp(48)))
        }
    }

    private fun buildTopBackButton(reason: String): Button {
        return Button(this).apply {
            text = "‹"
            textSize = 30f
            typeface = Typeface.DEFAULT_BOLD
            isAllCaps = false
            contentDescription = "返回主页"
            setTextColor(Color.parseColor("#111827"))
            background = roundedBackground(
                fillColor = Color.TRANSPARENT,
                strokeColor = Color.TRANSPARENT,
                radiusDp = 8,
            )
            setOnClickListener { goHomeFromSubpage(reason) }
        }
    }

    private fun buildJobDetailInfoCard(job: TranscriptionJob): LinearLayout {
        return buildSection(
            "文件信息",
            null,
            TextView(this).apply {
                text = buildString {
                    appendLine("文件名：${job.sourceName}")
                    appendLine("状态：${jobStatusText(job.status)}")
                    appendLine("进度：${job.progressPercent}%")
                    appendLine("音频时长：${formatJobDuration(job.durationMs)}")
                    appendLine("来源：${job.sourceType}")
                    job.errorMessage?.takeIf { it.isNotBlank() }?.let { appendLine("提示：$it") }
                    job.serverRecordingId?.takeIf { it.isNotBlank() }?.let { appendLine("服务器记录：$it") }
                }.trim()
                textSize = 14f
                setInfoBoxStyle(this, Color.parseColor("#F4F7FB"))
            },
        )
    }

    private fun buildJobUploadServerCard(job: TranscriptionJob): LinearLayout {
        val progress = job.progressPercent.coerceIn(0, 100)
        val canUpload = job.status in setOf(
            TranscriptionJobStatus.LOCAL_COMPLETED,
            TranscriptionJobStatus.UPLOAD_FAILED,
        )
        val statusBox = TextView(this).apply {
            text = when (job.status) {
                TranscriptionJobStatus.UPLOADING -> "${job.errorMessage ?: "上传中"} $progress%"
                TranscriptionJobStatus.UPLOADED -> "上传成功：${job.serverRecordingId ?: "服务器已接收"}"
                TranscriptionJobStatus.UPLOAD_FAILED -> job.errorMessage ?: "上传失败"
                TranscriptionJobStatus.LOCAL_COMPLETED -> "将上传原始 WAV 和同名 TXT"
                else -> "本地分析完成后才可上传"
            }
            textSize = 14f
            setInfoBoxStyle(
                this,
                if (job.status == TranscriptionJobStatus.UPLOAD_FAILED) Color.parseColor("#FFF7E8")
                else Color.parseColor("#F4F7FB"),
                if (job.status == TranscriptionJobStatus.UPLOAD_FAILED) Color.parseColor("#F0D6A8")
                else Color.parseColor("#E5EAF2"),
            )
        }
        val uploadProgressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            this.progress = progress
            visibility = if (job.status in setOf(
                    TranscriptionJobStatus.UPLOADING,
                    TranscriptionJobStatus.UPLOADED,
                    TranscriptionJobStatus.UPLOAD_FAILED,
                )
            ) View.VISIBLE else View.GONE
        }
        val children = mutableListOf<View>(statusBox, uploadProgressBar)
        if (canUpload) {
            children += Button(this).apply {
                text = if (job.status == TranscriptionJobStatus.UPLOAD_FAILED) "重新上传 WAV+TXT" else "上传 WAV+TXT"
                stylePrimaryButton(this)
                isEnabled = activeTranscriptionJobId == null
                setOnClickListener { uploadJob(job.id) }
            }
        }
        return buildSection("上传服务器", null, *children.toTypedArray())
    }

    private fun buildJobTranscriptCard(job: TranscriptionJob): LinearLayout {
        val transcriptFile = job.transcriptPath?.let(::File)?.takeIf { it.isFile }
        val transcriptPreview = transcriptFile
            ?.useLines(Charsets.UTF_8) { lines ->
                lines
                    .map { it.trimEnd() }
                    .filter { it.isNotBlank() }
                    .take(12)
                    .joinToString("\n")
            }
            ?: "TXT 文件尚未生成"
        val children = mutableListOf<View>(
            TextView(this).apply {
                text = transcriptPreview
                textSize = 14f
                setInfoBoxStyle(this, Color.parseColor("#F7FAF8"))
            }
        )
        if (transcriptFile != null) {
            children += buildTxtActionPanel(transcriptFile)
        }
        return buildSection(
            "识别文本预览",
            transcriptFile?.let { "${it.name} · 可保存到手机、查看全文或分享" },
            *children.toTypedArray(),
        )
    }

    private fun buildResultCompleteCard(state: ResultPageState): LinearLayout {
        return buildCard().apply {
            orientation = LinearLayout.VERTICAL
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "✓"
                            textSize = 18f
                            gravity = Gravity.CENTER
                            typeface = Typeface.DEFAULT_BOLD
                            setTextColor(Color.parseColor("#10A66E"))
                            background = roundedBackground(
                                fillColor = Color.parseColor("#EAFBF2"),
                                strokeColor = Color.TRANSPARENT,
                                radiusDp = 12,
                            )
                        },
                        LinearLayout.LayoutParams(dp(82), dp(82)).apply {
                            setMargins(0, 0, dp(16), 0)
                        },
                    )
                    addView(
                        LinearLayout(this@MainActivity).apply {
                            orientation = LinearLayout.VERTICAL
                            addView(
                                TextView(this@MainActivity).apply {
                                    text = if (state.txtSaveError == null) "识别完成" else "识别完成，TXT 保存失败"
                                    textSize = 24f
                                    typeface = Typeface.DEFAULT_BOLD
                                    setTextColor(Color.parseColor("#111827"))
                                },
                                matchWrap(),
                            )
                            addView(
                                TextView(this@MainActivity).apply {
                                    text = if (state.txtSaveError == null) {
                                        "TXT 结果已生成，可以继续上传服务器"
                                    } else {
                                        state.txtSaveError
                                    }
                                    textSize = 15f
                                    setTextColor(Color.parseColor("#5D6675"))
                                    setPadding(0, dp(4), 0, dp(10))
                                },
                                matchWrap(),
                            )
                            addView(
                                TextView(this@MainActivity).apply {
                                    text = "♫ ${state.displayName}\n${formatTxtTimestamp(state.output.stats.totalDurationMs)} · ${formatBytes(state.audioSizeBytes)}"
                                    textSize = 14f
                                    setTextColor(Color.parseColor("#5D6675"))
                                },
                                matchWrap(),
                            )
                        },
                        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
                    )
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "TXT"
                            textSize = 24f
                            gravity = Gravity.CENTER
                            typeface = Typeface.DEFAULT_BOLD
                            setTextColor(Color.parseColor("#8AAEF8"))
                            background = roundedBackground(
                                fillColor = Color.parseColor("#EEF4FF"),
                                strokeColor = Color.TRANSPARENT,
                                radiusDp = 14,
                            )
                        },
                        LinearLayout.LayoutParams(dp(96), dp(116)),
                    )
                },
                matchWrap(),
            )
        }
    }

    private fun buildResultStatsCard(stats: TranscriptionStats, segmentCount: Int): LinearLayout {
        return buildCard().apply {
            orientation = LinearLayout.HORIZONTAL
            addView(buildStatCell("总时长", formatTxtTimestamp(stats.totalDurationMs), "#1263EA"), statCellParams())
            addView(buildStatCell("人声时长", formatTxtTimestamp(stats.speechDurationMs), "#16A34A"), statCellParams())
            addView(buildStatCell("跳过噪音", formatTxtTimestamp(stats.skippedNoSpeechDurationMs), "#F97316"), statCellParams())
            addView(buildStatCell("分段数量", "$segmentCount 段", "#7C3AED"), statCellParams())
        }
    }

    private fun buildTxtFileCard(state: ResultPageState): LinearLayout {
        val transcriptFile = state.txtFile?.takeIf { it.isFile && it.length() > 0L }
        val description = when {
            transcriptFile != null -> "${transcriptFile.name} · 已生成应用内副本，请保存到手机以便在文件管理器中查看"
            !state.txtSaveError.isNullOrBlank() -> state.txtSaveError
            else -> "TXT 尚未生成"
        }
        return buildSection(
            "TXT 结果文件",
            description,
            buildTxtActionPanel(transcriptFile),
        )
    }

    private fun buildTxtActionPanel(transcriptFile: File?): LinearLayout {
        val fileReady = transcriptFile?.let { it.isFile && it.length() > 0L } == true
        val saveButton = Button(this).apply {
            text = "保存到手机"
            isEnabled = fileReady
            stylePrimaryButton(this)
            setOnClickListener {
                transcriptFile?.let(::exportTxtToUserLocation)
            }
        }
        val viewButton = Button(this).apply {
            text = "查看全文"
            isEnabled = fileReady
            styleSecondaryButton(this)
            setOnClickListener {
                transcriptFile?.let(::showTxtFile)
            }
        }
        val shareButton = Button(this).apply {
            text = "分享 TXT"
            isEnabled = fileReady
            styleSecondaryButton(this)
            setOnClickListener {
                transcriptFile?.let(::shareTxtFile)
            }
        }
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(saveButton, matchWrap())
            addView(buildButtonRow(viewButton, shareButton), childWrap(topMargin = 8))
        }
    }

    private fun buildUploadServerCard(state: ResultPageState): LinearLayout {
        val uploadedJob = state.jobId?.let { TranscriptionJobRepository.get(this, it) }
        val uploadText = when (uploadedJob?.status) {
            TranscriptionJobStatus.UPLOADED -> "已上传：${uploadedJob.serverRecordingId ?: "服务器已接收"}"
            TranscriptionJobStatus.UPLOADING -> "${uploadedJob.errorMessage ?: "上传中"}：${uploadedJob.progressPercent}%"
            TranscriptionJobStatus.UPLOAD_FAILED -> "上次上传失败，可重试上传原始 WAV 和同名 TXT"
            else -> "将上传原始 WAV 和同名 TXT 到 OSS，并在服务器登记记录"
        }
        val progress = uploadedJob?.progressPercent?.coerceIn(0, 100) ?: 0
        val statusBox = TextView(this).apply {
            text = when (uploadedJob?.status) {
                TranscriptionJobStatus.UPLOADING -> "${uploadedJob.errorMessage ?: "上传进度"} $progress%"
                TranscriptionJobStatus.UPLOADED -> "上传进度 100%"
                TranscriptionJobStatus.UPLOAD_FAILED -> uploadedJob.errorMessage ?: "上传失败"
                else -> if (state.jobId == null) "内置测试音频不创建上传任务" else "等待上传 WAV+TXT"
            }
            textSize = 14f
            setInfoBoxStyle(
                this,
                if (uploadedJob?.status == TranscriptionJobStatus.UPLOAD_FAILED) Color.parseColor("#FFF7E8")
                else Color.parseColor("#F4F7FB"),
                if (uploadedJob?.status == TranscriptionJobStatus.UPLOAD_FAILED) Color.parseColor("#F0D6A8")
                else Color.parseColor("#E5EAF2"),
            )
        }
        val uploadProgressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            this.progress = progress
            visibility = if (uploadedJob?.status in setOf(
                    TranscriptionJobStatus.UPLOADING,
                    TranscriptionJobStatus.UPLOADED,
                    TranscriptionJobStatus.UPLOAD_FAILED,
                )
            ) View.VISIBLE else View.GONE
        }
        val children = mutableListOf<View>(statusBox, uploadProgressBar)
        val jobId = state.jobId
        if (jobId != null) {
            children += Button(this).apply {
                text = if (uploadedJob?.status == TranscriptionJobStatus.UPLOAD_FAILED) "重新上传 WAV+TXT" else "上传 WAV+TXT"
                stylePrimaryButton(this)
                isEnabled = uploadedJob?.status != TranscriptionJobStatus.UPLOADING
                setOnClickListener { uploadJob(jobId) }
            }
        }
        return buildSection(
            "上传服务器",
            uploadText,
            *children.toTypedArray(),
        )
    }

    private fun buildTranscriptPreviewCard(state: ResultPageState): LinearLayout {
        val preview = buildTranscriptPreview(state.output)
        return buildSection(
            "识别文本预览",
            null,
            TextView(this).apply {
                text = preview
                textSize = 15f
                setInfoBoxStyle(this, Color.parseColor("#F4F7FB"))
            },
            TextView(this).apply {
                text = "仅预览前 5 段，可在上方查看全文、保存到手机或分享 TXT。"
                textSize = 13f
                setTextColor(Color.parseColor("#6B7280"))
            },
        )
    }

    private fun buildHomeHeader(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "LightASR"
                            textSize = 30f
                            typeface = Typeface.DEFAULT_BOLD
                            setTextColor(Color.parseColor("#111827"))
                        },
                        matchWrap(),
                    )
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "离线语音转文字工作台"
                            textSize = 16f
                            setTextColor(Color.parseColor("#5D6675"))
                            setPadding(0, dp(6), 0, dp(12))
                        },
                        matchWrap(),
                    )
                },
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
            )
            addView(
                TextView(this@MainActivity).apply {
                    text = "?"
                    textSize = 20f
                    typeface = Typeface.DEFAULT_BOLD
                    gravity = Gravity.CENTER
                    setTextColor(Color.parseColor("#4B5563"))
                    background = roundedBackground(
                        fillColor = Color.WHITE,
                        strokeColor = Color.parseColor("#D9E2F0"),
                        radiusDp = 18,
                    )
                },
                LinearLayout.LayoutParams(dp(38), dp(38)),
            )
        }
    }

    private fun buildStartAnalysisCard(): LinearLayout {
        return buildCard().apply {
            orientation = LinearLayout.VERTICAL
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "开始一次新的录音整理"
                            textSize = 22f
                            typeface = Typeface.DEFAULT_BOLD
                            setTextColor(Color.parseColor("#111827"))
                        },
                        matchWrap(),
                    )
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "适合处理 1 小时以上 WAV 录音"
                            textSize = 15f
                            setTextColor(Color.parseColor("#5D6675"))
                            setPadding(0, dp(6), 0, dp(12))
                        },
                        matchWrap(),
                    )
                    detachFromParent(pickAudioButton)
                    addView(pickAudioButton, matchWrap())
                    detachFromParent(selectedAudioView)
                    addView(selectedAudioView, childWrap(topMargin = 10))
                    detachFromParent(transcribeSelectedButton)
                    addView(transcribeSelectedButton, childWrap(topMargin = 8))
                },
                matchWrap(),
            )
        }
    }

    private fun buildStatusCard(): LinearLayout {
        val latestJob = TranscriptionJobRepository.list(this).firstOrNull()
        val statusText = if (activeTranscriptionJobId != null) {
            "处理中"
        } else {
            "空闲"
        }
        val detailText = latestJob?.let {
            "最近一次任务：${it.sourceName} · ${jobStatusText(it.status)}"
        } ?: "系统就绪，可开始新的音频分析任务"
        return buildCard().apply {
            orientation = LinearLayout.VERTICAL
            addView(
                TextView(this@MainActivity).apply {
                    text = "当前状态"
                    textSize = 18f
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(Color.parseColor("#111827"))
                },
                matchWrap(),
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    addView(
                        TextView(this@MainActivity).apply {
                            text = statusText
                            textSize = 24f
                            typeface = Typeface.DEFAULT_BOLD
                            gravity = Gravity.CENTER
                            setTextColor(if (activeTranscriptionJobId == null) Color.parseColor("#138A3D") else Color.parseColor("#1263EA"))
                            background = roundedBackground(
                                fillColor = if (activeTranscriptionJobId == null) Color.parseColor("#E7F8EA") else Color.parseColor("#EEF4FF"),
                                strokeColor = Color.TRANSPARENT,
                                radiusDp = 8,
                            )
                        },
                        LinearLayout.LayoutParams(dp(104), dp(58)).apply {
                            setMargins(0, 0, dp(18), 0)
                        },
                    )
                    addView(
                        TextView(this@MainActivity).apply {
                            text = detailText
                            textSize = 15f
                            setTextColor(Color.parseColor("#374151"))
                        },
                        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
                    )
                },
                childWrap(topMargin = 14),
            )
            addView(statusView, childWrap(topMargin = 12))
            addView(progressBar, childWrap(topMargin = 8))
        }
    }

    private fun buildRecentTasksCard(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    addView(sectionTitle("最近任务"), LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
                },
                matchWrap(),
            )
            addView(jobHistoryActions, childWrap(topMargin = 12))
        }
    }

    private fun buildAirecReceiverCard(): LinearLayout {
        return buildCard().apply {
            orientation = LinearLayout.VERTICAL
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "▰"
                            textSize = 26f
                            gravity = Gravity.CENTER
                            setTextColor(Color.parseColor("#16A34A"))
                            background = roundedBackground(
                                fillColor = Color.parseColor("#F0FDF4"),
                                strokeColor = Color.TRANSPARENT,
                                radiusDp = 12,
                            )
                        },
                        LinearLayout.LayoutParams(dp(66), dp(66)).apply {
                            setMargins(0, 0, dp(14), 0)
                        },
                    )
                    addView(
                        LinearLayout(this@MainActivity).apply {
                            orientation = LinearLayout.VERTICAL
                            addView(
                                TextView(this@MainActivity).apply {
                                    text = "AIREC 接收模式"
                                    textSize = 18f
                                    typeface = Typeface.DEFAULT_BOLD
                                    setTextColor(Color.parseColor("#111827"))
                                },
                                matchWrap(),
                            )
                            addView(airecReceiverStatusView, childWrap(topMargin = 8))
                        },
                        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
                    )
                },
                matchWrap(),
            )
            addView(
                TextView(this@MainActivity).apply {
                    text = AirecNetwork.uploadUrl()
                    textSize = 14f
                    setTextColor(Color.parseColor("#111827"))
                    setInfoBoxStyle(this, Color.parseColor("#F8FAFD"))
                },
                childWrap(topMargin = 12),
            )
            addView(buildButtonRow(startAirecReceiverButton, stopAirecReceiverButton, copyAirecUrlButton), childWrap(topMargin = 12))
            addView(airecUploadRecordsView, childWrap(topMargin = 12))
            addView(airecUploadRecordActions, childWrap(topMargin = 8))
        }
    }

    private fun buildVoiceprintEntryCard(): LinearLayout {
        return buildCard().apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            isClickable = true
            contentDescription = "进入声纹识别页面"
            setOnClickListener { showVoiceprintPage() }
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "声纹识别（实验功能）"
                            textSize = 18f
                            typeface = Typeface.DEFAULT_BOLD
                            setTextColor(Color.parseColor("#111827"))
                        },
                        matchWrap(),
                    )
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "注册员工声音样本，后续用于标注不同说话人"
                            textSize = 14f
                            setTextColor(Color.parseColor("#5D6675"))
                            setPadding(0, dp(5), 0, 0)
                        },
                        matchWrap(),
                    )
                },
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
            )
            addView(
                TextView(this@MainActivity).apply {
                    text = "⌄"
                    textSize = 24f
                    gravity = Gravity.CENTER
                    setTextColor(Color.parseColor("#374151"))
                },
                LinearLayout.LayoutParams(dp(44), dp(44)),
            )
        }
    }

    private fun buildVoiceprintHeader(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(buildTopBackButton("voiceprint header back"), LinearLayout.LayoutParams(dp(56), dp(48)))
            addView(
                TextView(this@MainActivity).apply {
                    text = "声纹识别"
                    textSize = 26f
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(Color.parseColor("#111827"))
                    setPadding(dp(12), 0, 0, 0)
                },
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
            )
        }
    }

    private fun buildVoiceprintMarkerText(): TextView {
        return TextView(this).apply {
            text = "当前声纹模块用于验证员工注册、样本保存和匹配流程。真实声纹模型不可用时会显示“模型未就绪”，不会影响正式录音转文字。"
            textSize = 14f
            setInfoBoxStyle(this, Color.parseColor("#FFF7E8"), Color.parseColor("#F0D6A8"))
        }
    }

    private fun buildSection(
        title: String,
        subtitle: String?,
        vararg children: View,
    ): LinearLayout {
        return buildCard().apply {
            addView(
                TextView(this@MainActivity).apply {
                    text = title
                    textSize = 19f
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(Color.parseColor("#111827"))
                },
                matchWrap(),
            )

            if (!subtitle.isNullOrBlank()) {
                addView(
                    TextView(this@MainActivity).apply {
                        text = subtitle
                        textSize = 14f
                        setTextColor(Color.parseColor("#5D6675"))
                    },
                    childWrap(topMargin = 5),
                )
            }

            children.forEach { child ->
                detachFromParent(child)
                addView(child, childWrap())
            }
        }
    }

    private fun startAirecReceiver() {
        requestNotificationPermissionIfNeeded()
        val intent = Intent(this, AirecReceiverService::class.java)
            .setAction(AirecReceiverService.ACTION_START)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        setStatus("正在开启 AIREC 接收模式...")
        refreshAirecReceiverUi()
    }

    private fun stopAirecReceiver() {
        val intent = Intent(this, AirecReceiverService::class.java)
            .setAction(AirecReceiverService.ACTION_STOP)
        startService(intent)
        setStatus("正在停止 AIREC 接收模式...")
        refreshAirecReceiverUi()
    }

    private fun copyAirecUploadUrl() {
        val url = AirecNetwork.uploadUrl()
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("AIREC upload URL", url))
        setStatus("AIREC 上传地址已复制")
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 18080)
        }
    }

    private fun refreshAirecReceiverUi() {
        val url = AirecNetwork.uploadUrl()
        val stateText = when {
            AirecReceiverService.isRunning -> "接收中"
            AirecReceiverService.lastError != null -> "错误"
            else -> "未开启"
        }
        val errorText = AirecReceiverService.lastError?.let { "\n错误：$it" }.orEmpty()
        airecReceiverStatusView.text = """
AIREC 接收模式：$stateText
上传地址：$url
保存目录：Android/data/$packageName/files/Incoming/{sn}/{fileName}$errorText
""".trim()
        startAirecReceiverButton.isEnabled = !AirecReceiverService.isRunning
        stopAirecReceiverButton.isEnabled = AirecReceiverService.isRunning
        copyAirecUrlButton.isEnabled = true
        val records = AirecUploadRepository.list(this)
        airecUploadRecordsView.text = buildAirecUploadRecordsText(records)
        refreshAirecUploadRecordActions(records)
    }

    private fun buildAirecUploadRecordsText(
        records: List<AirecUploadRecord> = AirecUploadRepository.list(this),
    ): String {
        if (records.isEmpty()) {
            return "最近收到的录音：暂无"
        }

        return buildString {
            appendLine("最近收到的录音：")
            records.take(10).forEachIndexed { index, record ->
                appendLine()
                appendLine("${index + 1}. ${record.fileName}")
                appendLine("sn: ${record.sn}")
                appendLine("大小: ${record.fileSize} bytes")
                appendLine("接收时间: ${formatAirecUploadTime(record.uploadTimeMs)}")
                appendLine("状态: ${record.status}")
                appendLine("路径: ${record.savedPath}")
            }
        }.trim()
    }

    private fun refreshAirecUploadRecordActions(records: List<AirecUploadRecord>) {
        airecUploadRecordActions.removeAllViews()

        if (records.isEmpty()) {
            return
        }

        val hint = TextView(this).apply {
            text = "点击下面按钮，可直接把接收到的录音设为当前分析文件："
            textSize = 13f
            setPadding(0, dp(8), 0, dp(4))
        }
        airecUploadRecordActions.addView(hint, matchWrap())

        records.take(10).forEachIndexed { index, record ->
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, dp(4), 0, dp(4))
            }

            val selectButton = Button(this).apply {
                text = "选择 ${index + 1}"
                setOnClickListener {
                    selectAirecUploadRecord(record, startTranscription = false)
                }
            }
            val transcribeButton = Button(this).apply {
                text = "分析 ${index + 1}"
                isEnabled = recognizer != null
                setOnClickListener {
                    selectAirecUploadRecord(record, startTranscription = true)
                }
            }

            row.addView(
                selectButton,
                LinearLayout.LayoutParams(
                    0,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    1f,
                ).apply { setMargins(0, 0, dp(6), 0) },
            )
            row.addView(
                transcribeButton,
                LinearLayout.LayoutParams(
                    0,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    1f,
                ).apply { setMargins(dp(6), 0, 0, 0) },
            )
            airecUploadRecordActions.addView(row, matchWrap())
        }
    }

    private fun selectAirecUploadRecord(
        record: AirecUploadRecord,
        startTranscription: Boolean,
    ) {
        val file = File(record.savedPath)
        if (!file.exists() || !file.isFile) {
            setStatus("接收录音文件不存在，请重新上传：${record.fileName}")
            refreshAirecReceiverUi()
            return
        }
        if (file.length() <= 0L) {
            setStatus("接收录音文件为空，请重新上传：${record.fileName}")
            return
        }

        val selection = SelectedAudioFile(
            uri = Uri.fromFile(file),
            displayName = record.fileName.ifBlank { file.name },
            mimeType = "audio/wav",
            sizeBytes = file.length(),
        )
        selectedAudioFile = selection
        selectedAudioView.text = buildSelectedAudioText(selection)
        updateSelectedAudioButtonState()
        updateVoiceprintButtonState()

        if (startTranscription) {
            setStatus("已选择接收录音，开始分析：${selection.displayName}")
            transcribeSelectedAudio()
        } else {
            setStatus("已选择接收录音：${selection.displayName}")
        }
    }

    private fun handleShareIntent(intent: Intent?) {
        if (intent == null) return
        val action = intent.action ?: return
        if (
            action != Intent.ACTION_SEND &&
            action != Intent.ACTION_SEND_MULTIPLE &&
            action != Intent.ACTION_VIEW
        ) return

        val uris = extractSharedUris(intent)
        Log.i(TAG, "share intent received ${buildShareIntentDebugText(intent)}, uriCount=${uris.size}")
        if (uris.isEmpty()) {
            addSharedImportFailureRecord(
                reason = "没有从分享内容中读取到音频文件",
                originalUri = buildShareIntentDebugText(intent),
            )
            setStatus("没有从分享内容中读取到音频文件；已在分享接收记录里写入诊断信息。")
            return
        }

        setStatus("正在导入音频... (${uris.size} 个文件)")
        progressBar.visibility = View.VISIBLE

        thread(name = "import-shared-audio") {
            val imported = mutableListOf<SharedImportResult>()
            val failures = mutableListOf<String>()

            uris.forEachIndexed { index, uri ->
                runOnUiThread {
                    setStatus("正在导入音频 ${index + 1}/${uris.size} ...")
                }
                try {
                    imported += importSharedAudio(uri)
                } catch (t: Throwable) {
                    failures += "${uri}: ${t.message ?: t::class.java.simpleName}"
                    Log.w(TAG, "shared audio import failed uri=$uri", t)
                }
            }

            runOnUiThread {
                progressBar.visibility = View.GONE
                if (imported.isNotEmpty()) {
                    val first = imported.first()
                    sharedAudioRecords.addAll(0, imported.map { it.record })
                    selectedAudioFile = first.selectedAudioFile
                    selectedAudioView.text = buildSelectedAudioText(first.selectedAudioFile)
                    updateSelectedAudioButtonState()
                    updateVoiceprintButtonState()
                    refreshSharedImportRecordsUi()
                }
                failures.forEach { failure ->
                    addSharedImportFailureRecord(
                        reason = "导入失败",
                        originalUri = failure,
                    )
                }

                val statusText = buildString {
                    if (imported.isNotEmpty()) {
                        append("分享导入完成：${imported.size} 个 WAV")
                    }
                    if (failures.isNotEmpty()) {
                        if (isNotEmpty()) append("\n")
                        append("导入失败：${failures.size} 个")
                        append("\n当前第一版只支持 WAV 文件，请分享 .wav 音频。")
                    }
                    if (isEmpty()) append("分享导入失败")
                }
                setStatus(statusText)
            }
        }
    }

    private fun extractSharedUris(intent: Intent): List<Uri> {
        val result = linkedMapOf<String, Uri>()

        fun add(uri: Uri?) {
            if (uri != null) result[uri.toString()] = uri
        }

        add(intent.data)

        intent.clipData?.let { clipData ->
            for (i in 0 until clipData.itemCount) {
                val item = clipData.getItemAt(i)
                add(item.uri)
                add(item.intent?.data)
            }
        }

        if (intent.action == Intent.ACTION_SEND_MULTIPLE) {
            sharedStreamUriList(intent).forEach { add(it) }
        } else {
            add(sharedStreamUri(intent))
        }

        return result.values.toList()
    }

    private fun addSharedImportFailureRecord(reason: String, originalUri: String) {
        Log.w(TAG, "shared audio import diagnostic reason=$reason, detail=$originalUri")
        sharedAudioRecords.add(
            0,
            SharedAudioRecord(
                fileName = reason,
                mimeType = null,
                fileSizeBytes = 0L,
                originalUri = originalUri,
                savedPath = "",
                receivedAt = System.currentTimeMillis(),
                status = SharedImportStatus.FAILED,
            )
        )
        refreshSharedImportRecordsUi()
    }

    private fun buildShareIntentDebugText(intent: Intent): String {
        val clipCount = intent.clipData?.itemCount ?: 0
        val extraKeys = intent.extras?.keySet()?.joinToString(",").orEmpty()
        return "action=${intent.action}, type=${intent.type}, data=${intent.data}, clipCount=$clipCount, extras=$extraKeys"
    }

    private fun sharedStreamUri(intent: Intent): Uri? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra(Intent.EXTRA_STREAM) as? Uri
        }
    }

    private fun sharedStreamUriList(intent: Intent): List<Uri> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java).orEmpty()
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableArrayListExtra<Uri>(Intent.EXTRA_STREAM).orEmpty()
        }
    }

    private fun importSharedAudio(uri: Uri): SharedImportResult {
        val info = queryAudioFileInfo(uri)
        val mime = info.mimeType
        val safeName = buildSafeSharedAudioFileName(info.displayName)
        val target = allocateSharedAudioTarget(safeName)
        val part = File(target.parentFile, "${target.name}.part")

        if (part.exists() && !part.delete()) {
            error("无法清理旧临时文件：${part.absolutePath}")
        }

        try {
            val copyDiagnostic = copySharedUriToPartWithRetry(uri, part)
            require(isRiffWaveFile(part)) {
                "当前第一版只支持 WAV 文件，请分享 .wav 音频。$copyDiagnostic"
            }
            if (!part.renameTo(target)) {
                error("无法保存分享音频：${target.absolutePath}")
            }

            val record = SharedAudioRecord(
                fileName = target.name,
                mimeType = mime,
                fileSizeBytes = target.length(),
                originalUri = uri.toString(),
                savedPath = target.absolutePath,
                receivedAt = System.currentTimeMillis(),
                status = SharedImportStatus.PENDING_TRANSCRIPTION,
            )
            Log.i(
                TAG,
                "shared audio imported uri=$uri, target=${target.absolutePath}, bytes=${target.length()}, $copyDiagnostic"
            )
            return SharedImportResult(
                record = record,
                selectedAudioFile = SelectedAudioFile(
                    uri = Uri.fromFile(target),
                    displayName = target.name,
                    mimeType = "audio/wav",
                    sizeBytes = target.length(),
                ),
            )
        } catch (t: Throwable) {
            if (part.exists()) {
                try {
                    part.delete()
                } catch (_: Throwable) {
                }
            }
            throw t
        }
    }

    private fun copySharedUriToPartWithRetry(uri: Uri, part: File): String {
        val attempts = mutableListOf<String>()
        var lastError: Throwable? = null

        for (attempt in 1..SHARED_IMPORT_RETRY_COUNT) {
            if (part.exists()) {
                try {
                    part.delete()
                } catch (_: Throwable) {
                }
            }

            try {
                val bytes = copySharedUriOnce(uri, part)
                attempts += "attempt=$attempt bytes=$bytes"
                Log.i(TAG, "shared audio copy attempt=$attempt uri=$uri bytes=$bytes")

                if (bytes > 12L) {
                    return "复制诊断：${attempts.joinToString("; ")}"
                }
            } catch (t: Throwable) {
                lastError = t
                attempts += "attempt=$attempt error=${t.message ?: t::class.java.simpleName}"
                Log.w(TAG, "shared audio copy attempt failed attempt=$attempt uri=$uri", t)
            }

            if (attempt < SHARED_IMPORT_RETRY_COUNT) {
                Thread.sleep(SHARED_IMPORT_RETRY_DELAY_MS)
            }
        }

        if (part.exists()) {
            try {
                part.delete()
            } catch (_: Throwable) {
            }
        }

        val errorSuffix = lastError?.let { "；最后错误：${it.message ?: it::class.java.simpleName}" }.orEmpty()
        error("分享音频为空或文件过小（${attempts.joinToString("; ")}）$errorSuffix")
    }

    private fun copySharedUriOnce(uri: Uri, part: File): Long {
        val input = contentResolver.openInputStream(uri)
            ?: error("无法读取分享音频：$uri")

        input.use { source ->
            FileOutputStream(part).use { output ->
                val buffer = ByteArray(SHARED_IMPORT_BUFFER_SIZE)
                var total = 0L
                while (true) {
                    val read = source.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    total += read.toLong()
                }
                output.fd.sync()
                return total
            }
        }
    }

    private fun buildSafeSharedAudioFileName(displayName: String?): String {
        val fallback = "shared_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.wav"
        val original = displayName
            ?.substringAfterLast('/')
            ?.substringAfterLast('\\')
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?: fallback

        val sanitized = original
            .replace(Regex("""[\u0000-\u001F\u007F\\/:*?"<>|]"""), "_")
            .replace(Regex("""_+"""), "_")
            .trim('.', ' ', '_')
            .take(SHARED_IMPORT_MAX_NAME_LENGTH)
            .ifBlank { fallback }

        val withExtension = if (sanitized.lowercase(Locale.US).endsWith(".wav")) {
            sanitized
        } else {
            "${sanitized.substringBeforeLast('.', sanitized)}.wav"
        }
        return withExtension.ifBlank { fallback }
    }

    private fun allocateSharedAudioTarget(fileName: String): File {
        val sharedDir = File(File(getExternalFilesDir(null) ?: filesDir, "Incoming"), "shared")
        if (!sharedDir.exists() && !sharedDir.mkdirs()) {
            error("无法创建分享导入目录：${sharedDir.absolutePath}")
        }

        val base = fileName.substringBeforeLast('.', fileName)
        val ext = fileName.substringAfterLast('.', "wav")
        var candidate = File(sharedDir, "$base.$ext")
        var index = 1
        while (candidate.exists() || File(sharedDir, "${candidate.name}.part").exists()) {
            candidate = File(sharedDir, "${base}_$index.$ext")
            index++
        }
        return candidate
    }

    private fun isRiffWaveFile(file: File): Boolean {
        val header = ByteArray(12)
        FileInputStream(file).use { input ->
            val read = input.read(header)
            return read == 12 &&
                String(header, 0, 4, Charsets.US_ASCII) == "RIFF" &&
                String(header, 8, 4, Charsets.US_ASCII) == "WAVE"
        }
    }

    private fun refreshSharedImportRecordsUi() {
        sharedImportRecordsView.text = buildSharedImportRecordsText()
    }

    private fun buildSharedImportRecordsText(): String {
        if (sharedAudioRecords.isEmpty()) {
            return "分享接收记录：暂无"
        }

        return buildString {
            appendLine("分享接收记录：")
            sharedAudioRecords.take(10).forEachIndexed { index, record ->
                appendLine()
                appendLine("${index + 1}. ${record.fileName}")
                if (record.status == SharedImportStatus.FAILED || record.originalUri.isNotBlank()) {
                    appendLine("诊断: ${record.originalUri}")
                }
                appendLine("MIME: ${record.mimeType ?: "未知"}")
                appendLine("大小: ${record.fileSizeBytes} bytes")
                appendLine("接收时间: ${formatAirecUploadTime(record.receivedAt)}")
                appendLine("状态: ${record.status}")
                appendLine("路径: ${record.savedPath}")
            }
        }.trim()
    }

    private fun pickAudioFile() {
        audioFilePicker.launch(
            arrayOf(
                "audio/wav",
                "audio/x-wav",
                "audio/*",
            )
        )
    }

    private fun onAudioFileSelected(uri: Uri) {
        runCatching {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        val info = queryAudioFileInfo(uri)
        selectedAudioFile = info
        selectedAudioView.text = buildSelectedAudioText(info)
        updateSelectedAudioButtonState()
        updateVoiceprintButtonState()

        if (!isSupportedWav(info)) {
            setStatus("当前第一版只支持 WAV 文件，请选择 .wav 音频。")
        }
    }

    private fun queryAudioFileInfo(uri: Uri): SelectedAudioFile {
        var displayName: String? = null
        var sizeBytes: Long? = null

        contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
            null,
            null,
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0 && !cursor.isNull(nameIndex)) {
                    displayName = cursor.getString(nameIndex)
                }

                val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) {
                    sizeBytes = cursor.getLong(sizeIndex)
                }
            }
        }

        return SelectedAudioFile(
            uri = uri,
            displayName = displayName,
            mimeType = contentResolver.getType(uri),
            sizeBytes = sizeBytes,
        )
    }

    private fun buildSelectedAudioText(file: SelectedAudioFile): String {
        return buildString {
            appendLine("已选择：${file.displayName ?: "未知文件"}")
            append("大小：${formatBytes(file.sizeBytes)}")
            if (!isSupportedWav(file)) {
                appendLine()
                append("当前仅支持 WAV 文件")
            }
        }
    }

    private fun isSupportedWav(fileInfo: SelectedAudioFile): Boolean {
        val name = fileInfo.displayName?.lowercase(Locale.US)
        val mime = fileInfo.mimeType?.lowercase(Locale.US)
        return name?.endsWith(".wav") == true ||
            mime == "audio/wav" ||
            mime == "audio/x-wav"
    }

    private fun updateSelectedAudioButtonState() {
        transcribeSelectedButton.isEnabled =
            recognizer != null && selectedAudioFile?.let { isSupportedWav(it) } == true
    }

    private fun updateVoiceprintButtonState() {
        val hasEmployee = selectedVoiceprintEmployeeId != null
        val hasSupportedWav = selectedAudioFile?.let { isSupportedWav(it) } == true
        addVoiceSampleButton.isEnabled = hasEmployee && hasSupportedWav
        enrollVoiceprintButton.isEnabled = hasEmployee
        identifySpeakerButton.isEnabled = hasSupportedWav
        deleteEmployeeButton.isEnabled = hasEmployee
    }

    private fun initRecognizerAsync() {
        progressBar.visibility = View.VISIBLE

        thread(name = "init-asr") {
            try {
                val missing = mutableListOf<String>()
                if (!assetFileExists("$MODEL_DIR/model.int8.onnx")) missing += "$MODEL_DIR/model.int8.onnx"
                if (!assetFileExists("$MODEL_DIR/tokens.txt")) missing += "$MODEL_DIR/tokens.txt"
                if (!assetFileExists(VAD_MODEL_ASSET)) missing += VAD_MODEL_ASSET
                if (!assetFileExists(TEST_WAV)) missing += TEST_WAV

                if (missing.isNotEmpty()) {
                    runOnUiThread {
                        progressBar.visibility = View.GONE
                        setStatus("缺少资源文件：\n${missing.joinToString("\n")}")
                    }
                    return@thread
                }

                val config = OfflineRecognizerConfig(
                    featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
                    modelConfig = OfflineModelConfig(
                        paraformer = OfflineParaformerModelConfig(
                            model = "$MODEL_DIR/model.int8.onnx",
                        ),
                        tokens = "$MODEL_DIR/tokens.txt",
                        modelType = "paraformer",
                    ),
                )

                recognizer = OfflineRecognizer(assetManager = assets, config = config)
                speechGate = SpeechGate(
                    assetManager = assets,
                    config = SpeechGateConfig(
                        modelAssetPath = VAD_MODEL_ASSET,
                        minSpeechMs = NATURAL_UTTERANCE_MIN_SPEECH_MS,
                        minSpeechRatio = 0.005f,
                        speechPadMs = 0L,
                        mergeGapMs = 0L,
                        modelMinSilenceDurationSec = 0.25f,
                        modelMinSpeechDurationSec = 0.18f,
                        modelMaxSpeechDurationSec = 30.0f,
                    ),
                )

                runOnUiThread {
                    progressBar.visibility = View.GONE
                    updateSelectedAudioButtonState()
                    refreshAirecReceiverUi()
                    setStatus("ASR 和人声检测模型已就绪，可以选择 WAV 文件开始分析。")
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    setStatus("模型初始化失败：${t.message ?: t::class.java.simpleName}")
                }
            }
        }
    }

    private fun transcribeSelectedAudio(resumeJobId: String? = null) {
        val currentRecognizer = recognizer ?: run { setStatus("模型尚未就绪"); return }
        if (activeTranscriptionJobId != null) {
            setStatus("已有录音任务正在运行，请等待完成或在通知栏取消。")
            return
        }
        val selection = selectedAudioFile ?: run { setStatus("请先选择 WAV 文件"); return }
        if (!isSupportedWav(selection)) {
            setStatus("当前第一版只支持 WAV 文件，请选择 .wav 音频。")
            return
        }
        val displayName = selection.displayName ?: "selected_audio.wav"
        val job = if (resumeJobId != null) {
            TranscriptionJobRepository.get(this, resumeJobId) ?: run {
                setStatus("找不到需要继续的任务"); return
            }
        } else {
            TranscriptionJobRepository.create(
                this, displayName, sourceTypeForSelection(selection), selection.uri.toString()
            )
        }
        activeTranscriptionJobId = job.id
        requestNotificationPermissionIfNeeded()
        progressBar.visibility = View.VISIBLE
        transcribeSelectedButton.isEnabled = false
        copyButton.isEnabled = false
        resultView.text = ""
        setStatus(if (resumeJobId == null) "正在准备录音任务..." else "正在从检查点继续任务...")
        refreshJobHistoryUi()
        TranscriptionForegroundService.start(this, job.id, "LightASR 正在分析", displayName)

        thread(name = "transcription-job-" + job.id.take(8)) {
            try {
                val wavFile = prepareJobAudio(job, selection)
                TranscriptionJobRepository.update(this, job.id) {
                    it.copy(status = TranscriptionJobStatus.TRANSCRIBING, errorMessage = null)
                }
                val output = recognizeWavFile(currentRecognizer, wavFile, displayName, job.id)
                val txtFile = saveTimestampedTxt(displayName, buildTimestampedTxt(displayName, output))
                lastOutput = buildSelectedOutput(displayName, output, txtFile, null)
                TranscriptionJobRepository.update(this, job.id) {
                    it.copy(
                        transcriptPath = txtFile.absolutePath,
                        durationMs = (output.durationSec * 1000.0).roundToLong(),
                        nextChunkIndex = output.stats.totalChunks,
                        progressPercent = 100,
                        status = TranscriptionJobStatus.LOCAL_COMPLETED,
                        errorMessage = null,
                    )
                }
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    updateSelectedAudioButtonState()
                    copyButton.isEnabled = true
                    resultView.text = lastOutput
                    showResultPage(
                        ResultPageState(
                            displayName = displayName,
                            output = output,
                            txtFile = txtFile,
                            txtSaveError = null,
                            jobId = job.id,
                            audioSizeBytes = wavFile.length(),
                        )
                    )
                    setStatus(if (!output.stats.hasAnySpeech) NO_SPEECH_MESSAGE else "本地分析完成：" + displayName)
                    refreshJobHistoryUi()
                }
            } catch (t: Throwable) {
                val canceled = t is TranscriptionCanceledException
                val message = t.message ?: t::class.java.simpleName
                TranscriptionJobRepository.update(this, job.id) {
                    it.copy(
                        status = if (canceled) TranscriptionJobStatus.CANCELED else TranscriptionJobStatus.FAILED,
                        errorMessage = message,
                    )
                }
                lastOutput = if (canceled) "任务已取消" else "识别失败：\n" + message
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    updateSelectedAudioButtonState()
                    resultView.text = lastOutput
                    setStatus(lastOutput)
                    refreshJobHistoryUi()
                }
            } finally {
                activeTranscriptionJobId = null
                TranscriptionForegroundService.stop(this, job.id)
                if (isDestroyed) {
                    runCatching { currentRecognizer.release() }
                    runCatching { speechGate?.release() }
                }
            }
        }
    }

    private fun prepareJobAudio(job: TranscriptionJob, selection: SelectedAudioFile): File {
        job.audioPath?.let { path ->
            val file = File(path)
            if (file.isFile && file.length() > 44L) {
                val hash = job.audioSha256 ?: sha256(file)
                TranscriptionJobRepository.update(this, job.id) {
                    it.copy(audioPath = file.absolutePath, audioSha256 = hash)
                }
                return file
            }
        }
        val direct = if (selection.uri.scheme.equals("file", true)) selection.uri.path?.let(::File) else null
        val appRoot = getExternalFilesDir(null)?.canonicalFile
        if (direct != null && direct.isFile && direct.length() > 44L && appRoot != null &&
            direct.canonicalFile.toPath().startsWith(appRoot.toPath())
        ) {
            val hash = sha256(direct)
            TranscriptionJobRepository.update(this, job.id) {
                it.copy(audioPath = direct.absolutePath, audioSha256 = hash)
            }
            return direct
        }

        val root = File(getExternalFilesDir(null) ?: filesDir, "Recordings/" + job.id)
        check(root.exists() || root.mkdirs()) { "无法创建任务录音目录" }
        val safeName = buildSafeSharedAudioFileName(selection.displayName).ifBlank { "selected_audio.wav" }
        val target = File(root, safeName)
        val part = File(root, safeName + ".part")
        part.delete()
        val expected = selection.sizeBytes ?: 0L
        if (expected > 0L && root.usableSpace < expected + 64L * 1024L * 1024L) {
            error("手机剩余空间不足，无法保存待分析录音。")
        }

        val digest = MessageDigest.getInstance("SHA-256")
        val input = contentResolver.openInputStream(selection.uri)
            ?: error("无法读取所选音频：" + selection.uri)
        input.use { source ->
            FileOutputStream(part).use { output ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val count = source.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    digest.update(buffer, 0, count)
                }
                output.flush()
                output.fd.sync()
            }
        }
        check(part.length() > 44L) { "复制后的 WAV 文件为空" }
        if (target.exists()) target.delete()
        if (!part.renameTo(target)) {
            part.copyTo(target, overwrite = true)
            part.delete()
        }
        val hash = digest.digest().joinToString("") { "%02x".format(it) }
        TranscriptionJobRepository.update(this, job.id) {
            it.copy(audioPath = target.absolutePath, audioSha256 = hash)
        }
        return target
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun sourceTypeForSelection(selection: SelectedAudioFile): String {
        val path = selection.uri.path.orEmpty().replace('\\', '/')
        return when {
            path.contains("/Incoming/shared/", true) -> "shared"
            path.contains("/Incoming/", true) -> "airec"
            else -> "local"
        }
    }

    private fun loadServerConfigUi() {
        if (!::serverUrlInput.isInitialized) return
        val config = LightAsrServerConfigRepository.load(this)
        serverUrlInput.setText(config.baseUrl)
        serverTokenInput.setText(config.uploadToken)
    }

    private fun saveServerConfig() {
        try {
            LightAsrServerConfigRepository.save(
                this, serverUrlInput.text?.toString().orEmpty(), serverTokenInput.text?.toString().orEmpty()
            )
            setStatus("服务器设置已保存")
        } catch (t: Throwable) {
            setStatus("服务器设置保存失败：" + (t.message ?: t::class.java.simpleName))
        }
    }

    private fun refreshJobHistoryUi() {
        if (!::jobHistoryActions.isInitialized) return
        val jobs = latestJobsByAudioName(TranscriptionJobRepository.list(this))
        refreshJobHistoryActions(jobs)
    }

    private fun latestJobsByAudioName(jobs: List<TranscriptionJob>): List<TranscriptionJob> {
        val seen = linkedSetOf<String>()
        val result = mutableListOf<TranscriptionJob>()
        jobs.sortedByDescending { it.updatedAt }.forEach { job ->
            val key = File(job.sourceName).name.trim().lowercase(Locale.ROOT)
            if (seen.add(key)) result += job
        }
        return result.take(8)
    }

    private fun refreshResultPageIfVisible() {
        val state = latestResultPageState ?: return
        if (state.jobId == null) return
        if (currentPage != LightAsrPage.RESULT) return
        runCatching { showResultPage(state, scrollToTop = false) }
            .onFailure { Log.w(TAG, "result page refresh failed", it) }
    }

    private fun refreshJobDetailPageIfVisible() {
        val jobId = activeJobDetailId ?: return
        if (currentPage != LightAsrPage.JOB_DETAIL) return
        runCatching { showJobDetailPage(jobId, scrollToTop = false) }
            .onFailure { Log.w(TAG, "job detail page refresh failed", it) }
    }

    private fun refreshJobHistoryActions(jobs: List<TranscriptionJob>) {
        if (!::jobHistoryActions.isInitialized) return
        jobHistoryActions.removeAllViews()
        if (jobs.isEmpty()) {
            jobHistoryActions.addView(TextView(this).apply {
                text = "暂无分析任务"
                textSize = 14f
                setInfoBoxStyle(this, Color.parseColor("#F7FAF8"))
            }, matchWrap())
            return
        }
        jobs.take(5).forEachIndexed { index, job ->
            val item = buildRecentJobItem(job)
            jobHistoryActions.addView(item, if (index == 0) matchWrap() else childWrap(topMargin = 8))
        }
    }

    private fun buildRecentJobItem(job: TranscriptionJob): LinearLayout {
        val canResume = job.status in setOf(
            TranscriptionJobStatus.INTERRUPTED,
            TranscriptionJobStatus.FAILED,
            TranscriptionJobStatus.CANCELED,
        ) && job.audioPath?.let { File(it).isFile } == true
        val canUpload = job.status in setOf(
            TranscriptionJobStatus.LOCAL_COMPLETED,
            TranscriptionJobStatus.UPLOAD_FAILED,
        )
        val canViewResult = job.status in setOf(
            TranscriptionJobStatus.LOCAL_COMPLETED,
            TranscriptionJobStatus.UPLOADING,
            TranscriptionJobStatus.UPLOADED,
            TranscriptionJobStatus.UPLOAD_FAILED,
        ) || job.transcriptPath?.let { File(it).isFile } == true
        val buttons = mutableListOf<Button>()
        if (canViewResult) {
            buttons += Button(this).apply {
                text = "查看结果"
                styleSecondaryButton(this)
                setOnClickListener { showJobDetailPage(job.id) }
            }
        }
        if (canResume) {
            buttons += Button(this).apply {
                text = "继续"
                styleSecondaryButton(this)
                isEnabled = recognizer != null && activeTranscriptionJobId == null
                setOnClickListener { resumeJob(job.id) }
            }
        }
        if (canUpload) {
            buttons += Button(this).apply {
                text = if (job.status == TranscriptionJobStatus.UPLOAD_FAILED) "重新上传" else "上传"
                styleSecondaryButton(this)
                isEnabled = activeTranscriptionJobId == null
                setOnClickListener { uploadJob(job.id) }
            }
        }

        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = roundedBackground(
                fillColor = Color.parseColor("#F7FAF8"),
                strokeColor = Color.parseColor("#E5EAF2"),
                radiusDp = 8,
            )
            addView(TextView(this@MainActivity).apply {
                text = job.sourceName
                textSize = 15f
                typeface = Typeface.DEFAULT_BOLD
                setTextColor(Color.parseColor("#111827"))
            }, matchWrap())
            addView(TextView(this@MainActivity).apply {
                text = jobStatusText(job.status) + " · " + job.progressPercent + "% · " +
                    formatJobDuration(job.durationMs) + " · " + job.sourceType
                textSize = 13f
                setTextColor(Color.parseColor("#5D6675"))
                setPadding(0, dp(4), 0, 0)
            }, matchWrap())
            if (job.totalChunks > 0) {
                addView(TextView(this@MainActivity).apply {
                    text = "进度：" + job.nextChunkIndex + "/" + job.totalChunks + " chunk"
                    textSize = 13f
                    setTextColor(Color.parseColor("#5D6675"))
                    setPadding(0, dp(4), 0, 0)
                }, matchWrap())
            }
            job.serverRecordingId?.let {
                addView(TextView(this@MainActivity).apply {
                    text = "服务器记录：$it"
                    textSize = 13f
                    setTextColor(Color.parseColor("#5D6675"))
                    setPadding(0, dp(4), 0, 0)
                }, matchWrap())
            }
            job.errorMessage?.let {
                addView(TextView(this@MainActivity).apply {
                    text = "提示：$it"
                    textSize = 13f
                    setTextColor(Color.parseColor("#8A4B0F"))
                    setPadding(0, dp(4), 0, 0)
                }, matchWrap())
            }
            if (buttons.isNotEmpty()) {
                addView(buildButtonRow(*buttons.toTypedArray()), childWrap(topMargin = 8))
            }
        }
    }

    private fun resumeJob(jobId: String) {
        val job = TranscriptionJobRepository.get(this, jobId) ?: return
        val audio = job.audioPath?.let(::File)
        if (audio == null || !audio.isFile) {
            setStatus("任务原音频不存在，无法继续")
            return
        }
        selectedAudioFile = SelectedAudioFile(Uri.fromFile(audio), job.sourceName, "audio/wav", audio.length())
        selectedAudioView.text = buildSelectedAudioText(selectedAudioFile!!)
        updateSelectedAudioButtonState()
        transcribeSelectedAudio(job.id)
    }

    private fun uploadJob(jobId: String) {
        if (activeTranscriptionJobId != null) { setStatus("已有任务正在运行"); return }
        val job = TranscriptionJobRepository.get(this, jobId) ?: return
        val audio = job.audioPath?.let(::File)
        val transcript = job.transcriptPath?.let(::File)
        val hash = job.audioSha256
        if (audio == null || !audio.isFile || transcript == null || !transcript.isFile || hash.isNullOrBlank()) {
            setStatus("上传所需的 WAV、TXT 或 SHA256 不完整")
            return
        }
        activeTranscriptionJobId = job.id
        TranscriptionJobRepository.update(this, job.id) {
            it.copy(status = TranscriptionJobStatus.UPLOADING, progressPercent = 0, errorMessage = null)
        }
        runCatching { TranscriptionForegroundService.start(this, job.id, "LightASR 正在上传", job.sourceName) }
            .onFailure { Log.w(TAG, "upload foreground service start failed", it) }
        refreshJobHistoryUi()
        refreshResultPageIfVisible()
        refreshJobDetailPageIfVisible()
        val uploadTxtName = audio.name.substringBeforeLast('.', audio.name) + ".txt"
        setStatus("正在上传：${audio.name} + $uploadTxtName")

        thread(name = "upload-job-" + job.id.take(8)) {
            var lastProgress = -1
            try {
                val response = RecordingUploadClient(LightAsrServerConfigRepository.load(this)).upload(
                    audio, transcript, job.sourceName, job.sourceType,
                    recordingTimeIsoForSource(job.sourceName), job.durationMs, hash,
                    packageManager.getPackageInfo(packageName, 0).versionName ?: "unknown", deviceIdForUpload(),
                    onStage = { stage ->
                        TranscriptionJobRepository.update(this, job.id) {
                            it.copy(status = TranscriptionJobStatus.UPLOADING, errorMessage = stage)
                        }
                        runOnUiThread {
                            setStatus(stage + "：" + job.sourceName)
                            refreshJobHistoryUi()
                            refreshResultPageIfVisible()
                            refreshJobDetailPageIfVisible()
                        }
                    },
                    onProgress = { progress ->
                    if (progress >= lastProgress + 2) {
                        lastProgress = progress
                        TranscriptionJobRepository.update(this, job.id) { it.copy(progressPercent = progress) }
                        runCatching {
                            TranscriptionForegroundService.progress(
                                this, job.id, "LightASR 正在上传", job.sourceName + "  " + progress + "%", progress
                            )
                        }.onFailure { Log.w(TAG, "upload foreground progress failed", it) }
                        runOnUiThread {
                            setStatus("正在上传：${audio.name} + $uploadTxtName  $progress%")
                            refreshJobHistoryUi()
                            refreshResultPageIfVisible()
                            refreshJobDetailPageIfVisible()
                        }
                    }
                    },
                )
                TranscriptionJobRepository.update(this, job.id) {
                    it.copy(
                        status = TranscriptionJobStatus.UPLOADED,
                        progressPercent = 100,
                        serverRecordingId = response.recordingId,
                        errorMessage = null,
                    )
                }
                runOnUiThread {
                    setStatus("上传完成：" + response.recordingId)
                    refreshJobHistoryUi()
                    refreshResultPageIfVisible()
                    refreshJobDetailPageIfVisible()
                }
            } catch (t: Throwable) {
                val message = t.message ?: t::class.java.simpleName
                TranscriptionJobRepository.update(this, job.id) {
                    it.copy(status = TranscriptionJobStatus.UPLOAD_FAILED, errorMessage = message)
                }
                runOnUiThread {
                    setStatus("上传失败，可在任务记录中重试：" + message)
                    refreshJobHistoryUi()
                    refreshResultPageIfVisible()
                    refreshJobDetailPageIfVisible()
                }
            } finally {
                activeTranscriptionJobId = null
                runCatching { TranscriptionForegroundService.stop(this, job.id) }
                    .onFailure { Log.w(TAG, "upload foreground service stop failed", it) }
            }
        }
    }

    private fun recordingTimeIsoForSource(sourceFileName: String): String? {
        val epochMs = parseRecordingStartEpochMsFromAirecFileName(sourceFileName) ?: return null
        return SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.US).apply {
            timeZone = TimeZone.getTimeZone(DEFAULT_RECORDING_TIMEZONE_ID)
        }.format(Date(epochMs))
    }

    private fun deviceIdForUpload(): String {
        val id = Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)
            ?.replace(Regex("[^A-Za-z0-9_.-]"), "_").orEmpty().ifBlank { "unknown" }
        return "android-" + id
    }

    private fun jobStatusText(status: TranscriptionJobStatus): String = when (status) {
        TranscriptionJobStatus.PREPARING -> "准备中"
        TranscriptionJobStatus.TRANSCRIBING -> "分析中"
        TranscriptionJobStatus.INTERRUPTED -> "已中断，可继续"
        TranscriptionJobStatus.LOCAL_COMPLETED -> "本地分析完成"
        TranscriptionJobStatus.FAILED -> "分析失败"
        TranscriptionJobStatus.CANCELED -> "已取消"
        TranscriptionJobStatus.UPLOADING -> "上传中"
        TranscriptionJobStatus.UPLOADED -> "已上传"
        TranscriptionJobStatus.UPLOAD_FAILED -> "上传失败，可重试"
    }

    private fun createVoiceprintEmployee() {
        val name = employeeNameInput.text?.toString().orEmpty()
        val storeName = employeeStoreInput.text?.toString().orEmpty()
        try {
            val employee = voiceprintManager.createEmployee(name, storeName)
            selectedVoiceprintEmployeeId = employee.id
            employeeNameInput.setText("")
            employeeStoreInput.setText("")
            updateVoiceprintEmployeeView()
            lastOutput = "employee created id=${employee.id} name=${employee.name} store=${employee.storeName ?: ""}"
            resultView.text = lastOutput
            copyButton.isEnabled = true
            setStatus("已创建员工：${employee.name}")
        } catch (t: Throwable) {
            setStatus("创建员工失败：${t.message ?: t::class.java.simpleName}")
        }
    }

    private fun selectNextVoiceprintEmployee() {
        val employees = voiceprintManager.listEmployees()
        if (employees.isEmpty()) {
            selectedVoiceprintEmployeeId = null
            updateVoiceprintEmployeeView()
            setStatus("暂无员工，请先创建员工")
            return
        }

        val currentIndex = employees.indexOfFirst { it.id == selectedVoiceprintEmployeeId }
        val nextIndex = if (currentIndex < 0) 0 else (currentIndex + 1) % employees.size
        selectedVoiceprintEmployeeId = employees[nextIndex].id
        updateVoiceprintEmployeeView()
        setStatus("当前员工：${employees[nextIndex].name}")
    }

    private fun addVoiceprintSampleFromSelected() {
        val employeeId = selectedVoiceprintEmployeeId ?: run {
            setStatus("请先创建或选择员工")
            return
        }
        val currentSelection = selectedAudioFile ?: run {
            setStatus("请先选择 WAV 文件")
            return
        }
        if (!isSupportedWav(currentSelection)) {
            setStatus("当前声纹样本暂只支持标准 PCM WAV，后续再接 MediaExtractor 转码。")
            return
        }

        setVoiceprintBusy(true)
        setStatus("正在添加声纹样本...")
        thread(name = "voiceprint-add-sample") {
            try {
                val wavFile = copySelectedUriToCache(currentSelection.uri, currentSelection.displayName)
                val sample = voiceprintManager.addVoiceSample(employeeId, wavFile)
                val employee = voiceprintManager.getEmployee(employeeId)
                lastOutput = buildVoiceSampleAddedText(employee, sample)
                runOnUiThread {
                    setVoiceprintBusy(false)
                    updateVoiceprintEmployeeView()
                    resultView.text = lastOutput
                    copyButton.isEnabled = true
                    setStatus("声纹样本已添加：${employee?.name ?: employeeId}")
                }
            } catch (t: Throwable) {
                val message = t.message ?: t::class.java.simpleName
                lastOutput = "声纹样本添加失败：\n$message"
                runOnUiThread {
                    setVoiceprintBusy(false)
                    resultView.text = lastOutput
                    copyButton.isEnabled = true
                    setStatus("声纹样本添加失败：$message")
                }
            }
        }
    }

    private fun enrollSelectedVoiceprintEmployee() {
        val employeeId = selectedVoiceprintEmployeeId ?: run {
            setStatus("请先创建或选择员工")
            return
        }

        setVoiceprintBusy(true)
        setStatus("正在执行声纹注册...")
        thread(name = "voiceprint-enroll") {
            try {
                val voiceprint = voiceprintManager.enrollEmployee(employeeId)
                lastOutput = """
voiceprint template saved employee=${voiceprint.employeeId}
员工：${voiceprint.employeeName}
门店：${voiceprint.storeName ?: "未填写"}
embeddingDim：${voiceprint.embeddingDim}
modelVersion：${voiceprint.modelVersion}
sampleCount：${voiceprint.sampleCount}
""".trim()
                runOnUiThread {
                    setVoiceprintBusy(false)
                    updateVoiceprintEmployeeView()
                    resultView.text = lastOutput
                    copyButton.isEnabled = true
                    setStatus("声纹注册完成：${voiceprint.employeeName}")
                }
            } catch (t: Throwable) {
                val message = t.message ?: t::class.java.simpleName
                lastOutput = "声纹注册失败：\n$message"
                runOnUiThread {
                    setVoiceprintBusy(false)
                    resultView.text = lastOutput
                    copyButton.isEnabled = true
                    setStatus(
                        if (message.contains("voiceprint model not available")) {
                            "声纹模型未就绪：当前 Android runtime 暂不支持 CAM++ embedding。"
                        } else {
                            "声纹注册失败：$message"
                        }
                    )
                }
            }
        }
    }

    private fun identifySelectedSpeaker() {
        val currentSelection = selectedAudioFile ?: run {
            setStatus("请先选择 WAV 文件")
            return
        }
        if (!isSupportedWav(currentSelection)) {
            setStatus("当前声纹识别暂只支持标准 PCM WAV，后续再接 MediaExtractor 转码。")
            return
        }

        setVoiceprintBusy(true)
        setStatus("正在识别所选音频说话人...")
        thread(name = "voiceprint-identify") {
            try {
                val wavFile = copySelectedUriToCache(currentSelection.uri, currentSelection.displayName)
                val result = voiceprintManager.identifySpeaker(wavFile)
                lastOutput = buildVoiceprintMatchText(result)
                runOnUiThread {
                    setVoiceprintBusy(false)
                    resultView.text = lastOutput
                    copyButton.isEnabled = true
                    setStatus(
                        if (result.reason == "model_not_available") {
                            "声纹模型未就绪：当前 Android runtime 暂不支持 CAM++ embedding。"
                        } else {
                            "声纹识别完成：${result.employeeName ?: "unknown"}"
                        }
                    )
                }
            } catch (t: Throwable) {
                val message = t.message ?: t::class.java.simpleName
                lastOutput = "声纹识别失败：\n$message"
                runOnUiThread {
                    setVoiceprintBusy(false)
                    resultView.text = lastOutput
                    copyButton.isEnabled = true
                    setStatus("声纹识别失败：$message")
                }
            }
        }
    }

    private fun showVoiceprintEmployees() {
        val runtimeCheck = voiceprintManager.runtimeCheck()
        val employees = voiceprintManager.listEmployees()
        lastOutput = """
声纹模型状态：
supported=${runtimeCheck.supported}
reason=${runtimeCheck.reason}
modelVersion=${voiceprintManager.modelVersion}

员工列表：
${buildEmployeesText(employees)}

${voiceprintManager.runMatcherSelfCheck()}
""".trim()
        resultView.text = lastOutput
        copyButton.isEnabled = true
        setStatus("员工列表已刷新")
    }

    private fun deleteSelectedVoiceprintEmployee() {
        val employeeId = selectedVoiceprintEmployeeId ?: run {
            setStatus("请先选择员工")
            return
        }
        val employeeName = voiceprintManager.getEmployee(employeeId)?.name ?: employeeId.toString()

        try {
            voiceprintManager.deleteEmployee(employeeId)
            selectedVoiceprintEmployeeId = voiceprintManager.listEmployees().firstOrNull()?.id
            updateVoiceprintEmployeeView()
            lastOutput = "employee deleted id=$employeeId name=$employeeName"
            resultView.text = lastOutput
            copyButton.isEnabled = true
            setStatus("已删除员工：$employeeName")
        } catch (t: Throwable) {
            setStatus("删除员工失败：${t.message ?: t::class.java.simpleName}")
        }
    }

    private fun updateVoiceprintEmployeeView() {
        val employees = voiceprintManager.listEmployees()
        if (selectedVoiceprintEmployeeId == null && employees.isNotEmpty()) {
            selectedVoiceprintEmployeeId = employees.first().id
        }

        val selected = selectedVoiceprintEmployeeId?.let { voiceprintManager.getEmployee(it) }
        if (selected == null) {
            selectedVoiceprintEmployeeId = null
            voiceprintEmployeeView.text = "当前员工：未选择\n声纹模型：${voiceprintManager.runtimeCheck().reason}"
        } else {
            voiceprintEmployeeView.text = """
当前员工：${selected.name}
员工ID：${selected.id}
门店：${selected.storeName ?: "未填写"}
已注册：${if (selected.enrolled) "是" else "否"}
样本数：${selected.sampleCount}
声纹模型：${voiceprintManager.runtimeCheck().reason}
""".trim()
        }
        updateVoiceprintButtonState()
    }

    private fun setVoiceprintBusy(busy: Boolean) {
        progressBar.visibility = if (busy) View.VISIBLE else View.GONE
        createEmployeeButton.isEnabled = !busy
        nextEmployeeButton.isEnabled = !busy
        listEmployeesButton.isEnabled = !busy
        if (busy) {
            addVoiceSampleButton.isEnabled = false
            enrollVoiceprintButton.isEnabled = false
            identifySpeakerButton.isEnabled = false
            deleteEmployeeButton.isEnabled = false
        } else {
            updateVoiceprintButtonState()
        }
    }

    private fun buildVoiceSampleAddedText(employee: Employee?, sample: VoiceSample): String {
        return """
voice sample added employee=${employee?.id ?: sample.employeeId}
员工：${employee?.name ?: "未知"}
门店：${employee?.storeName ?: "未填写"}
sampleId：${sample.id}
duration：${"%.2f".format(Locale.US, sample.durationSec)} 秒
sampleRate：${sample.sampleRate} Hz
file：${sample.filePath}
sampleCount：${employee?.sampleCount ?: "未知"}
""".trim()
    }

    private fun buildVoiceprintMatchText(result: VoiceprintMatchResult): String {
        return """
识别结果：${result.employeeName ?: "unknown"}
门店：${result.storeName ?: "未知"}
相似度：${"%.3f".format(Locale.US, result.top1Score)}
第二候选：${result.top2Score?.let { "%.3f".format(Locale.US, it) } ?: "无"}
margin：${result.scoreMargin?.let { "%.3f".format(Locale.US, it) } ?: "无"}
阈值：${"%.2f".format(Locale.US, voiceprintManager.config.threshold)}
判断：${if (result.accepted) "通过" else "未通过"}
原因：${result.reason}
""".trim()
    }

    private fun buildEmployeesText(employees: List<Employee>): String {
        if (employees.isEmpty()) return "暂无员工"
        return employees.joinToString("\n") { employee ->
            "id=${employee.id}, name=${employee.name}, store=${employee.storeName ?: "未填写"}, " +
                "enrolled=${employee.enrolled}, sampleCount=${employee.sampleCount}"
        }
    }

    private fun copyAssetToCache(assetName: String, cacheFileName: String): File {
        val target = File(cacheDir, cacheFileName)
        assets.open(assetName).use { input ->
            target.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        require(target.length() > 0L) { "缓存内置音频失败：文件为空" }
        Log.i(TAG, "asset copied to cache asset=$assetName, cache=${target.absolutePath}, bytes=${target.length()}")
        return target
    }

    private fun copySelectedUriToCache(uri: Uri, displayName: String?): File {
        val target = File(cacheDir, "selected_audio.wav")
        val input = if (uri.scheme == "file") {
            FileInputStream(File(uri.path ?: error("无法读取本地音频路径：$uri")))
        } else {
            contentResolver.openInputStream(uri)
        }
            ?: error("无法读取所选音频：$uri")

        input.use { source ->
            target.outputStream().use { output ->
                source.copyTo(output, bufferSize = SHARED_IMPORT_BUFFER_SIZE)
            }
        }

        require(target.length() > 0L) {
            "所选音频复制失败：${displayName ?: uri} 为空"
        }

        Log.i(
            TAG,
            "selected uri copied to cache name=${displayName ?: "unknown"}, uri=$uri, " +
                "cache=${target.absolutePath}, bytes=${target.length()}"
        )
        return target
    }

    private fun recognizeWavFile(
        currentRecognizer: OfflineRecognizer,
        wavFile: File,
        sourceFileName: String? = wavFile.name,
        jobId: String? = null,
    ): RecognitionOutput {
        val currentSpeechGate = speechGate ?: error("人声检测模型尚未就绪，不能直接执行 ASR")
        val header = parseWavHeader(wavFile)
        val durationSec = header.durationMs.toDouble() / 1000.0
        Log.i(
            TAG,
            "recognize wav file=${wavFile.name}, path=${wavFile.absolutePath}, " +
                "sampleRate=${header.sampleRate}, channels=${header.channels}, " +
                "bitsPerSample=${header.bitsPerSample}, dataStart=${header.dataStart}, " +
                "dataSize=${header.dataSize}, totalFrames=${header.totalFrames}, " +
                "durationSec=${"%.2f".format(Locale.US, durationSec)}, " +
                "segmentMs=$CHUNKED_SEGMENT_MS, overlapMs=$CHUNKED_OVERLAP_MS"
        )

        val segmentation = try {
            buildNaturalUtteranceSegmentRanges(
                wavFile = wavFile,
                header = header,
                currentSpeechGate = currentSpeechGate,
            )
        } catch (t: Throwable) {
            Log.w(TAG, "natural utterance VAD failed, fallback to fixed chunks: ${t.message ?: t::class.java.simpleName}")
            val fallbackRanges = buildChunkedSegmentRanges(header.durationMs)
            SegmentationResult(
                mode = "fixed-${CHUNKED_SEGMENT_MS / 1000}s-fallback-overlap-${formatSeconds(CHUNKED_OVERLAP_MS)}s",
                fallbackToFixedSegments = true,
                rawSegmentCount = fallbackRanges.size,
                mergedSegmentCount = fallbackRanges.size,
                segments = fallbackRanges,
            )
        }
        val ranges = segmentation.segments
        if (jobId != null) {
            val checkpoint = TranscriptionCheckpointStore.load(this, jobId)
            TranscriptionJobRepository.update(this, jobId) {
                it.copy(
                    durationMs = header.durationMs,
                    totalChunks = ranges.size,
                    nextChunkIndex = checkpoint?.nextRangeIndex ?: 0,
                    progressPercent = if (ranges.isEmpty()) 0 else
                        ((checkpoint?.nextRangeIndex ?: 0) * 100 / ranges.size).coerceIn(0, 99),
                    status = TranscriptionJobStatus.TRANSCRIBING,
                )
            }
        }
        val chunkResult = recognizeChunkedWav(
            currentRecognizer = currentRecognizer,
            currentSpeechGate = currentSpeechGate,
            wavFile = wavFile,
            header = header,
            ranges = ranges,
            jobId = jobId,
        )
        val rawSegments = chunkResult.segments
        val segments = dedupRecognitionSegments(rawSegments)
        val timestampedSentences = dedupTimestampedSentences(
            buildTimestampedSentencesFromSegments(segments)
        )
        val mergedText = buildMergedText(segments)
        val recordingTimeContext = recordingTimeContextForSource(sourceFileName ?: wavFile.name)
        val transcriptSegments = buildTimestampedTranscriptSegments(
            segments = segments,
            recordingTimeContext = recordingTimeContext,
        )
        Log.i(
            TAG,
            "timestamped transcript built sentenceCount=${timestampedSentences.size}, " +
                "absoluteSegmentCount=${transcriptSegments.size}, " +
                "rawSegments=${rawSegments.size}, dedupSegments=${segments.size}, " +
                "speechChunks=${chunkResult.stats.speechChunks}, skippedChunks=${chunkResult.stats.skippedChunks}"
        )

        return RecognitionOutput(
            text = if (chunkResult.stats.hasAnySpeech) {
                TimestampedTranscriptWriter.buildDisplayText(recordingTimeContext, transcriptSegments)
            } else {
                NO_SPEECH_MESSAGE
            },
            durationSec = durationSec,
            segments = segments,
            timestampedSentences = timestampedSentences,
            transcriptSegments = transcriptSegments,
            mergedText = mergedText,
            segmentationMode = segmentation.mode,
            fallbackToFixedSegments = segmentation.fallbackToFixedSegments,
            rawSegmentCount = segmentation.rawSegmentCount,
            mergedSegmentCount = segmentation.mergedSegmentCount,
            sampleRate = header.sampleRate,
            channels = header.channels,
            bitsPerSample = header.bitsPerSample,
            stats = chunkResult.stats,
            recordingTimeContext = recordingTimeContext,
        )
    }

    private fun parseWavHeader(wavFile: File): WavHeader {
        RandomAccessFile(wavFile, "r").use { input ->
            if (input.length() < 44L) {
                error("WAV 文件太小，无法读取 header")
            }

            val riff = input.readFourCc()
            if (riff != "RIFF") {
                error("不是标准 WAV 文件：缺少 RIFF 标记")
            }

            input.readUInt32Le()
            val wave = input.readFourCc()
            if (wave != "WAVE") {
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
                        if (chunkSize < 16L) {
                            error("WAV fmt chunk 异常：长度不足")
                        }
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
                if (audioFormat != null && dataStart != null) {
                    break
                }
            }

            val format = audioFormat ?: error("WAV 文件缺少 fmt chunk")
            val channelCount = channels ?: error("WAV 文件缺少声道信息")
            val rate = sampleRate ?: error("WAV 文件缺少采样率信息")
            val bytesPerSecond = byteRate ?: error("WAV 文件缺少 byteRate 信息")
            val align = blockAlign ?: error("WAV 文件缺少 blockAlign 信息")
            val bitDepth = bitsPerSample ?: error("WAV 文件缺少位深信息")
            val pcmStart = dataStart ?: error("WAV 文件缺少 data chunk")
            val pcmSize = dataSize ?: error("WAV 文件 data chunk 为空")

            if (format != 1) {
                error("当前只支持标准 PCM WAV，不支持压缩 WAV 或浮点 WAV。audioFormat=$format")
            }
            if (channelCount <= 0) {
                error("WAV 声道数无效：$channelCount")
            }
            if (rate <= 0) {
                error("WAV 采样率无效：$rate")
            }
            if (bitDepth !in listOf(8, 16, 24, 32)) {
                error("当前只支持 8/16/24/32-bit PCM WAV，当前位深：$bitDepth")
            }

            val expectedAlign = channelCount * (bitDepth / 8)
            if (align != expectedAlign) {
                error("WAV blockAlign 异常：当前 $align，预期 $expectedAlign")
            }
            if (pcmSize <= 0L) {
                error("WAV data chunk 为空")
            }

            val header = WavHeader(
                audioFormat = format,
                channels = channelCount,
                sampleRate = rate,
                byteRate = bytesPerSecond,
                blockAlign = align,
                bitsPerSample = bitDepth,
                dataStart = pcmStart,
                dataSize = pcmSize,
            )
            Log.i(
                TAG,
                "wav header ok audioFormat=${header.audioFormat}, channels=${header.channels}, " +
                    "sampleRate=${header.sampleRate}, byteRate=${header.byteRate}, " +
                    "blockAlign=${header.blockAlign}, bits=${header.bitsPerSample}, " +
                    "dataStart=${header.dataStart}, dataSize=${header.dataSize}, " +
                    "durationMs=${header.durationMs}"
            )
            return header
        }
    }

    private fun buildNaturalUtteranceSegmentRanges(
        wavFile: File,
        header: WavHeader,
        currentSpeechGate: SpeechGate,
    ): SegmentationResult {
        if (header.durationMs <= 0L) {
            return SegmentationResult(
                mode = "natural-utterance-vad-empty",
                fallbackToFixedSegments = false,
                rawSegmentCount = 0,
                mergedSegmentCount = 0,
                segments = emptyList(),
            )
        }

        val scanStepMs = (NATURAL_VAD_SCAN_WINDOW_MS - NATURAL_VAD_SCAN_OVERLAP_MS)
            .coerceAtLeast(1L)
        val rawSpeechSegments = mutableListOf<SpeechSegment>()
        RandomAccessFile(wavFile, "r").use { input ->
            var reusableBytes = ByteArray(0)
            var scanStartMs = 0L
            while (scanStartMs < header.durationMs) {
                val scanEndMs = min(scanStartMs + NATURAL_VAD_SCAN_WINDOW_MS, header.durationMs)
                val (nextReusableBytes, scanAudio) = readPcmChunkAsMonoFloat(
                    input = input,
                    header = header,
                    range = SegmentRange(
                        startMs = scanStartMs,
                        endMs = scanEndMs,
                        cutMode = "natural-vad-scan",
                    ),
                    reusableBytes = reusableBytes,
                )
                reusableBytes = nextReusableBytes
                if (scanAudio.samples.isNotEmpty()) {
                    val vadResult = currentSpeechGate.analyze(
                        chunkStartMs = scanAudio.startMs,
                        samples = scanAudio.samples,
                        sampleRate = header.sampleRate,
                    )
                    rawSpeechSegments += vadResult.rawSpeechSegments
                }
                if (scanEndMs >= header.durationMs) break
                scanStartMs += scanStepMs
            }
        }

        val utterances = NaturalUtteranceSegmenter.assemble(
            rawSegments = rawSpeechSegments,
            audioDurationMs = header.durationMs,
            config = NaturalUtteranceConfig(
                mergeGapMs = NATURAL_UTTERANCE_MERGE_GAP_MS,
                speechPaddingMs = NATURAL_UTTERANCE_PADDING_MS,
                minSpeechMs = NATURAL_UTTERANCE_MIN_SPEECH_MS,
            ),
        )
        val ranges = utterances.mapIndexed { index, utterance ->
            SegmentRange(
                startMs = utterance.startMs,
                endMs = utterance.endMs,
                index = index + 1,
                cutMode = "natural-vad-utterance",
                targetEndMs = utterance.endMs,
                adjustedEndMs = utterance.endMs,
            )
        }

        Log.i(
            TAG,
            "natural utterance ranges built raw=${rawSpeechSegments.size}, " +
                "utterances=${ranges.size}, scanWindowMs=$NATURAL_VAD_SCAN_WINDOW_MS, " +
                "scanOverlapMs=$NATURAL_VAD_SCAN_OVERLAP_MS, " +
                "mergeGapMs=$NATURAL_UTTERANCE_MERGE_GAP_MS, paddingMs=$NATURAL_UTTERANCE_PADDING_MS"
        )
        ranges.forEach { range ->
            Log.i(TAG, "natural utterance index=${range.index} startMs=${range.startMs} endMs=${range.endMs}")
        }

        return SegmentationResult(
            mode = "natural-utterance-vad-gap-${NATURAL_UTTERANCE_MERGE_GAP_MS}ms-pad-${NATURAL_UTTERANCE_PADDING_MS}ms",
            fallbackToFixedSegments = false,
            rawSegmentCount = rawSpeechSegments.size,
            mergedSegmentCount = ranges.size,
            segments = ranges,
        )
    }

    private fun buildVadAdjustedSegmentRanges(
        wavFile: File,
        header: WavHeader,
        chunkMs: Long = CHUNKED_SEGMENT_MS,
        overlapMs: Long = CHUNKED_OVERLAP_MS,
        vadSearchRadiusMs: Long = VAD_BOUNDARY_SEARCH_RADIUS_MS,
        vadMinSilenceMs: Long = VAD_BOUNDARY_MIN_SILENCE_MS,
    ): List<SegmentRange> {
        if (header.durationMs <= 0L) return emptyList()

        Log.i(
            TAG,
            "vad adjusted ranges building durationMs=${header.durationMs}, chunkMs=$chunkMs, " +
                "overlapMs=$overlapMs, searchRadiusMs=$vadSearchRadiusMs, minSilenceMs=$vadMinSilenceMs"
        )

        val ranges = mutableListOf<SegmentRange>()
        var startMs = 0L
        var index = 1
        val stepFallbackMs = (chunkMs - overlapMs).coerceAtLeast(1L)

        while (startMs < header.durationMs) {
            val targetEndMs = min(startMs + chunkMs, header.durationMs)
            if (targetEndMs >= header.durationMs) {
                ranges += SegmentRange(
                    startMs = startMs,
                    endMs = header.durationMs,
                    index = index,
                    cutMode = "final",
                    targetEndMs = targetEndMs,
                    adjustedEndMs = header.durationMs,
                )
                break
            }

            val adjustedEndMs = findBestSilenceCutNear(
                wavFile = wavFile,
                header = header,
                targetMs = targetEndMs,
                searchRadiusMs = vadSearchRadiusMs,
                minSilenceMs = vadMinSilenceMs,
            )
            val useVadCut = adjustedEndMs != null && adjustedEndMs > startMs + 5_000L
            val endMs = if (useVadCut) adjustedEndMs!! else targetEndMs
            val cutMode = if (useVadCut) "vad-silence" else "fixed-fallback"

            if (useVadCut) {
                Log.i(
                    TAG,
                    "vad silence cut found target=${formatSec(targetEndMs)} adjusted=${formatSec(endMs)}"
                )
            } else {
                Log.i(
                    TAG,
                    "vad no silence found target=${formatSec(targetEndMs)} fallback fixed"
                )
            }

            ranges += SegmentRange(
                startMs = startMs,
                endMs = endMs,
                index = index,
                cutMode = cutMode,
                targetEndMs = targetEndMs,
                adjustedEndMs = endMs,
            )

            val nextStartMs = max(0L, endMs - overlapMs)
            startMs = if (nextStartMs <= startMs + 500L) {
                startMs + stepFallbackMs
            } else {
                nextStartMs
            }
            index += 1
        }

        Log.i(TAG, "vad adjusted ranges built count=${ranges.size}")
        return ranges
    }

    private fun findBestSilenceCutNear(
        wavFile: File,
        header: WavHeader,
        targetMs: Long,
        searchRadiusMs: Long,
        minSilenceMs: Long,
    ): Long? {
        val probeStartMs = max(0L, targetMs - searchRadiusMs)
        val probeEndMs = min(header.durationMs, targetMs + searchRadiusMs)
        if (probeEndMs <= probeStartMs) return null

        Log.i(
            TAG,
            "vad probe target=${formatSec(targetMs)} probe=${formatSec(probeStartMs)}-${formatSec(probeEndMs)}"
        )

        val chunkAudio = RandomAccessFile(wavFile, "r").use { input ->
            readPcmChunkAsMonoFloat(
                input = input,
                header = header,
                range = SegmentRange(
                    startMs = probeStartMs,
                    endMs = probeEndMs,
                    cutMode = "vad-probe",
                ),
                reusableBytes = ByteArray(0),
            ).second
        }
        if (chunkAudio.samples.isEmpty()) return null

        val frameSize = max(1, ((header.sampleRate * VAD_BOUNDARY_FRAME_MS) / 1000L).toInt())
        val frameDbList = calculateFrameDbList(chunkAudio.samples, frameSize)
        if (frameDbList.isEmpty()) return null

        val sortedDb = frameDbList.sorted()
        val noiseFloorDb = percentile(sortedDb, 0.20)
        val thresholdDb = (noiseFloorDb + VAD_BOUNDARY_NOISE_MARGIN_DB).coerceIn(-55.0, -35.0)
        Log.i(
            TAG,
            "vad threshold db=${"%.1f".format(Locale.US, thresholdDb)} " +
                "noiseFloor=${"%.1f".format(Locale.US, noiseFloorDb)}"
        )

        val candidates = findSilenceCandidates(
            frameDbList = frameDbList,
            thresholdDb = thresholdDb,
            frameMs = VAD_BOUNDARY_FRAME_MS,
            minSilenceMs = minSilenceMs,
            probeStartMs = probeStartMs,
            probeEndMs = probeEndMs,
        )
        val best = candidates.minByOrNull { abs(it.centerMs - targetMs) }
        if (best == null) {
            Log.i(TAG, "vad no silence candidate target=${formatSec(targetMs)}")
            return null
        }

        Log.i(
            TAG,
            "vad silence candidate target=${formatSec(targetMs)} " +
                "start=${formatSec(best.startMs)} end=${formatSec(best.endMs)} " +
                "center=${formatSec(best.centerMs)} durationMs=${best.durationMs}"
        )
        return best.centerMs
    }

    private fun calculateFrameDbList(samples: FloatArray, frameSize: Int): List<Double> {
        val frameDbList = mutableListOf<Double>()
        var start = 0

        while (start < samples.size) {
            val end = min(samples.size, start + frameSize)
            var sumSquares = 0.0
            for (i in start until end) {
                val value = samples[i].toDouble()
                sumSquares += value * value
            }
            val count = max(1, end - start)
            val rms = sqrt(sumSquares / count.toDouble())
            frameDbList += 20.0 * log10(rms + 1e-9)
            start += frameSize
        }

        return frameDbList
    }

    private fun findSilenceCandidates(
        frameDbList: List<Double>,
        thresholdDb: Double,
        frameMs: Long,
        minSilenceMs: Long,
        probeStartMs: Long,
        probeEndMs: Long,
    ): List<SilenceCandidate> {
        val candidates = mutableListOf<SilenceCandidate>()
        var silenceStartFrame: Int? = null

        for (frameIndex in frameDbList.indices) {
            val isSilent = frameDbList[frameIndex] < thresholdDb
            if (isSilent && silenceStartFrame == null) {
                silenceStartFrame = frameIndex
            }

            val reachedEnd = frameIndex == frameDbList.lastIndex
            if ((!isSilent || reachedEnd) && silenceStartFrame != null) {
                val startFrame = silenceStartFrame
                val endFrameExclusive = if (isSilent && reachedEnd) frameIndex + 1 else frameIndex
                val startMs = probeStartMs + startFrame.toLong() * frameMs
                val endMs = min(probeStartMs + endFrameExclusive.toLong() * frameMs, probeEndMs)
                if (endMs - startMs >= minSilenceMs) {
                    candidates += SilenceCandidate(startMs = startMs, endMs = endMs)
                }
                silenceStartFrame = null
            }
        }

        return candidates
    }

    private fun buildChunkedSegmentRanges(durationMs: Long): List<SegmentRange> {
        if (durationMs <= 0L) return emptyList()

        val stepMs = (CHUNKED_SEGMENT_MS - CHUNKED_OVERLAP_MS).coerceAtLeast(1L)
        val ranges = mutableListOf<SegmentRange>()
        var startMs = 0L
        var index = 1

        while (startMs < durationMs) {
            val endMs = min(startMs + CHUNKED_SEGMENT_MS, durationMs)
            ranges += SegmentRange(
                startMs = startMs,
                endMs = endMs,
                index = index,
                cutMode = if (endMs >= durationMs) "final" else "fixed-fallback",
                targetEndMs = endMs,
                adjustedEndMs = endMs,
            )
            if (endMs >= durationMs) break
            startMs += stepMs
            index += 1
        }

        Log.i(
            TAG,
            "chunked ranges built durationMs=$durationMs, segmentMs=$CHUNKED_SEGMENT_MS, " +
                "overlapMs=$CHUNKED_OVERLAP_MS, stepMs=$stepMs, count=${ranges.size}"
        )
        return ranges
    }

    private fun recognizeChunkedWav(
        currentRecognizer: OfflineRecognizer,
        currentSpeechGate: SpeechGate,
        wavFile: File,
        header: WavHeader,
        ranges: List<SegmentRange>,
        jobId: String? = null,
    ): ChunkRecognitionResult {
        val checkpoint = jobId?.let { TranscriptionCheckpointStore.load(this, it) }
        val segments = checkpoint?.segments?.toMutableList() ?: mutableListOf()
        var speechDurationMs = checkpoint?.speechDurationMs ?: 0L
        var skippedNoSpeechDurationMs = checkpoint?.skippedNoSpeechDurationMs ?: 0L
        var speechChunks = checkpoint?.speechChunks ?: 0
        var skippedChunks = checkpoint?.skippedChunks ?: 0
        var asrProcessedDurationMs = checkpoint?.asrProcessedDurationMs ?: 0L
        val resumeAt = (checkpoint?.nextRangeIndex ?: 0).coerceIn(0, ranges.size)

        Log.i(TAG, "chunk recognition resumeAt=" + resumeAt + " total=" + ranges.size +
            " restoredSegments=" + segments.size)

        RandomAccessFile(wavFile, "r").use { input ->
            var reusableBytes = ByteArray(0)
            for ((position, range) in ranges.withIndex()) {
                if (position < resumeAt) continue
                if (TranscriptionForegroundService.isCancellationRequested(jobId)) {
                    throw TranscriptionCanceledException()
                }

                val (nextReusableBytes, chunkAudio) = readPcmChunkAsMonoFloat(
                    input = input,
                    header = header,
                    range = range,
                    reusableBytes = reusableBytes,
                )
                reusableBytes = nextReusableBytes

                if (chunkAudio.samples.isEmpty()) {
                    Log.w(TAG, "skip empty chunk index=" + (position + 1))
                    persistJobCheckpoint(
                        jobId, position, ranges.size, header.durationMs, range.endMs,
                        speechDurationMs, skippedNoSpeechDurationMs, speechChunks,
                        skippedChunks, asrProcessedDurationMs, null,
                    )
                    continue
                }

                val chunkDurationMs = chunkAudio.endMs - chunkAudio.startMs
                val vadResult = currentSpeechGate.analyze(
                    chunkStartMs = chunkAudio.startMs,
                    samples = chunkAudio.samples,
                    sampleRate = header.sampleRate,
                )

                if (!vadResult.hasSpeech) {
                    skippedChunks += 1
                    skippedNoSpeechDurationMs += chunkDurationMs
                    Log.i(TAG, "skip no-speech chunk index=" + (position + 1) +
                        " startMs=" + chunkAudio.startMs + " endMs=" + chunkAudio.endMs)
                    persistJobCheckpoint(
                        jobId, position, ranges.size, header.durationMs, chunkAudio.endMs,
                        speechDurationMs, skippedNoSpeechDurationMs, speechChunks,
                        skippedChunks, asrProcessedDurationMs, null,
                    )
                    if (!isDestroyed) runOnUiThread {
                        setStatus("正在识别... 已跳过无人声 " + formatTxtTimestamp(chunkAudio.endMs) +
                            " / " + formatTxtTimestamp(header.durationMs))
                    }
                    continue
                }

                speechChunks += 1
                speechDurationMs += vadResult.totalSpeechMs
                asrProcessedDurationMs += chunkDurationMs
                Log.i(TAG, "recognize chunk index=" + (position + 1) +
                    " startMs=" + chunkAudio.startMs + " endMs=" + chunkAudio.endMs +
                    " durationMs=" + chunkDurationMs + " mode=" + range.cutMode +
                    " sampleCount=" + chunkAudio.samples.size +
                    " vadSpeechMs=" + vadResult.totalSpeechMs +
                    " usedHeap=" + currentUsedHeapBytes())

                val rawSegmentText = recognizeSamples(
                    currentRecognizer = currentRecognizer,
                    samples = chunkAudio.samples,
                    sampleRate = header.sampleRate,
                    chunkIndex = range.index.takeIf { it > 0 } ?: position + 1,
                    chunkDurationMs = chunkDurationMs,
                    logNativeTimestampDetail = position == resumeAt,
                )
                val segmentText = formatNaturalUtteranceText(rawSegmentText)
                val segment = RecognitionSegment(
                    index = segments.size + 1,
                    startMs = chunkAudio.startMs,
                    endMs = chunkAudio.endMs,
                    text = segmentText,
                    cutMode = range.cutMode + "+speechgate-full-chunk",
                    startUs = chunkAudio.startUs,
                    endUs = chunkAudio.endUs,
                )
                segments += segment
                persistJobCheckpoint(
                    jobId, position, ranges.size, header.durationMs, chunkAudio.endMs,
                    speechDurationMs, skippedNoSpeechDurationMs, speechChunks,
                    skippedChunks, asrProcessedDurationMs, segment,
                )

                Log.i(TAG, "transcript segment index=" + segment.index +
                    " relativeStartUs=" + chunkAudio.startUs +
                    " relativeEndUs=" + chunkAudio.endUs +
                    " textLength=" + segmentText.length)
                if (!isDestroyed) runOnUiThread {
                    setStatus("正在识别... 已处理 " + formatTxtTimestamp(chunkAudio.endMs) +
                        " / " + formatTxtTimestamp(header.durationMs))
                }
            }
        }

        val stats = TranscriptionStats(
            totalDurationMs = header.durationMs,
            speechDurationMs = speechDurationMs,
            skippedNoSpeechDurationMs = skippedNoSpeechDurationMs,
            totalChunks = ranges.size,
            speechChunks = speechChunks,
            skippedChunks = skippedChunks,
            asrProcessedDurationMs = asrProcessedDurationMs,
        )
        Log.i(TAG, "speech gate summary totalMs=" + stats.totalDurationMs +
            " speechMs=" + stats.speechDurationMs +
            " skippedNoSpeechMs=" + stats.skippedNoSpeechDurationMs +
            " totalChunks=" + stats.totalChunks +
            " speechChunks=" + stats.speechChunks +
            " skippedChunks=" + stats.skippedChunks)
        return ChunkRecognitionResult(segments = segments, stats = stats)
    }

    private fun persistJobCheckpoint(
        jobId: String?,
        rangePosition: Int,
        totalRanges: Int,
        totalDurationMs: Long,
        processedEndMs: Long,
        speechDurationMs: Long,
        skippedNoSpeechDurationMs: Long,
        speechChunks: Int,
        skippedChunks: Int,
        asrProcessedDurationMs: Long,
        segment: RecognitionSegment?,
    ) {
        if (jobId == null) return
        val nextRange = rangePosition + 1
        val progress = if (totalRanges <= 0) 0 else
            ((nextRange.toLong() * 100L) / totalRanges.toLong()).toInt().coerceIn(0, 99)
        TranscriptionCheckpointStore.saveAfterChunk(
            this, jobId, rangePosition, nextRange,
            speechDurationMs, skippedNoSpeechDurationMs, speechChunks,
            skippedChunks, asrProcessedDurationMs, segment,
        )
        TranscriptionJobRepository.update(this, jobId) {
            it.copy(
                durationMs = totalDurationMs,
                totalChunks = totalRanges,
                nextChunkIndex = nextRange,
                progressPercent = progress,
                status = TranscriptionJobStatus.TRANSCRIBING,
                errorMessage = null,
            )
        }
        TranscriptionForegroundService.progress(
            this,
            jobId,
            "LightASR 正在分析",
            formatTxtTimestamp(processedEndMs) + " / " + formatTxtTimestamp(totalDurationMs),
            progress,
        )
    }

    private fun readPcmChunkAsMonoFloat(
        input: RandomAccessFile,
        header: WavHeader,
        range: SegmentRange,
        reusableBytes: ByteArray,
    ): Pair<ByteArray, ChunkedAudio> {
        val startFrame = msToFrameIndex(range.startMs, header.sampleRate, header.totalFrames)
        val endFrame = msToFrameIndex(range.endMs, header.sampleRate, header.totalFrames)
            .coerceAtLeast(startFrame)
        val frameCount = (endFrame - startFrame).coerceAtLeast(0L)
        if (frameCount == 0L) {
            return reusableBytes to ChunkedAudio(range.startMs, range.endMs, FloatArray(0))
        }

        val byteCountLong = frameCount * header.blockAlign.toLong()
        if (byteCountLong > Int.MAX_VALUE) {
            error("单个分块过大：$byteCountLong bytes，请缩短分块时长")
        }
        val byteCount = byteCountLong.toInt()
        val buffer = if (reusableBytes.size >= byteCount) reusableBytes else ByteArray(byteCount)
        val dataOffset = header.dataStart + startFrame * header.blockAlign.toLong()

        input.seek(dataOffset)
        input.readFully(buffer, 0, byteCount)

        val samples = FloatArray(frameCount.toInt())
        decodePcmToMonoFloat(
            bytes = buffer,
            byteCount = byteCount,
            header = header,
            out = samples,
        )

        return buffer to ChunkedAudio(
            startMs = framesToMs(startFrame, header.sampleRate),
            endMs = framesToMs(endFrame, header.sampleRate),
            samples = samples,
            startUs = framesToUs(startFrame, header.sampleRate),
            endUs = framesToUs(endFrame, header.sampleRate),
        )
    }

    private fun extractSpeechRegionSamples(
        chunkAudio: ChunkedAudio,
        region: SpeechRegion,
        sampleRate: Int,
    ): FloatArray {
        if (sampleRate <= 0 || chunkAudio.samples.isEmpty()) return FloatArray(0)

        val startIndex = region.startSample
            .coerceIn(0L, chunkAudio.samples.size.toLong())
            .toInt()
        val endIndex = region.endSample
            .coerceIn(startIndex.toLong(), chunkAudio.samples.size.toLong())
            .toInt()
        val length = endIndex - startIndex
        if (length <= 0) return FloatArray(0)

        val out = FloatArray(length)
        chunkAudio.samples.copyInto(
            destination = out,
            destinationOffset = 0,
            startIndex = startIndex,
            endIndex = endIndex,
        )
        return out
    }

    private fun decodePcmToMonoFloat(
        bytes: ByteArray,
        byteCount: Int,
        header: WavHeader,
        out: FloatArray,
    ) {
        val bytesPerSample = header.bytesPerSample
        val frameCount = byteCount / header.blockAlign

        for (frameIndex in 0 until frameCount) {
            val frameOffset = frameIndex * header.blockAlign
            var sum = 0.0

            for (channel in 0 until header.channels) {
                val sampleOffset = frameOffset + channel * bytesPerSample
                sum += decodePcmSample(bytes, sampleOffset, header.bitsPerSample).toDouble()
            }

            out[frameIndex] = (sum / header.channels.toDouble()).toFloat()
        }
    }

    private fun decodePcmSample(bytes: ByteArray, offset: Int, bitsPerSample: Int): Float {
        return when (bitsPerSample) {
            8 -> {
                val value = bytes[offset].toInt() and 0xFF
                ((value - 128).toFloat() / 128.0f).coerceIn(-1.0f, 1.0f)
            }
            16 -> {
                val value = ((bytes[offset].toInt() and 0xFF) or (bytes[offset + 1].toInt() shl 8)).toShort()
                (value.toFloat() / 32768.0f).coerceIn(-1.0f, 1.0f)
            }
            24 -> {
                var value = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16)
                if ((value and 0x800000) != 0) {
                    value = value or -0x1000000
                }
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

    private fun segmentSamplesForRecognition(
        samples: FloatArray,
        sampleRate: Int,
        durationMs: Long,
    ): SegmentationResult {
        val fixedSegments = buildFixedSegmentRanges(durationMs)
        Log.i(
            TAG,
            "segmentation mode=fixed-${FIXED_SEGMENT_SECONDS}s, sampleRate=$sampleRate, " +
                "sampleCount=${samples.size}, durationMs=$durationMs, segmentCount=${fixedSegments.size}"
        )

        return SegmentationResult(
            mode = "fixed-${"%.0f".format(Locale.US, FIXED_SEGMENT_SECONDS)}s",
            fallbackToFixedSegments = false,
            rawSegmentCount = 0,
            mergedSegmentCount = fixedSegments.size,
            segments = fixedSegments,
        )
    }

    private fun detectEnergyVadSegments(
        samples: FloatArray,
        sampleRate: Int,
        durationMs: Long,
        config: EnergyVadConfig,
    ): SegmentationResult {
        if (samples.isEmpty()) {
            Log.w(TAG, "energy VAD skipped because samples are empty")
            return SegmentationResult(
                mode = "VAD",
                fallbackToFixedSegments = false,
                rawSegmentCount = 0,
                mergedSegmentCount = 0,
                segments = emptyList(),
            )
        }

        val frameSize = max(1, ((sampleRate * config.frameMs) / 1000L).toInt())
        val hopSize = max(1, ((sampleRate * config.hopMs) / 1000L).toInt())
        val energies = calculateFrameEnergies(samples, frameSize, hopSize)
        val threshold = calculateEnergyThreshold(energies)
        val flags = energies.map { it >= threshold }

        Log.i(
            TAG,
            "energy VAD params frameMs=${config.frameMs}, hopMs=${config.hopMs}, " +
                "minSpeechMs=${config.minSpeechMs}, minSilenceMs=${config.minSilenceMs}, " +
                "paddingMs=${config.speechPaddingMs}, mergeGapMs=${config.mergeGapMs}, " +
                "minSegmentMs=${config.minSegmentMs}, maxSegmentMs=${config.maxSegmentMs}, " +
                "frameSize=$frameSize, hopSize=$hopSize, frameCount=${energies.size}, " +
                "threshold=${"%.6f".format(Locale.US, threshold)}"
        )

        val rawSegments = detectRawSpeechRanges(
            speechFlags = flags,
            sampleRate = sampleRate,
            frameSize = frameSize,
            hopSize = hopSize,
            durationMs = durationMs,
            config = config,
        )
        val paddedSegments = rawSegments
            .map { range ->
                SegmentRange(
                    startMs = (range.startMs - config.speechPaddingMs).coerceAtLeast(0L),
                    endMs = (range.endMs + config.speechPaddingMs).coerceAtMost(durationMs),
                )
            }
            .filter { it.endMs - it.startMs >= config.minSegmentMs }
        val mergedSegments = mergeCloseSegments(paddedSegments, config.mergeGapMs)
        val splitSegments = splitLongSegments(
            segments = mergedSegments,
            maxSegmentMs = config.maxSegmentMs,
            minSegmentMs = config.minSegmentMs,
        )

        Log.i(
            TAG,
            "energy VAD counts raw=${rawSegments.size}, afterPadding=${paddedSegments.size}, " +
                "merged=${mergedSegments.size}, split=${splitSegments.size}"
        )
        splitSegments.forEachIndexed { index, segment ->
            Log.i(
                TAG,
                "energy VAD segment index=${index + 1}, startMs=${segment.startMs}, " +
                    "endMs=${segment.endMs}, durationMs=${segment.endMs - segment.startMs}"
            )
        }

        return SegmentationResult(
            mode = "VAD",
            fallbackToFixedSegments = false,
            rawSegmentCount = rawSegments.size,
            mergedSegmentCount = mergedSegments.size,
            segments = splitSegments,
        )
    }

    private fun calculateFrameEnergies(
        samples: FloatArray,
        frameSize: Int,
        hopSize: Int,
    ): List<Double> {
        val energies = mutableListOf<Double>()
        var start = 0

        while (start < samples.size) {
            val end = min(samples.size, start + frameSize)
            var sumSquares = 0.0
            for (i in start until end) {
                val value = samples[i].toDouble()
                sumSquares += value * value
            }
            val count = max(1, end - start)
            energies += sqrt(sumSquares / count.toDouble())
            start += hopSize
        }

        return energies
    }

    private fun calculateEnergyThreshold(energies: List<Double>): Double {
        if (energies.isEmpty()) return Double.MAX_VALUE

        val sorted = energies.sorted()
        val low = percentile(sorted, 0.10)
        val median = percentile(sorted, 0.50)
        val high = percentile(sorted, 0.90)
        val peak = sorted.last()
        val dynamicThreshold = low + (high - low).coerceAtLeast(0.0) * 0.35
        val threshold = maxOf(
            dynamicThreshold,
            median * 0.55,
            peak * 0.08,
            0.003,
        )

        Log.i(
            TAG,
            "energy stats low=${"%.6f".format(Locale.US, low)}, " +
                "median=${"%.6f".format(Locale.US, median)}, high=${"%.6f".format(Locale.US, high)}, " +
                "peak=${"%.6f".format(Locale.US, peak)}, threshold=${"%.6f".format(Locale.US, threshold)}"
        )

        return threshold
    }

    private fun percentile(sortedValues: List<Double>, percentile: Double): Double {
        if (sortedValues.isEmpty()) return 0.0
        val index = ((sortedValues.size - 1) * percentile)
            .roundToInt()
            .coerceIn(0, sortedValues.lastIndex)
        return sortedValues[index]
    }

    private fun detectRawSpeechRanges(
        speechFlags: List<Boolean>,
        sampleRate: Int,
        frameSize: Int,
        hopSize: Int,
        durationMs: Long,
        config: EnergyVadConfig,
    ): List<SegmentRange> {
        val minSpeechFrames = framesForDuration(config.minSpeechMs, config.hopMs)
        val minSilenceFrames = framesForDuration(config.minSilenceMs, config.hopMs)
        val ranges = mutableListOf<SegmentRange>()
        var inSpeech = false
        var speechRun = 0
        var silenceRun = 0
        var speechStartFrame = 0

        for (frameIndex in speechFlags.indices) {
            val isSpeech = speechFlags[frameIndex]

            if (!inSpeech) {
                if (isSpeech) {
                    speechRun += 1
                    if (speechRun >= minSpeechFrames) {
                        inSpeech = true
                        speechStartFrame = frameIndex - speechRun + 1
                        silenceRun = 0
                    }
                } else {
                    speechRun = 0
                }
                continue
            }

            if (isSpeech) {
                silenceRun = 0
            } else {
                silenceRun += 1
                if (silenceRun >= minSilenceFrames) {
                    val firstSilenceFrame = frameIndex - silenceRun + 1
                    val lastSpeechFrame = (firstSilenceFrame - 1).coerceAtLeast(speechStartFrame)
                    ranges += SegmentRange(
                        startMs = frameStartMs(speechStartFrame, hopSize, sampleRate),
                        endMs = frameEndMs(lastSpeechFrame, frameSize, hopSize, sampleRate, durationMs),
                    )
                    inSpeech = false
                    speechRun = 0
                    silenceRun = 0
                }
            }
        }

        if (inSpeech) {
            ranges += SegmentRange(
                startMs = frameStartMs(speechStartFrame, hopSize, sampleRate),
                endMs = durationMs,
            )
        }

        return ranges
    }

    private fun mergeCloseSegments(
        segments: List<SegmentRange>,
        mergeGapMs: Long,
    ): List<SegmentRange> {
        if (segments.isEmpty()) return emptyList()

        val merged = mutableListOf<SegmentRange>()
        var current = segments.first()

        for (next in segments.drop(1)) {
            current = if (next.startMs - current.endMs <= mergeGapMs) {
                SegmentRange(current.startMs, max(current.endMs, next.endMs))
            } else {
                merged += current
                next
            }
        }
        merged += current

        return merged
    }

    private fun splitLongSegments(
        segments: List<SegmentRange>,
        maxSegmentMs: Long,
        minSegmentMs: Long,
    ): List<SegmentRange> {
        val split = mutableListOf<SegmentRange>()

        for (segment in segments) {
            var startMs = segment.startMs
            while (startMs < segment.endMs) {
                val endMs = min(startMs + maxSegmentMs, segment.endMs)
                if (endMs - startMs >= minSegmentMs) {
                    split += SegmentRange(startMs, endMs)
                }
                startMs = endMs
            }
        }

        return split
    }

    private fun recognizeSegmentRanges(
        currentRecognizer: OfflineRecognizer,
        samples: FloatArray,
        sampleRate: Int,
        ranges: List<SegmentRange>,
    ): List<RecognitionSegment> {
        val segments = mutableListOf<RecognitionSegment>()

        for ((position, range) in ranges.withIndex()) {
            val startSample = msToSampleIndex(range.startMs, sampleRate, samples.size)
            val endSample = msToSampleIndex(range.endMs, sampleRate, samples.size)
                .coerceAtLeast(startSample)
            if (endSample <= startSample) {
                Log.w(
                    TAG,
                    "skip empty segment index=${position + 1}, startMs=${range.startMs}, endMs=${range.endMs}"
                )
                continue
            }

            val segmentSamples = samples.copyOfRange(startSample, endSample)
            Log.i(
                TAG,
                "recognize segment index=${position + 1}, startMs=${range.startMs}, " +
                    "endMs=${range.endMs}, durationMs=${range.endMs - range.startMs}, " +
                    "sampleCount=${segmentSamples.size}"
            )
            val segmentText = recognizeSamples(currentRecognizer, segmentSamples, sampleRate)
            segments += RecognitionSegment(
                index = position + 1,
                startMs = range.startMs,
                endMs = range.endMs,
                text = segmentText,
            )
            Log.i(
                TAG,
                "segment result index=${position + 1}, textLength=${segmentText.length}, text=$segmentText"
            )
        }

        return segments
    }

    private fun buildFixedSegmentRanges(durationMs: Long): List<SegmentRange> {
        if (durationMs <= 0L) return emptyList()

        val chunkMs = (FIXED_SEGMENT_SECONDS * 1000.0).roundToLong().coerceAtLeast(1L)
        val segments = mutableListOf<SegmentRange>()
        var startMs = 0L

        while (startMs < durationMs) {
            val endMs = min(startMs + chunkMs, durationMs)
            segments += SegmentRange(startMs, endMs)
            startMs = endMs
        }

        return segments
    }

    private fun framesForDuration(durationMs: Long, hopMs: Long): Int {
        return ceil(durationMs.toDouble() / hopMs.toDouble()).toInt().coerceAtLeast(1)
    }

    private fun samplesToMs(sampleCount: Int, sampleRate: Int): Long {
        return ((sampleCount.toDouble() * 1000.0) / sampleRate.toDouble()).roundToLong()
    }

    private fun msToFrameIndex(ms: Long, sampleRate: Int, totalFrames: Long): Long {
        return ((ms.toDouble() * sampleRate.toDouble()) / 1000.0)
            .roundToLong()
            .coerceIn(0L, totalFrames)
    }

    private fun framesToMs(frameIndex: Long, sampleRate: Int): Long {
        return ((frameIndex.toDouble() * 1000.0) / sampleRate.toDouble())
            .roundToLong()
            .coerceAtLeast(0L)
    }

    private fun framesToUs(frameIndex: Long, sampleRate: Int): Long {
        if (sampleRate <= 0) return 0L
        return ((frameIndex.coerceAtLeast(0L) * 1_000_000L) + sampleRate.toLong() / 2L) /
            sampleRate.toLong()
    }

    private fun usToMs(us: Long): Long {
        return ((us.coerceAtLeast(0L) + 500L) / 1000L)
    }

    private fun frameStartMs(frameIndex: Int, hopSize: Int, sampleRate: Int): Long {
        val sampleIndex = frameIndex.toLong() * hopSize.toLong()
        return ((sampleIndex.toDouble() * 1000.0) / sampleRate.toDouble()).roundToLong()
    }

    private fun frameEndMs(
        frameIndex: Int,
        frameSize: Int,
        hopSize: Int,
        sampleRate: Int,
        durationMs: Long,
    ): Long {
        val sampleIndex = frameIndex.toLong() * hopSize.toLong() + frameSize.toLong()
        return ((sampleIndex.toDouble() * 1000.0) / sampleRate.toDouble())
            .roundToLong()
            .coerceAtMost(durationMs)
    }

    private fun msToSampleIndex(ms: Long, sampleRate: Int, sampleCount: Int): Int {
        return ((ms.toDouble() * sampleRate.toDouble()) / 1000.0)
            .roundToInt()
            .coerceIn(0, sampleCount)
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
        if (b0 < 0 || b1 < 0) {
            error("WAV header 读取失败：文件提前结束")
        }
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

    private fun currentUsedHeapBytes(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }

    private fun formatSeconds(ms: Long): String {
        return "%.1f".format(Locale.US, ms.toDouble() / 1000.0)
    }

    private fun formatSec(ms: Long): String {
        return "%.3f".format(Locale.US, ms.toDouble() / 1000.0)
    }

    private fun recognizeSamples(
        currentRecognizer: OfflineRecognizer,
        samples: FloatArray,
        sampleRate: Int,
        chunkIndex: Int? = null,
        chunkDurationMs: Long? = null,
        logNativeTimestampDetail: Boolean = false,
    ): String {
        val normalization = SafeAudioNormalizer.normalize(samples)
        Log.i(
            TAG,
            "audio normalization chunk=${chunkIndex ?: -1}, " +
                "inputRmsDb=${"%.2f".format(Locale.US, normalization.inputRmsDb)}, " +
                "outputRmsDb=${"%.2f".format(Locale.US, normalization.outputRmsDb)}, " +
                "inputPeak=${"%.4f".format(Locale.US, normalization.inputPeak)}, " +
                "outputPeak=${"%.4f".format(Locale.US, normalization.outputPeak)}, " +
                "gain=${"%.3f".format(Locale.US, normalization.gain)}, " +
                "dc=${"%.6f".format(Locale.US, normalization.removedDcOffset)}, " +
                "clipped=${normalization.clippedSamples}, applied=${normalization.applied}"
        )
        val stream = currentRecognizer.createStream()
        try {
            stream.acceptWaveform(normalization.samples, sampleRate)
            currentRecognizer.decode(stream)
            val result = currentRecognizer.getResult(stream)
            logNativeTimestampProbe(
                chunkIndex = chunkIndex,
                chunkDurationMs = chunkDurationMs,
                textLength = result.text.length,
                tokensSize = result.tokens.size,
                timestamps = result.timestamps,
                durations = result.durations,
                logDetail = logNativeTimestampDetail,
            )
            val rawText = result.text.trim()
            val correction = localSemanticCorrector.correct(rawText)
            if (correction.changed) {
                val rules = correction.appliedRules.joinToString(",") {
                    "${it.category}:${it.observed}->${it.canonical}(${it.occurrences})"
                }
                Log.i(
                    TAG,
                    "local semantic correction chunk=${chunkIndex ?: -1}, " +
                        "rules=$rules, raw=$rawText, corrected=${correction.correctedText}"
                )
            }
            return correction.correctedText
        } finally {
            stream.release()
        }
    }

    private fun logNativeTimestampProbe(
        chunkIndex: Int?,
        chunkDurationMs: Long?,
        textLength: Int,
        tokensSize: Int,
        timestamps: FloatArray,
        durations: FloatArray,
        logDetail: Boolean,
    ) {
        val prefix = if (chunkIndex != null) {
            "native timestamp probe chunk=$chunkIndex"
        } else {
            "native timestamp probe"
        }
        Log.i(
            TAG,
            "$prefix text length=$textLength, tokens size=$tokensSize, " +
                "timestamps size=${timestamps.size}, durations size=${durations.size}"
        )

        if (!logDetail) return

        val firstTimestamps = timestamps.take(5).joinToString(",") {
            "%.3f".format(Locale.US, it)
        }
        val firstDurations = durations.take(5).joinToString(",") {
            "%.3f".format(Locale.US, it)
        }
        val chunkDurationSec = chunkDurationMs?.toDouble()?.div(1000.0)
        val timestampsIncreasing = timestamps
            .asSequence()
            .zipWithNext()
            .all { (left, right) -> right >= left }
        val timestampsInRange = if (chunkDurationSec != null) {
            timestamps.all { it >= 0.0f && it <= chunkDurationSec.toFloat() + 0.5f }
        } else {
            false
        }
        val durationsPositive = durations.all { it > 0.0f }

        Log.i(
            TAG,
            "$prefix first timestamps=$firstTimestamps, first durations=$firstDurations, " +
                "timestampsIncreasing=$timestampsIncreasing, timestampsInChunkRange=$timestampsInRange, " +
                "durationsPositive=$durationsPositive"
        )
    }

    private fun buildSegmentDisplayText(segments: List<RecognitionSegment>): String {
        val visibleSegments = segments.filter { it.text.isNotBlank() }
        if (visibleSegments.isEmpty()) return ""
        return visibleSegments.joinToString("\n") { segment ->
            val text = segment.text.ifBlank { "（未识别出文本）" }
            "[${formatTxtTimestamp(segment.startMs)} - ${formatTxtTimestamp(segment.endMs)}] $text"
        }
    }

    private fun recordingTimeContextForSource(sourceFileName: String): RecordingTimeContext {
        parseRecordingStartEpochMsFromAirecFileName(sourceFileName)?.let { startEpochMs ->
            Log.i(
                TAG,
                "recording start time parsed from fileName=$sourceFileName, " +
                    "timezone=$DEFAULT_RECORDING_TIMEZONE_ID"
            )
            return RecordingTimeContext(
                recordingStartEpochMs = startEpochMs,
                timezoneId = DEFAULT_RECORDING_TIMEZONE_ID,
                source = "filename-yyyyMMddHHmmss",
            )
        }

        return defaultRecordingTimeContext()
    }

    private fun parseRecordingStartEpochMsFromAirecFileName(sourceFileName: String): Long? {
        val baseName = File(sourceFileName).name
        val match = Regex("""^(\d{14})\.wav$""", RegexOption.IGNORE_CASE)
            .matchEntire(baseName)
            ?: return null
        val formatter = SimpleDateFormat("yyyyMMddHHmmss", Locale.US).apply {
            isLenient = false
            timeZone = TimeZone.getTimeZone(DEFAULT_RECORDING_TIMEZONE_ID)
        }
        return try {
            formatter.parse(match.groupValues[1])?.time
        } catch (t: Throwable) {
            Log.w(TAG, "invalid recording start fileName=$sourceFileName", t)
            null
        }
    }

    private fun defaultRecordingTimeContext(): RecordingTimeContext {
        val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).apply {
            isLenient = false
            timeZone = TimeZone.getTimeZone(DEFAULT_RECORDING_TIMEZONE_ID)
        }
        val start = formatter.parse(DEFAULT_RECORDING_START_TEXT)
            ?: error("invalid default recording start time: $DEFAULT_RECORDING_START_TEXT")
        return RecordingTimeContext(
            recordingStartEpochMs = start.time,
            timezoneId = DEFAULT_RECORDING_TIMEZONE_ID,
            source = "fallback-default",
        )
    }

    private fun buildTimestampedTranscriptSegments(
        segments: List<RecognitionSegment>,
        recordingTimeContext: RecordingTimeContext,
    ): List<TimestampedTranscriptSegment> {
        val transcriptSegments = mutableListOf<TimestampedTranscriptSegment>()

        for (segment in segments) {
            if (segment.text.isBlank() || segment.endUs <= segment.startUs) continue
            val timestampSource = if (segment.cutMode.startsWith("natural-vad-utterance")) {
                TimestampSource.VAD_BOUNDARY
            } else {
                TimestampSource.ESTIMATED
            }

            val sentenceTexts = splitTranscriptSentences(segment.text)
                .filter { it.isNotBlank() }
            if (sentenceTexts.isEmpty()) continue

            if (sentenceTexts.size == 1) {
                transcriptSegments += buildTranscriptSegment(
                    relativeStartUs = segment.startUs,
                    relativeEndUs = segment.endUs,
                    text = sentenceTexts.first(),
                    timestampSource = timestampSource,
                    recordingTimeContext = recordingTimeContext,
                )
                continue
            }

            val lengths = sentenceTexts.map { effectiveNormalizedTextLength(it) }
            val totalLength = lengths.sum().coerceAtLeast(1)
            val segmentDurationUs = (segment.endUs - segment.startUs).coerceAtLeast(1L)
            var cursorUs = segment.startUs

            sentenceTexts.forEachIndexed { index, sentenceText ->
                val endUs = if (index == sentenceTexts.lastIndex) {
                    segment.endUs
                } else {
                    val estimatedDurationUs = segmentDurationUs * lengths[index].toLong() / totalLength.toLong()
                    (cursorUs + estimatedDurationUs)
                        .coerceAtLeast(cursorUs + 200_000L)
                        .coerceAtMost(segment.endUs)
                }
                if (endUs > cursorUs) {
                    transcriptSegments += buildTranscriptSegment(
                        relativeStartUs = cursorUs,
                        relativeEndUs = endUs,
                        text = sentenceText,
                        timestampSource = timestampSource,
                        recordingTimeContext = recordingTimeContext,
                    )
                }
                cursorUs = endUs
            }
        }

        return dedupTimestampedTranscriptSegments(transcriptSegments)
    }

    private fun buildTranscriptSegment(
        relativeStartUs: Long,
        relativeEndUs: Long,
        text: String,
        timestampSource: TimestampSource,
        recordingTimeContext: RecordingTimeContext,
    ): TimestampedTranscriptSegment {
        return TimestampedTranscriptSegment(
            relativeStartUs = relativeStartUs,
            relativeEndUs = relativeEndUs,
            absoluteStartEpochMs = recordingTimeContext.recordingStartEpochMs + relativeStartUs / 1000L,
            absoluteEndEpochMs = recordingTimeContext.recordingStartEpochMs + relativeEndUs / 1000L,
            speaker = "unknown",
            text = text.trim(),
            timestampSource = timestampSource,
        )
    }

    private fun dedupTimestampedTranscriptSegments(
        segments: List<TimestampedTranscriptSegment>,
    ): List<TimestampedTranscriptSegment> {
        if (segments.isEmpty()) return emptyList()
        val result = mutableListOf<TimestampedTranscriptSegment>()
        var removed = 0
        var dropped = 0

        for (segment in segments) {
            val rawText = segment.text.trim()
            if (rawText.isBlank()) continue

            val currentNorm = normalizeTranscriptText(rawText)
            val previous = result.lastOrNull()
            val overlapsPrevious = previous != null &&
                segment.relativeStartUs < previous.relativeEndUs
            if (previous != null && overlapsPrevious) {
                val previousNorm = normalizeTranscriptText(previous.text)
                when {
                    currentNorm.isEmpty() -> continue
                    previousNorm == currentNorm -> {
                        dropped += 1
                        Log.i(TAG, "dedup transcript row dropped exact text=$rawText")
                        continue
                    }
                    previousNorm.contains(currentNorm) && currentNorm.length >= DEDUP_MIN_OVERLAP_CHARS -> {
                        dropped += 1
                        Log.i(TAG, "dedup transcript row dropped contained text=$rawText")
                        continue
                    }
                    currentNorm.contains(previousNorm) &&
                        previousNorm.length >= DEDUP_MIN_OVERLAP_CHARS &&
                        previousNorm.length.toDouble() / currentNorm.length.toDouble() >= 0.55 -> {
                        result[result.lastIndex] = segment.copy(
                            relativeStartUs = previous.relativeStartUs,
                            absoluteStartEpochMs = previous.absoluteStartEpochMs,
                        )
                        removed += previous.text.length
                        Log.i(TAG, "dedup transcript row replaced previous with longer text=$rawText")
                        continue
                    }
                }
            }

            val cutIndex = if (previous != null && overlapsPrevious) {
                findNormalizedPrefixOverlapCutIndex(
                    left = previous.text,
                    right = rawText,
                    maxCheck = DEDUP_MAX_CHECK_CHARS,
                ).coerceIn(0, rawText.length)
            } else 0
            val cleanedText = trimLeadingDedupSeparators(rawText.substring(cutIndex))
            removed += rawText.length - cleanedText.length
            if (cleanedText.isBlank()) {
                dropped += 1
                Log.i(TAG, "dedup transcript row dropped overlap-only text=$rawText")
                continue
            }

            result += segment.copy(text = cleanedText)
        }

        Log.i(
            TAG,
            "dedup transcript rows raw=${segments.size}, kept=${result.size}, dropped=$dropped, removedChars=$removed"
        )
        return result
    }

    private fun dedupRecognitionSegments(
        segments: List<RecognitionSegment>,
    ): List<RecognitionSegment> {
        if (segments.isEmpty()) return emptyList()

        val result = mutableListOf<RecognitionSegment>()
        var removedChars = 0
        var droppedSegments = 0

        for (segment in segments) {
            val rawText = segment.text.trim()
            if (rawText.isBlank()) {
                droppedSegments += 1
                continue
            }

            val previousSegment = result.lastOrNull()
            val overlapsPrevious = previousSegment != null &&
                segment.startUs < previousSegment.endUs
            val cutIndex = if (previousSegment != null && overlapsPrevious) {
                findNormalizedPrefixOverlapCutIndex(
                    left = previousSegment.text,
                    right = rawText,
                    maxCheck = DEDUP_MAX_CHECK_CHARS,
                ).coerceIn(0, rawText.length)
            } else {
                0
            }
            val cleanedText = trimLeadingDedupSeparators(rawText.substring(cutIndex))
            removedChars += rawText.length - cleanedText.length
            if (cutIndex > 0) {
                Log.i(
                    TAG,
                    "dedup segment overlap index=${segment.index}, " +
                        "gapMs=${segment.startMs - (previousSegment?.endMs ?: segment.startMs)}, " +
                        "cutChars=$cutIndex"
                )
            }

            if (cleanedText.isBlank()) {
                droppedSegments += 1
                Log.i(TAG, "dedup segment dropped index=${segment.index}, reason=overlap-only")
                continue
            }

            result += segment.copy(text = cleanedText)
        }

        Log.i(
            TAG,
            "dedup segments raw=${segments.size}, kept=${result.size}, " +
                "dropped=$droppedSegments, removedChars=$removedChars"
        )
        return result
    }

    private fun buildTimestampedSentencesFromSegments(
        segments: List<RecognitionSegment>,
    ): List<TimestampedSentence> {
        val result = mutableListOf<TimestampedSentence>()

        for (segment in segments) {
            val sentenceTexts = splitTranscriptSentences(segment.text)
                .filter { it.isNotBlank() }
            if (sentenceTexts.isEmpty()) continue

            val lengths = sentenceTexts.map { effectiveNormalizedTextLength(it) }
            val totalLength = lengths.sum().coerceAtLeast(1)
            val segmentStartSec = segment.startMs.toDouble() / 1000.0
            val segmentEndSec = segment.endMs.toDouble() / 1000.0
            val segmentDurationSec = (segmentEndSec - segmentStartSec).coerceAtLeast(0.001)
            var cursorSec = segmentStartSec

            for ((index, sentenceText) in sentenceTexts.withIndex()) {
                val endSec = if (index == sentenceTexts.lastIndex) {
                    segmentEndSec
                } else {
                    val estimatedDuration = segmentDurationSec * lengths[index].toDouble() / totalLength.toDouble()
                    (cursorSec + estimatedDuration)
                        .coerceAtLeast(cursorSec + 0.2)
                        .coerceAtMost(segmentEndSec)
                }
                if (endSec > cursorSec) {
                    result += TimestampedSentence(
                        startSec = cursorSec,
                        endSec = endSec,
                        text = sentenceText,
                        sourceSegmentIndex = segment.index,
                        timeMode = if (segment.cutMode.startsWith("natural-vad-utterance")) {
                            "vad-boundary"
                        } else {
                            "estimated-from-chunk"
                        },
                    )
                }
                cursorSec = endSec
            }
        }

        return result
    }

    private fun splitTranscriptSentences(text: String): List<String> {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return emptyList()

        val sentenceEndings = setOf('。', '！', '？', '；', '.', '!', '?', ';')
        val sentences = mutableListOf<String>()
        val current = StringBuilder()

        for (char in trimmed) {
            current.append(char)
            if (char in sentenceEndings) {
                val sentence = current.toString().trim()
                if (sentence.isNotEmpty()) {
                    sentences += sentence
                }
                current.clear()
            }
        }

        val tail = current.toString().trim()
        if (tail.isNotEmpty()) {
            sentences += tail
        }

        return sentences.ifEmpty { listOf(trimmed) }
            .filter { it.isNotBlank() }
    }

    private fun formatNaturalUtteranceText(text: String): String {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return trimmed
        if (trimmed.last() in setOf('。', '！', '？', '；', '.', '!', '?', ';')) return trimmed

        val normalized = normalizeTranscriptText(trimmed)
        val questionPrefixes = listOf(
            "要不要", "有没有", "是不是", "多少", "几", "哪", "怎么", "为什么",
            "能不能", "可不可以", "是否", "谁", "什么",
        )
        return when {
            normalized.endsWith("吗") || normalized.endsWith("么") || normalized.endsWith("呢") ||
                questionPrefixes.any { normalized.startsWith(it) } -> "$trimmed？"
            normalized.endsWith("啊") || normalized.endsWith("呀") || normalized.endsWith("啦") ||
                normalized.endsWith("喽") -> "$trimmed！"
            else -> "$trimmed。"
        }
    }

    private fun splitLongTranscriptSentence(sentence: String): List<String> {
        val trimmed = sentence.trim()
        if (trimmed.isEmpty()) return emptyList()
        if (effectiveNormalizedTextLength(trimmed) <= TRANSCRIPT_MAX_SENTENCE_NORM_CHARS) {
            return listOf(trimmed)
        }

        val softBreaks = setOf('，', ',', '、', '：', ':')
        val clauses = mutableListOf<String>()
        val current = StringBuilder()
        for (char in trimmed) {
            current.append(char)
            if (char in softBreaks) {
                val clause = current.toString().trim()
                if (clause.isNotEmpty()) clauses += clause
                current.clear()
            }
        }
        val tail = current.toString().trim()
        if (tail.isNotEmpty()) clauses += tail

        val packed = mutableListOf<String>()
        val buffer = StringBuilder()
        for (clause in clauses.ifEmpty { listOf(trimmed) }) {
            val candidate = (buffer.toString() + clause).trim()
            if (buffer.isNotEmpty() &&
                effectiveNormalizedTextLength(candidate) > TRANSCRIPT_MAX_SENTENCE_NORM_CHARS
            ) {
                packed += buffer.toString().trim()
                buffer.clear()
            }
            if (effectiveNormalizedTextLength(clause) > TRANSCRIPT_MAX_SENTENCE_NORM_CHARS) {
                if (buffer.isNotEmpty()) {
                    packed += buffer.toString().trim()
                    buffer.clear()
                }
                packed += splitByNormalizedLength(clause, TRANSCRIPT_MAX_SENTENCE_NORM_CHARS)
            } else {
                buffer.append(clause)
            }
        }
        if (buffer.isNotEmpty()) packed += buffer.toString().trim()
        return packed.filter { it.isNotBlank() }
    }

    private fun splitByNormalizedLength(text: String, maxNormLength: Int): List<String> {
        val result = mutableListOf<String>()
        var start = 0
        while (start < text.length) {
            var normCount = 0
            var end = start
            while (end < text.length && normCount < maxNormLength) {
                if (text[end].isLetterOrDigit()) normCount += 1
                end += 1
            }
            if (end <= start) end = min(text.length, start + maxNormLength)
            result += text.substring(start, end).trim()
            start = end
        }
        return result.filter { it.isNotBlank() }
    }

    private fun effectiveNormalizedTextLength(text: String): Int {
        return normalizeTranscriptText(text).length.coerceAtLeast(1)
    }

    private fun normalizeTranscriptText(text: String): String {
        return text
            .lowercase(Locale.ROOT)
            .filter { it.isLetterOrDigit() }
    }

    private fun findNormalizedPrefixOverlapCutIndex(
        left: String,
        right: String,
        maxCheck: Int,
    ): Int {
        if (left.isBlank() || right.isBlank()) return 0

        val leftNorm = normalizeTranscriptText(left).takeLast(maxCheck)
        val rightNorm = normalizeTranscriptText(right)
        if (leftNorm.isEmpty() || rightNorm.isEmpty()) return 0

        val minOverlap = DEDUP_MIN_OVERLAP_CHARS
        val fuzzyMinOverlap = DEDUP_FUZZY_MIN_OVERLAP_CHARS
        val fuzzyThreshold = DEDUP_FUZZY_THRESHOLD

        val leftNearTail = leftNorm.takeLast(min(leftNorm.length, max(80, rightNorm.length * 2)))
        if (rightNorm.length >= minOverlap &&
            rightNorm.length <= maxCheck &&
            (leftNearTail.endsWith(rightNorm) || (rightNorm.length >= 12 && leftNearTail.contains(rightNorm)))
        ) {
            return right.length
        }

        var bestCutIndex = 0
        var bestNormLength = 0
        for (rawEnd in 1..right.length) {
            val prefixNorm = normalizeTranscriptText(right.substring(0, rawEnd))
            val prefixLength = prefixNorm.length
            if (prefixLength < minOverlap || prefixLength < bestNormLength) continue
            if (prefixLength > maxCheck) break
            if (prefixLength > leftNorm.length) continue

            val leftSuffix = leftNorm.takeLast(prefixLength)
            val prefixNearTail = leftNorm.takeLast(min(leftNorm.length, max(80, prefixLength * 2)))
            val isExactOverlap = leftSuffix == prefixNorm
            val isFuzzyOverlap = prefixLength >= fuzzyMinOverlap &&
                normalizedSimilarity(leftSuffix, prefixNorm) >= fuzzyThreshold
            val isContainedNearTail = prefixLength >= 12 &&
                prefixNearTail.contains(prefixNorm)

            if (isExactOverlap || isFuzzyOverlap || isContainedNearTail) {
                bestCutIndex = rawEnd
                bestNormLength = prefixLength
            }
        }

        return bestCutIndex
    }

    private fun trimLeadingDedupSeparators(text: String): String {
        var index = 0
        while (index < text.length) {
            val char = text[index]
            if (char.isLetterOrDigit()) break
            index += 1
        }
        return text.substring(index).trim()
    }

    private fun normalizedSimilarity(left: String, right: String): Double {
        if (left.isEmpty() && right.isEmpty()) return 1.0
        if (left.isEmpty() || right.isEmpty()) return 0.0

        val distance = levenshteinDistance(left, right)
        return 1.0 - distance.toDouble() / max(left.length, right.length).toDouble()
    }

    private fun levenshteinDistance(left: String, right: String): Int {
        var previous = IntArray(right.length + 1) { it }
        var current = IntArray(right.length + 1)

        for (i in 1..left.length) {
            current[0] = i
            for (j in 1..right.length) {
                val substitutionCost = if (left[i - 1] == right[j - 1]) 0 else 1
                current[j] = minOf(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitutionCost,
                )
            }
            val swap = previous
            previous = current
            current = swap
        }

        return previous[right.length]
    }

    private fun splitChineseSentences(text: String): List<String> {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return emptyList()

        val sentenceEndings = setOf('。', '！', '？', '；', '.', '!', '?', ';')
        val sentences = mutableListOf<String>()
        val current = StringBuilder()

        for (char in trimmed) {
            current.append(char)
            if (char in sentenceEndings) {
                val sentence = current.toString().trim()
                if (sentence.isNotEmpty()) {
                    sentences += sentence
                }
                current.clear()
            }
        }

        val tail = current.toString().trim()
        if (tail.isNotEmpty()) {
            sentences += tail
        }

        return sentences.ifEmpty { listOf(trimmed) }
    }

    private fun effectiveTextLength(text: String): Int {
        return text
            .replace(Regex("\\s+"), "")
            .replace(Regex("[，,。.!！?？；;：:]"), "")
            .length
            .coerceAtLeast(1)
    }

    private fun dedupTimestampedSentences(
        sentences: List<TimestampedSentence>,
    ): List<TimestampedSentence> {
        if (sentences.isEmpty()) return emptyList()

        val result = mutableListOf<TimestampedSentence>()
        for (sentence in sentences) {
            if (sentence.text.isBlank()) continue

            val previous = result.lastOrNull()
            if (previous == null) {
                result += sentence
                continue
            }

            if (sentence.startSec >= previous.endSec) {
                result += sentence
                continue
            }

            val previousNorm = normalizeForDedup(previous.text)
            val currentNorm = normalizeForDedup(sentence.text)
            if (currentNorm.isEmpty()) continue

            when {
                previousNorm == currentNorm -> {
                    Log.i(TAG, "dedup sentence exact source=${sentence.sourceSegmentIndex}")
                }
                previousNorm.contains(currentNorm) -> {
                    Log.i(TAG, "dedup sentence contained source=${sentence.sourceSegmentIndex}")
                }
                currentNorm.contains(previousNorm) -> {
                    result[result.lastIndex] = sentence.copy(startSec = previous.startSec)
                    Log.i(TAG, "dedup sentence replaced source=${sentence.sourceSegmentIndex}")
                }
                hasLargeNormalizedOverlap(previousNorm, currentNorm) -> {
                    Log.i(TAG, "dedup sentence overlap source=${sentence.sourceSegmentIndex}")
                }
                else -> result += sentence
            }
        }

        return result
    }

    private fun normalizeForDedup(text: String): String {
        if (System.currentTimeMillis() >= 0L) {
            return normalizeTranscriptText(text)
        }

        return text
            .replace(Regex("\\s+"), "")
            .replace(Regex("[，,。.!！?？；;：:]"), "")
    }

    private fun hasLargeNormalizedOverlap(left: String, right: String): Boolean {
        val maxLength = min(40, min(left.length, right.length))
        if (maxLength < 4) return false

        for (length in maxLength downTo 4) {
            val suffixPrefix = left.takeLast(length) == right.take(length)
            val prefixSuffix = right.takeLast(length) == left.take(length)
            val coversMostOfShorter = length.toDouble() / min(left.length, right.length).toDouble() >= 0.8
            if ((suffixPrefix || prefixSuffix) && coversMostOfShorter) {
                return true
            }
        }

        return false
    }

    private fun buildTimestampedSentenceDisplayText(
        sentences: List<TimestampedSentence>,
    ): String {
        if (sentences.isEmpty()) return ""
        return sentences.joinToString("\n") { sentence ->
            "[${formatTxtTimestamp(sentence.startSec)} - ${formatTxtTimestamp(sentence.endSec)}] ${sentence.text}"
        }
    }

    private fun buildFullOutput(
        displayName: String,
        output: RecognitionOutput,
        txtFile: File?,
        txtSaveError: String?,
    ): String {
        return buildRecognitionOutput(displayName, output, txtFile, txtSaveError)
    }

    private fun buildSelectedOutput(
        displayName: String,
        output: RecognitionOutput,
        txtFile: File?,
        txtSaveError: String?,
    ): String {
        return buildRecognitionOutput(displayName, output, txtFile, txtSaveError)
    }

    private fun buildRecognitionOutput(
        displayName: String,
        output: RecognitionOutput,
        txtFile: File?,
        txtSaveError: String?,
    ): String {
        val txtStatus = if (txtSaveError == null) {
            "TXT 已保存：\n${txtFile?.absolutePath ?: "未知路径"}"
        } else {
            "TXT 保存失败：\n$txtSaveError"
        }

        return """
识别完成：$displayName
切分模式：${output.segmentationMode}
分段数量：${output.segments.size}

识别统计：
${buildTranscriptionStatsText(output.stats)}

识别文本：
${output.text}

$txtStatus
""".trim()
    }

    private fun buildTimestampedTxt(
        sourceFileName: String,
        output: RecognitionOutput,
    ): String {
        return buildTimestampedSentenceTxt(sourceFileName, output)
    }

    private fun buildTimestampedSentenceTxt(
        sourceFileName: String,
        output: RecognitionOutput,
    ): String {
        if (System.currentTimeMillis() >= 0L) {
            return TimestampedTranscriptWriter.buildTxt(
                sourceFileName = sourceFileName,
                recordingTimeContext = output.recordingTimeContext,
                durationSec = output.durationSec,
                sampleRate = output.sampleRate,
                channels = output.channels,
                bitsPerSample = output.bitsPerSample,
                segmentationMode = output.segmentationMode,
                stats = output.stats,
                transcriptSegments = output.transcriptSegments,
            )
        }

        return """
文件名：$sourceFileName
识别模式：chunked-large-wav-speechgate-silero-vad
输出模式：timestamped-dedup-segment-transcript
切分方式：15 秒目标窗口，1.5 秒重叠，先用 Silero VAD 拦截无人声 chunk，再做 VAD 静音切点微调
音频时长：${formatTxtTimestamp((output.durationSec * 1000.0).roundToLong())}
采样率：${output.sampleRate} Hz
声道数：${output.channels}
位深：${output.bitsPerSample}-bit
分块数量：${output.segments.size}
句子数量：${output.timestampedSentences.size}
时间戳模式：estimated-from-chunk
时间戳说明：当前句子时间戳由分块时间范围和文本长度估算生成，适合快速定位音频位置；如需精确到词级，需要接入支持 timestamp 的 ASR 输出或额外 forced alignment。

==============================
识别统计
==============================

${buildTranscriptionStatsText(output.stats)}

==============================
时间戳分段稿（已去重）
==============================

${if (output.stats.hasAnySpeech) buildSegmentDisplayText(output.segments) else NO_SPEECH_MESSAGE}

==============================
合并稿
==============================

${if (output.stats.hasAnySpeech) output.mergedText else ""}
""".trimEnd() + "\n"
    }

    private fun buildTranscriptionStatsText(stats: TranscriptionStats): String {
        return """
总时长：${formatTxtTimestamp(stats.totalDurationMs)}
检测到的人声时长：${formatTxtTimestamp(stats.speechDurationMs)}
跳过的无人声时长：${formatTxtTimestamp(stats.skippedNoSpeechDurationMs)}
总 chunk 数量：${stats.totalChunks}
送入 ASR 的 chunk 数量：${stats.speechChunks}
跳过的 chunk 数量：${stats.skippedChunks}
ASR 实际处理时长：${formatTxtTimestamp(stats.asrProcessedDurationMs)}
""".trim()
    }

    private fun buildTimestampedSentenceText(sentences: List<TimestampedSentence>): String {
        if (sentences.isEmpty()) return ""
        return sentences.joinToString("\n") { sentence ->
            "[${formatTxtTimestamp(sentence.startSec)} - ${formatTxtTimestamp(sentence.endSec)}] ${sentence.text}"
        }
    }

    private fun buildMergedText(segments: List<RecognitionSegment>): String {
        return mergeTextsWithSimpleDedup(
            segments
            .map { it.text.trim() }
            .filter { it.isNotEmpty() }
        )
    }

    private fun mergeTextsWithSimpleDedup(texts: List<String>): String {
        if (texts.isEmpty()) return ""

        val merged = StringBuilder(texts.first())
        for (text in texts.drop(1)) {
            val overlap = findSuffixPrefixOverlap(
                left = merged.toString(),
                right = text,
                maxCheck = DEDUP_MAX_CHECK_CHARS,
            )
            merged.append(trimLeadingDedupSeparators(text.substring(overlap)))
        }

        return merged.toString()
    }

    private fun findSuffixPrefixOverlap(
        left: String,
        right: String,
        maxCheck: Int,
    ): Int {
        return findNormalizedPrefixOverlapCutIndex(
            left = left,
            right = right,
            maxCheck = maxCheck,
        )
    }

    private fun formatTxtTimestamp(ms: Long): String {
        val totalCentiseconds = (ms.toDouble() / 10.0).roundToLong().coerceAtLeast(0L)
        val centiseconds = totalCentiseconds % 100
        val totalSeconds = totalCentiseconds / 100
        val s = totalSeconds % 60
        val totalMinutes = totalSeconds / 60
        val m = totalMinutes % 60
        val h = totalMinutes / 60
        return "%02d:%02d:%02d.%02d".format(Locale.US, h, m, s, centiseconds)
    }

    private fun formatTxtTimestamp(seconds: Double): String {
        return formatTxtTimestamp((seconds * 1000.0).roundToLong())
    }

    private fun exportTxtToUserLocation(file: File) {
        if (!isUsableTxtFile(file)) {
            showTxtOperationError("无法保存", "TXT 文件不存在或为空，请重新完成一次识别。")
            return
        }
        pendingTxtExportPath = file.absolutePath
        txtDocumentCreator.launch(file.name)
    }

    private fun finishTxtExport(destinationUri: Uri?) {
        val sourcePath = pendingTxtExportPath
        pendingTxtExportPath = null
        if (destinationUri == null) {
            Toast.makeText(this, "已取消保存", Toast.LENGTH_SHORT).show()
            return
        }
        val sourceFile = sourcePath?.let(::File)
        if (sourceFile == null || !isUsableTxtFile(sourceFile)) {
            showTxtOperationError("保存失败", "找不到待保存的 TXT 文件，请返回任务记录后重试。")
            return
        }

        thread(name = "txt-export") {
            val result = runCatching {
                val output = contentResolver.openOutputStream(destinationUri, "w")
                    ?: error("系统未提供可写入的目标文件")
                sourceFile.inputStream().use { input ->
                    output.use { target -> input.copyTo(target) }
                }
            }
            runOnUiThread {
                result.onSuccess {
                    setStatus("TXT 已保存到您选择的位置")
                    showTxtSavedDialog(destinationUri)
                }.onFailure { error ->
                    Log.e(TAG, "TXT export failed uri=$destinationUri", error)
                    showTxtOperationError("保存失败", error.message ?: error::class.java.simpleName)
                }
            }
        }
    }

    private fun showTxtSavedDialog(uri: Uri) {
        AlertDialog.Builder(this)
            .setTitle("TXT 已保存")
            .setMessage("文件已经保存到您选择的位置，可在手机文件管理器中找到。")
            .setPositiveButton("打开查看") { _, _ -> openTxtUri(uri) }
            .setNegativeButton("完成", null)
            .show()
    }

    private fun showTxtFile(file: File) {
        if (!isUsableTxtFile(file)) {
            showTxtOperationError("无法查看", "TXT 文件不存在或为空，请重新完成一次识别。")
            return
        }
        thread(name = "txt-preview") {
            val result = runCatching { file.readText(Charsets.UTF_8) }
            runOnUiThread {
                result.onSuccess { content -> showTxtContentDialog(file.name, content) }
                    .onFailure { error ->
                        Log.e(TAG, "TXT preview failed path=${file.absolutePath}", error)
                        showTxtOperationError("读取失败", error.message ?: error::class.java.simpleName)
                    }
            }
        }
    }

    private fun showTxtContentDialog(fileName: String, content: String) {
        val contentView = TextView(this).apply {
            text = content
            textSize = 14f
            setTextIsSelectable(true)
            setPadding(dp(18), dp(12), dp(18), dp(12))
        }
        val scrollView = ScrollView(this).apply {
            addView(contentView, matchWrap())
        }
        AlertDialog.Builder(this)
            .setTitle(fileName)
            .setView(scrollView)
            .setPositiveButton("复制全文") { _, _ ->
                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                clipboard.setPrimaryClip(ClipData.newPlainText(fileName, content))
                Toast.makeText(this, "全文已复制", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("关闭", null)
            .show()
    }

    private fun shareTxtFile(file: File) {
        if (!isUsableTxtFile(file)) {
            showTxtOperationError("无法分享", "TXT 文件不存在或为空，请重新完成一次识别。")
            return
        }
        runCatching {
            val contentUri = FileProvider.getUriForFile(
                this,
                "$packageName.fileprovider",
                file,
            )
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_SUBJECT, file.name)
                putExtra(Intent.EXTRA_STREAM, contentUri)
                clipData = ClipData.newRawUri(file.name, contentUri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "分享识别结果"))
        }.onFailure { error ->
            Log.e(TAG, "TXT share failed path=${file.absolutePath}", error)
            showTxtOperationError("分享失败", error.message ?: "没有可用于分享 TXT 的应用")
        }
    }

    private fun openTxtUri(uri: Uri) {
        runCatching {
            val viewIntent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "text/plain")
                clipData = ClipData.newRawUri("TXT", uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(viewIntent)
        }.onFailure { error ->
            Log.w(TAG, "No app can open exported TXT uri=$uri", error)
            showTxtOperationError("无法打开", "TXT 已保存，但手机中没有可打开纯文本文件的应用。")
        }
    }

    private fun showTxtOperationError(title: String, message: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("知道了", null)
            .show()
    }

    private fun isUsableTxtFile(file: File): Boolean {
        return file.isFile && file.length() > 0L
    }

    private fun saveTimestampedTxt(
        sourceFileName: String?,
        content: String,
    ): File {
        val documentsDir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: error("无法获取 App Documents 目录")
        if (!documentsDir.exists() && !documentsDir.mkdirs()) {
            error("无法创建 App Documents 目录：${documentsDir.absolutePath}")
        }

        val target = File(documentsDir, buildTxtFileName(sourceFileName))
        target.writeText(content, Charsets.UTF_8)
        require(target.length() > 0L) {
            "TXT 保存失败：文件为空"
        }
        Log.i(TAG, "timestamped sentence TXT saved path=${target.absolutePath}, bytes=${target.length()}")
        return target
    }

    private fun buildTxtFileName(sourceFileName: String?): String {
        val cleanName = sourceFileName
            ?.trim()
            ?.substringAfterLast('/')
            ?.substringAfterLast('\\')
            .orEmpty()
        val dotIndex = cleanName.lastIndexOf('.')
        if (dotIndex <= 0) return "selected_audio.txt"

        val baseName = cleanName
            .substring(0, dotIndex)
            .replace(Regex("""[\\/:*?"<>|]"""), "_")
            .ifBlank { "selected_audio" }
        return "$baseName.txt"
    }

    private fun assetFileExists(path: String): Boolean {
        val dir = path.substringBeforeLast('/', "")
        val fileName = path.substringAfterLast('/')
        return assets.list(dir)?.contains(fileName) == true
    }

    private fun copyResult() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("ASR result", lastOutput))
        setStatus("结果已复制")
    }

    private fun setStatus(message: String) {
        statusView.text = message
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun buildCard(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
            background = roundedBackground(
                fillColor = Color.WHITE,
                strokeColor = Color.parseColor("#E5EAF2"),
                radiusDp = 12,
            )
        }
    }

    private fun buildButtonRow(vararg buttons: Button): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            buttons.forEachIndexed { index, button ->
                detachFromParent(button)
                addView(
                    button,
                    LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                        if (index > 0) setMargins(dp(8), 0, 0, 0)
                    },
                )
            }
        }
    }

    private fun sectionTitle(text: String): TextView {
        return TextView(this).apply {
            this.text = text
            textSize = 20f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#111827"))
        }
    }

    private fun buildStatCell(title: String, value: String, colorHex: String): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            addView(
                TextView(this@MainActivity).apply {
                    text = title
                    textSize = 13f
                    gravity = Gravity.CENTER
                    setTextColor(Color.parseColor("#111827"))
                },
                matchWrap(),
            )
            addView(
                TextView(this@MainActivity).apply {
                    text = value
                    textSize = 18f
                    gravity = Gravity.CENTER
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(Color.parseColor(colorHex))
                    setPadding(0, dp(6), 0, 0)
                },
                matchWrap(),
            )
        }
    }

    private fun statCellParams() = LinearLayout.LayoutParams(
        0,
        LinearLayout.LayoutParams.WRAP_CONTENT,
        1f,
    )

    private fun buildTranscriptPreview(output: RecognitionOutput): String {
        if (!output.stats.hasAnySpeech) return NO_SPEECH_MESSAGE
        val lines = output.transcriptSegments
            .take(5)
            .map { segment ->
                "[${formatTxtTimestamp(segment.relativeStartUs / 1_000L)}] ${segment.text}"
            }
        if (lines.isNotEmpty()) return lines.joinToString("\n\n")
        return output.mergedText
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .take(5)
            .joinToString("\n\n")
            .ifBlank { output.text.take(500) }
    }

    private fun formatBytes(bytes: Long?): String {
        val value = bytes ?: return "大小未知"
        if (value < 1024L) return "$value B"
        val units = arrayOf("KB", "MB", "GB")
        var amount = value.toDouble() / 1024.0
        var index = 0
        while (amount >= 1024.0 && index < units.lastIndex) {
            amount /= 1024.0
            index++
        }
        return String.format(Locale.US, "%.2f %s", amount, units[index])
    }

    private fun stylePrimaryButton(button: Button) {
        button.setTextColor(Color.WHITE)
        button.textSize = 15f
        button.typeface = Typeface.DEFAULT_BOLD
        button.background = roundedBackground(
            fillColor = Color.parseColor("#1263EA"),
            strokeColor = Color.parseColor("#1263EA"),
            radiusDp = 8,
        )
    }

    private fun styleSecondaryButton(button: Button) {
        button.setTextColor(Color.parseColor("#1263EA"))
        button.textSize = 15f
        button.background = roundedBackground(
            fillColor = Color.WHITE,
            strokeColor = Color.parseColor("#1263EA"),
            radiusDp = 8,
        )
    }

    private fun setInputStyle(input: EditText) {
        input.textSize = 14f
        input.setPadding(dp(12), dp(8), dp(12), dp(8))
        input.background = roundedBackground(
            fillColor = Color.parseColor("#F8FAFD"),
            strokeColor = Color.parseColor("#D9E2F0"),
            radiusDp = 8,
        )
    }

    private fun setInfoBoxStyle(
        view: TextView,
        fillColor: Int,
        strokeColor: Int = Color.parseColor("#E5EAF2"),
    ) {
        view.setTextColor(Color.parseColor("#374151"))
        view.setPadding(dp(12), dp(10), dp(12), dp(10))
        view.background = roundedBackground(
            fillColor = fillColor,
            strokeColor = strokeColor,
            radiusDp = 8,
        )
    }

    private fun roundedBackground(fillColor: Int, strokeColor: Int, radiusDp: Int): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            setColor(fillColor)
            cornerRadius = dp(radiusDp).toFloat()
            if (strokeColor != Color.TRANSPARENT) {
                setStroke(dp(1), strokeColor)
            }
        }
    }

    private fun detachFromParent(view: View) {
        (view.parent as? ViewGroup)?.removeView(view)
    }

    private fun formatJobDuration(durationMs: Long?): String {
        val value = durationMs ?: 0L
        return if (value > 0L) formatTxtTimestamp(value) else "--:--"
    }

    private fun matchWrap() = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT,
    )

    private fun sectionWrap() = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT,
    ).apply {
        setMargins(0, dp(8), 0, dp(10))
    }

    private fun wrapWrap() = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.WRAP_CONTENT,
        LinearLayout.LayoutParams.WRAP_CONTENT,
    )

    private fun childWrap(topMargin: Int = 10) = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT,
    ).apply {
        setMargins(0, dp(topMargin), 0, 0)
    }

    override fun onStart() {
        super.onStart()
        if (!airecReceiverRegistered) {
            val filter = IntentFilter().apply {
                addAction(AirecReceiverService.ACTION_STATE_CHANGED)
                addAction(AirecReceiverService.ACTION_UPLOAD_RECEIVED)
                addAction(TranscriptionForegroundService.ACTION_JOB_UPDATED)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                registerReceiver(airecReceiverUpdates, filter, Context.RECEIVER_NOT_EXPORTED)
            } else {
                @Suppress("DEPRECATION")
                registerReceiver(airecReceiverUpdates, filter)
            }
            airecReceiverRegistered = true
        }
        activeTranscriptionJobId = TranscriptionForegroundService.activeJobId
        refreshAirecReceiverUi()
        refreshJobHistoryUi()
    }

    override fun onStop() {
        if (airecReceiverRegistered) {
            unregisterReceiver(airecReceiverUpdates)
            airecReceiverRegistered = false
        }
        super.onStop()
    }

    override fun onDestroy() {
        if (activeTranscriptionJobId == null) {
            recognizer?.release()
            recognizer = null
            speechGate?.release()
            speechGate = null
        }
        super.onDestroy()
    }
}
