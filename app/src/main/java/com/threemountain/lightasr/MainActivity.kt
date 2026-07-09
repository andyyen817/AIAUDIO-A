package com.threemountain.lightasr

import android.Manifest
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
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.getFeatureConfig
import com.k2fsa.sherpa.onnx.getOfflineModelConfig
import com.threemountain.lightasr.voiceprint.Employee
import com.threemountain.lightasr.voiceprint.VoiceSample
import com.threemountain.lightasr.voiceprint.VoiceprintManager
import com.threemountain.lightasr.voiceprint.VoiceprintMatchResult
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
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
private const val MODEL_TYPE = 14
private const val MODEL_DIR = "sherpa-onnx-paraformer-zh-small-2024-03-09"
private const val VAD_MODEL_ASSET = "models/vad/silero_vad.onnx"
private const val TEST_WAV = "test.wav"
private const val NO_SPEECH_MESSAGE = "未检测到有效人声，已跳过识别。"
private const val FIXED_SEGMENT_SECONDS = 15.0
private const val CHUNKED_SEGMENT_MS = 15_000L
private const val CHUNKED_OVERLAP_MS = 1_500L
private const val VAD_BOUNDARY_SEARCH_RADIUS_MS = 2_000L
private const val VAD_BOUNDARY_MIN_SILENCE_MS = 300L
private const val VAD_BOUNDARY_FRAME_MS = 20L
private const val VAD_BOUNDARY_NOISE_MARGIN_DB = 8.0
private const val VAD_FRAME_MS = 30L
private const val VAD_HOP_MS = 10L
private const val VAD_MIN_SPEECH_MS = 300L
private const val VAD_MIN_SILENCE_MS = 500L
private const val VAD_SPEECH_PADDING_MS = 200L
private const val VAD_MAX_SEGMENT_MS = 15_000L
private const val VAD_MERGE_GAP_MS = 300L
private const val VAD_MIN_SEGMENT_MS = 500L
private const val SHARED_IMPORT_BUFFER_SIZE = 64 * 1024
private const val SHARED_IMPORT_MAX_NAME_LENGTH = 120
private const val SHARED_IMPORT_RETRY_COUNT = 8
private const val SHARED_IMPORT_RETRY_DELAY_MS = 500L
private const val DEFAULT_RECORDING_START_TEXT = "2026-07-09 10:00:00.000"
private const val DEFAULT_RECORDING_TIMEZONE_ID = "Asia/Shanghai"
private const val DEDUP_MAX_CHECK_CHARS = 320
private const val DEDUP_ACCUMULATED_TAIL_CHARS = 2_000
private const val DEDUP_MIN_OVERLAP_CHARS = 4
private const val DEDUP_FUZZY_MIN_OVERLAP_CHARS = 8
private const val DEDUP_FUZZY_THRESHOLD = 0.82

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

    private lateinit var pickAudioButton: Button
    private lateinit var transcribeButton: Button
    private lateinit var transcribeSelectedButton: Button
    private lateinit var copyButton: Button
    private lateinit var startAirecReceiverButton: Button
    private lateinit var stopAirecReceiverButton: Button
    private lateinit var copyAirecUrlButton: Button
    private lateinit var airecReceiverStatusView: TextView
    private lateinit var airecUploadRecordsView: TextView
    private lateinit var airecUploadRecordActions: LinearLayout
    private lateinit var sharedImportRecordsView: TextView
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

    private lateinit var voiceprintManager: VoiceprintManager
    private var recognizer: OfflineRecognizer? = null
    private var speechGate: SpeechGate? = null
    private var lastOutput: String = ""
    private var selectedAudioFile: SelectedAudioFile? = null
    private var selectedVoiceprintEmployeeId: Long? = null
    private val sharedAudioRecords = mutableListOf<SharedAudioRecord>()
    private var airecReceiverRegistered = false
    private val airecReceiverUpdates = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            refreshAirecReceiverUi()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        voiceprintManager = VoiceprintManager.create(this, assets)
        buildUi()
        refreshAirecReceiverUi()
        refreshSharedImportRecordsUi()
        updateVoiceprintEmployeeView()
        initRecognizerAsync()
        handleShareIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleShareIntent(intent)
    }

    private fun buildUi() {
        val scrollView = ScrollView(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val padding = dp(16)
            setPadding(padding, padding, padding, padding)
            gravity = Gravity.CENTER_HORIZONTAL
        }

        val title = TextView(this).apply {
            text = "LightASR 声纹测试版"
            textSize = 22f
            gravity = Gravity.CENTER
        }

        val hint = TextView(this).apply {
            text = "本地 ASR + 3D-Speaker 声纹测试版，不上传服务器。"
            textSize = 14f
            setPadding(0, dp(8), 0, dp(12))
        }

        pickAudioButton = Button(this).apply {
            text = "选择音频文件"
            setOnClickListener { pickAudioFile() }
        }

        selectedAudioView = TextView(this).apply {
            text = "未选择文件"
            textSize = 14f
            val padding = dp(12)
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(0xFFEFF3F8.toInt())
        }

        transcribeButton = Button(this).apply {
            text = "识别内置测试音频"
            isEnabled = false
            setOnClickListener { transcribeBundledWav() }
        }

        transcribeSelectedButton = Button(this).apply {
            text = "识别所选音频"
            isEnabled = false
            setOnClickListener { transcribeSelectedAudio() }
        }

        copyButton = Button(this).apply {
            text = "复制结果"
            isEnabled = false
            setOnClickListener { copyResult() }
        }

        airecReceiverStatusView = TextView(this).apply {
            textSize = 14f
            val padding = dp(12)
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(0xFFEFF3F8.toInt())
        }

        startAirecReceiverButton = Button(this).apply {
            text = "开启接收模式"
            setOnClickListener { startAirecReceiver() }
        }

        stopAirecReceiverButton = Button(this).apply {
            text = "停止接收模式"
            setOnClickListener { stopAirecReceiver() }
        }

        copyAirecUrlButton = Button(this).apply {
            text = "复制上传地址"
            setOnClickListener { copyAirecUploadUrl() }
        }

        airecUploadRecordsView = TextView(this).apply {
            textSize = 14f
            val padding = dp(12)
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(0xFFF3F6F4.toInt())
        }

        airecUploadRecordActions = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        sharedImportRecordsView = TextView(this).apply {
            textSize = 14f
            val padding = dp(12)
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(0xFFEFF3F8.toInt())
        }

        val voiceprintMarker = TextView(this).apply {
            text = "如果你看到这一行，说明当前安装的是带声纹模型入口的测试版。"
            textSize = 14f
            val padding = dp(12)
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(0xFFFFF6E6.toInt())
        }

        employeeNameInput = EditText(this).apply {
            setHint("员工姓名，例如：张三")
            setSingleLine(true)
        }

        employeeStoreInput = EditText(this).apply {
            setHint("门店，可选")
            setSingleLine(true)
        }

        voiceprintEmployeeView = TextView(this).apply {
            textSize = 14f
            val padding = dp(12)
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(0xFFFFF6E6.toInt())
        }

        createEmployeeButton = Button(this).apply {
            text = "创建员工"
            setOnClickListener { createVoiceprintEmployee() }
        }

        nextEmployeeButton = Button(this).apply {
            text = "选择下一个员工"
            setOnClickListener { selectNextVoiceprintEmployee() }
        }

        addVoiceSampleButton = Button(this).apply {
            text = "添加所选音频为声纹样本"
            isEnabled = false
            setOnClickListener { addVoiceprintSampleFromSelected() }
        }

        enrollVoiceprintButton = Button(this).apply {
            text = "执行声纹注册"
            isEnabled = false
            setOnClickListener { enrollSelectedVoiceprintEmployee() }
        }

        identifySpeakerButton = Button(this).apply {
            text = "识别所选音频说话人"
            isEnabled = false
            setOnClickListener { identifySelectedSpeaker() }
        }

        listEmployeesButton = Button(this).apply {
            text = "查看员工列表"
            setOnClickListener { showVoiceprintEmployees() }
        }

        deleteEmployeeButton = Button(this).apply {
            text = "删除当前员工"
            isEnabled = false
            setOnClickListener { deleteSelectedVoiceprintEmployee() }
        }

        progressBar = ProgressBar(this).apply {
            visibility = View.GONE
        }

        statusView = TextView(this).apply {
            text = "正在初始化模型..."
            textSize = 14f
            setPadding(0, dp(12), 0, dp(8))
        }

        resultView = TextView(this).apply {
            text = ""
            textSize = 15f
            movementMethod = ScrollingMovementMethod()
            val padding = dp(12)
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(0xFFF3F6F4.toInt())
        }

        root.addView(title, matchWrap())
        root.addView(hint, matchWrap())
        root.addView(
            buildSection(
                "正式音频分析",
                "选择 WAV 后进入本地 VAD + ASR",
                0xFFEFF3F8.toInt(),
                pickAudioButton,
                selectedAudioView,
                transcribeSelectedButton,
                transcribeButton,
            ),
            sectionWrap(),
        )
        root.addView(
            buildSection(
                "Android 分享导入",
                "从其他录音 App 分享历史 WAV",
                0xFFF3F6F4.toInt(),
                sharedImportRecordsView,
            ),
            sectionWrap(),
        )
        root.addView(
            buildSection(
                "AIREC 局域网接收",
                "同一 Wi-Fi 下接收 AIREC 自动上传",
                0xFFEFF3F8.toInt(),
                airecReceiverStatusView,
                startAirecReceiverButton,
                stopAirecReceiverButton,
                copyAirecUrlButton,
                airecUploadRecordsView,
                airecUploadRecordActions,
            ),
            sectionWrap(),
        )
        root.addView(
            buildSection(
                "声纹识别 / 员工注册",
                "员工样本、注册和所选音频说话人识别",
                0xFFFFF6E6.toInt(),
                voiceprintMarker,
                employeeNameInput,
                employeeStoreInput,
                voiceprintEmployeeView,
                createEmployeeButton,
                nextEmployeeButton,
                addVoiceSampleButton,
                enrollVoiceprintButton,
                identifySpeakerButton,
                listEmployeesButton,
                deleteEmployeeButton,
            ),
            sectionWrap(),
        )
        root.addView(
            buildSection(
                "运行状态与结果",
                null,
                0xFFF7F7F7.toInt(),
                progressBar,
                statusView,
                resultView,
                copyButton,
            ),
            sectionWrap(),
        )

        scrollView.addView(root)
        setContentView(scrollView)
    }

    private fun buildSection(
        title: String,
        subtitle: String?,
        backgroundColor: Int,
        vararg children: View,
    ): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val padding = dp(12)
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(backgroundColor)

            addView(
                TextView(this@MainActivity).apply {
                    text = title
                    textSize = 18f
                    setPadding(0, 0, 0, dp(4))
                },
                matchWrap(),
            )

            if (!subtitle.isNullOrBlank()) {
                addView(
                    TextView(this@MainActivity).apply {
                        text = subtitle
                        textSize = 13f
                        setPadding(0, 0, 0, dp(8))
                    },
                    matchWrap(),
                )
            }

            children.forEach { child ->
                val params = if (child is ProgressBar) wrapWrap() else matchWrap()
                addView(child, params)
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
            sharedStreamUriList(intent)?.forEach { add(it) }
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
        return """
已选择文件
文件名: ${file.displayName ?: "未知"}
MIME: ${file.mimeType ?: "未知"}
大小: ${file.sizeBytes?.toString() ?: "未知"} bytes
Uri: ${file.uri}
""".trim()
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
        transcribeButton.isEnabled = false
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
                    modelConfig = getOfflineModelConfig(MODEL_TYPE)!!,
                )

                recognizer = OfflineRecognizer(assetManager = assets, config = config)
                speechGate = SpeechGate(
                    assetManager = assets,
                    config = SpeechGateConfig(modelAssetPath = VAD_MODEL_ASSET),
                )

                runOnUiThread {
                    progressBar.visibility = View.GONE
                    transcribeButton.isEnabled = true
                    updateSelectedAudioButtonState()
                    refreshAirecReceiverUi()
                    setStatus("ASR 和人声检测模型已就绪，可以识别内置测试音频。")
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    transcribeButton.isEnabled = false
                    setStatus("模型初始化失败：${t.message ?: t::class.java.simpleName}")
                }
            }
        }
    }

    private fun transcribeBundledWav() {
        val currentRecognizer = recognizer ?: run {
            setStatus("模型尚未就绪")
            return
        }

        progressBar.visibility = View.VISIBLE
        transcribeButton.isEnabled = false
        transcribeSelectedButton.isEnabled = false
        copyButton.isEnabled = false
        resultView.text = ""
        setStatus("正在识别 $TEST_WAV ...")

        thread(name = "transcribe-bundled-wav") {
            try {
                val wavFile = copyAssetToCache(TEST_WAV, "builtin_test.wav")
                val output = recognizeWavFile(
                    currentRecognizer = currentRecognizer,
                    wavFile = wavFile,
                    sourceFileName = TEST_WAV,
                )
                val txtContent = buildTimestampedTxt(
                    sourceFileName = TEST_WAV,
                    output = output,
                )
                var txtFile: File? = null
                var txtSaveError: String? = null
                try {
                    txtFile = saveTimestampedTxt(TEST_WAV, txtContent)
                } catch (t: Throwable) {
                    txtSaveError = t.message ?: t::class.java.simpleName
                }
                lastOutput = buildFullOutput(TEST_WAV, output, txtFile, txtSaveError)

                runOnUiThread {
                    progressBar.visibility = View.GONE
                    transcribeButton.isEnabled = true
                    updateSelectedAudioButtonState()
                    copyButton.isEnabled = lastOutput.isNotBlank()
                    resultView.text = lastOutput
                    setStatus(
                        if (!output.stats.hasAnySpeech) {
                            NO_SPEECH_MESSAGE
                        } else if (txtSaveError == null) {
                            "识别完成：$TEST_WAV"
                        } else {
                            "TXT 保存失败：$txtSaveError"
                        }
                    )
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    transcribeButton.isEnabled = true
                    updateSelectedAudioButtonState()
                    setStatus("识别失败：${t.message ?: t::class.java.simpleName}")
                }
            }
        }
    }

    private fun transcribeSelectedAudio() {
        val currentRecognizer = recognizer ?: run {
            setStatus("模型尚未就绪")
            return
        }

        val currentSelection = selectedAudioFile ?: run {
            setStatus("请先选择 WAV 文件")
            return
        }

        if (!isSupportedWav(currentSelection)) {
            setStatus("当前第一版只支持 WAV 文件，请选择 .wav 音频。")
            return
        }

        progressBar.visibility = View.VISIBLE
        transcribeButton.isEnabled = false
        transcribeSelectedButton.isEnabled = false
        copyButton.isEnabled = false
        resultView.text = ""
        setStatus("正在识别所选音频...")

        thread(name = "transcribe-selected-wav") {
            try {
                val wavFile = copySelectedUriToCache(
                    currentSelection.uri,
                    currentSelection.displayName,
                )
                val displayName = currentSelection.displayName ?: "所选音频"
                val output = recognizeWavFile(
                    currentRecognizer = currentRecognizer,
                    wavFile = wavFile,
                    sourceFileName = displayName,
                )
                val txtContent = buildTimestampedTxt(
                    sourceFileName = displayName,
                    output = output,
                )
                var txtFile: File? = null
                var txtSaveError: String? = null
                try {
                    txtFile = saveTimestampedTxt(currentSelection.displayName, txtContent)
                } catch (t: Throwable) {
                    txtSaveError = t.message ?: t::class.java.simpleName
                }
                lastOutput = buildSelectedOutput(displayName, output, txtFile, txtSaveError)

                runOnUiThread {
                    progressBar.visibility = View.GONE
                    transcribeButton.isEnabled = true
                    updateSelectedAudioButtonState()
                    copyButton.isEnabled = lastOutput.isNotBlank()
                    resultView.text = lastOutput
                    setStatus(
                        if (!output.stats.hasAnySpeech) {
                            NO_SPEECH_MESSAGE
                        } else if (txtSaveError == null) {
                            "识别完成：$displayName"
                        } else {
                            "TXT 保存失败：$txtSaveError"
                        }
                    )
                }
            } catch (t: Throwable) {
                val message = t.message ?: t::class.java.simpleName
                lastOutput = "识别失败：\n$message"

                runOnUiThread {
                    progressBar.visibility = View.GONE
                    transcribeButton.isEnabled = true
                    updateSelectedAudioButtonState()
                    copyButton.isEnabled = false
                    resultView.text = lastOutput
                    setStatus("识别失败：$message")
                }
            }
        }
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

        var segmentationMode = "chunked-${CHUNKED_SEGMENT_MS / 1000}s-speechgate-full-chunk-asr-overlap-${formatSeconds(CHUNKED_OVERLAP_MS)}s"
        val ranges = try {
            buildVadAdjustedSegmentRanges(
                wavFile = wavFile,
                header = header,
                chunkMs = CHUNKED_SEGMENT_MS,
                overlapMs = CHUNKED_OVERLAP_MS,
                vadSearchRadiusMs = VAD_BOUNDARY_SEARCH_RADIUS_MS,
                vadMinSilenceMs = VAD_BOUNDARY_MIN_SILENCE_MS,
            )
        } catch (t: Throwable) {
            Log.w(TAG, "VAD 切点微调失败，回退固定分块: ${t.message ?: t::class.java.simpleName}")
            segmentationMode = "chunked-${CHUNKED_SEGMENT_MS / 1000}s-speechgate-full-chunk-asr-overlap-${formatSeconds(CHUNKED_OVERLAP_MS)}s"
            buildChunkedSegmentRanges(header.durationMs)
        }
        val chunkResult = recognizeChunkedWav(
            currentRecognizer = currentRecognizer,
            currentSpeechGate = currentSpeechGate,
            wavFile = wavFile,
            header = header,
            ranges = ranges,
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
            segmentationMode = segmentationMode,
            fallbackToFixedSegments = false,
            rawSegmentCount = rawSegments.size,
            mergedSegmentCount = segments.size,
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
    ): ChunkRecognitionResult {
        val segments = mutableListOf<RecognitionSegment>()
        var speechDurationMs = 0L
        var skippedNoSpeechDurationMs = 0L
        var speechChunks = 0
        var skippedChunks = 0
        var asrProcessedDurationMs = 0L

        RandomAccessFile(wavFile, "r").use { input ->
            var reusableBytes = ByteArray(0)

            for ((position, range) in ranges.withIndex()) {
                val (nextReusableBytes, chunkAudio) = readPcmChunkAsMonoFloat(
                    input = input,
                    header = header,
                    range = range,
                    reusableBytes = reusableBytes,
                )
                reusableBytes = nextReusableBytes

                if (chunkAudio.samples.isEmpty()) {
                    Log.w(TAG, "skip empty chunk index=${position + 1}, startMs=${range.startMs}, endMs=${range.endMs}")
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
                    Log.i(
                        TAG,
                        "skip no-speech chunk index=${range.index.takeIf { it > 0 } ?: position + 1}, " +
                            "startMs=${chunkAudio.startMs}, endMs=${chunkAudio.endMs}, " +
                            "durationMs=$chunkDurationMs, speechRatio=${"%.4f".format(Locale.US, vadResult.speechRatio)}"
                    )
                    runOnUiThread {
                        setStatus(
                            "正在识别... 已跳过无人声 ${formatTxtTimestamp(chunkAudio.endMs)} / " +
                                "${formatTxtTimestamp(header.durationMs)}"
                        )
                    }
                    continue
                }

                speechChunks += 1
                speechDurationMs += vadResult.totalSpeechMs
                asrProcessedDurationMs += chunkDurationMs

                Log.i(
                    TAG,
                    "recognize chunk index=${range.index.takeIf { it > 0 } ?: position + 1}, " +
                        "startMs=${chunkAudio.startMs}, " +
                        "endMs=${chunkAudio.endMs}, durationMs=$chunkDurationMs, " +
                        "mode=${range.cutMode}, sampleCount=${chunkAudio.samples.size}, " +
                        "vadSpeechMs=${vadResult.totalSpeechMs}, " +
                        "vadSpeechRatio=${"%.4f".format(Locale.US, vadResult.speechRatio)}, " +
                        "asrInput=full-speechgate-approved-chunk, " +
                        "usedHeap=${currentUsedHeapBytes()}"
                )

                val segmentText = recognizeSamples(
                    currentRecognizer = currentRecognizer,
                    samples = chunkAudio.samples,
                    sampleRate = header.sampleRate,
                    chunkIndex = range.index.takeIf { it > 0 } ?: position + 1,
                    chunkDurationMs = chunkDurationMs,
                    logNativeTimestampDetail = position == 0,
                )
                val segmentIndex = segments.size + 1
                segments += RecognitionSegment(
                    index = segmentIndex,
                    startMs = chunkAudio.startMs,
                    endMs = chunkAudio.endMs,
                    text = segmentText,
                    cutMode = "${range.cutMode}+speechgate-full-chunk",
                    startUs = chunkAudio.startUs,
                    endUs = chunkAudio.endUs,
                )

                Log.i(
                    TAG,
                    "transcript segment index=$segmentIndex, chunk=${range.index.takeIf { it > 0 } ?: position + 1}, " +
                        "relativeStartUs=${chunkAudio.startUs}, relativeEndUs=${chunkAudio.endUs}, " +
                        "timestampSource=${TimestampSource.ESTIMATED}, textLength=${segmentText.length}"
                )
                runOnUiThread {
                    setStatus(
                        "正在识别... 已处理 ${formatTxtTimestamp(chunkAudio.endMs)} / " +
                            "${formatTxtTimestamp(header.durationMs)}"
                    )
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
        Log.i(
            TAG,
            "speech gate summary totalMs=${stats.totalDurationMs}, speechMs=${stats.speechDurationMs}, " +
                "skippedNoSpeechMs=${stats.skippedNoSpeechDurationMs}, totalChunks=${stats.totalChunks}, " +
                "speechChunks=${stats.speechChunks}, skippedChunks=${stats.skippedChunks}, " +
                "asrProcessedMs=${stats.asrProcessedDurationMs}"
        )

        return ChunkRecognitionResult(segments = segments, stats = stats)
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
        val stream = currentRecognizer.createStream()
        try {
            stream.acceptWaveform(samples, sampleRate)
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
            return result.text.trim()
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

            val sentenceTexts = splitTranscriptSentences(segment.text)
                .filter { it.isNotBlank() }
            if (sentenceTexts.isEmpty()) continue

            if (sentenceTexts.size == 1) {
                transcriptSegments += buildTranscriptSegment(
                    relativeStartUs = segment.startUs,
                    relativeEndUs = segment.endUs,
                    text = sentenceTexts.first(),
                    timestampSource = TimestampSource.ESTIMATED,
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
                        timestampSource = TimestampSource.ESTIMATED,
                        recordingTimeContext = recordingTimeContext,
                    )
                }
                cursorUs = endUs
            }
        }

        return transcriptSegments
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

    private fun dedupRecognitionSegments(
        segments: List<RecognitionSegment>,
    ): List<RecognitionSegment> {
        if (segments.isEmpty()) return emptyList()

        val result = mutableListOf<RecognitionSegment>()
        val accumulatedText = StringBuilder()
        var removedChars = 0
        var droppedSegments = 0

        for (segment in segments) {
            val rawText = segment.text.trim()
            if (rawText.isBlank()) {
                droppedSegments += 1
                continue
            }

            val previousSegment = result.lastOrNull()
            val canDedupAgainstPrevious = previousSegment != null &&
                segment.startMs <= previousSegment.endMs + CHUNKED_OVERLAP_MS + 500L
            val cutIndex = if (canDedupAgainstPrevious) {
                findNormalizedPrefixOverlapCutIndex(
                    left = accumulatedText.toString(),
                    right = rawText,
                    maxCheck = DEDUP_MAX_CHECK_CHARS,
                ).coerceIn(0, rawText.length)
            } else {
                0
            }
            val cleanedText = trimLeadingDedupSeparators(rawText.substring(cutIndex))
            removedChars += rawText.length - cleanedText.length

            if (cleanedText.isBlank()) {
                droppedSegments += 1
                Log.i(TAG, "dedup segment dropped index=${segment.index}, reason=overlap-only")
                continue
            }

            result += segment.copy(text = cleanedText)
            accumulatedText.append(cleanedText)
            if (accumulatedText.length > DEDUP_ACCUMULATED_TAIL_CHARS) {
                accumulatedText.delete(0, accumulatedText.length - DEDUP_ACCUMULATED_TAIL_CHARS)
            }
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
                        timeMode = "estimated-from-chunk",
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

        if (rightNorm.length >= minOverlap &&
            rightNorm.length <= maxCheck &&
            leftNorm.contains(rightNorm)
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
            val leftNearTail = leftNorm.takeLast(min(leftNorm.length, max(80, prefixLength * 2)))
            val isExactOverlap = leftSuffix == prefixNorm
            val isFuzzyOverlap = prefixLength >= fuzzyMinOverlap &&
                normalizedSimilarity(leftSuffix, prefixNorm) >= fuzzyThreshold
            val isContainedNearTail = prefixLength >= 12 &&
                leftNearTail.contains(prefixNorm)

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
                mergedText = if (output.stats.hasAnySpeech) output.mergedText else "",
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

    override fun onStart() {
        super.onStart()
        if (!airecReceiverRegistered) {
            val filter = IntentFilter().apply {
                addAction(AirecReceiverService.ACTION_STATE_CHANGED)
                addAction(AirecReceiverService.ACTION_UPLOAD_RECEIVED)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                registerReceiver(airecReceiverUpdates, filter, Context.RECEIVER_NOT_EXPORTED)
            } else {
                @Suppress("DEPRECATION")
                registerReceiver(airecReceiverUpdates, filter)
            }
            airecReceiverRegistered = true
        }
        refreshAirecReceiverUi()
    }

    override fun onStop() {
        if (airecReceiverRegistered) {
            unregisterReceiver(airecReceiverUpdates)
            airecReceiverRegistered = false
        }
        super.onStop()
    }

    override fun onDestroy() {
        recognizer?.release()
        recognizer = null
        super.onDestroy()
    }
}
