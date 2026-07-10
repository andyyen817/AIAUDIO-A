package com.threemountain.lightasr

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import java.util.concurrent.ConcurrentHashMap

class TranscriptionForegroundService : Service() {
    private var currentJobId: String? = null

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            TranscriptionJobRepository.markInterruptedJobs(this)
            stopSelf()
            return START_NOT_STICKY
        }

        val jobId = intent.getStringExtra(EXTRA_JOB_ID)
        when (intent.action) {
            ACTION_START -> {
                currentJobId = jobId
                activeJobId = jobId
                isRunning = true
                if (jobId != null) cancellationRequests.remove(jobId)
                startForeground(
                    NOTIFICATION_ID,
                    notification(
                        title = intent.getStringExtra(EXTRA_TITLE) ?: "LightASR 正在处理录音",
                        message = intent.getStringExtra(EXTRA_MESSAGE) ?: "正在准备...",
                        progress = intent.getIntExtra(EXTRA_PROGRESS, 0),
                        jobId = jobId,
                    )
                )
            }
            ACTION_PROGRESS -> {
                currentJobId = jobId ?: currentJobId
                val manager = getSystemService(NotificationManager::class.java)
                manager.notify(
                    NOTIFICATION_ID,
                    notification(
                        title = intent.getStringExtra(EXTRA_TITLE) ?: "LightASR 正在处理录音",
                        message = intent.getStringExtra(EXTRA_MESSAGE) ?: "处理中",
                        progress = intent.getIntExtra(EXTRA_PROGRESS, 0),
                        jobId = currentJobId,
                    )
                )
                currentJobId?.let { broadcastUpdate(it) }
            }
            ACTION_CANCEL -> {
                if (jobId != null) {
                    cancellationRequests += jobId
                    TranscriptionJobRepository.update(this, jobId) {
                        it.copy(status = TranscriptionJobStatus.CANCELED, errorMessage = "用户已取消任务")
                    }
                    broadcastUpdate(jobId)
                }
            }
            ACTION_STOP -> {
                if (jobId != null) cancellationRequests.remove(jobId)
                if (jobId != null) broadcastUpdate(jobId)
                activeJobId = null
                isRunning = false
                stopForegroundCompat()
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        isRunning = false
        activeJobId = null
        super.onDestroy()
    }

    private fun notification(title: String, message: String, progress: Int, jobId: String?): Notification {
        val openIntent = Intent(this, MainActivity::class.java)
        val openPendingIntent = PendingIntent.getActivity(
            this,
            21001,
            openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }
        builder
            .setContentTitle(title)
            .setContentText(message)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openPendingIntent)
            .setProgress(100, progress.coerceIn(0, 100), false)

        if (jobId != null) {
            val cancelIntent = Intent(this, TranscriptionForegroundService::class.java)
                .setAction(ACTION_CANCEL)
                .putExtra(EXTRA_JOB_ID, jobId)
            val cancelPendingIntent = PendingIntent.getService(
                this,
                21002,
                cancelIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
            builder.addAction(
                Notification.Action.Builder(
                    null,
                    "取消",
                    cancelPendingIntent,
                ).build()
            )
        }
        return builder.build()
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "录音分析任务",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "显示长录音分析和上传进度"
            }
        )
    }

    private fun stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
    }

    private fun broadcastUpdate(jobId: String) {
        sendBroadcast(
            Intent(ACTION_JOB_UPDATED)
                .setPackage(packageName)
                .putExtra(EXTRA_JOB_ID, jobId)
        )
    }

    companion object {
        const val ACTION_JOB_UPDATED = "com.threemountain.lightasr.TRANSCRIPTION_JOB_UPDATED"
        private const val ACTION_START = "com.threemountain.lightasr.TRANSCRIPTION_START"
        private const val ACTION_PROGRESS = "com.threemountain.lightasr.TRANSCRIPTION_PROGRESS"
        private const val ACTION_CANCEL = "com.threemountain.lightasr.TRANSCRIPTION_CANCEL"
        private const val ACTION_STOP = "com.threemountain.lightasr.TRANSCRIPTION_STOP"
        private const val EXTRA_JOB_ID = "jobId"
        private const val EXTRA_TITLE = "title"
        private const val EXTRA_MESSAGE = "message"
        private const val EXTRA_PROGRESS = "progress"
        private const val CHANNEL_ID = "lightasr_transcription_tasks"
        private const val NOTIFICATION_ID = 21000
        private val cancellationRequests = ConcurrentHashMap.newKeySet<String>()
        @Volatile var isRunning: Boolean = false
            private set
        @Volatile var activeJobId: String? = null
            private set

        fun start(context: Context, jobId: String, title: String, message: String) {
            val intent = Intent(context, TranscriptionForegroundService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_JOB_ID, jobId)
                .putExtra(EXTRA_TITLE, title)
                .putExtra(EXTRA_MESSAGE, message)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun progress(context: Context, jobId: String, title: String, message: String, progress: Int) {
            context.startService(
                Intent(context, TranscriptionForegroundService::class.java)
                    .setAction(ACTION_PROGRESS)
                    .putExtra(EXTRA_JOB_ID, jobId)
                    .putExtra(EXTRA_TITLE, title)
                    .putExtra(EXTRA_MESSAGE, message)
                    .putExtra(EXTRA_PROGRESS, progress)
            )
        }

        fun stop(context: Context, jobId: String) {
            context.startService(
                Intent(context, TranscriptionForegroundService::class.java)
                    .setAction(ACTION_STOP)
                    .putExtra(EXTRA_JOB_ID, jobId)
            )
        }

        fun isCancellationRequested(jobId: String?): Boolean =
            jobId != null && cancellationRequests.contains(jobId)
    }
}

class TranscriptionCanceledException : IllegalStateException("任务已取消")
