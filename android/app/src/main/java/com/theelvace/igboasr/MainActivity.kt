package com.theelvace.igboasr

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.*

class MainActivity : AppCompatActivity() {

    private lateinit var btnRecord: Button
    private lateinit var tvStatus: TextView
    private lateinit var tvTranscript: TextView
    private lateinit var btnClear: Button

    private var ortEnv: OrtEnvironment? = null
    private var encoderSession: OrtSession? = null
    private var crossAttnSession: OrtSession? = null
    private var decoderSession: OrtSession? = null
    private var vocab: Map<Int, String> = emptyMap()
    private var decoderInputNames: List<String> = emptyList()

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private val recordedSamples = mutableListOf<Short>()
    private var isRecording = false

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    private val SAMPLE_RATE = 16000
    private val RECORD_PERMISSION_CODE = 101
    private val N_MELS = 80
    private val N_FRAMES = 3000
    private val NUM_LAYERS = 12
    private val NUM_HEADS = 12
    private val HEAD_DIM = 64
    private val ENC_SEQ = 1500

    private val SOT = 50258L
    // 50325 = <|yo|>, reused as a stand-in since whisper has no igbo token
    private val LANG_TOKEN = 50325L
    private val TRANSCRIBE = 50359L
    private val NO_TIMESTAMPS = 50363L
    private val EOT = 50257L
    private val SPECIAL_TOKENS = setOf(SOT, LANG_TOKEN, TRANSCRIBE, NO_TIMESTAMPS, EOT)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnRecord = findViewById(R.id.btnRecord)
        tvStatus = findViewById(R.id.tvStatus)
        tvTranscript = findViewById(R.id.tvTranscript)
        btnClear = findViewById(R.id.btnClear)

        requestMicPermission()
        loadModels()

        btnRecord.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> startRecording()
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> stopRecordingAndTranscribe()
            }
            true
        }

        btnClear.setOnClickListener {
            tvTranscript.text = ""
            tvStatus.text = "Ready"
        }
    }

    private fun loadModels() {
        tvStatus.text = "Loading models..."
        scope.launch(Dispatchers.IO) {
            try {
                val vocabJson = assets.open("whisper_vocab.json").bufferedReader().readText()
                val jsonObj = JSONObject(vocabJson)
                val map = mutableMapOf<Int, String>()
                jsonObj.keys().forEach { key -> map[key.toInt()] = jsonObj.getString(key) }
                vocab = map

                ortEnv = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions()

                val encoderFile  = File(filesDir, "whisper_encoder_int8.onnx")
                val crossFile    = File(filesDir, "whisper_cross_attn_init_int8.onnx")
                val decoderFile  = File(filesDir, "whisper_decoder_kvcache_int8.onnx")

                if (!encoderFile.exists()) {
                    withContext(Dispatchers.Main) { tvStatus.text = "Copying encoder..." }
                    assets.open("whisper_encoder_int8.onnx").use { i ->
                        encoderFile.outputStream().use { o -> i.copyTo(o) }
                    }
                }
                if (!crossFile.exists()) {
                    withContext(Dispatchers.Main) { tvStatus.text = "Copying cross-attn..." }
                    assets.open("whisper_cross_attn_init_int8.onnx").use { i ->
                        crossFile.outputStream().use { o -> i.copyTo(o) }
                    }
                }
                if (!decoderFile.exists()) {
                    withContext(Dispatchers.Main) { tvStatus.text = "Copying decoder..." }
                    assets.open("whisper_decoder_kvcache_int8.onnx").use { i ->
                        decoderFile.outputStream().use { o -> i.copyTo(o) }
                    }
                }

                withContext(Dispatchers.Main) { tvStatus.text = "Loading encoder..." }
                encoderSession = ortEnv!!.createSession(encoderFile.absolutePath, opts)

                withContext(Dispatchers.Main) { tvStatus.text = "Loading cross-attn..." }
                crossAttnSession = ortEnv!!.createSession(crossFile.absolutePath, opts)

                withContext(Dispatchers.Main) { tvStatus.text = "Loading decoder..." }
                decoderSession = ortEnv!!.createSession(decoderFile.absolutePath, opts)
                decoderInputNames = decoderSession!!.inputNames.toList()

                withContext(Dispatchers.Main) {
                    tvStatus.text = "Ready — hold button to speak"
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) return

        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize * 4
        )
        recordedSamples.clear()
        isRecording = true
        audioRecord?.startRecording()

        recordingThread = Thread {
            val buffer = ShortArray(1024)
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (read > 0) {
                    synchronized(recordedSamples) {
                        recordedSamples.addAll(buffer.take(read).toList())
                    }
                }
            }
        }
        recordingThread?.start()

        tvStatus.text = "Recording..."
        btnRecord.backgroundTintList = android.content.res.ColorStateList.valueOf(
            android.graphics.Color.parseColor("#FF4444")
        )
    }

    private fun stopRecordingAndTranscribe() {
        btnRecord.backgroundTintList = android.content.res.ColorStateList.valueOf(
            android.graphics.Color.parseColor("#1DB954")
        )
        tvStatus.text = "Transcribing..."

        isRecording = false
        recordingThread?.join()
        recordingThread = null

        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        val samplesSnapshot: List<Short>
        synchronized(recordedSamples) {
            samplesSnapshot = recordedSamples.toList()
        }

        val audioFloats = FloatArray(samplesSnapshot.size) { samplesSnapshot[it] / 32768.0f }

        Thread {
            try {
                val serviceIntent = Intent(this, InferenceService::class.java)
                startForegroundService(serviceIntent)

                val transcript = transcribe(audioFloats)

                stopService(serviceIntent)

                runOnUiThread {
                    if (transcript.isNotEmpty()) tvTranscript.append(transcript + " ")
                    tvStatus.text = "Done — hold to speak again"
                }
            } catch (e: Exception) {
                runOnUiThread { tvStatus.text = "Error: ${e.message}" }
            }
        }.start()
    }

    private fun transcribe(audio: FloatArray): String {
        val env = ortEnv ?: throw IllegalStateException("ORT not initialized")
        val enc = encoderSession ?: throw IllegalStateException("Encoder not loaded")
        val cross = crossAttnSession ?: throw IllegalStateException("Cross-attn not loaded")
        val dec = decoderSession ?: throw IllegalStateException("Decoder not loaded")

        val melStart = System.nanoTime()

        val melFeatures = getMelFromServer(audio)

        val melEnd = System.nanoTime()
        android.util.Log.d("BENCHMARK", "Mel extraction: ${(melEnd - melStart) / 1_000_000}ms")
        val inputTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(melFeatures),
            longArrayOf(1, N_MELS.toLong(), N_FRAMES.toLong())
        )

        val encStart = System.nanoTime()
        val encOut = enc.run(mapOf("input_features" to inputTensor))
        val encoderHidden = encOut[0].value as Array<Array<FloatArray>>
        inputTensor.close()
        encOut.close()
        val encEnd = System.nanoTime()
        android.util.Log.d("BENCHMARK", "Encoder inference: ${(encEnd - encStart) / 1_000_000}ms")

        // flatten encoder output to row-major for the cross-attn session
        val encFlat = FloatArray(ENC_SEQ * 768)
        for (t in 0 until ENC_SEQ)
            for (d in 0 until 768)
                encFlat[t * 768 + d] = encoderHidden[0][t][d]

        val encTensorForCross = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(encFlat), longArrayOf(1, ENC_SEQ.toLong(), 768)
        )
        val crossStart = System.nanoTime()
        val crossOut = cross.run(mapOf("encoder_hidden_states" to encTensorForCross))
        encTensorForCross.close()
        val crossEnd = System.nanoTime()
        android.util.Log.d("BENCHMARK", "Cross-attn init: ${(crossEnd - crossStart) / 1_000_000}ms")

        val crossKCache = Array(NUM_LAYERS) { FloatArray(NUM_HEADS * ENC_SEQ * HEAD_DIM) }
        val crossVCache = Array(NUM_LAYERS) { FloatArray(NUM_HEADS * ENC_SEQ * HEAD_DIM) }
        for (i in 0 until NUM_LAYERS) {
            val ckArr = crossOut[i * 2].value as Array<Array<Array<FloatArray>>>
            val cvArr = crossOut[i * 2 + 1].value as Array<Array<Array<FloatArray>>>
            for (h in 0 until NUM_HEADS)
                for (s in 0 until ENC_SEQ)
                    for (d in 0 until HEAD_DIM) {
                        crossKCache[i][h * ENC_SEQ * HEAD_DIM + s * HEAD_DIM + d] = ckArr[0][h][s][d]
                        crossVCache[i][h * ENC_SEQ * HEAD_DIM + s * HEAD_DIM + d] = cvArr[0][h][s][d]
                    }
        }
        crossOut.close()

        val selfKCache = Array(NUM_LAYERS) { FloatArray(0) }
        val selfVCache = Array(NUM_LAYERS) { FloatArray(0) }
        var selfCacheSeqLen = 0

        val prefix = longArrayOf(SOT, LANG_TOKEN, TRANSCRIBE, NO_TIMESTAMPS)
        val tokenIds = mutableListOf<Long>()
        val recentTokens = mutableListOf<Long>()
        val maxNewTokens = 20
        val decoderStepTimes = mutableListOf<Long>()

        for (stepIdx in 0 until (prefix.size + maxNewTokens)) {
            val stepStart = System.nanoTime()
            val inputToken = if (stepIdx < prefix.size) prefix[stepIdx]
            else tokenIds.lastOrNull() ?: break

            if (stepIdx >= prefix.size && (tokenIds.isEmpty() || tokenIds.last() == EOT)) break

            val feeds = mutableMapOf<String, OnnxTensor>()

            feeds["input_ids"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(inputToken)), longArrayOf(1, 1)
            )

            for (i in 0 until NUM_LAYERS) {
                feeds["past_self_k_$i"] = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(selfKCache[i]),
                    longArrayOf(1, NUM_HEADS.toLong(), selfCacheSeqLen.toLong(), HEAD_DIM.toLong())
                )
                feeds["past_self_v_$i"] = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(selfVCache[i]),
                    longArrayOf(1, NUM_HEADS.toLong(), selfCacheSeqLen.toLong(), HEAD_DIM.toLong())
                )
            }

            for (i in 0 until NUM_LAYERS) {
                feeds["past_cross_k_$i"] = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(crossKCache[i]),
                    longArrayOf(1, NUM_HEADS.toLong(), ENC_SEQ.toLong(), HEAD_DIM.toLong())
                )
                feeds["past_cross_v_$i"] = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(crossVCache[i]),
                    longArrayOf(1, NUM_HEADS.toLong(), ENC_SEQ.toLong(), HEAD_DIM.toLong())
                )
            }

            val decOut = dec.run(feeds)
            val logits = decOut[0].value as Array<Array<FloatArray>>
            val lastLogits = logits[0][0]

            val newSeqLen = selfCacheSeqLen + 1
            // decoder outputs: logits, then present self_k/self_v per layer
            for (i in 0 until NUM_LAYERS) {
                val baseIdx = 1 + i * 2
                val skArr = decOut[baseIdx].value as Array<Array<Array<FloatArray>>>
                val newSK = FloatArray(NUM_HEADS * newSeqLen * HEAD_DIM)
                for (h in 0 until NUM_HEADS)
                    for (s in 0 until newSeqLen)
                        for (d in 0 until HEAD_DIM)
                            newSK[h * newSeqLen * HEAD_DIM + s * HEAD_DIM + d] = skArr[0][h][s][d]
                selfKCache[i] = newSK

                val svArr = decOut[baseIdx + 1].value as Array<Array<Array<FloatArray>>>
                val newSV = FloatArray(NUM_HEADS * newSeqLen * HEAD_DIM)
                for (h in 0 until NUM_HEADS)
                    for (s in 0 until newSeqLen)
                        for (d in 0 until HEAD_DIM)
                            newSV[h * newSeqLen * HEAD_DIM + s * HEAD_DIM + d] = svArr[0][h][s][d]
                selfVCache[i] = newSV
            }

            selfCacheSeqLen++
            feeds.values.forEach { it.close() }
            decOut.close()

            val stepEnd = System.nanoTime()
            decoderStepTimes.add((stepEnd - stepStart) / 1_000_000)

            // only sample once the forced prefix has been fed
            if (stepIdx >= prefix.size - 1) {
                val suppressedLogits = lastLogits.copyOf()

                // suppress recently emitted tokens to curb repetition loops
                recentTokens.takeLast(16).forEach { tok ->
                    if (tok < suppressedLogits.size) suppressedLogits[tok.toInt()] = -1e9f
                }

                val nextToken = suppressedLogits.indices
                    .maxByOrNull { suppressedLogits[it] }?.toLong() ?: EOT
                tokenIds.add(nextToken)
                recentTokens.add(nextToken)

                // bail out if the last 4 tokens repeat the previous 4
                if (tokenIds.size >= 8) {
                    val last4 = tokenIds.takeLast(4)
                    val prev4 = tokenIds.dropLast(4).takeLast(4)
                    if (last4 == prev4) break
                }

                if (nextToken == EOT) break
            }
        }

        val totalDecoderTime = decoderStepTimes.sum()
        val avgStepTime = if (decoderStepTimes.isNotEmpty()) totalDecoderTime / decoderStepTimes.size else 0
        val tokensGenerated = tokenIds.size
        val tokensPerSecond = if (totalDecoderTime > 0) (tokensGenerated * 1000.0 / totalDecoderTime) else 0.0

        android.util.Log.d("BENCHMARK", "Decoder total: ${totalDecoderTime}ms across ${decoderStepTimes.size} steps")
        android.util.Log.d("BENCHMARK", "Decoder avg per-step: ${avgStepTime}ms")
        android.util.Log.d("BENCHMARK", "Tokens generated: $tokensGenerated")
        android.util.Log.d("BENCHMARK", "Tokens/sec: ${String.format("%.2f", tokensPerSecond)}")

        return tokenIds
            .filter { it !in SPECIAL_TOKENS }
            .mapNotNull { vocab[it.toInt()] }
            .joinToString("")
            .trim()
    }

    private fun computeLogMel(audio: FloatArray): FloatArray {
        val nFft = 400
        val hopLength = 160
        val target = SAMPLE_RATE * 30
        val padded = FloatArray(target)
        audio.copyInto(padded, 0, 0, minOf(audio.size, target))

        val window = FloatArray(nFft) { i ->
            (0.5 * (1.0 - cos(2.0 * PI * i / nFft))).toFloat()
        }

        val melFilters = buildMelFilterbank(N_MELS, nFft, SAMPLE_RATE)
        val numFrames = (target - nFft) / hopLength + 1
        val melSpec = Array(N_MELS) { FloatArray(numFrames) }

        for (frame in 0 until numFrames) {
            val start = frame * hopLength
            val windowed = FloatArray(nFft) { i -> padded[start + i] * window[i] }
            val fftMag = fftMagnitude(windowed)
            for (mel in 0 until N_MELS) {
                var sum = 0f
                for (f in fftMag.indices) sum += melFilters[mel][f] * fftMag[f]
                melSpec[mel][frame] = max(sum, 1e-10f)
            }
        }

        val logMel = FloatArray(N_MELS * N_FRAMES)
        var maxVal = Float.NEGATIVE_INFINITY
        for (mel in 0 until N_MELS) {
            for (f in 0 until minOf(numFrames, N_FRAMES)) {
                val v = log10(melSpec[mel][f])
                logMel[mel * N_FRAMES + f] = v
                if (v > maxVal) maxVal = v
            }
        }
        for (i in logMel.indices) {
            logMel[i] = max(logMel[i], maxVal - 8.0f)
            logMel[i] = (logMel[i] + 4.0f) / 4.0f
        }
        return logMel
    }

    private fun getMelFromServer(audio: FloatArray): FloatArray {
        val serverUrl = "http://192.168.18.5:8765/mel"

        val pcmBytes = java.io.ByteArrayOutputStream()
        val buf = java.nio.ByteBuffer.allocate(2)
        buf.order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (sample in audio) {
            buf.clear()
            buf.putShort((sample * 32768).toInt().toShort())
            pcmBytes.write(buf.array())
        }
        val pcmData = pcmBytes.toByteArray()

        // POST to server
        val url = java.net.URL(serverUrl)
        val conn = url.openConnection() as java.net.HttpURLConnection
        conn.requestMethod = "POST"
        conn.doOutput = true
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=boundary123")
        conn.connectTimeout = 10000
        conn.readTimeout = 30000

        val body = buildMultipart(pcmData)
        conn.setRequestProperty("Content-Length", body.size.toString())
        conn.outputStream.write(body)

        val responseBytes = conn.inputStream.readBytes()
        conn.disconnect()

        val floatBuf = java.nio.ByteBuffer.wrap(responseBytes)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
        val result = FloatArray(80 * 3000)
        floatBuf.get(result)
        return result
    }

    private fun buildMultipart(pcmData: ByteArray): ByteArray {
        val boundary = "boundary123"
        val out = java.io.ByteArrayOutputStream()
        val header = "--$boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.pcm\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        out.write(header.toByteArray())
        out.write(pcmData)
        out.write("\r\n--$boundary--\r\n".toByteArray())
        return out.toByteArray()
    }

    private fun buildMelFilterbank(nMels: Int, nFft: Int, sr: Int): Array<FloatArray> {
        val nFreqs = nFft / 2 + 1
        fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
        fun melToHz(mel: Double) = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
        val melMin = hzToMel(0.0)
        val melMax = hzToMel(sr / 2.0)
        val melPoints = DoubleArray(nMels + 2) { i ->
            melToHz(melMin + i * (melMax - melMin) / (nMels + 1))
        }
        val freqBins = DoubleArray(nFreqs) { i -> i * sr.toDouble() / nFft }
        val filters = Array(nMels) { FloatArray(nFreqs) }
        for (m in 0 until nMels) {
            val lo = melPoints[m]; val center = melPoints[m + 1]; val hi = melPoints[m + 2]
            for (f in 0 until nFreqs) {
                val freq = freqBins[f]
                filters[m][f] = when {
                    freq < lo || freq > hi -> 0f
                    freq <= center -> ((freq - lo) / (center - lo)).toFloat()
                    else -> ((hi - freq) / (hi - center)).toFloat()
                }
            }
        }
        return filters
    }

    private fun fftMagnitude(signal: FloatArray): FloatArray {
        var n = 1
        while (n < signal.size) n = n shl 1
        val real = DoubleArray(n) { if (it < signal.size) signal[it].toDouble() else 0.0 }
        val imag = DoubleArray(n)
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val tr = real[i]; real[i] = real[j]; real[j] = tr
                val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
            }
        }
        var mlen = 2
        while (mlen <= n) {
            val angle = -2.0 * PI / mlen
            val wr = cos(angle); val wi = sin(angle)
            var i = 0
            while (i < n) {
                var cr = 1.0; var ci = 0.0
                for (k in 0 until mlen / 2) {
                    val ur = real[i+k]; val ui = imag[i+k]
                    val vr = real[i+k+mlen/2]*cr - imag[i+k+mlen/2]*ci
                    val vi = real[i+k+mlen/2]*ci + imag[i+k+mlen/2]*cr
                    real[i+k] = ur+vr; imag[i+k] = ui+vi
                    real[i+k+mlen/2] = ur-vr; imag[i+k+mlen/2] = ui-vi
                    val ncr = cr*wr - ci*wi; ci = cr*wi + ci*wr; cr = ncr
                }
                i += mlen
            }
            mlen = mlen shl 1
        }
        val nFreqs = signal.size / 2 + 1
        return FloatArray(nFreqs) { i -> sqrt(real[i]*real[i] + imag[i]*imag[i]).toFloat() }
    }

    private fun requestMicPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_PERMISSION_CODE
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        isRecording = false
        encoderSession?.close()
        crossAttnSession?.close()
        decoderSession?.close()
        ortEnv?.close()
    }
}