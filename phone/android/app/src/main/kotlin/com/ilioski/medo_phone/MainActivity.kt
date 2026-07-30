package com.ilioski.medo_phone

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.net.Uri
import android.provider.MediaStore
import android.util.Base64
import androidx.core.content.FileProvider
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * Native device features for the MEDO phone skills, exposed over one
 * MethodChannel ("medo/native"). Everything is best-effort: on any failure a
 * handler replies false/null so the Dart skill can speak a graceful message.
 */
class MainActivity : FlutterActivity() {
    private val channel = "medo/native"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channel)
            .setMethodCallHandler { call, result ->
                try {
                    when (call.method) {
                        "flashlight" -> result.success(setFlashlight(call.argument("on") ?: false))
                        "volume" -> result.success(
                            changeVolume(call.argument("set"), call.argument("step") ?: 0)
                        )
                        "openApp" -> result.success(openApp(call.argument("query") ?: ""))
                        "dial" -> result.success(launch(Intent.ACTION_DIAL, "tel:", call.argument("value") ?: ""))
                        "openUrl" -> result.success(launch(Intent.ACTION_VIEW, "", call.argument("value") ?: ""))
                        "sms" -> result.success(sms(call.argument("number") ?: "", call.argument("body") ?: ""))
                        "captureImage" -> captureImage(result)
                        else -> result.notImplemented()
                    }
                } catch (e: Exception) {
                    result.success(if (call.method == "openApp") null else false)
                }
            }
    }

    private fun setFlashlight(on: Boolean): Boolean {
        val cm = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val id = cm.cameraIdList.firstOrNull { camId ->
            cm.getCameraCharacteristics(camId)
                .get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
        } ?: return false
        cm.setTorchMode(id, on)
        return true
    }

    private fun changeVolume(setPercent: Int?, step: Int): Boolean {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val stream = AudioManager.STREAM_MUSIC
        val max = am.getStreamMaxVolume(stream)
        when {
            setPercent != null -> {
                val target = setPercent.coerceIn(0, 100) * max / 100
                am.setStreamVolume(stream, target, AudioManager.FLAG_SHOW_UI)
            }
            step > 0 -> am.adjustStreamVolume(stream, AudioManager.ADJUST_RAISE, AudioManager.FLAG_SHOW_UI)
            step < 0 -> am.adjustStreamVolume(stream, AudioManager.ADJUST_LOWER, AudioManager.FLAG_SHOW_UI)
        }
        return true
    }

    /** Fuzzy-launch an installed app by (partial) label. Returns the label. */
    private fun openApp(query: String): String? {
        val q = query.trim().lowercase()
        if (q.isEmpty()) return null
        val pm = packageManager
        val main = Intent(Intent.ACTION_MAIN, null).addCategory(Intent.CATEGORY_LAUNCHER)
        val apps = pm.queryIntentActivities(main, 0)
        val match = apps.firstOrNull { it.loadLabel(pm).toString().lowercase() == q }
            ?: apps.firstOrNull { it.loadLabel(pm).toString().lowercase().contains(q) }
            ?: return null
        val pkg = match.activityInfo.packageName
        val launch = pm.getLaunchIntentForPackage(pkg) ?: return null
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        startActivity(launch)
        return match.loadLabel(pm).toString()
    }

    private fun launch(action: String, scheme: String, value: String): Boolean {
        val uri = if (scheme.isEmpty()) Uri.parse(value) else Uri.parse(scheme + value)
        val intent = Intent(action, uri).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return if (intent.resolveActivity(packageManager) != null) {
            startActivity(intent); true
        } else false
    }

    private fun sms(number: String, body: String): Boolean {
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$number"))
            .putExtra("sms_body", body)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return if (intent.resolveActivity(packageManager) != null) {
            startActivity(intent); true
        } else false
    }

    // --- camera: capture one photo and hand back a base64 JPEG ---------------

    private val reqImageCapture = 4210
    private var captureResult: MethodChannel.Result? = null
    private var captureFile: File? = null

    /** Launch the system camera into a FileProvider URI; reply on result. */
    private fun captureImage(result: MethodChannel.Result) {
        if (captureResult != null) { // a capture is already in flight
            result.success(null)
            return
        }
        try {
            val file = File.createTempFile("medo_cam_", ".jpg", cacheDir)
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                .putExtra(MediaStore.EXTRA_OUTPUT, uri)
                .addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            if (intent.resolveActivity(packageManager) == null) {
                file.delete()
                result.success(null)
                return
            }
            captureFile = file
            captureResult = result
            startActivityForResult(intent, reqImageCapture)
        } catch (e: Exception) {
            result.success(null)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != reqImageCapture) return
        val result = captureResult
        val file = captureFile
        captureResult = null
        captureFile = null
        val encoded =
            if (resultCode == RESULT_OK && file != null) encodeJpeg(file) else null
        file?.delete()
        result?.success(encoded)
    }

    /** Decode, downscale to <=1024 px, and base64-encode as JPEG for the model. */
    private fun encodeJpeg(file: File): String? {
        return try {
            var bmp = BitmapFactory.decodeFile(file.absolutePath) ?: return null
            val longest = maxOf(bmp.width, bmp.height)
            if (longest > 1024) {
                val scale = 1024f / longest
                bmp = Bitmap.createScaledBitmap(
                    bmp, (bmp.width * scale).toInt(), (bmp.height * scale).toInt(), true
                )
            }
            val out = ByteArrayOutputStream()
            bmp.compress(Bitmap.CompressFormat.JPEG, 85, out)
            Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
        } catch (e: Exception) {
            null
        }
    }
}
