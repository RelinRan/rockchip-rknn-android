package androidx.runtime.rknn.internal

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.runtime.rknn.data.RknnImage
import kotlin.math.roundToInt

/**
 * Provides the `ImagePreprocessor` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `ImagePreprocessor` where its surrounding API requires this contract.
 */
internal object ImagePreprocessor {

    /**
     * Executes `toRgbImage` for the RKNN runtime contract.
     * @param bitmap Source bitmap to preprocess and run through the model.
     * @param targetWidth Requested output width in pixels.
     * @param targetHeight Requested output height in pixels.
     */
    fun toRgbImage(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): RknnImage {
        require(targetWidth > 0 && targetHeight > 0) { "Invalid model input size" }
        val scale = minOf(targetWidth.toFloat() / bitmap.width, targetHeight.toFloat() / bitmap.height)
        val scaledWidth = (bitmap.width * scale).roundToInt().coerceAtLeast(1)
        val scaledHeight = (bitmap.height * scale).roundToInt().coerceAtLeast(1)
        val resized = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        val paddingLeft = (targetWidth - scaledWidth) / 2
        val paddingTop = (targetHeight - scaledHeight) / 2
        val letterboxed = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        Canvas(letterboxed).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(resized, paddingLeft.toFloat(), paddingTop.toFloat(), Paint(Paint.FILTER_BITMAP_FLAG))
        }
        if (resized !== bitmap && !resized.isRecycled) resized.recycle()
        val bytes = bitmapToRgbBytes(letterboxed)
        if (!letterboxed.isRecycled) letterboxed.recycle()
        return RknnImage(
            width = targetWidth,
            height = targetHeight,
            channels = 3,
            bytes = bytes,
            originalWidth = bitmap.width,
            originalHeight = bitmap.height,
            scale = scale,
            paddingLeft = paddingLeft,
            paddingTop = paddingTop,
        )
    }

    private fun bitmapToRgbBytes(bitmap: Bitmap): ByteArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val out = ByteArray(bitmap.width * bitmap.height * 3)
        var offset = 0
        for (pixel in pixels) {
            out[offset++] = ((pixel shr 16) and 0xFF).toByte()
            out[offset++] = ((pixel shr 8) and 0xFF).toByte()
            out[offset++] = (pixel and 0xFF).toByte()
        }
        return out
    }
}
