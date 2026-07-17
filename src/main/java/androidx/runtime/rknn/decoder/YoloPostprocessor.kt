package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnBoundingBox
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.data.RknnKeypoint
import androidx.runtime.rknn.data.RknnSegmentationMask
import androidx.runtime.rknn.data.RknnOrientedBox
import androidx.runtime.rknn.data.RknnPoint
import kotlin.math.abs

/**
 * Provides the `YoloCandidate` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `YoloCandidate` where its surrounding API requires this contract.
 */
internal data class YoloCandidate(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val classId: Int,
    val score: Float,
    val keyPoints: List<RknnKeypoint> = emptyList(),
    val categoryScores: List<Pair<Int, Float>> = listOf(classId to score),
    val segmentationMask: RknnSegmentationMask? = null,
    val orientedBox: RknnOrientedBox? = null,
)

/**
 * Provides the `YoloPostprocessor` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `YoloPostprocessor` where its surrounding API requires this contract.
 */
internal object YoloPostprocessor {
    /**
     * Executes `process` for the RKNN runtime contract.
     * @param candidates Value supplied for `candidates`.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param config Model or runtime configuration used by the operation.
     * @param applyNms Value supplied for `applyNms`.
     */
    fun process(
        candidates: List<YoloCandidate>,
        image: RknnImage,
        config: RknnModelConfig,
        applyNms: Boolean,
    ): List<RknnDetection> {
        val valid = candidates.asSequence()
            .filter { it.classId in config.labels.indices && it.score.isFinite() && it.score >= config.scoreThreshold }
            .filter { listOf(it.left, it.top, it.right, it.bottom).all(Float::isFinite) }
            .filter { it.right > it.left && it.bottom > it.top }
            .sortedByDescending { it.score }
            .toList()
        val selected = if (!applyNms) valid else {
            val kept = ArrayList<YoloCandidate>()
            valid.forEach { candidate ->
                if (kept.none { it.classId == candidate.classId && iou(it, candidate) > config.nmsThreshold }) {
                    kept += candidate
                }
            }
            kept
        }
        return selected.take(config.maxResults).mapNotNull { candidate ->
            val mappedOrientedBox = candidate.orientedBox?.let { oriented ->
                RknnOrientedBox(
                    centerX = unmap(oriented.centerX, image.paddingLeft, image.scale, image.originalWidth),
                    centerY = unmap(oriented.centerY, image.paddingTop, image.scale, image.originalHeight),
                    width = oriented.width / image.scale,
                    height = oriented.height / image.scale,
                    rotationRadians = oriented.rotationRadians,
                )
            }
            val box = mappedOrientedBox?.boundingBox() ?: RknnBoundingBox(
                left = unmap(candidate.left, image.paddingLeft, image.scale, image.originalWidth),
                top = unmap(candidate.top, image.paddingTop, image.scale, image.originalHeight),
                right = unmap(candidate.right, image.paddingLeft, image.scale, image.originalWidth),
                bottom = unmap(candidate.bottom, image.paddingTop, image.scale, image.originalHeight),
            )
            if (box.width <= 0f || box.height <= 0f) return@mapNotNull null
            RknnDetection(
                categories = candidate.categoryScores.map { (classId, score) ->
                    RknnCategory(classId, config.labels[classId], score)
                },
                boundingBox = box,
                keyPoints = candidate.keyPoints.map { point ->
                    point.copy(
                        x = unmap(point.x, image.paddingLeft, image.scale, image.originalWidth),
                        y = unmap(point.y, image.paddingTop, image.scale, image.originalHeight),
                    )
                },
                segmentationMask = candidate.segmentationMask,
                orientedBox = mappedOrientedBox,
            )
        }
    }

    private fun unmap(value: Float, padding: Int, scale: Float, limit: Int): Float =
        ((value - padding) / scale).coerceIn(0f, limit.toFloat())

    private fun iou(a: YoloCandidate, b: YoloCandidate): Float {
        if (a.orientedBox != null && b.orientedBox != null) {
            return orientedIou(a.orientedBox, b.orientedBox)
        }
        val width = (minOf(a.right, b.right) - maxOf(a.left, b.left)).coerceAtLeast(0f)
        val height = (minOf(a.bottom, b.bottom) - maxOf(a.top, b.top)).coerceAtLeast(0f)
        val intersection = width * height
        val union = (a.right - a.left) * (a.bottom - a.top) +
            (b.right - b.left) * (b.bottom - b.top) - intersection
        return intersection / union.coerceAtLeast(1e-6f)
    }

    private fun orientedIou(a: RknnOrientedBox, b: RknnOrientedBox): Float {
        val intersectionPolygon = clipPolygon(a.corners, b.corners)
        val intersection = polygonArea(intersectionPolygon)
        val union = a.width * a.height + b.width * b.height - intersection
        return intersection / union.coerceAtLeast(1e-6f)
    }

    private fun clipPolygon(subject: List<RknnPoint>, clip: List<RknnPoint>): List<RknnPoint> {
        var output = subject
        clip.indices.forEach { index ->
            val edgeStart = clip[index]
            val edgeEnd = clip[(index + 1) % clip.size]
            val input = output
            output = mutableListOf()
            if (input.isEmpty()) return@forEach
            var previous = input.last()
            input.forEach { current ->
                val currentInside = isInside(current, edgeStart, edgeEnd)
                val previousInside = isInside(previous, edgeStart, edgeEnd)
                when {
                    currentInside && !previousInside -> {
                        lineIntersection(previous, current, edgeStart, edgeEnd)?.let { output += it }
                        output += current
                    }
                    currentInside -> output += current
                    previousInside -> lineIntersection(previous, current, edgeStart, edgeEnd)?.let { output += it }
                }
                previous = current
            }
        }
        return output
    }

    private fun isInside(point: RknnPoint, start: RknnPoint, end: RknnPoint): Boolean =
        (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (point.x - start.x) >= -1e-5f

    private fun lineIntersection(
        a: RknnPoint,
        b: RknnPoint,
        c: RknnPoint,
        d: RknnPoint,
    ): RknnPoint? {
        val abX = b.x - a.x
        val abY = b.y - a.y
        val cdX = d.x - c.x
        val cdY = d.y - c.y
        val denominator = abX * cdY - abY * cdX
        if (abs(denominator) < 1e-6f) return null
        val t = ((c.x - a.x) * cdY - (c.y - a.y) * cdX) / denominator
        return RknnPoint(a.x + t * abX, a.y + t * abY)
    }

    private fun polygonArea(points: List<RknnPoint>): Float {
        if (points.size < 3) return 0f
        var sum = 0f
        points.indices.forEach { index ->
            val next = (index + 1) % points.size
            sum += points[index].x * points[next].y - points[next].x * points[index].y
        }
        return abs(sum) / 2f
    }
}
