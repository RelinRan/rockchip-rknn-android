package androidx.runtime.rknn.tracking

import androidx.runtime.rknn.data.RknnBoundingBox
import androidx.runtime.rknn.data.RknnDetection

/** ByteTrack 风格的高低置信度二阶段目标跟踪器。 */
class RknnTracker(private val config: RknnTrackerConfig = RknnTrackerConfig()) {
    private data class Track(
        val id: Long,
        var detection: RknnDetection,
        var state: RknnTrackState,
        var age: Int = 1,
        var hits: Int = 1,
        var missed: Int = 0,
        var velocityX: Float = 0f,
        var velocityY: Float = 0f,
    )

    private val tracks = mutableListOf<Track>()
    private var nextId = 1L

    @Synchronized
    fun update(detections: List<RknnDetection>): List<RknnTrackedDetection> {
        tracks.forEach { it.age++ }
        val high = detections.filter { score(it) >= config.highScoreThreshold }.toMutableList()
        val low = detections.filter { score(it) in config.lowScoreThreshold..<config.highScoreThreshold }.toMutableList()
        val unmatched = tracks.toMutableList()

        associate(unmatched, high)
        associate(unmatched, low)

        unmatched.forEach { track ->
            track.missed++
            track.state = RknnTrackState.LOST
        }
        high.forEach { detection ->
            tracks += Track(
                id = nextId++,
                detection = detection,
                state = if (config.minConfirmedFrames == 1) RknnTrackState.TRACKED else RknnTrackState.TENTATIVE,
            )
        }
        tracks.removeAll { it.missed > config.maxLostFrames }
        return tracks.map { it.toResult() }
    }

    @Synchronized
    fun reset() {
        tracks.clear()
        nextId = 1L
    }

    private fun associate(unmatched: MutableList<Track>, detections: MutableList<RknnDetection>) {
        while (unmatched.isNotEmpty() && detections.isNotEmpty()) {
            var bestTrack: Track? = null
            var bestDetection: RknnDetection? = null
            var bestScore = Float.NEGATIVE_INFINITY
            unmatched.forEach { track -> detections.forEach { detection ->
                val matchScore = matchScore(track, detection) ?: return@forEach
                if (matchScore > bestScore) {
                    bestScore = matchScore
                    bestTrack = track
                    bestDetection = detection
                }
            } }
            val track = bestTrack ?: break
            val detection = bestDetection ?: break
            val previous = track.detection.boundingBox
            val current = detection.boundingBox
            track.velocityX = current.centerX - previous.centerX
            track.velocityY = current.centerY - previous.centerY
            track.detection = detection
            track.missed = 0
            track.hits++
            track.state = if (track.hits >= config.minConfirmedFrames) RknnTrackState.TRACKED else RknnTrackState.TENTATIVE
            unmatched.remove(track)
            detections.remove(detection)
        }
    }

    private fun Track.toResult() = RknnTrackedDetection(id, detection, state, age, missed)

    private fun score(detection: RknnDetection): Float = detection.categories.maxOfOrNull { it.score } ?: 0f

    private fun matchScore(track: Track, detection: RknnDetection): Float? {
        if (track.detection.categories.firstOrNull()?.index != detection.categories.firstOrNull()?.index) return null
        val predicted = predict(track)
        val overlap = iou(predicted, detection.boundingBox)
        if (overlap >= config.matchIouThreshold) return 2f + overlap

        val dx = detection.boundingBox.centerX - predicted.centerX
        val dy = detection.boundingBox.centerY - predicted.centerY
        val distance = kotlin.math.sqrt(dx * dx + dy * dy)
        val scale = maxOf(
            predicted.width,
            predicted.height,
            detection.boundingBox.width,
            detection.boundingBox.height,
        ).coerceAtLeast(1f)
        val ratio = distance / scale
        return if (ratio <= config.maxCenterDistanceRatio) 1f - ratio / config.maxCenterDistanceRatio else null
    }

    private fun predict(track: Track): RknnBoundingBox {
        val steps = track.missed + 1
        val dx = track.velocityX * steps
        val dy = track.velocityY * steps
        val box = track.detection.boundingBox
        return RknnBoundingBox(box.left + dx, box.top + dy, box.right + dx, box.bottom + dy)
    }

    private fun iou(a: RknnBoundingBox, b: RknnBoundingBox): Float {
        val width = (minOf(a.right, b.right) - maxOf(a.left, b.left)).coerceAtLeast(0f)
        val height = (minOf(a.bottom, b.bottom) - maxOf(a.top, b.top)).coerceAtLeast(0f)
        val intersection = width * height
        return intersection / (a.width * a.height + b.width * b.height - intersection).coerceAtLeast(1e-6f)
    }
}
