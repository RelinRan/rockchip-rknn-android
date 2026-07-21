package androidx.runtime.rknn.tracking

import androidx.runtime.rknn.data.RknnBoundingBox
import androidx.runtime.rknn.data.RknnDetection

/**
 * ByteTrack-style tracker with high- and low-confidence association stages.
 *
 * Feed detections in frame order and reset when the video source changes:
 * ```kotlin
 * val tracker = RknnTracker()
 * val tracked = tracker.update(result.detections)
 * tracker.reset()
 * ```
 *
 * @param config Association thresholds and track lifecycle limits.
 */
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
        var embedding: FloatArray? = null,
    )

    private val tracks = mutableListOf<Track>()
    private var nextId = 1L

    /**
     * Associates detections from the next video frame with existing tracks.
     *
     * @param detections Detections from one frame in source image coordinates.
     * @return Current tracks, including temporarily lost tracks.
     */
    @Synchronized
    fun update(detections: List<RknnDetection>): List<RknnTrackedDetection> =
        updateInputs(detections.map(::RknnTrackingInput))

    /** Associates detections with optional ReID appearance embeddings. */
    @Synchronized
    fun updateInputs(inputs: List<RknnTrackingInput>): List<RknnTrackedDetection> {
        tracks.forEach { it.age++ }
        val high = inputs.filter { score(it.detection) >= config.highScoreThreshold }.toMutableList()
        val low = inputs.filter {
            score(it.detection) in config.lowScoreThreshold..<config.highScoreThreshold
        }.toMutableList()
        val unmatched = tracks.toMutableList()

        associate(unmatched, high) { it.missed == 0 }
        associate(unmatched, high) { true }
        associate(unmatched, low) { it.missed == 0 }
        associate(unmatched, low) { true }

        if (config.singleTargetRecovery && unmatched.size == 1 && high.size + low.size == 1) {
            val track = unmatched.single()
            val input = high.firstOrNull() ?: low.single()
            if (track.detection.categories.firstOrNull()?.index == input.detection.categories.firstOrNull()?.index &&
                appearanceAllowsMatch(track, input)
            ) {
                updateTrack(track, input)
                unmatched.clear()
                high.remove(input)
                low.remove(input)
            }
        }

        val matchedTracks = tracks.filterNot { it in unmatched }
        high.removeAll { input ->
            matchedTracks.any { track ->
                track.detection.categories.firstOrNull()?.index ==
                    input.detection.categories.firstOrNull()?.index &&
                    iou(track.detection.boundingBox, input.detection.boundingBox) >=
                    config.duplicateIouThreshold
            }
        }

        unmatched.forEach { track ->
            track.missed++
            track.state = RknnTrackState.LOST
        }
        high.forEach { input ->
            tracks += Track(
                id = nextId++,
                detection = input.detection,
                state = if (config.minConfirmedFrames == 1) RknnTrackState.TRACKED else RknnTrackState.TENTATIVE,
                embedding = normalizedEmbedding(input.embedding),
            )
        }
        tracks.removeAll { it.missed > config.maxLostFrames }
        return tracks.map { it.toResult() }
    }

    /** Removes every track and restarts track identifiers from 1. */
    @Synchronized
    fun reset() {
        tracks.clear()
        nextId = 1L
    }

    private fun associate(
        unmatched: MutableList<Track>,
        inputs: MutableList<RknnTrackingInput>,
        trackFilter: (Track) -> Boolean,
    ) {
        val candidates = unmatched.filter(trackFilter).toMutableList()
        while (candidates.isNotEmpty() && inputs.isNotEmpty()) {
            var bestTrack: Track? = null
            var bestInput: RknnTrackingInput? = null
            var bestScore = Float.NEGATIVE_INFINITY
            candidates.forEach { track -> inputs.forEach { input ->
                val matchScore = matchScore(track, input) ?: return@forEach
                if (matchScore > bestScore) {
                    bestScore = matchScore
                    bestTrack = track
                    bestInput = input
                }
            } }
            val track = bestTrack ?: break
            val input = bestInput ?: break
            updateTrack(track, input)
            candidates.remove(track)
            unmatched.remove(track)
            inputs.remove(input)
        }
    }

    private fun updateTrack(track: Track, input: RknnTrackingInput) {
        val detection = input.detection
        val previous = track.detection.boundingBox
        val current = detection.boundingBox
        track.velocityX = current.centerX - previous.centerX
        track.velocityY = current.centerY - previous.centerY
        track.detection = detection
        updateEmbedding(track, input.embedding)
        track.missed = 0
        track.hits++
        track.state = if (track.hits >= config.minConfirmedFrames) RknnTrackState.TRACKED else RknnTrackState.TENTATIVE
    }

    private fun Track.toResult() = RknnTrackedDetection(id, detection, state, age, missed)

    private fun score(detection: RknnDetection): Float = detection.categories.maxOfOrNull { it.score } ?: 0f

    private fun matchScore(track: Track, input: RknnTrackingInput): Float? {
        val detection = input.detection
        if (track.detection.categories.firstOrNull()?.index != detection.categories.firstOrNull()?.index) return null
        val predictedScore = boxMatchScore(predict(track), detection.boundingBox)
        val trustedScore = if (track.missed > 0) {
            boxMatchScore(track.detection.boundingBox, detection.boundingBox)
        } else {
            null
        }
        val spatialScore = listOfNotNull(predictedScore, trustedScore).maxOrNull()
        if (!config.reIdEnabled) return spatialScore
        val appearance = appearanceSimilarity(track.embedding, input.embedding) ?: return spatialScore
        if (appearance < config.reIdSimilarityThreshold) return null
        if (spatialScore != null) return spatialScore + appearance * config.reIdWeight
        return appearance * config.reIdWeight
    }

    private fun appearanceAllowsMatch(track: Track, input: RknnTrackingInput): Boolean {
        if (!config.reIdEnabled) return true
        val similarity = appearanceSimilarity(track.embedding, input.embedding) ?: return true
        return similarity >= config.reIdSimilarityThreshold
    }

    private fun updateEmbedding(track: Track, candidate: FloatArray?) {
        val normalized = normalizedEmbedding(candidate) ?: return
        val previous = track.embedding
        if (previous == null || previous.size != normalized.size) {
            track.embedding = normalized
            return
        }
        val mixed = FloatArray(previous.size) {
            previous[it] * config.reIdMomentum + normalized[it] * (1f - config.reIdMomentum)
        }
        track.embedding = normalizedEmbedding(mixed)
    }

    private fun appearanceSimilarity(first: FloatArray?, second: FloatArray?): Float? {
        val a = normalizedEmbedding(first) ?: return null
        val b = normalizedEmbedding(second) ?: return null
        if (a.size != b.size) return null
        var similarity = 0f
        for (index in a.indices) similarity += a[index] * b[index]
        return similarity.coerceIn(-1f, 1f)
    }

    private fun normalizedEmbedding(values: FloatArray?): FloatArray? {
        if (values == null || values.isEmpty() || values.any { !it.isFinite() }) return null
        var squaredNorm = 0.0
        values.forEach { squaredNorm += it.toDouble() * it.toDouble() }
        val norm = kotlin.math.sqrt(squaredNorm).toFloat()
        if (!norm.isFinite() || norm <= 1e-12f) return null
        return FloatArray(values.size) { values[it] / norm }
    }

    private fun boxMatchScore(predicted: RknnBoundingBox, current: RknnBoundingBox): Float? {
        val overlap = iou(predicted, current)
        if (overlap >= config.matchIouThreshold) return 2f + overlap

        val dx = current.centerX - predicted.centerX
        val dy = current.centerY - predicted.centerY
        val distance = kotlin.math.sqrt(dx * dx + dy * dy)
        val scale = maxOf(
            predicted.width,
            predicted.height,
            current.width,
            current.height,
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
