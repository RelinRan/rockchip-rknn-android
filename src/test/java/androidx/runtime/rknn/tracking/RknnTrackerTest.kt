package androidx.runtime.rknn.tracking

import androidx.runtime.rknn.data.RknnBoundingBox
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnDetection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RknnTrackerTest {
    @Test
    fun `keeps stable id and confirms track`() {
        val tracker = RknnTracker(RknnTrackerConfig(minConfirmedFrames = 2))

        val first = tracker.update(listOf(detection(0.9f, 0f))).single()
        val second = tracker.update(listOf(detection(0.8f, 1f))).single()

        assertEquals(first.trackId, second.trackId)
        assertEquals(RknnTrackState.TENTATIVE, first.state)
        assertEquals(RknnTrackState.TRACKED, second.state)
    }

    @Test
    fun `recovers track with low confidence detection`() {
        val tracker = RknnTracker(RknnTrackerConfig(minConfirmedFrames = 1))
        val id = tracker.update(listOf(detection(0.9f, 0f))).single().trackId

        val recovered = tracker.update(listOf(detection(0.2f, 1f))).single()

        assertEquals(id, recovered.trackId)
        assertEquals(RknnTrackState.TRACKED, recovered.state)
    }

    @Test
    fun `keeps lost track then removes and resets ids`() {
        val tracker = RknnTracker(RknnTrackerConfig(minConfirmedFrames = 1, maxLostFrames = 1))
        tracker.update(listOf(detection(0.9f, 0f)))

        assertEquals(RknnTrackState.LOST, tracker.update(emptyList()).single().state)
        assertTrue(tracker.update(emptyList()).isEmpty())
        tracker.reset()
        assertEquals(1L, tracker.update(listOf(detection(0.9f, 0f))).single().trackId)
    }

    @Test
    fun `keeps id when target moves quickly between frames`() {
        val tracker = RknnTracker(
            RknnTrackerConfig(
                minConfirmedFrames = 1,
                matchIouThreshold = 0.3f,
                maxCenterDistanceRatio = 1.5f,
            ),
        )
        val firstId = tracker.update(listOf(detection(0.9f, 0f))).single().trackId

        val second = tracker.update(listOf(detection(0.9f, 70f))).single { it.state == RknnTrackState.TRACKED }
        val third = tracker.update(listOf(detection(0.9f, 150f))).single { it.state == RknnTrackState.TRACKED }

        assertEquals(firstId, second.trackId)
        assertEquals(firstId, third.trackId)
    }

    @Test
    fun `does not reuse id for a distant new target`() {
        val tracker = RknnTracker(
            RknnTrackerConfig(minConfirmedFrames = 1, maxCenterDistanceRatio = 1.5f),
        )
        val firstId = tracker.update(listOf(detection(0.9f, 0f))).single().trackId

        val tracks = tracker.update(listOf(detection(0.9f, 400f)))
        val newTrack = tracks.single { it.state == RknnTrackState.TRACKED }

        assertTrue(newTrack.trackId != firstId)
    }

    private fun detection(score: Float, offset: Float) = RknnDetection(
        listOf(RknnCategory(0, "person", score)),
        RknnBoundingBox(offset, offset, 100f + offset, 100f + offset),
    )
}
