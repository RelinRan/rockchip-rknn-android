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

    @Test
    fun `does not create a second id for an overlapping duplicate detection`() {
        val tracker = RknnTracker(RknnTrackerConfig(minConfirmedFrames = 1))
        val firstId = tracker.update(listOf(detection(0.9f, 100f))).single().trackId

        val duplicateFrame = tracker.update(
            listOf(detection(0.9f, 100f), detection(0.8f, 105f)),
        )

        assertEquals(1, duplicateFrame.count { it.state == RknnTrackState.TRACKED })
        assertEquals(firstId, duplicateFrame.single { it.state == RknnTrackState.TRACKED }.trackId)
    }

    @Test
    fun `single target mode keeps id when the only person jumps across the frame`() {
        val tracker = RknnTracker(
            RknnTrackerConfig(
                minConfirmedFrames = 1,
                singleTargetRecovery = true,
            ),
        )
        val id = tracker.update(listOf(detection(0.9f, 0f))).single().trackId

        val moved = tracker.update(listOf(detection(0.9f, 700f)))
            .single { it.state == RknnTrackState.TRACKED }

        assertEquals(id, moved.trackId)
        assertEquals(1, tracker.update(listOf(detection(0.9f, 20f))).count { it.state == RknnTrackState.TRACKED })
    }

    @Test
    fun `recovers lost track from last trusted position when velocity prediction overshoots`() {
        val tracker = RknnTracker(
            RknnTrackerConfig(minConfirmedFrames = 1, maxLostFrames = 5),
        )
        val id = tracker.update(listOf(detection(0.9f, 0f))).single().trackId
        tracker.update(listOf(detection(0.9f, 70f)))
        repeat(3) { tracker.update(emptyList()) }

        val recovered = tracker.update(listOf(detection(0.9f, 75f)))
            .single { it.state == RknnTrackState.TRACKED }

        assertEquals(id, recovered.trackId)
    }

    @Test
    fun `reid recovers a lost track after a distant move`() {
        val tracker = RknnTracker(
            RknnTrackerConfig(
                minConfirmedFrames = 1,
                maxLostFrames = 5,
                reIdEnabled = true,
                reIdSimilarityThreshold = 0.8f,
            ),
        )
        val id = tracker.updateInputs(
            listOf(RknnTrackingInput(detection(0.9f, 0f), floatArrayOf(1f, 0f))),
        ).single().trackId
        tracker.update(emptyList())

        val recovered = tracker.updateInputs(
            listOf(RknnTrackingInput(detection(0.9f, 500f), floatArrayOf(0.99f, 0.01f))),
        ).single { it.state == RknnTrackState.TRACKED }

        assertEquals(id, recovered.trackId)
    }

    @Test
    fun `reid keeps the active id while one person moves across distant stations`() {
        val tracker = RknnTracker(
            RknnTrackerConfig(
                minConfirmedFrames = 1,
                maxLostFrames = 30,
                reIdEnabled = true,
                reIdSimilarityThreshold = 0.8f,
            ),
        )
        val embedding = floatArrayOf(1f, 0f)
        val initialId = tracker.updateInputs(
            listOf(RknnTrackingInput(detection(0.9f, 0f), embedding)),
        ).single { it.state == RknnTrackState.TRACKED }.trackId

        listOf(400f, 800f, 0f).forEach { stationOffset ->
            val current = tracker.updateInputs(
                listOf(RknnTrackingInput(detection(0.9f, stationOffset), embedding)),
            ).single { it.state == RknnTrackState.TRACKED }

            assertEquals(initialId, current.trackId)
        }
    }

    @Test
    fun `reid keeps two visually distinct tracks from swapping ids`() {
        val tracker = RknnTracker(
            RknnTrackerConfig(
                minConfirmedFrames = 1,
                reIdEnabled = true,
                reIdSimilarityThreshold = 0.8f,
            ),
        )
        val first = tracker.updateInputs(
            listOf(
                RknnTrackingInput(detection(0.9f, 0f), floatArrayOf(1f, 0f)),
                RknnTrackingInput(detection(0.9f, 300f), floatArrayOf(0f, 1f)),
            ),
        ).filter { it.state == RknnTrackState.TRACKED }.associateBy { it.detection.boundingBox.left }
        tracker.update(emptyList())

        val moved = tracker.updateInputs(
            listOf(
                RknnTrackingInput(detection(0.9f, 300f), floatArrayOf(1f, 0f)),
                RknnTrackingInput(detection(0.9f, 0f), floatArrayOf(0f, 1f)),
            ),
        ).filter { it.state == RknnTrackState.TRACKED }.associateBy { it.detection.boundingBox.left }

        assertEquals(first.getValue(0f).trackId, moved.getValue(300f).trackId)
        assertEquals(first.getValue(300f).trackId, moved.getValue(0f).trackId)
    }

    @Test
    fun `disabled reid does not recover a distant track`() {
        val tracker = RknnTracker(
            RknnTrackerConfig(minConfirmedFrames = 1, maxLostFrames = 5, reIdEnabled = false),
        )
        val id = tracker.updateInputs(
            listOf(RknnTrackingInput(detection(0.9f, 0f), floatArrayOf(1f, 0f))),
        ).single().trackId
        tracker.update(emptyList())

        val current = tracker.updateInputs(
            listOf(RknnTrackingInput(detection(0.9f, 500f), floatArrayOf(1f, 0f))),
        ).single { it.state == RknnTrackState.TRACKED }

        assertTrue(current.trackId != id)
    }

    private fun detection(score: Float, offset: Float) = RknnDetection(
        listOf(RknnCategory(0, "person", score)),
        RknnBoundingBox(offset, offset, 100f + offset, 100f + offset),
    )
}
