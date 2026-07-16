package androidx.runtime.rknn

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test

class RknnOptionsTest {
    @Test
    fun resolveModelRoot_usesSharedExternalStorageByDefault() {
        val externalStorage = File("/storage/emulated/0")

        val root = RknnOptions().resolveModelRoot(externalStorage)

        assertEquals(File("/storage/emulated/0/AiHandHygiene/model"), root)
    }

    @Test
    fun modelRoot_preservesAbsoluteModelDirectory() {
        val root = File("/models/rknn").absoluteFile

        val options = RknnOptions(modelRoot = root)

        assertEquals(root, options.modelRoot)
    }
}
