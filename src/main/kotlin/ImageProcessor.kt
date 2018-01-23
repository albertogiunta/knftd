import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.javacpp.indexer.UByteIndexer
import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_imgproc
import org.bytedeco.javacpp.opencv_imgproc.getRotationMatrix2D
import org.bytedeco.javacpp.opencv_imgproc.warpAffine

class ImageProcessor {
    //-------------------Contrast params
    private val contrastRType = -1
    private val contrastAlpha = 1.55
    private val contrastBeta = -25.5

    //-------------------Hough params
    private var houghCounter = 1
    private val distanceResolutionInPixels: Double = 1.0 // rho
    private val angleResolutionInRadians: Double = Math.PI / 180 // theta
    private var minimumVotes = 1000
    private var minimumVotesToConsider = 3
    private var minimumVotesStep = 100


    private var finalTheta: Double = 0.0 // degrees
    private var lines = opencv_core.Mat()
    private val binaryThreshold = 110.0
    private val maxRhoThreshold = 500


    fun convertToGreyscale(source: opencv_core.Mat, dest: opencv_core.Mat = source) = opencv_imgproc.cvtColor(source, dest, opencv_imgproc.CV_BGR2GRAY)

    fun increaseContrast(source: opencv_core.Mat) = source.convertTo(source, contrastRType, contrastAlpha, contrastBeta)

    //fun increaseContrast2(source: opencv_core.Mat) = source.convertTo(source, rType, alpha2, beta2)

    fun applySobel(source: opencv_core.Mat) {
        val gradX = opencv_core.Mat()
        val gradY = opencv_core.Mat()
        val absGradX = opencv_core.Mat()
        val absGradY = opencv_core.Mat()
        opencv_imgproc.Sobel(source, gradX, opencv_core.CV_16S, 1, 0, 3, 1.0, 0.0, opencv_core.BORDER_DEFAULT)
        opencv_imgproc.Sobel(source, gradY, opencv_core.CV_16S, 0, 1, 3, 1.0, 0.0, opencv_core.BORDER_DEFAULT)
        opencv_core.convertScaleAbs(gradX, absGradX)
        opencv_core.convertScaleAbs(gradY, absGradY)
        opencv_core.addWeighted(absGradX, 0.5, absGradY, 0.5, 0.0, source)
    }

    fun reduceColor(source: opencv_core.Mat) {
        val div = 64
        val indexer = source.createIndexer() as UByteIndexer
        // Total number of elements, combining components from each channel
        val nbElements = source.rows() * source.cols() * source.channels()
        for (i in 0 until nbElements) {
            // Convert to integer, byte is treated as an unsigned value
            val v = indexer.get(i.toLong())
            // Use integer division to reduce number of values
            val newV = v / div * div + div / 2
            // Put back into the image
            indexer.put(i.toLong(), newV)
        }
    }


    fun applyOtsu(source: opencv_core.Mat, dest: opencv_core.Mat = source) = opencv_imgproc.threshold(source, dest, 0.0, 255.0, opencv_imgproc.THRESH_BINARY + opencv_imgproc.THRESH_OTSU)

    fun applyBinary(source: opencv_core.Mat, dest: opencv_core.Mat = source) = opencv_imgproc.threshold(source, dest, binaryThreshold, 255.0, opencv_imgproc.THRESH_BINARY)

    fun crop(source: opencv_core.Mat, rect: opencv_core.Rect): opencv_core.Mat = opencv_core.Mat(source, rect)

    fun applyHough(source: opencv_core.Mat, matForDisplay: opencv_core.Mat) {

        lines = opencv_core.Mat()
        opencv_imgproc.HoughLines(source, lines, distanceResolutionInPixels, angleResolutionInRadians, minimumVotes)

        val indexer = lines.createIndexer() as FloatRawIndexer
        val houghResult = opencv_core.Mat().also { source.copyTo(it) }
        val horizontalLinesList = mutableListOf<Line>()
        val verticalLinesList = mutableListOf<Line>()

        for (i in 0 until lines.rows()) {
            val rho = indexer.get(i.toLong(), 0, 0)
            val thetaDeg = indexer.get(i.toLong(), 0, 1).toDouble().toDegrees()
            var theta = indexer.get(i.toLong(), 0, 1).toDouble()
            lateinit var p1: opencv_core.Point
            lateinit var p2: opencv_core.Point

            if (thetaDeg <= 45 || thetaDeg >= 135) {
                // ~vertical line
                if (thetaDeg < 10 || thetaDeg > 170) {
                    p1 = opencv_core.Point(Math.round(rho / Math.cos(theta)).toInt(), 0) // point of intersection of the line with first row
                    p2 = opencv_core.Point(
                            Math.round((rho - houghResult.rows() * Math.sin(theta)) / Math.cos(theta)).toInt(),
                            houghResult.rows()
                    ) // point of intersection of the line with last row
                    if (thetaDeg > 90) theta = -(Math.PI - theta)
                    verticalLinesList.add(Line(rho, theta, p1, p2))
                }
            } else {
                // ~horizontal line
                if ((thetaDeg < 95 && thetaDeg > 85)) {
                    p1 = opencv_core.Point(0, Math.round(rho / Math.sin(theta)).toInt()) // point of intersection of the line with first column
                    p2 = opencv_core.Point(
                            houghResult.cols(),
                            Math.round((rho - houghResult.cols() * Math.cos(theta)) / Math.sin(theta)).toInt()
                    ) // point of intersection of the line with last column
                    horizontalLinesList.add(Line(rho, theta, p1, p2))
                }
            }
        }

        horizontalLinesList.addAll(horizontalLinesList.sortedBy { it.rho }.toMutableList())
        verticalLinesList.addAll(verticalLinesList.sortedBy { it.rho }.toMutableList())

        fun removeLines(list: MutableList<Line>) {
            for (i in 0 until list.size) {
                for (j in i + 1 until list.size - 1) {
                    if (j < list.size) {
                        val shouldBeRemoved = Math.abs(list[i].rho - list[j].rho) < maxRhoThreshold
                        if (shouldBeRemoved) {
                            list.removeAt(j)
                        } else break
                    }
                }
            }
        }

        removeLines(horizontalLinesList)
        removeLines(verticalLinesList)

        val vertTheta = if (verticalLinesList.isNotEmpty()) verticalLinesList.map { it.theta.toDegrees() }.average() else 0.0
        val horTheta = if (horizontalLinesList.isNotEmpty()) horizontalLinesList.map { it.theta.toDegrees() }.average() - 90 else 0.0
        finalTheta = listOf(vertTheta, horTheta).average()
        println("$vertTheta $horTheta $finalTheta")

        val res2 = matForDisplay.clone()
        //draw lines
        horizontalLinesList.forEach { opencv_imgproc.line(res2, it.p1, it.p2, opencv_core.Scalar.RED, 1, opencv_core.LINE_8, 0) }
        verticalLinesList.forEach { opencv_imgproc.line(res2, it.p1, it.p2, opencv_core.Scalar.RED, 1, opencv_core.LINE_8, 0) }


        if (verticalLinesList.size < minimumVotesToConsider && horizontalLinesList.size < minimumVotesToConsider) {
            minimumVotes -= minimumVotesStep
            applyHough(source, matForDisplay)
        } else {
            //res2.clone().resizeSelf().show("applied hough n°$houghCounter | minimumVotes: $minimumVotes")
            houghCounter += 1
            minimumVotes = 1000
        }
    }

    fun rotate(image : opencv_core.Mat){

            if (finalTheta != 0.0) {
                warpAffine(image, image, getRotationMatrix2D(opencv_core.Point2f((image.size().width() / 2).toFloat(), (image.size().height() / 2).toFloat()), finalTheta, 1.0), image.size())
            }

    }


}