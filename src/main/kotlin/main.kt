import FileUtils.getImg
import HOUGH.angleResolutionInRadians
import HOUGH.distanceResolutionInPixels
import HOUGH.finalTheta
import HOUGH.houghCounter
import HOUGH.lines
import HOUGH.minimumVotes
import HOUGH.scalar
import IMG.originalImg
import IMG.resizeRatio
import LINE_SHRINKING.maxRhoTheresold
import net.sourceforge.tess4j.Tesseract
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.*
import org.bytedeco.javacv.OpenCVFrameConverter
import java.io.File
import java.lang.Math.*
import javax.imageio.ImageIO

object IMG {
    val resizeRatio = 0.2
    val originalImg: Mat = imread(getImg("dado.jpg"), CV_LOAD_IMAGE_UNCHANGED).also { resizeSelf(it) }
    val imgConverter = OpenCVFrameConverter.ToMat()
}

object CANNY {
    val threshold = 70.0
    val apertureSize = 3
}

object HOUGH {
    var houghCounter = 1
    val distanceResolutionInPixels: Double = 1.0 // rho
    val angleResolutionInRadians: Double = PI / 180 // theta
    val minimumVotes = 200
    var finalTheta: Double = 0.0
    var lines = Mat()
    val scalar = Scalar(0.0, 0.0, 0.0, 0.0)
    val scalar2 = Scalar(0.0, 0.0, 255.0, 0.0)
}

object LINE_SHRINKING {
    val maxRhoTheresold = 50
}

fun main(args: Array<String>) {
    runMainAlgorithm()
//    ocr()
}

fun runMainAlgorithm() {
    // copy original     image into a new one to be used for filter applying
    val modifiedImg = cloneImageFrom(originalImg)

    // 1) convert to greyscale
    convertToGreyscale(originalImg, modifiedImg)

    // 2) apply canny edge detection
    applyCanny(modifiedImg)

    // 3) apply Hough transform
    applyHough(modifiedImg, false)

    // 4) correct rotation w/ average horizontal 0
    originalImg.rotateToTheta()

    // NB now originalImg is rotated, let's re-run the previous steps

    // 5) convert to greyscale
    convertToGreyscale(originalImg, modifiedImg)

    // 2) apply canny edge detection
    applyCanny(modifiedImg)

    // 4) correct rotation w/ average horizontal 0
    applyHough(modifiedImg, true)

}

fun cloneImageFrom(img: Mat) = Mat(img.size().width(), img.size().height(), IPL_DEPTH_8U, 1)

fun resizeSelf(img: Mat) = resize(img, img, Size((img.size().width() * resizeRatio).toInt(), (img.size().height() * resizeRatio).toInt()))

fun convertToGreyscale(source: Mat, dest: Mat = source) = cvtColor(source, dest, CV_BGR2GRAY)

fun applyCanny(source: Mat, dest: Mat = source) = Canny(source, dest, CANNY.threshold, (CANNY.threshold * 1), CANNY.apertureSize, true)

fun applyHough(source: Mat, shrinkLines: Boolean) {

    lines = Mat()
    HoughLines(source, lines, distanceResolutionInPixels, angleResolutionInRadians, minimumVotes)

    val indexer = lines.createIndexer() as FloatRawIndexer
    val houghResult = Mat().also { source.copyTo(it) }
    val horizontalLinesList = mutableListOf<Line>()
    val verticalLinesList = mutableListOf<Line>()

    for (i in 0 until lines.rows()) {
        val rho = indexer.get(i.toLong(), 0, 0)
        val theta = indexer.get(i.toLong(), 0, 1).toDouble()
        lateinit var p1: Point
        lateinit var p2: Point

        if (theta <= PI / 4.0 || theta >= 3.0 * PI / 4.0) {
            // ~vertical line
            if (theta < 0.3141 || theta > 2.5132) {
                p1 = Point(round(rho / cos(theta)).toInt(), 0) // point of intersection of the line with first row
                p2 = Point(round((rho - houghResult.rows() * sin(theta)) / cos(theta)).toInt(), houghResult.rows()) // point of intersection of the line with last row
                verticalLinesList.add(Line(rho, theta, p1, p2))
                finalTheta = theta
            }
        } else {
            // ~horizontal line
            if ((theta < 1.60 && theta > 1.55)) {
                p1 = Point(0, round(rho / sin(theta)).toInt()) // point of intersection of the line with first column
                p2 = Point(houghResult.cols(), round((rho - houghResult.cols() * cos(theta)) / sin(theta)).toInt()) // point of intersection of the line with last column
                horizontalLinesList.add(Line(rho, theta, p1, p2))
            }
        }
    }

    horizontalLinesList.addAll(horizontalLinesList.sortedBy { it.rho }.toMutableList())
    verticalLinesList.addAll(verticalLinesList.sortedBy { it.rho }.toMutableList())

    if (!shrinkLines) {
        println("last theta " + finalTheta)
//        finalTheta = vertLinesList.map { Math.abs(it.theta) }.average()
        println("mean theta " + finalTheta)
    }

    fun removeLines(list: MutableList<Line>) {
        for (i in 0 until list.size) {
            for (j in i + 1 until list.size - 1) {
                if (j < list.size) {
                    val shouldBeRemoved = Math.abs(list[i].rho - list[j].rho) < maxRhoTheresold
                    if (shouldBeRemoved) {
                        list.removeAt(j)
                    } else break
                }
            }
        }
    }

    // uncomment to display lines before the shrinking process
/*
    val res1 = Mat().also { originalImg.copyTo(it) }
    horLinesList.forEach { line(res1, it.p1, it.p2, scalar, 1, LINE_8, 0) }
    vertLinesList.forEach { line(res1, it.p1, it.p2, scalar, 1, LINE_8, 0) }
    res1.show("Non rimosse $houghCounter")
*/

    if (shrinkLines) {
        println("Result $houghCounter, NON RIMOSSE ${horizontalLinesList.size} ${verticalLinesList.size}")
        removeLines(horizontalLinesList)
        removeLines(verticalLinesList)
        println("Result $houghCounter, RIMOSSE ${horizontalLinesList.size} ${verticalLinesList.size}")
    }

    val res2 = Mat().also { originalImg.copyTo(it) }
    horizontalLinesList.forEach { line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0) }
    verticalLinesList.forEach { line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0) }
    res2.show("Rimosse $houghCounter")

    houghCounter = houghCounter + 1
}

fun Mat.rotateToTheta() {
    if (finalTheta != 0.0) {
        warpAffine(this, this, getRotationMatrix2D(Point2f((this.size().width() / 2).toFloat(), (this.size().height() / 2).toFloat()), 180 - finalTheta.toDegrees(), 1.0), this.size())
    }
}

fun ocr() {
    val tess = Tesseract()
    tess.setLanguage("ita")
    val file = File(getImg("dado.jpg"))
    val image = ImageIO.read(file)
    val r = tess.doOCR(file)
    val words = tess.getWords(image, 0)
    words.forEach { println("|${it.text}|") }
}

data class Line(val rho: Float, val theta: Double, val p1: Point, val p2: Point)