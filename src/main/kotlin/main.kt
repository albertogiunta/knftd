import FileUtils.getImg
import HOUGH.angleResolutionInRadians
import HOUGH.distanceResolutionInPixels
import HOUGH.finalTheta
import HOUGH.houghCounter
import HOUGH.lines
import HOUGH.minimumVotes
import HOUGH.scalar
import HOUGH.scalar2
import IMG.originalImg
import IMG.originalImgNotResized
import IMG.resizeRatio
import LINE_SHRINKING.maxRhoTheresold
import net.sourceforge.tess4j.Tesseract
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.*
import org.bytedeco.javacv.OpenCVFrameConverter
import java.lang.Math.*


object IMG {
    val imgName = "olioantiorario" + "." + "jpg"
    val resizeRatio = 0.6
    val originalImgNotResized: Mat = imread(getImg(imgName), CV_LOAD_IMAGE_UNCHANGED)
    val originalImg: Mat = imread(getImg(imgName), CV_LOAD_IMAGE_UNCHANGED).also { resizeSelf(it) }
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
    val minimumVotes = 150
    var finalTheta: Double = 0.0 // degrees
    var lines = Mat()
    val scalar = Scalar(0.0, 0.0, 0.0, 0.0)
    val scalar2 = Scalar(0.0, 0.0, 255.0, 0.0)
}

object LINE_SHRINKING {
    val maxRhoTheresold = 500
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
    increaseContrast(modifiedImg)

    // 2) apply canny edge detection
    applyCanny(modifiedImg)

    // 3) apply Hough transform
    applyHough(modifiedImg)

    // 4) correct rotation w/ average horizontal 0
    originalImg.rotateToTheta()
    originalImgNotResized.rotateToTheta()

    // NB now originalImg is rotated, let's re-run the previous steps

    // 5) convert to greyscale
    convertToGreyscale(originalImgNotResized, modifiedImg)
    increaseContrast(modifiedImg)

    // 2) apply canny edge detection
//    applyCanny(modifiedImg)

    // 4) correct rotation w/ average horizontal 0
//    applyHough(modifiedImg)

    applyOtsu(modifiedImg)
//    modifiedImg.show()

    ocr(modifiedImg)

}

fun cloneImageFrom(img: Mat) = Mat(img.size().width(), img.size().height(), IPL_DEPTH_8U, 1)

fun resizeSelf(img: Mat) = resize(img, img, Size((img.size().width() * resizeRatio).toInt(), (img.size().height() * resizeRatio).toInt()))

fun convertToGreyscale(source: Mat, dest: Mat = source) = cvtColor(source, dest, CV_BGR2GRAY)

fun increaseContrast(source: Mat) = source.convertTo(source, -1, 1.5, -100.0)

fun applyCanny(source: Mat, dest: Mat = source) = Canny(source, dest, CANNY.threshold, (CANNY.threshold * 1), CANNY.apertureSize, true)

fun Mat.rotateToTheta() {
    if (finalTheta != 0.0) {
        warpAffine(this, this, getRotationMatrix2D(Point2f((this.size().width() / 2).toFloat(), (this.size().height() / 2).toFloat()), finalTheta, 1.0), this.size())
    }
}

fun applyOtsu(source: Mat, dest: Mat = source) = threshold(source, dest, 0.0, 255.0, THRESH_OTSU)

fun applyHough(source: Mat) {

    lines = Mat()
    HoughLines(source, lines, distanceResolutionInPixels, angleResolutionInRadians, minimumVotes)

    val indexer = lines.createIndexer() as FloatRawIndexer
    val houghResult = Mat().also { source.copyTo(it) }
    val horizontalLinesList = mutableListOf<Line>()
    val verticalLinesList = mutableListOf<Line>()

    for (i in 0 until lines.rows()) {
        val rho = indexer.get(i.toLong(), 0, 0)
        val thetaDeg = indexer.get(i.toLong(), 0, 1).toDouble().toDegrees()
        var theta = indexer.get(i.toLong(), 0, 1).toDouble()
        lateinit var p1: Point
        lateinit var p2: Point

        if (thetaDeg <= 45 || thetaDeg >= 135) {
            // ~vertical line
            if (thetaDeg < 10 || thetaDeg > 170) {
                p1 = Point(round(rho / cos(theta)).toInt(), 0) // point of intersection of the line with first row
                p2 = Point(round((rho - houghResult.rows() * sin(theta)) / cos(theta)).toInt(), houghResult.rows()) // point of intersection of the line with last row
                if (thetaDeg > 90) theta = -(PI - theta)
                verticalLinesList.add(Line(rho, theta, p1, p2))
            }
        } else {
            // ~horizontal line
            if ((thetaDeg < 95 && thetaDeg > 85)) {
                p1 = Point(0, round(rho / sin(theta)).toInt()) // point of intersection of the line with first column
                p2 = Point(houghResult.cols(), round((rho - houghResult.cols() * cos(theta)) / sin(theta)).toInt()) // point of intersection of the line with last column
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
                    val shouldBeRemoved = Math.abs(list[i].rho - list[j].rho) < maxRhoTheresold
                    if (shouldBeRemoved) {
                        list.removeAt(j)
                    } else break
                }
            }
        }
    }

    println("Result $houghCounter, NON RIMOSSE ${horizontalLinesList.size} ${verticalLinesList.size}")
    removeLines(horizontalLinesList)
    removeLines(verticalLinesList)
    println("Result $houghCounter, RIMOSSE ${horizontalLinesList.size} ${verticalLinesList.size}")

    verticalLinesList.forEach { println(it.theta.toDegrees()) }
    finalTheta = if (verticalLinesList.isNotEmpty()) verticalLinesList.map { it.theta.toDegrees() }.average() else 0.0
    println("mean theta rad " + finalTheta)
    println("mean theta deg " + finalTheta * 180 / PI)

    val res2 = Mat().also { originalImg.copyTo(it) }
    //draw lines
    horizontalLinesList.forEach { line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0) }
    verticalLinesList.forEach {
        if (it.theta == 0.0) line(res2, it.p1, it.p2, scalar2, 1, LINE_8, 0)
        else line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0)
    }
//    res2.show("HOUGH (lines removed) $houghCounter")

    houghCounter = houghCounter + 1
}

fun ocr(source: Mat) {
    val a: Mat = source.clone()

//    val horizontalsize = a.cols() / 30
//    val horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 50))
//    morphologyEx(a, a, MORPH_OPEN, horizontalStructure)
//    erode(source, a, horizontalStructure)
//    dilate(a, a, horizontalStructure2)
//    bitwise_not(source, source)
//    bitwise_or(source, a, source)

    val tess = Tesseract()
    tess.setLanguage("ita")
    tess.setTessVariable("tessedit_pageseg_mode", "11")
    tess.setTessVariable("load_system_dawg", "F")
    tess.setTessVariable("load_freq_dawg", "F")
    tess.setTessVariable("enable_new_segsearch", "1")
    tess.setTessVariable("language_model_penalty_non_dict_word", "10000000")

    val imgForOCR = source.toBufferedImage()

    val nutriList = listOf<String>("nutrizionale", "nutrizionali")
    var words = tess.getWords(imgForOCR, 0)

    words.forEach { println(it.text) }

    val y = words.filter { it.text.contains("nutrizion", true) }.map { it.boundingBox.y }[0] - 25
    val x = words.filter { it.text.contains("sale", true) }.map { it.boundingBox.x }[0] - 20

    val r = Rect(x, y, source.size().width() - x, source.size().height() - y)
    val cropped = Mat(source, r)
    cropped.show("cropped")

    source.show()
}

data class Line(val rho: Float, val theta: Double, val p1: Point, val p2: Point)