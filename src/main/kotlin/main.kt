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
    val resizeRatio = 0.6
    val originalImg: Mat = imread(getImg("olioantiorario.jpg"), CV_LOAD_IMAGE_UNCHANGED).also { resizeSelf(it) }
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
//    modifiedImg.show("grey")

    // 2) apply canny edge detection
    applyCanny(modifiedImg)

    // 3) apply Hough transform
    applyHough(modifiedImg, true)
    // 4) correct rotation w/ average horizontal 0
    originalImg.rotateToTheta()

    originalImg.show("rotated")

    // NB now originalImg is rotated, let's re-run the previous steps
/*
    // 5) convert to greyscale
    convertToGreyscale(originalImg, modifiedImg)

    // 2) apply canny edge detection
    applyCanny(modifiedImg)

    // 4) correct rotation w/ average horizontal 0
    applyHough(modifiedImg, true)

    convertToGreyscale(originalImg, modifiedImg)

    applyOtsu(modifiedImg)

    ocr(modifiedImg)
*/

}

fun cloneImageFrom(img: Mat) = Mat(img.size().width(), img.size().height(), IPL_DEPTH_8U, 1)

fun resizeSelf(img: Mat) = resize(img, img, Size((img.size().width() * resizeRatio).toInt(), (img.size().height() * resizeRatio).toInt()))

fun convertToGreyscale(source: Mat, dest: Mat = source) = cvtColor(source, dest, CV_BGR2GRAY)

fun applyCanny(source: Mat, dest: Mat = source) = Canny(source, dest, CANNY.threshold, (CANNY.threshold * 1), CANNY.apertureSize, true)

fun Mat.rotateToTheta() {
    if (finalTheta != 0.0) {
        warpAffine(this, this, getRotationMatrix2D(Point2f((this.size().width() / 2).toFloat(), (this.size().height() / 2).toFloat()), finalTheta, 1.0), this.size())
    }
}

fun applyOtsu(source: Mat, dest: Mat = source) = threshold(source, dest, 0.0, 255.0, THRESH_OTSU)

fun applyHough(source: Mat, shrinkLines: Boolean) {

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

//        if (theta <= PI / 4.0 || theta >= 3.0 * PI / 4.0) {
        if (thetaDeg <= 45 || thetaDeg >= 135) {
            // ~vertical line
//            if (theta < 0.3141 || theta > 2.5132) {
            if (thetaDeg < 2 || thetaDeg > 178) {
                p1 = Point(round(rho / cos(theta)).toInt(), 0) // point of intersection of the line with first row
                p2 = Point(round((rho - houghResult.rows() * sin(theta)) / cos(theta)).toInt(), houghResult.rows()) // point of intersection of the line with last row
                if (thetaDeg > 90) theta = -(PI - theta)
                verticalLinesList.add(Line(rho, theta, p1, p2))
            }
        } else {
            // ~horizontal line
//            if ((theta < 1.60 && theta > 1.55)) {
            if ((thetaDeg < 92 && thetaDeg > 88)) {
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

    // uncomment to display lines before the shrinking process
//    val res1 = Mat().also { originalImg.copyTo(it) }
//    horizontalLinesList.forEach { line(res1, it.p1, it.p2, scalar, 1, LINE_8, 0) }
//    verticalLinesList.forEach { line(res1, it.p1, it.p2, scalar, 1, LINE_8, 0) }
//    res1.show("Non rimosse $houghCounter")

//    if (shrinkLines) {
    println("Result $houghCounter, NON RIMOSSE ${horizontalLinesList.size} ${verticalLinesList.size}")
    removeLines(horizontalLinesList)
    removeLines(verticalLinesList)
    println("Result $houghCounter, RIMOSSE ${horizontalLinesList.size} ${verticalLinesList.size}")
//    }

    //    if (!shrinkLines) {
    verticalLinesList.forEach { println(it.theta.toDegrees()) }
    finalTheta = if (verticalLinesList.isNotEmpty()) verticalLinesList.map { it.theta.toDegrees() }.average() else 0.0
    println("mean theta rad " + finalTheta)
//    println("mean theta ° " + finalTheta.toDegrees())
//    }


    val res2 = Mat().also { originalImg.copyTo(it) }
    //draw lines
    horizontalLinesList.forEach { line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0) }
    verticalLinesList.forEach {
        if (it.theta == 0.0) line(res2, it.p1, it.p2, scalar2, 1, LINE_8, 0)
        else line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0)
    }
    res2.show("Rimosse $houghCounter")

    houghCounter = houghCounter + 1
}

fun ocr(source: Mat) {

    Mat.zeros(Size(11, 11), CV_8UC1)
    val horizontalsize = source.cols() / 30
    val horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1))
    val horizontalStructure2 = getStructuringElement(MORPH_RECT, Size(horizontalsize, 5))
    val a: Mat = source.clone()
    a.show()
    erode(source, a, horizontalStructure)
    dilate(a, a, horizontalStructure2)
    a.show()
    bitwise_not(source, source)
    bitwise_or(source, a, source)
    source.show()
//    a.show()
//    source.show()

    val tess = Tesseract()
    tess.setLanguage("ita")
    tess.setTessVariable("tessedit_pageseg_mode", "11")
    tess.setTessVariable("load_system_dawg", "F")
    tess.setTessVariable("load_freq_dawg", "F")
    tess.setTessVariable("enable_new_segsearch", "1")
    tess.setTessVariable("language_model_penalty_non_dict_word", "10000000")

//    val imgForOCR = ImageIO.read(File(getImg("dado2.jpg")))

//    imwrite("ehm.jpg", source)
//    val imgForOCR= ImageIO.read(File(getImg("ehm.jpg")))

    val imgForOCR = source.toBufferedImage()

//    val r = tess.doOCR(imgForOCR)
//    println(r)
//    source.show()
}

data class Line(val rho: Float, val theta: Double, val p1: Point, val p2: Point)