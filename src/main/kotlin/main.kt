import CANVAS.canvasFrame
import FileUtils.getImg
import IMG.imgConverter
import IMG.originalImg
import net.sourceforge.tess4j.Tesseract
import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.*
import org.bytedeco.javacv.CanvasFrame
import org.bytedeco.javacv.OpenCVFrameConverter
import java.io.File
import java.lang.Math.*


object IMG {
    val originalImg: Mat = imread(getImg("dado.jpg"), CV_LOAD_IMAGE_UNCHANGED).also { resizeSelf(it) }
    val imgConverter = OpenCVFrameConverter.ToMat()
}

object CANVAS {
    val canvasFrame = CanvasFrame("KNFTD - Kotlin Nutritional Facts Table Detector").apply {
        isResizable = false
        setCanvasSize(originalImg.size().width(), originalImg.size().height())
    }
}

object CANNY {
    val threshold = 40.0
    val apertureSize = 3
}

fun main(args: Array<String>) {
    runMainAlgorithm()
}

fun runMainAlgorithm() {
    // copy original     image into a new one to be used for filter applying
    val modifiedImg = cloneImageFrom(originalImg)

    // 1) convert to greyscale
    cvtColor(originalImg, modifiedImg, CV_BGR2GRAY)

    // 2) apply canny edge detection
    Canny(modifiedImg, modifiedImg, CANNY.threshold, (CANNY.threshold * 1), CANNY.apertureSize, true)

    // 3) apply Hough transform
    val distanceResolutionInPixels = 0.3
    val angleResolutionInRadians = 0.015
    val minimumVotes = 150

    val lines = Mat()

    HoughLines(
            modifiedImg,
            lines,
            distanceResolutionInPixels,
            angleResolutionInRadians,
            minimumVotes)

    val indexer = lines.createIndexer() as FloatRawIndexer
    val result = Mat().also { modifiedImg.copyTo(it) }

    for (i in 0 until lines.rows()) {
        val rho = indexer.get(i.toLong(), 0, 0)
        val theta = indexer.get(i.toLong(), 0, 1).toDouble()

        val pair = if (theta < PI / 4.0 || theta > 3.0 * PI / 4.0) {
            // ~vertical line
            // point of intersection of the line with first row
            val p1 = Point(round(rho / cos(theta)).toInt(), 0)
            // point of intersection of the line with last row
            val p2 = Point(round((rho - result.rows() * sin(theta)) / cos(theta)).toInt(), result.rows())
            Pair(p1, p2)
        } else {
            // ~horizontal line
            // point of intersection of the line with first column
            val p1 = Point(0, round(rho / sin(theta)).toInt())
            // point of intersection of the line with last column
            val p2 = Point(result.cols(), round((rho - result.cols() * cos(theta)) / sin(theta)).toInt())
            Pair(p1, p2)
        }

        // draw a white line
        line(result, pair.first, pair.second, Scalar(255.0, 0.0, 255.0, 0.0), 1, LINE_8, 0)
    }

    // 4) correct rotation w/ average horizontal 0

    // final) show image to screen
    canvasFrame.showImage(imgConverter.convert(toMat8U(result)))
}

fun toMat8U(src: Mat, doScaling: Boolean = true): Mat {
    val minVal = DoublePointer(Double.MAX_VALUE)
    val maxVal = DoublePointer(Double.MIN_VALUE)
    minMaxLoc(src, minVal, maxVal, null, null, Mat())
    val min = minVal.get(0)
    val max = maxVal.get(0)
    val (scale, offset) = if (doScaling) {
        val s = 255.toDouble() / (max - min)
        Pair(s, -min * s)
    } else Pair(1.toDouble(), 0.0)

    val dest = Mat()
    src.convertTo(dest, CV_8U, scale, offset)
    return dest
}

fun ocr() {
    val tess = Tesseract()
    tess.setLanguage("ita")
    val image = File(getImg("biscotti.jpg"))
    val r = tess.doOCR(image)
    println(r)
}

fun resizeSelf(img: Mat) = resize(img, img, Size((img.size().width() * 0.2).toInt(), (img.size().height() * 0.2).toInt()))

fun cloneImageFrom(img: Mat) = Mat(img.size().width(), img.size().height(), IPL_DEPTH_8U, 1)