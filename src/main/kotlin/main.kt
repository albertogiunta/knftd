import CANVAS.canvasFrame
import FileUtils.getImg
import IMG.imgConverter
import IMG.originalImg
import IMG.resizeRatio
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
import javax.imageio.ImageIO


object IMG {
    val resizeRatio = 0.2
    val originalImg: Mat = imread(getImg("biscotti.jpg"), CV_LOAD_IMAGE_UNCHANGED).also { resizeSelf(it) }
    val imgConverter = OpenCVFrameConverter.ToMat()
}

object CANVAS {
    val canvasFrame = CanvasFrame("KNFTD - Kotlin Nutritional Facts Table Detector").apply {
        isResizable = false
        setCanvasSize(originalImg.size().width(), originalImg.size().height())
    }
}

object CANNY {
    val threshold = 60.0
    val apertureSize = 3
}

fun main(args: Array<String>) {
    runMainAlgorithm()
//    ocr()
}

fun runMainAlgorithm() {
    // copy original     image into a new one to be used for filter applying
    val modifiedImg = cloneImageFrom(originalImg)

    // 1) convert to greyscale
    cvtColor(originalImg, modifiedImg, CV_BGR2GRAY)

    // 2) apply canny edge detection
    Canny(modifiedImg, modifiedImg, CANNY.threshold, (CANNY.threshold * 1), CANNY.apertureSize, true)

    // 3) apply Hough transform
    val distanceResolutionInPixels: Double = 1.0 // rho
    val angleResolutionInRadians: Double = PI / 180 // theta
    val minimumVotes = 200

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

        if (theta <= PI / 4.0 || theta >= 3.0 * PI / 4.0) {
            // ~vertical line
            if (theta < 0.3141 || theta > 2.5132) {
                val p1 = Point(round(rho / cos(theta)).toInt(), 0) // point of intersection of the line with first row
                val p2 = Point(round((rho - result.rows() * sin(theta)) / cos(theta)).toInt(), result.rows()) // point of intersection of the line with last row
                line(result, p1, p2, Scalar(255.0, 0.0, 255.0, 0.0), 1, LINE_8, 0)
            }
        } else {
            // ~horizontal line
            if ((theta < 1.60 && theta > 1.55)) {
                val p1 = Point(0, round(rho / sin(theta)).toInt()) // point of intersection of the line with first column
                val p2 = Point(result.cols(), round((rho - result.cols() * cos(theta)) / sin(theta)).toInt()) // point of intersection of the line with last column
                line(result, p1, p2, Scalar(255.0, 0.0, 255.0, 0.0), 1, LINE_8, 0)
            }
        }
    }

    // 4) correct rotation w/ average horizontal 0

    // final) show image to screen
    canvasFrame.showImage(imgConverter.convert(toMat8U(result)))
//    canvasFrame.showImage(imgConverter.convert(modifiedImg))
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
    val file = File(getImg("dado.jpg"))
    val image = ImageIO.read(file)
    val r = tess.doOCR(file)
    val words = tess.getWords(image, 0)
    words.forEach { println("|${it.text}|") }
}

fun resizeSelf(img: Mat) = resize(img, img, Size((img.size().width() * resizeRatio).toInt(), (img.size().height() * resizeRatio).toInt()))

fun cloneImageFrom(img: Mat) = Mat(img.size().width(), img.size().height(), IPL_DEPTH_8U, 1)