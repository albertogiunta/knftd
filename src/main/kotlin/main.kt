import CANVAS.canvasFrame
import FileUtils.getImg
import IMG.imgConverter
import IMG.originalImg
import net.sourceforge.tess4j.Tesseract
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.*
import org.bytedeco.javacv.CanvasFrame
import org.bytedeco.javacv.OpenCVFrameConverter
import java.io.File

object IMG {
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
    val threshold = 20.0
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

    // 4) correct rotation w/ average horizontal 0

    // final) show image to screen
    canvasFrame.showImage(imgConverter.convert(modifiedImg))
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