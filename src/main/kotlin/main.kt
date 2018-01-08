import FileUtils.getImg
import org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U
import org.bytedeco.javacpp.opencv_core.Mat
import org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.*
import org.bytedeco.javacv.CanvasFrame
import org.bytedeco.javacv.OpenCVFrameConverter

fun main(args: Array<String>) {
    cannyWithMatLoader()
}

fun cannyWithMatLoader() {
    val threshold = 20.0
    val apertureSize = 3
    val converter = OpenCVFrameConverter.ToMat()
    val canvasFrame = CanvasFrame("Canny filter")
    val originalImg = imread(getImg("test2.png"), CV_LOAD_IMAGE_UNCHANGED)
    with(canvasFrame) {
        isResizable = false
        setCanvasSize(originalImg.size().width(), originalImg.size().height())
    }

    val modifiedImg = Mat(originalImg.size().width(), originalImg.size().height(), IPL_DEPTH_8U, 1)
    cvtColor(originalImg, modifiedImg, CV_BGR2GRAY)
    Canny(modifiedImg, modifiedImg, threshold, (threshold * 1), apertureSize, true)

    canvasFrame.showImage(converter.convert(modifiedImg))

}