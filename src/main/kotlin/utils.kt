import IMG.imgConverter
import net.sourceforge.tess4j.ITessAPI
import net.sourceforge.tess4j.Tesseract
import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacv.CanvasFrame
import org.bytedeco.javacv.Java2DFrameConverter
import java.awt.Image
import java.awt.image.BufferedImage
import java.lang.Math.PI

object FileUtils {

    fun getRes() = "${System.getProperty("user.dir")}/res/"

    fun getImg(imgName: String) = "${getRes()}$imgName"

    fun getImgIgnorante(imgName: String) = "${System.getProperty("user.dir")}/$imgName"

}

fun Image.toBufferedImage(): BufferedImage {
    if (this is BufferedImage) {
        return this
    }
    val bufferedImage = BufferedImage(this.getWidth(null), this.getHeight(null), BufferedImage.TYPE_INT_ARGB)

    val graphics2D = bufferedImage.createGraphics()
    graphics2D.drawImage(this, 0, 0, null)
    graphics2D.dispose()

    return bufferedImage
}

fun Double.toDegrees() = this * 180 / PI

fun opencv_core.Mat.toBufferedImage(): BufferedImage = Java2DFrameConverter().convert(imgConverter.convert(this))

fun opencv_core.Mat.toMat8U(doScaling: Boolean = true): opencv_core.Mat {
    val minVal = DoublePointer(Double.MAX_VALUE)
    val maxVal = DoublePointer(Double.MIN_VALUE)
    opencv_core.minMaxLoc(this, minVal, maxVal, null, null, opencv_core.Mat())
    val min = minVal.get(0)
    val max = maxVal.get(0)
    val (scale, offset) = if (doScaling) {
        val s = 255.toDouble() / (max - min)
        Pair(s, -min * s)
    } else Pair(1.toDouble(), 0.0)

    val dest = opencv_core.Mat()
    this.convertTo(dest, opencv_core.CV_8U, scale, offset)
    return dest
}

fun opencv_core.Mat.show(title: String = "") {
    CanvasFrame(title).apply {
        isResizable = true
        setCanvasSize(this@show.size().width(), this@show.size().height())
        showImage(imgConverter.convert(this@show))
    }
}

fun getTesseractInstance(): Tesseract {
    val tess = Tesseract()
    tess.setLanguage("ita")
    tess.setTessVariable("tessedit_pageseg_mode", "11")
    tess.setTessVariable("load_system_dawg", "F")
    tess.setTessVariable("load_freq_dawg", "F")
    tess.setTessVariable("enable_new_segsearch", "1")
    tess.setTessVariable("language_model_penalty_non_dict_word", "10000000")
    tess.setTessVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.'")

    return tess
}

fun opencv_core.Mat.getWords() = getTesseractInstance().getWords(this.toBufferedImage(), ITessAPI.TessPageIteratorLevel.RIL_WORD)

fun opencv_core.Mat.getWordsAndPrint() = getTesseractInstance().getWords(this.toBufferedImage(), ITessAPI.TessPageIteratorLevel.RIL_WORD).forEach { println(it.text) }
