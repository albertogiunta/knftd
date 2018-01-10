import java.awt.Image
import java.awt.image.BufferedImage

object FileUtils {

    fun getRes() = "${System.getProperty("user.dir")}/res/"

    fun getImg(imgName: String) = "${getRes()}$imgName"

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