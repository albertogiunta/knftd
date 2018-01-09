object FileUtils {

    fun getRes() = "${System.getProperty("user.dir")}/res/"

    fun getImg(imgName: String) = "${getRes()}$imgName"

}