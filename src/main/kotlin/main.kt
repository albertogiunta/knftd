import DICT.dictionaryProperties
import DICT.dictionaryX
import DICT.dictionaryY
import DICT.distanceThresh
import DICT.leven
import DICT.words
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
import IMG.properties
import IMG.resizeRatio
import LINE_SHRINKING.maxRhoTheresold
import info.debatty.java.stringsimilarity.NormalizedLevenshtein
import net.sourceforge.tess4j.Word
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.*
import org.bytedeco.javacv.OpenCVFrameConverter
import java.lang.Math.*

object DICT {
    val dictionaryProperties = listOf("energia", "energetico", "grassi", "acidi", "saturi", "insaturi", "monoinsaturi", "polinsaturi", "carboidrati", "zuccheri", "proteine", "fibre", "sale", "sodio", "fibre", "fibra", "alimentare", "amido")
    val dictionaryY = listOf("valori", "informazioni", "tabella", "dichiarazione", "nutrizionale", "nutrizionali")
    val dictionaryX = listOf("grassi", "carboidrati", "proteine", "sale", "sodio")
    val words = mutableListOf<Word>()
    val leven = NormalizedLevenshtein()
    val distanceThresh = 0.5
}

object IMG {
    val imgName = "biscotti" + "." + "jpg"
    val resizeRatio = 0.6
    val originalImgNotResized: Mat = imread(getImg(imgName), CV_LOAD_IMAGE_UNCHANGED)
    val originalImg: Mat = imread(getImg(imgName), CV_LOAD_IMAGE_UNCHANGED).also { resizeSelf(it) }
    val imgConverter = OpenCVFrameConverter.ToMat()
    val properties = mutableListOf<CustomDistance>()
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
//    getRectForCrop()
}

fun runMainAlgorithm() {
    // copy original     image into a new one to be used for filter applying
    var imgForHough = originalImg.clone()

    // 1) convert to greyscale
    convertToGreyscale(imgForHough)
    // 1 bis) convert to greyscale
    increaseContrast(imgForHough)

    // 2) apply canny edge detection
    applyCanny(imgForHough)

    // 3) apply Hough transform
    applyHough(imgForHough)

    // 4) correct rotation w/ average horizontal 0
    originalImgNotResized.rotateToTheta()

    // NB now originalImg is rotated, let's re-run the previous steps

    val imgForOCR = originalImgNotResized.clone()
    // 1) convert to greyscale
    convertToGreyscale(imgForOCR)
    // 1 bis) convert to greyscale
    increaseContrast(imgForOCR)

    // 2) apply otsu binary filter
    applyOtsu(imgForOCR)

    val rect = getRectForCrop(imgForOCR)

    val croppedOriginalNotResized = crop(originalImgNotResized, rect)

    // NB now original image is cropped to be only the table

    val imgForHough2 = croppedOriginalNotResized.clone()
    resizeSelf(imgForHough2)

    // 1) convert to greyscale
    convertToGreyscale(imgForHough2)
    increaseContrast(imgForHough2)

    // 2) apply canny edge detection
    applyCanny(imgForHough2)

    // 3) apply Hough transform
    applyHough(imgForHough2)

    // 4) correct rotation w/ average horizontal 0
    croppedOriginalNotResized.rotateToTheta()

    val imgForOCR2 = croppedOriginalNotResized.clone()
    // 1) convert to greyscale
    convertToGreyscale(imgForOCR2)
    // 1 bis) convert to greyscale
    increaseContrast(imgForOCR2)

    // 2) apply otsu binary filter
    applyBinary(imgForOCR2)

    words.clear()
    words.addAll(imgForOCR2.getWords())

    extractNutritionalPropertyNames()

    extractNutritionalPropertiesValues()
}

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
fun applyBinary(source: Mat, dest: Mat = source) = threshold(source, dest, 50.0, 255.0, THRESH_BINARY)

fun crop(source: Mat, rect: Rect): Mat = Mat(source, rect)

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

//    println("Result $houghCounter, NON RIMOSSE ${horizontalLinesList.size} ${verticalLinesList.size}")
    removeLines(horizontalLinesList)
    removeLines(verticalLinesList)
//    println("Result $houghCounter, RIMOSSE ${horizontalLinesList.size} ${verticalLinesList.size}")

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

fun getRectForCrop(source: Mat): Rect {

    words.clear()
    words.addAll(source.getWords())

    val cropOffset = 0
    val wordY = words.map { word -> dictionaryY.map { dictWord -> CustomDistance(word, dictWord, leven.distance(word.text.toLowerCase(), dictWord)) }.minBy { it.distance } }.minBy { it!!.distance }!!.ocrWord
    val wordX = words.map { word -> dictionaryX.map { dictWord -> CustomDistance(word, dictWord, leven.distance(word.text.toLowerCase(), dictWord)) }.minBy { it.distance } }.minBy { it!!.distance }!!.ocrWord
    val x = wordX.boundingBox.x - cropOffset
    val y = wordY.boundingBox.y - cropOffset

    println(x)
    println(y)

    return Rect(x, y, source.size().width() - x, source.size().height() - y)
}

fun extractNutritionalPropertyNames() {

//    words.forEach { println(it) }

    words
            .forEach { ocrWord ->
                dictionaryProperties
                        .map { dictWord -> CustomDistance(ocrWord, dictWord, leven.distance(ocrWord.text.toLowerCase(), dictWord)) }
                        .filter { it.distance < distanceThresh }
                        .forEach { properties.add(it) }
            }

    properties
            .map { main ->
                properties
                        .filter { it.ocrWord == main.ocrWord }
                        .minBy { it.distance }
            }
            .distinctBy { it!!.ocrWord }
//            .forEach { println(it) }
}

fun extractNutritionalPropertiesValues() {

    val maxXX = properties.maxBy { it.ocrWord.boundingBox.x + it.ocrWord.boundingBox.width }!!
    val maxX = maxXX.ocrWord.boundingBox.x + maxXX.ocrWord.boundingBox.width
    val numRows = properties.mapIndexed { i, p -> if (i + 1 < properties.size) Math.abs(p.ocrWord.boundingBox.y - properties[i + 1].ocrWord.boundingBox.y) else 0 }.filter { it > 5 }.count() + 1

    // valori più a sinistra dei nomi delle proprietà nutrizionali
    val allRightValues = words.filter { it.boundingBox.x > maxX }.sortedBy { it.boundingBox.x }.take(numRows + 10).sortedBy { it.boundingBox.y }

    allRightValues.sortedBy { it.boundingBox.y }.forEach { println(it.text) }

    val shrinkedList = mutableListOf<Word>()
    val muSet = setOf("9", "g", "kcal", "kJ")

    allRightValues.mapIndexed { i, p ->
        if (i + 1 < allRightValues.size) {
            if ((muSet.contains(allRightValues[i + 1].text)) && Math.abs(p.boundingBox.y - allRightValues[i + 1].boundingBox.y) < 20) {
                shrinkedList.add(Word(p.text + " g", p.confidence, p.boundingBox))
            } else if (!muSet.contains(p.text)) {
                shrinkedList.add(Word(p.text, p.confidence, p.boundingBox))
            }
        }
    }

    shrinkedList.sortedBy { it.boundingBox.x }.take(numRows + 5).sortedBy { it.boundingBox.y }.forEach { println(it) }
}

data class Line(val rho: Float, val theta: Double, val p1: Point, val p2: Point)

data class CustomDistance(val ocrWord: Word, val dictWord: String, val distance: Double)