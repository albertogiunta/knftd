import CANNY.binaryThreshold
import DICT.alignedYMargin
import DICT.dictionaryProperties
import DICT.dictionaryX
import DICT.dictionaryY
import DICT.distanceThresh
import DICT.leven
import DICT.lineMergingYDistance
import DICT.lineMergingYDistanceForValuesAndPossibleMUOnNextLine
import DICT.muSet
import DICT.numberOfRowsToAddToTheActualNumberOfRows
import DICT.shrinkedList
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
    val dictionaryX = listOf("grassi", "carboidrati", "proteine", "sale")
    val words = mutableListOf<Word>()
    val leven = NormalizedLevenshtein()
    val shrinkedList = mutableListOf<Word>()
    val distanceThresh = 0.5
    val lineMergingYDistance = 5
    val lineMergingYDistanceForValuesAndPossibleMUOnNextLine = 5
    val numberOfRowsToAddToTheActualNumberOfRows = 10
    val alignedYMargin = 20
    val muSet = setOf("9", "g", "kcal", "kJ")
}

object IMG {
    val imgName = "libere" + "/" + "olio3" + "." + "jpg"
    val resizeRatio = 0.5
    val originalImgNotResized: Mat = imread(getImg(imgName), CV_LOAD_IMAGE_UNCHANGED)
    val originalImg: Mat = imread(getImg(imgName), CV_LOAD_IMAGE_UNCHANGED).also { it.resizeSelf() }
    val imgConverter = OpenCVFrameConverter.ToMat()
    var properties = mutableListOf<CustomDistance>()
}

object CANNY {
    val threshold = 70.0
    val apertureSize = 3
    val binaryThreshold = 50.0
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
    // copy original image into a new one to be used for filter applying
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

    imgForOCR.show("otsu")

    val rect = getRectForCrop(imgForOCR)

    val croppedOriginalNotResized = crop(originalImgNotResized, rect)

    // NB now original image is cropped to be only the table

    val imgForHough2 = croppedOriginalNotResized.clone()
    imgForHough2.resizeSelf()

    // 1) convert to greyscale
    convertToGreyscale(imgForHough2)
    increaseContrast(imgForHough2)

    // 2) apply canny edge detection
    applyCanny(imgForHough2)

    // 3) apply Hough transform
    applyHough(imgForHough2)

    // 4) correct rotation w/ average horizontal 0
    croppedOriginalNotResized.rotateToTheta()

    croppedOriginalNotResized.clone().resizeSelf().show("Final rotation (resized and cropped)")

    val imgForOCR2 = croppedOriginalNotResized.clone()
    // 1) convert to greyscale
    convertToGreyscale(imgForOCR2)
    // 1 bis) convert to greyscale
    increaseContrast(imgForOCR2)

    // 2) apply otsu binary filter
    applyBinary(imgForOCR2)

    imgForOCR2.clone().resizeSelf().show("After binarization (non otsu)")

    words.clear()
    words.addAll(imgForOCR2.getWords())

    extractNutritionalPropertyNames()

    extractNutritionalPropertiesValues()

    mergePropertiesWithValues()
}

fun Mat.resizeSelf(): Mat {
    resize(this, this, Size((this.size().width() * resizeRatio).toInt(), (this.size().height() * resizeRatio).toInt()))
    return this
}

fun convertToGreyscale(source: Mat, dest: Mat = source) = cvtColor(source, dest, CV_BGR2GRAY)

fun increaseContrast(source: Mat) = source.convertTo(source, -1, 1.5, -100.0)

fun applyCanny(source: Mat, dest: Mat = source) = Canny(source, dest, CANNY.threshold, (CANNY.threshold * 1), CANNY.apertureSize, true)

fun Mat.rotateToTheta() {
    if (finalTheta != 0.0) {
        warpAffine(this, this, getRotationMatrix2D(Point2f((this.size().width() / 2).toFloat(), (this.size().height() / 2).toFloat()), finalTheta, 1.0), this.size())
    }
}

fun applyOtsu(source: Mat, dest: Mat = source) = threshold(source, dest, 0.0, 255.0, THRESH_OTSU)

fun applyBinary(source: Mat, dest: Mat = source) = threshold(source, dest, binaryThreshold, 255.0, THRESH_BINARY)

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

    removeLines(horizontalLinesList)
    removeLines(verticalLinesList)

    finalTheta = if (verticalLinesList.isNotEmpty()) verticalLinesList.map { it.theta.toDegrees() }.average() else 0.0

    val res2 = Mat().also { originalImg.copyTo(it) }
    //draw lines
    horizontalLinesList.forEach { line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0) }
    verticalLinesList.forEach {
        if (it.theta == 0.0) line(res2, it.p1, it.p2, scalar2, 1, LINE_8, 0)
        else line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0)
    }

    houghCounter = houghCounter + 1
}

fun getRectForCrop(source: Mat): Rect {

    words.clear()
    words.addAll(source.getWords())

    val cropOffset = 0
    //Computes the levenstein distance for each word with selected dictionaries which contain the keywords for cropping
    val wordY = words.map { word -> dictionaryY.map { dictWord -> CustomDistance(word, dictWord, leven.distance(word.text.toLowerCase(), dictWord)) }.minBy { it.distance } }.minBy { it!!.distance }!!.ocrWord
    val wordX = words.map { word -> dictionaryX.map { dictWord -> CustomDistance(word, dictWord, leven.distance(word.text.toLowerCase(), dictWord)) }.minBy { it.distance } }.minBy { it!!.distance }!!.ocrWord
    val x = wordX.boundingBox.x - cropOffset
    val y = wordY.boundingBox.y - cropOffset

    return Rect(x, y, source.size().width() - x, source.size().height() - y)
}

fun extractNutritionalPropertyNames() {

    //Foreach word found by tesseract, it saves in a different structure only the words that are equal or similar to the ones in the dictionary
    words
            .forEach { ocrWord ->
                dictionaryProperties
                        .map { dictWord -> CustomDistance(ocrWord, dictWord, leven.distance(ocrWord.text.toLowerCase(), dictWord)) }
                        .filter { it.distance < distanceThresh }
                        .forEach { properties.add(it) }
            }

    //Contains indexes of words that have already been merged with a previous word
    val alreadyMergedPropertiesIndexes = mutableListOf<Int>()
    //Contains merged proprieties (like "acidi grassi saturi")
    val newProperties = mutableListOf<CustomDistance>()

    //Filters the words that match with more than one dictionary word, keeping only the one that matches more (has the smallest distance)
    properties = properties
            .map { main ->
                properties
                        .filter { it.ocrWord == main.ocrWord }
                        .minBy { it.distance }!!
            }
            .distinctBy { it.ocrWord }.toMutableList()



    properties.forEachIndexed { i, _ ->
        //If the word has not been merged already
        if (!alreadyMergedPropertiesIndexes.contains(i)) {
            //Save the Word representing the property
            newProperties.add(properties[i])
            var j = i + 1
            //Check next Words, if they have almost the same Y values it means that they are on the same line, and it add the text to the current Word text
            while (j < properties.size - 1 && Math.abs(properties[i].ocrWord.boundingBox.y - properties[j].ocrWord.boundingBox.y) < lineMergingYDistance) {
                newProperties[newProperties.size - 1] = CustomDistance(properties[j].ocrWord, newProperties.last().dictWord + " " + properties[j].dictWord, newProperties.last().distance)
                alreadyMergedPropertiesIndexes.add(j)
                j++
            }
        }
    }

    properties = newProperties
    newProperties.forEach { println(it) }
}

fun extractNutritionalPropertiesValues() {

    val maxXX = properties.maxBy { it.ocrWord.boundingBox.x + it.ocrWord.boundingBox.width }!!
    //X value from which starting to look for nutritional values
    val maxX = maxXX.ocrWord.boundingBox.x + maxXX.ocrWord.boundingBox.width
    // probably there's no need for this one because properties are already shrinked in lines before
    //val numRows = properties.mapIndexed { i, p -> if (i + 1 < properties.size) Math.abs(p.ocrWord.boundingBox.y - properties[i + 1].ocrWord.boundingBox.y) else 0 }.filter { it > 5 }.count() + 1
    val numRows = properties.size

    // Selects only the Words that have the smallest X values (as many as the number of proprieties (plus a margin of error)
    val allRightValues = words.filter { it.boundingBox.x > maxX }.sortedBy { it.boundingBox.x }.take(numRows + numberOfRowsToAddToTheActualNumberOfRows).sortedBy { it.boundingBox.y }

    //Fixes the error made with measurement units (often alone in one line, often a "9", etc) checking for each Word if the next one is a 9 and merging
    allRightValues.mapIndexed { i, p ->
        if (i + 1 < allRightValues.size) {
            if ((muSet.contains(allRightValues[i + 1].text)) && Math.abs(p.boundingBox.y - allRightValues[i + 1].boundingBox.y) < lineMergingYDistanceForValuesAndPossibleMUOnNextLine) {
                shrinkedList.add(Word(p.text + " g", p.confidence, p.boundingBox))
            } else if (!muSet.contains(p.text)) {
                shrinkedList.add(Word(p.text, p.confidence, p.boundingBox))
            }
        }
    }

//    shrinkedList.sortedBy { it.boundingBox.x }.take(numRows + 5).sortedBy { it.boundingBox.y }.forEach { println(it) }
}

fun mergePropertiesWithValues() {

    val map = mutableMapOf<CustomDistance, Word>()

    //For each property looks for a corresponding value almost on the same Y value (aligned)
    properties.forEach { prop ->
        shrinkedList
                .forEach { value ->
                    println("${prop.dictWord} - ${value.text} |  prop y ${prop.ocrWord.boundingBox.y} | value y ${value.boundingBox.y}")
                    if (Math.abs(value.boundingBox.y - prop.ocrWord.boundingBox.y) < alignedYMargin && (!map.containsKey(prop) || value.boundingBox.x < map[prop]!!.boundingBox.x)) {
                        map.put(prop, value)
                    }
                }
    }

    map.forEach { t, u -> println(t.dictWord + " " + u.text) }

}

data class Line(val rho: Float, val theta: Double, val p1: Point, val p2: Point)

data class CustomDistance(val ocrWord: Word, val dictWord: String, val distance: Double)